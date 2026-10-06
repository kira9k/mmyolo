import os
import sys
from tabnanny import check
import warnings
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e   # 0.12.0
#from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import get_symmetric_quantization_config, XNNPACKQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config, QuantizationConfig
from torch.ao.quantization.qconfig import QConfig


import torch
from mmdet.apis import init_detector
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist
from mmengine.runner import load_checkpoint

# Add MMYOLO ROOT to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from projects.easydeploy.model import DeployModel, MMYOLOBackend 

from mmyolo.models.dense_heads import YOLOv5HeadModule, YOLOv7HeadModule, YOLOv8HeadModule, YOLOv6HeadModule, YOLOXHead
from mmyolo.utils.save_model import save_model_proto

from edgeai_torchmodelopt import xmodelopt
from edgeai_torchmodelopt import xonnx

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=torch.jit.ScriptWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)


class UnnormalizeToTensor:
    def __call__(self, tensor):
        return tensor * 255

class CustomImageDataset(Dataset):
    """Кастомный датасет из папки с изображениями"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Путь к папке с изображениями
            transform (callable, optional): Трансформации для изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root, file))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0

def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model

def duplicate_yolov8_cls_convs(
    model: torch.nn.Module,
    output_channels: int = 16
) -> None:
    """
    Заменяет последние cls Conv2d YOLOv8:

        Conv2d(C, 1, 1) -> Conv2d(C, output_channels, 1)

    Каждый новый выходной фильтр является точной копией
    исходного обученного фильтра.
    """

    head_module = model.bbox_head.head_module

    for level, cls_branch in enumerate(head_module.cls_preds):
        old_conv = cls_branch[-1]
        print(old_conv)

        if not isinstance(old_conv, torch.nn.Conv2d):
            raise TypeError(
                f'cls_preds[{level}][-1] должен быть Conv2d, '
                f'но получен {type(old_conv)}'
            )

        print(
            f'До замены cls level {level}: '
            f'weight={tuple(old_conv.weight.shape)}, '
            f'bias={None if old_conv.bias is None else tuple(old_conv.bias.shape)}'
        )

        if old_conv.out_channels != 1:
            raise ValueError(
                f'Ожидался один выходной cls-канал, '
                f'но cls_preds[{level}][-1] имеет '
                f'{old_conv.out_channels} каналов'
            )

        new_conv = torch.nn.Conv2d(
            in_channels=old_conv.in_channels,
            out_channels=output_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )

        # Переносим новую свёртку на то же устройство
        # и в тот же dtype, что и исходная.
        new_conv = new_conv.to(
            device=old_conv.weight.device,
            dtype=old_conv.weight.dtype,
        )

        with torch.no_grad():
            # Было:
            # [1, C, 1, 1]
            #
            # Станет:
            # [16, C, 1, 1]
            new_conv.weight.copy_(
                old_conv.weight.repeat(output_channels, 1, 1, 1)
            )

            if old_conv.bias is not None:
                # Было [1], станет [16].
                new_conv.bias.copy_(
                    old_conv.bias.repeat(output_channels)
                )

        cls_branch[-1] = new_conv

        print(
            f'После замены cls level {level}: '
            f'weight={tuple(new_conv.weight.shape)}, '
            f'bias={None if new_conv.bias is None else tuple(new_conv.bias.shape)}'
        )

def main():
    checkpoint = "/mmyolo/work_dirs/yolov8_n_big_dataset/best_coco_bbox_mAP_epoch_499.pth"
    config = "/mmyolo/configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py"
    #config = "/mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"
    #checkpoint = "/mmyolo/work_dirs/yolov8_21_02_2025_multi_label_false_back_norm_people/best_coco_bbox_mAP_epoch_140.pth"
    device = "cpu"
    backend = MMYOLOBackend(MMYOLOBackend.ONNXRUNTIME)
    model_only = True
    postprocess_cfg = None
    outupt_names = None

    baseModel = build_model_from_cfg(config, checkpoint, device)

    surgery_fn = xmodelopt.surgery.v2.convert_to_lite_fx
    
    baseModel.backbone = surgery_fn(baseModel.backbone)
    baseModel.neck = surgery_fn(baseModel.neck)

    reg_max = None
    proj = None
    baseModel.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(baseModel.bbox_head.head_module)

    load_checkpoint(baseModel, checkpoint, map_location='cpu')

    ###для дублирования весов
    duplicate_yolov8_cls_convs(
        baseModel,
        output_channels=16
    )
    #####
    
    deploy_model = DeployModel(
        baseModel=baseModel, backend=backend, postprocess_cfg=postprocess_cfg)
    deploy_model.eval()


    with torch.no_grad():
        input = torch.rand(1,3,640,640)
        deploy_model(input)

    transform = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        UnnormalizeToTensor()
    ])

    for name, module in deploy_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(name, module)

    input = torch.rand(1, 3, 640, 640).to(torch.device("cpu"))
    exported_model = torch.export.export(deploy_model, (input,)).module()
    quantizer = XNNPACKQuantizer()
    


    quantizer.set_global(get_symmetric_quantization_config())
    
    ##Убираем слои из квантования
    
    quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.2.2", no_quant_config)
    quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.1.2", no_quant_config)
    quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.0.2", no_quant_config)
        
    # quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.2.1.conv", no_quant_config)
    # quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.1.1.conv", no_quant_config)
    # quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.0.1.conv", no_quant_config)
    
    # quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.2.0.conv", no_quant_config)
    # quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.1.0.conv", no_quant_config)
    # quantizer.set_module_name("baseModel.bbox_head.head_module.cls_preds.0.0.conv", no_quant_config)
    
    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model.to(device)

    calib_dataset = CustomImageDataset(
        root_dir="/mmyolo/data/v3/val/images",
        #root_dir="/mmyolo/data/test_data/Sasha/images",  
        transform=transform
    )

    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False, num_workers=20, drop_last=True)

    with torch.no_grad():
        pbar = tqdm(total=len(calib_loader), position=0)
        for images, _ in calib_loader:
            images = images.to(device)
            prepared_model(images)
            pbar.update()

    quantized_model = convert_pt2e(prepared_model)
    quantized_model.to(torch.device("cpu"))
    
    scripted = torch.jit.trace(quantized_model, input)
    scripted.save("/mmyolo/weights/full_quant_no_slice/yolov8_n_int8_ptq_no_slice.jit.pt")

    quantized_model.to(device)
    ref = quantized_model(input)

    quantized_ep = torch.export.export(quantized_model.to(torch.device("cpu")), (input.to(torch.device("cpu")),))
    torch.export.save(quantized_ep, "/mmyolo/weights/full_quant_no_slice/yolov8_n_int8_ptq_no_slice.pt2")




if __name__ == "__main__":
    main()
