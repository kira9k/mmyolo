import argparse
import hashlib
import json
import os
import random
import sys
import torch.nn as nn
import warnings
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
#torchao = 0.12.0
import torch
import torch.nn.functional as F
from mmdet.apis import init_detector
from mmengine.runner import load_checkpoint

# Add MMYOLO ROOT to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from projects.easydeploy.model import DeployModel, MMYOLOBackend
from projects.easydeploy.model.add_replacement import replace_exported_adds
from projects.easydeploy.model.pooling import get_surgery_kwargs, replace_maxpool5x5
from projects.easydeploy.model.quantization import (add_adjacent_convolutions,
                                                    convolution_precisions,
                                                    make_quantizer,
                                                    select_fp32_modules,
                                                    tensor_add_nodes,
                                                    validate_fp32_adds)

from mmyolo.datasets.transforms import LetterResize, YOLOv5KeepRatioResize
from mmyolo.models.dense_heads import YOLOv8HeadModule

from edgeai_torchmodelopt import xmodelopt

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=torch.jit.ScriptWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)


class CustomTransform:

    def __init__(self, upscale_factor=4):
        self.upscale_factor = upscale_factor

    def transform(self, img):
        return nn.PixelUnshuffle(downscale_factor=self.upscale_factor)(
            img.unsqueeze(0)).squeeze(0)


class UnnormalizeToTensor:

    def __call__(self, tensor):
        return tensor * 255


class FocusCalibrationTransform:
    """Match MMYOLO's grayscale resize/letterbox and normalized Focus4 input."""

    def __init__(self, image_size=(640, 640), mean=0., std=255., pad=114):
        if std <= 0:
            raise ValueError('Calibration std must be positive')
        self.image_size = tuple(image_size)
        self.mean = mean
        self.std = std
        scale = (image_size[1], image_size[0])
        self.resize = YOLOv5KeepRatioResize(scale=scale)
        self.letterbox = LetterResize(
            scale=scale, allow_scale_up=False, pad_val=dict(img=pad))

    def __call__(self, image):
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError('Expected a uint8 grayscale calibration image')
        results = dict(img=image, img_shape=image.shape, ori_shape=image.shape)
        results = self.letterbox(self.resize(results))
        image = results['img']
        if image.shape[:2] != self.image_size:
            raise ValueError('Letterbox did not produce the requested size')
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        tensor = (tensor - self.mean) / self.std
        return F.pixel_unshuffle(tensor[None, None], 4)[0].contiguous()


def build_focus_calibration_transform(cfg, image_size):
    preprocess = cfg.model.data_preprocessor
    if (preprocess.type.split('.')[-1] != 'YOLOv5FocusDetDataPreprocessor'
            or cfg.model.backbone.input_channels != 16
            or cfg.get('deploy_cfg', {}).get('input_spatial_divisor') != 4):
        raise ValueError(
            'Expected a Focus4 config with 16 packed input channels')
    mean, std = preprocess.get('mean', [0.]), preprocess.get('std', [255.])
    if len(mean) != 1 or len(std) != 1:
        raise ValueError(
            'Focus4 normalization must have one grayscale channel')
    pad = cfg.deploy_cfg.get('preprocess', {}).get('pad', 114)
    return FocusCalibrationTransform(image_size, mean[0], std[0], pad)


class CustomImageDataset(Dataset):
    """Кастомный датасет из папки с изображениями"""

    def __init__(self, root_dir, transform=None, max_images=None, seed=0):
        """
        Args:
            root_dir (str): Путь к папке с изображениями
            transform (callable, optional): Трансформации для изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root, file))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        self.image_paths.sort()
        random.Random(seed).shuffle(self.image_paths)
        if max_images is not None:
            if max_images < 1:
                raise ValueError('max_images must be positive')
            self.image_paths = self.image_paths[:max_images]
        print(f"Selected {len(self.image_paths)} calibration images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Cannot read calibration image: {img_path}')

        if self.transform:
            image = self.transform(image)

        return image, 0


def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    # init_detector loads permissively; pruning must never silently use partial weights.
    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
    model.eval()
    return model


def duplicate_yolov8_cls_convs(model: torch.nn.Module,
                               output_channels: int = 16) -> None:
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
            raise TypeError(f'cls_preds[{level}][-1] должен быть Conv2d, '
                            f'но получен {type(old_conv)}')

        print(
            f'До замены cls level {level}: '
            f'weight={tuple(old_conv.weight.shape)}, '
            f'bias={None if old_conv.bias is None else tuple(old_conv.bias.shape)}'
        )

        if old_conv.out_channels != 1:
            raise ValueError(f'Ожидался один выходной cls-канал, '
                             f'но cls_preds[{level}][-1] имеет '
                             f'{old_conv.out_channels} каналов')

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

        new_conv = new_conv.to(
            device=old_conv.weight.device,
            dtype=old_conv.weight.dtype,
        )

        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.repeat(output_channels, 1, 1, 1))

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias.repeat(output_channels))

        cls_branch[-1] = new_conv

        print(
            f'После замены cls level {level}: '
            f'weight={tuple(new_conv.weight.shape)}, '
            f'bias={None if new_conv.bias is None else tuple(new_conv.bias.shape)}'
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Quantize a Focus4 YOLOv8 model')
    parser.add_argument(
        '--config',
        default=
        '/mmyolo/configs/yolov8/yolov8_s_focus_add_spp_concat_surgery2.py')
    parser.add_argument(
        '--checkpoint',
        default='/mmyolo/work_dirs/yolov8_s_focus_add_spp_concat_surgery2/'
        'best_coco_bbox_mAP_epoch_10.pth')
    parser.add_argument(
        '--raw-output-format',
        choices=('combined', 'separate'),
        default='combined',
        help='Combine cls/bbox per scale (default) or return separate branches'
    )
    parser.add_argument(
        '--duplicate-cls-channels',
        type=int,
        default=1,
        help='Keep trained cls channels (1, default), or copy a single class '
        'into this many channels (e.g. 16)')
    parser.add_argument('--calib-dir', default='/mmyolo/data/v3/val/images')
    parser.add_argument('--num-calib-images', type=int, default=300)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640])
    parser.add_argument(
        '--replace-add-with-concat-conv', action='store_true',
        help='Replace equal-shape NCHW feature Add by Concat + fixed 1x1 Conv')
    parser.add_argument(
        '--keep-last-convs-fp32',
        action='store_true',
        help='Keep final cls and reg prediction convolutions in FP32')
    add_policy = parser.add_mutually_exclusive_group()
    add_policy.add_argument(
        '--keep-add-fp32',
        action='store_true',
        help='Keep Add/Add+ReLU and nearest convolutions on all input/output '
        'tensor paths in FP32')
    add_policy.add_argument(
        '--add-fp32-only',
        action='store_true',
        help='Keep only Add/Add+ReLU in FP32 without automatically excluding '
        'neighboring convolutions (previous --keep-add-fp32 behavior)')
    parser.add_argument(
        '--exclude-modules',
        action='extend',
        nargs='+',
        default=[],
        metavar='PATTERN',
        help='Keep named Conv2d layers or entire convolution blocks in FP32; '
        'supports shell-style patterns and repeated options')
    parser.add_argument(
        '--list-conv-modules',
        action='store_true',
        help='List convolution names after surgery and exit without calibration'
    )
    pooling = parser.add_mutually_exclusive_group()
    pooling.add_argument(
        '--maxpool-kernel-size',
        type=int,
        choices=(3, 5),
        default=None,
        help='SPPF pooling kernel: 5 keeps original pools; 3 decomposes them. '
        'Omit to keep the installed surgery defaults.')
    pooling.add_argument(
        '--maxpool-3x3',
        dest='maxpool_kernel_size',
        action='store_const',
        const=3,
        help='Alias for --maxpool-kernel-size 3')
    pooling.add_argument(
        '--maxpool-5x5',
        dest='maxpool_kernel_size',
        action='store_const',
        const=5,
        help='Alias for --maxpool-kernel-size 5')
    parser.add_argument(
        '--force', action='store_true', help='Overwrite existing outputs')
    parser.add_argument(
        '--work-dir',
        help='Output directory (default: checkpoint directory/pt)')
    args = parser.parse_args()
    args.keep_add_convs_fp32 = args.keep_add_fp32
    args.keep_add_fp32 = args.keep_add_fp32 or args.add_fp32_only
    if args.duplicate_cls_channels < 1:
        parser.error('--duplicate-cls-channels must be positive')
    if args.num_calib_images < 1 or args.num_workers < 0:
        parser.error(
            'Calibration image count must be positive and workers nonnegative')
    if len(args.img_size) == 1:
        args.img_size *= 2
    if (len(args.img_size) != 2
            or any(size < 32 or size % 32 for size in args.img_size)):
        parser.error(
            '--img-size must contain one or two positive multiples of 32')
    return args


def build_quantization_deploy_model(model,
                                    raw_output_format='combined',
                                    duplicate_cls_channels=1):
    if duplicate_cls_channels < 1:
        raise ValueError('duplicate_cls_channels must be positive')
    if raw_output_format not in ('combined', 'separate'):
        raise ValueError('raw_output_format must be combined or separate')
    if duplicate_cls_channels != 1:
        head_module = model.bbox_head.head_module
        if not isinstance(head_module, YOLOv8HeadModule):
            raise ValueError('Class duplication is supported only for YOLOv8')
        # Validate all scales before changing any trained weights.
        for branch in head_module.cls_preds:
            if (not isinstance(branch[-1], nn.Conv2d)
                    or branch[-1].out_channels != 1):
                raise ValueError(
                    'Class duplication requires single-class cls Conv2d layers'
                )
        duplicate_yolov8_cls_convs(
            model, output_channels=duplicate_cls_channels)
    return DeployModel(
        baseModel=model,
        backend=MMYOLOBackend.ONNXRUNTIME,
        postprocess_cfg=None,
        raw_output_format=raw_output_format)


def get_pt2e_quantization_api():
    """Use torchao when available, or the PT2E API bundled with PyTorch 2.7."""
    try:
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    except ImportError:
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)
        api = 'torch.ao'
    else:
        try:
            from torchao.quantization.pt2e.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config)
        except ImportError:
            from torch.ao.quantization.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config)
        api = 'torchao'
    return (prepare_pt2e, convert_pt2e, XNNPACKQuantizer,
            get_symmetric_quantization_config, api)


def apply_model_surgery(model, maxpool_kernel_size=None):
    kwargs = get_surgery_kwargs(maxpool_kernel_size)
    if maxpool_kernel_size == 3:
        replace_maxpool5x5(model)
    model.backbone = xmodelopt.surgery.v2.convert_to_lite_fx(
        model.backbone, **kwargs)
    model.neck = xmodelopt.surgery.v2.convert_to_lite_fx(model.neck, **kwargs)
    model.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(
        model.bbox_head.head_module)


def get_maxpool_kernels(graph):
    kernels = []
    for node in graph.nodes:
        if node.op == 'call_function' and str(
                node.target).startswith('aten.max_pool2d'):
            kernel = node.args[1] if len(
                node.args) > 1 else node.kwargs['kernel_size']
            kernels.append(
                [kernel, kernel] if isinstance(kernel, int) else list(kernel))
    return kernels


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = (
        Path(args.work_dir)
        if args.work_dir else Path(args.checkpoint).parent / 'pt')
    suffix = f'_int8_{args.raw_output_format}'
    if args.replace_add_with_concat_conv:
        suffix += '_addconcatconv'
    if args.duplicate_cls_channels != 1:
        suffix += f'_cls{args.duplicate_cls_channels}'
    if args.maxpool_kernel_size is not None:
        suffix += f'_pool{args.maxpool_kernel_size}x{args.maxpool_kernel_size}'
    if args.keep_last_convs_fp32:
        suffix += '_fp32last'
    if args.keep_add_fp32:
        suffix += '_fp32add'
        if args.keep_add_convs_fp32:
            suffix += '_convs'
    if args.exclude_modules:
        policy = json.dumps(sorted(set(args.exclude_modules))).encode()
        suffix += '_fp32sel_' + hashlib.sha256(policy).hexdigest()[:8]
    stem = Path(args.checkpoint).stem + suffix
    paths = {
        kind: output_dir / (stem + extension)
        for kind, extension in (('jit', '.jit.pt'), ('pt2', '.pt2'), ('report',
                                                                      '.json'))
    }
    if not args.list_conv_modules and not args.force and any(
            path.exists() for path in paths.values()):
        raise FileExistsError(
            'Quantized outputs already exist; use another --work-dir or --force'
        )

    model = build_model_from_cfg(args.config, args.checkpoint, 'cpu')
    apply_model_surgery(model, args.maxpool_kernel_size)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    deploy = build_quantization_deploy_model(
        model,
        raw_output_format=args.raw_output_format,
        duplicate_cls_channels=args.duplicate_cls_channels).eval()
    if args.list_conv_modules:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                print(
                    f'{name}: {module.in_channels} -> {module.out_channels}, '
                    f'kernel={module.kernel_size}')
        return
    fp32_modules, fp32_convs = select_fp32_modules(model, args.exclude_modules,
                                                   args.keep_last_convs_fp32)
    transform = build_focus_calibration_transform(model.cfg, args.img_size)
    dataset = CustomImageDataset(
        args.calib_dir,
        transform=transform,
        max_images=args.num_calib_images,
        seed=args.seed)
    example = dataset[0][0].unsqueeze(0)
    prepare_pt2e, convert_pt2e, quantizer_cls, qconfig_fn, api = get_pt2e_quantization_api(
    )
    with torch.no_grad():
        fp32_outputs = deploy(example)
    exported = torch.export.export(deploy, (example, )).module()
    add_fp32_convs = (add_adjacent_convolutions(exported)
                      if args.keep_add_convs_fp32 else [])
    add_replacements = (replace_exported_adds(exported)
                         if args.replace_add_with_concat_conv else [])
    if args.replace_add_with_concat_conv:
        print(f'Replaced {len(add_replacements)} feature Add operators by Concat+Conv')
        with torch.no_grad():
            torch.testing.assert_close(exported(example), fp32_outputs,
                                       rtol=1e-4, atol=1e-5)
    replacement_fp32 = [row['module'] for row in add_replacements
                        if args.keep_add_fp32]
    add_fp32_convs = sorted(set(add_fp32_convs))
    fp32_modules = sorted(set(fp32_modules) | set(add_fp32_convs) | set(replacement_fp32))
    fp32_convs = sorted(set(fp32_convs) | set(add_fp32_convs) | set(replacement_fp32))
    if args.keep_add_convs_fp32:
        print(f'Keeping {len(add_fp32_convs)} Add-adjacent convolutions in FP32')
    quantizer = make_quantizer(quantizer_cls, qconfig_fn(), exported,
                               fp32_modules, keep_add_fp32=args.keep_add_fp32)
    prepared = prepare_pt2e(exported, quantizer)
    fp32_add_nodes = validate_fp32_adds(prepared) if args.keep_add_fp32 else []
    if args.keep_add_fp32:
        print(f'Keeping {len(fp32_add_nodes)} tensor Add operators in FP32')
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    with torch.no_grad():
        for images, _ in tqdm(loader, desc='Calibration'):
            prepared(images)
    converted = convert_pt2e(prepared).cpu()
    if args.keep_add_fp32:
        converted_adds = validate_fp32_adds(converted)
        if len(converted_adds) != len(fp32_add_nodes):
            raise RuntimeError('Converted graph changed the number of FP32 Add operators')
    quantization_ops = [
        str(node.target) for node in converted.graph.nodes
        if 'quantized_decomposed.quantize_' in str(node.target)
    ]
    if not quantization_ops:
        raise RuntimeError('PT2E did not insert any quantization operators')
    precisions = convolution_precisions(converted)
    float_names = {
        row['module']
        for row in precisions
        if not row['quantized_weight'] and row['weight_dtype'] == 'float32'
    }
    missing = set(fp32_convs) - float_names
    if missing:
        raise RuntimeError(
            f'Requested FP32 convolutions were not preserved: {sorted(missing)}'
        )
    maxpool_kernels = get_maxpool_kernels(converted.graph)
    if args.maxpool_kernel_size is not None and any(
            kernel != [args.maxpool_kernel_size] * 2
            for kernel in maxpool_kernels):
        raise RuntimeError(
            'Quantized graph does not preserve the requested MaxPool kernel')
    with torch.no_grad():
        outputs = converted(example)
    fp32_flat, _ = torch.utils._pytree.tree_flatten(fp32_outputs)
    flat, _ = torch.utils._pytree.tree_flatten(outputs)
    if (len(flat) != len(fp32_flat) or any(
            value.shape != reference.shape or not torch.isfinite(value).all()
            for value, reference in zip(flat, fp32_flat))):
        raise RuntimeError(
            'Quantized outputs changed shape or contain non-finite values')

    preprocess = dict(
        model.cfg.deploy_cfg.get('preprocess', {}), source_size=args.img_size)
    extra_files = {'preprocess.json': json.dumps(preprocess)}
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(converted, example).save(
        str(paths['jit']), _extra_files=extra_files)
    torch.export.save(
        torch.export.export(converted, (example, )),
        paths['pt2'],
        extra_files=extra_files)
    report = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        model_surgery=2,
        quantizer='XNNPACKQuantizer',
        pt2e_api=api,
        torch_version=torch.__version__,
        input_shape=list(example.shape),
        input_dtype='float32',
        output_shapes=[list(value.shape) for value in flat],
        raw_output_format=args.raw_output_format,
        duplicate_cls_channels=args.duplicate_cls_channels,
        maxpool_kernel_size=args.maxpool_kernel_size,
        maxpool_kernels=maxpool_kernels,
        keep_last_convs_fp32=args.keep_last_convs_fp32,
        keep_add_fp32=args.keep_add_fp32,
        replace_add_with_concat_conv=args.replace_add_with_concat_conv,
        add_concat_conv_replacements=add_replacements,
        keep_add_convs_fp32=args.keep_add_convs_fp32,
        add_adjacent_fp32_convolutions=add_fp32_convs,
        fp32_add_nodes=fp32_add_nodes,
        tensor_add_count=len(tensor_add_nodes(converted.graph)),
        exclude_module_patterns=args.exclude_modules,
        fp32_modules=fp32_modules,
        fp32_convolutions=fp32_convs,
        convolution_precisions=precisions,
        calibration_images=dataset.image_paths,
        seed=args.seed,
        preprocess=preprocess,
        quantization_operator_count=len(quantization_ops))
    paths['report'].write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved: {paths["jit"]}, {paths["pt2"]}, {paths["report"]}')


if __name__ == "__main__":
    main()
