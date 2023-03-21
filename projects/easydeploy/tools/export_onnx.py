import argparse
import os
import sys
import warnings
from io import BytesIO
from pathlib import Path

import onnx
import torch
from mmdet.apis import init_detector
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist
from mmengine.runner import load_checkpoint

# Add MMYOLO ROOT to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from projects.easydeploy.model import DeployModel, MMYOLOBackend  # noqa E402

from mmyolo.models.dense_heads import YOLOv5HeadModule, YOLOv7HeadModule, YOLOv8HeadModule, YOLOv6HeadModule, YOLOXHead
from mmyolo.utils.save_model import save_model_proto

from edgeai_torchmodelopt import xmodelopt
from edgeai_torchmodelopt import xonnx

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=torch.jit.ScriptWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)

def str_or_none(v):
    return None if v.lower() in ('none', 'null', '') else v


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--model-surgery',
        type=int,
        default=0,
        help='create lite version of a model by applying a set of fx based transformations on the model')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--model-only', action='store_true', help='Export model only')
    parser.add_argument(
        '--work-dir', default='./work_dir', help='Path to save export model')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='Image size of height and width')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify onnx model by onnx-sim')
    parser.add_argument(
        '--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument(
        '--backend',
        type=str,
        default='onnxruntime',
        help='Backend for export onnx')
    parser.add_argument(
        '--pre-topk',
        type=int,
        default=1000,
        help='Postprocess pre topk bboxes feed into NMS')
    parser.add_argument(
        '--keep-topk',
        type=int,
        default=100,
        help='Postprocess keep topk bboxes out of NMS')
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.65,
        help='IoU threshold for NMS')
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.25,
        help='Score threshold for NMS')
    parser.add_argument(
        '--export-type',
        type=str_or_none,
        default=None,
        help='Make output compatible with the desired format (for example: MMDetection, YOLOv5). None: for default format.')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args

def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model

def main():
    args = parse_args()
    mkdir_or_exist(args.work_dir)
    backend = MMYOLOBackend(args.backend.lower())
    if backend in (MMYOLOBackend.ONNXRUNTIME, MMYOLOBackend.OPENVINO,
                   MMYOLOBackend.TENSORRT8, MMYOLOBackend.TENSORRT7):
        if not args.model_only:
            print_log('Export ONNX with bbox decoder and NMS ...')
    else:
        args.model_only = True
        print_log(f'Can not export postprocess for {args.backend.lower()}.\n'
                  f'Set "args.model_only=True" default.')
    if args.model_only:
        postprocess_cfg = None
        output_names = None
    else:
        postprocess_cfg = ConfigDict(
            pre_top_k=args.pre_topk,
            keep_top_k=args.keep_topk,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            backend=args.backend,
            export_type=args.export_type)
        
        if args.export_type in (None, 'MMYOLO'):
            output_names = ['num_dets', 'boxes', 'scores', 'labels']
        elif args.export_type == 'MMDetection':
            output_names = ['dets', 'labels']
        elif args.export_type == 'YOLOv5':
            output_names = ['detections']
        else:
           output_names = ['num_dets', 'boxes', 'scores', 'labels']

    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)

    if args.export_type is None:
        is_yolov5_or_yolov7 = isinstance(baseModel.bbox_head.head_module, (YOLOv5HeadModule, YOLOv7HeadModule, YOLOv8HeadModule))
        args.export_type = 'YOLOv5' if is_yolov5_or_yolov7 else args.export_type
        postprocess_cfg.export_type = args.export_type
        output_names = ['detections']
		    
    if args.model_surgery:
        surgery_fn = xmodelopt.surgery.v1.convert_to_lite_model if args.model_surgery == 1 \
                     else (xmodelopt.surgery.v2.convert_to_lite_fx if args.model_surgery == 2 else None)
        
        if isinstance(baseModel.bbox_head.head_module, (YOLOv6HeadModule)):
            #For YOLOv6 model, the flow is different as reparameterization has to happen before surgery as it is lost during fx based transformation
            deploy_model = DeployModel(
                baseModel=baseModel, backend=backend, postprocess_cfg=postprocess_cfg)
            deploy_model.baseModel.backbone = surgery_fn(deploy_model.baseModel.backbone)
            deploy_model.baseModel.neck = surgery_fn(deploy_model.baseModel.neck)
            # Only head_module of head goes through model_surgery as it contains all compute layers
            # deploy_model.baseModel.bbox_head.head_module = surgery_fn(deploy_model.baseModel.bbox_head.head_module)
            deploy_model.baseModel.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(deploy_model.baseModel.bbox_head.head_module)
        else:
            baseModel.backbone = surgery_fn(baseModel.backbone)
            baseModel.neck = surgery_fn(baseModel.neck)
            
            # Only head_module of head goes through model_surgery as it contains all compute layers
            if hasattr(baseModel.bbox_head.head_module, 'reg_max') and hasattr(baseModel.bbox_head.head_module, 'proj'):
                reg_max = baseModel.bbox_head.head_module.reg_max
                proj = baseModel.bbox_head.head_module.proj
            else:
                reg_max = None
                proj = None
            if isinstance(baseModel.bbox_head.head_module, (YOLOv8HeadModule,)):
                baseModel.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(baseModel.bbox_head.head_module)
            if not isinstance(baseModel.bbox_head.head_module, (YOLOv5HeadModule, YOLOv7HeadModule, YOLOv8HeadModule)):
                baseModel.bbox_head.head_module = surgery_fn(baseModel.bbox_head.head_module)
            if reg_max is not None and proj is not None:
                baseModel.bbox_head.head_module.reg_max = reg_max
                baseModel.bbox_head.head_module.proj = proj

    load_checkpoint(baseModel, args.checkpoint, map_location='cpu')

    deploy_model = DeployModel(
        baseModel=baseModel, backend=backend, postprocess_cfg=postprocess_cfg)
    deploy_model.eval()

    fake_input = torch.randn(args.batch_size, 3,
                             *args.img_size).to(args.device)
    # dry run
    fake_outputs = deploy_model(fake_input)

    save_onnx_path = os.path.join(
        args.work_dir,
        os.path.basename(args.checkpoint).replace('pth', 'onnx'))
    # export onnx
    with BytesIO() as f:
        torch.onnx.export(
            deploy_model,
            fake_input,
            f,
            input_names=['images'],
            output_names=output_names,
            opset_version=args.opset)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)

        # Fix tensorrt onnx output shape, just for view
        if not args.model_only and backend in (MMYOLOBackend.TENSORRT8,
                                               MMYOLOBackend.TENSORRT7):
            shapes = [
                args.batch_size, 1, args.batch_size, args.keep_topk, 4,
                args.batch_size, args.keep_topk, args.batch_size,
                args.keep_topk
            ]
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))
    if args.simplify:
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print_log(f'Simplify failure: {e}')
    onnx.save(onnx_model, save_onnx_path)
    print_log(f'ONNX export success, save into {save_onnx_path}')

    # check the layers names and shorten it required.
    if args.model_surgery:
        xonnx.prune_layer_names(save_onnx_path, save_onnx_path, opset_version=args.opset)

    onnx_model = onnx.load(save_onnx_path)
    save_prototxt_path = save_model_proto(baseModel,
                     onnx_model,
                     fake_input,
                     save_onnx_path,
                     output_names=output_names,
                     export_type=args.export_type)

    print(f'Prototxt export success, save into {save_prototxt_path}')

if __name__ == '__main__':
    main()
