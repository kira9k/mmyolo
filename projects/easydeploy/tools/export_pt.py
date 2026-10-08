import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from mmdet.apis import init_detector
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist
from mmengine.runner import load_checkpoint

# Add MMYOLO ROOT to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from projects.easydeploy.model import DeployModel, MMYOLOBackend  # noqa E402

from mmyolo.models.dense_heads import (YOLOv5HeadModule, YOLOv6HeadModule,
                                      YOLOv7HeadModule, YOLOv8HeadModule)

from edgeai_torchmodelopt import xmodelopt

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
            print_log('Export PyTorch with bbox decoder and NMS ...')
    else:
        args.model_only = True
        print_log(f'Can not export postprocess for {args.backend.lower()}.\n'
                  f'Set "args.model_only=True" default.')
    if args.model_only:
        postprocess_cfg = None
    else:
        postprocess_cfg = ConfigDict(
            pre_top_k=args.pre_topk,
            keep_top_k=args.keep_topk,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            backend=args.backend,
            export_type=args.export_type)

    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)
    input_channels = baseModel.backbone.input_channels
    deploy_cfg = baseModel.cfg.get('deploy_cfg', {})
    raw_output_format = deploy_cfg.get('raw_output_format', 'separate')
    input_divisor = deploy_cfg.get('input_spatial_divisor', 1)
    if (not isinstance(input_divisor, int) or input_divisor < 1
            or any(size % input_divisor for size in args.img_size)):
        raise ValueError('Image dimensions must be divisible by input_spatial_divisor')
    input_img_size = [size // input_divisor for size in args.img_size]

    if args.export_type is None and not args.model_only:
        is_yolov5_or_yolov7 = isinstance(baseModel.bbox_head.head_module, (YOLOv5HeadModule, YOLOv7HeadModule, YOLOv8HeadModule))
        args.export_type = 'YOLOv5' if is_yolov5_or_yolov7 else args.export_type
        postprocess_cfg.export_type = args.export_type
		    
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
        baseModel=baseModel, backend=backend, postprocess_cfg=postprocess_cfg,
        raw_output_format=raw_output_format)
    deploy_model.eval()

    fake_input = torch.randn(args.batch_size, input_channels,
                             *input_img_size).to(args.device)
    # dry run
    deploy_model(fake_input)

    save_pt_path = os.path.join(
        args.work_dir,
        os.path.basename(args.checkpoint).replace('pth', 'pt'))

    if args.model_surgery:
        traced_model = torch.jit.trace(
            deploy_model, fake_input, check_trace=False, strict=False)
        traced_model = torch.jit.freeze(traced_model)

        extra_files = {}
        if deploy_cfg.get('preprocess'):
            preprocess = dict(deploy_cfg['preprocess'], source_size=args.img_size)
            extra_files['preprocess.json'] = json.dumps(preprocess)
        torch.jit.save(traced_model, save_pt_path, _extra_files=extra_files)
        print_log(f'Optimized PyTorch model (traced) saved to {save_pt_path}')
    else:
        torch.save(deploy_model, save_pt_path)

if __name__ == '__main__':
    main()
