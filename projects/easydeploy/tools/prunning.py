import argparse
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from mmdet.apis import init_detector
from mmengine.runner import load_checkpoint

from edgeai_torchmodelopt import xmodelopt

def parse_args():
    parser = argparse.ArgumentParser(description='Prune YOLOv8 model')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='Ratio of channels to prune')
    parser.add_argument('--output', default='pruned_model.pth', help='Output path for the pruned model')
    return parser.parse_args()


def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model

def main():
    args = parse_args()

    checkpoint = args.checkpoint
    config = args.config

    device = 'cpu' if args.device == 'cpu' else 'cuda:0'
    
    base_model = build_model_from_cfg(args.config, args.checkpoint, args.device)

    surgery_fn = xmodelopt.surgery.v2.convert_to_lite_fx

    base_model.backbone = surgery_fn(base_model.backbone)
    base_model.neck = surgery_fn(base_model.neck)

    base_model.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(base_model.bbox_head.head_module)

    load_checkpoint(base_model, checkpoint, map_location=device)