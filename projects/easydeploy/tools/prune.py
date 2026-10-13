# Copyright (c) OpenMMLab. All rights reserved.
"""Structurally prune FP32 YOLOv8 while preserving its training surgery."""
import argparse
import json
import sys
from pathlib import Path

import torch
from mmdet.apis import init_detector
from mmengine.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mmyolo.utils import register_all_modules
from mmyolo.utils.structural_pruning import (RawYOLOv8, describe_structure,
                                             prune_yolov8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument(
        '--prune-ratio', '--prune_ratio', type=float, default=0.2)
    parser.add_argument('--round-to', type=int, default=16)
    parser.add_argument(
        '--scope',
        choices=('backbone', 'backbone-neck', 'all'),
        default='backbone-neck')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--model-surgery',
        type=int,
        choices=(0, 2),
        default=None,
        help='Surgery used during training; omit to inherit the pruned config '
        'or use 0 for an ordinary checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not 0 < args.prune_ratio < 1:
        parser.error('--prune-ratio must be between 0 and 1, exclusively')
    if args.round_to < 1:
        parser.error('--round-to must be positive')
    if len(args.img_size) == 1:
        args.img_size *= 2
    if (len(args.img_size) != 2
            or any(size < 32 or size % 32 for size in args.img_size)):
        parser.error(
            '--img-size must contain one or two positive multiples of 32')
    if Path(args.output).suffix != '.pth':
        parser.error('--output must end in .pth')
    output = Path(args.output).resolve()
    sources = {Path(args.config).resolve(), Path(args.checkpoint).resolve()}
    if sources.intersection(
        (output, output.with_suffix('.py'), output.with_suffix('.json'))):
        parser.error('Cannot overwrite the source config or checkpoint')
    return args


def prepare_surgery2_pruning_model(model, cfg, example):
    """Keep an eager C2f topology, verified against the training FX surgery."""
    from edgeai_torchmodelopt import xmodelopt
    from mmyolo.registry import MODELS

    state = model.state_dict()
    model.backbone = xmodelopt.surgery.v2.convert_to_lite_fx(model.backbone)
    model.neck = xmodelopt.surgery.v2.convert_to_lite_fx(model.neck)
    model.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(
        model.bbox_head.head_module)
    model.to(example.device).eval()
    model.load_state_dict(state, strict=True)
    # These configs pass act_cfg through all YOLOv8 compute blocks.
    # Other FX transformations must remain numerically equivalent or fail here.
    for part in (cfg.model.backbone, cfg.model.neck,
                 cfg.model.bbox_head.head_module):
        part.act_cfg = dict(type='ReLU', inplace=False)
    eager = MODELS.build(cfg.model).to(example.device).eval()
    eager.load_state_dict(state, strict=True)
    eager.cfg = cfg
    if hasattr(model, 'dataset_meta'):
        eager.dataset_meta = model.dataset_meta
    with torch.no_grad():
        torch.testing.assert_close(
            RawYOLOv8(eager)(example),
            RawYOLOv8(model)(example),
            rtol=1e-5,
            atol=1e-6)
    return eager


def main():
    args = parse_args()
    register_all_modules()
    output = Path(args.output).resolve()
    paths = [output, output.with_suffix('.py'), output.with_suffix('.json')]
    if not args.force and any(path.exists() for path in paths):
        raise FileExistsError(
            'Output already exists; choose another path or use --force')
    cfg = Config.fromfile(args.config)
    surgery = (
        args.model_surgery if args.model_surgery is not None else cfg.get(
            'pruning_model_surgery', 0))
    if surgery not in (0, 2):
        raise ValueError('Pruning supports training surgery versions 0 and 2')
    model = init_detector(cfg, args.checkpoint, device=args.device).eval()
    checkpoint = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    divisor = cfg.get('deploy_cfg', {}).get('input_spatial_divisor', 1)
    if (not isinstance(divisor, int) or divisor < 1
            or any(size % divisor for size in args.img_size)):
        raise ValueError('Invalid deploy_cfg.input_spatial_divisor')
    torch.manual_seed(0)
    example = torch.randn(
        1,
        model.backbone.input_channels,
        *(size // divisor for size in args.img_size),
        device=args.device)
    if surgery == 2:
        model = prepare_surgery2_pruning_model(model, cfg, example)
    report = prune_yolov8(
        model,
        example,
        ratio=args.prune_ratio,
        round_to=args.round_to,
        scope=args.scope)
    report.update(
        source_config=str(Path(args.config).resolve()),
        source_checkpoint=str(Path(args.checkpoint).resolve()),
        source_image_size=args.img_size,
        model_surgery=surgery,
        surgery_reference_checked=surgery == 2,
        quantized=False,
        finetuned=False)
    structure = describe_structure(model)
    cfg.model.pruning_structure = structure
    cfg.pruning_model_surgery = surgery
    cfg.load_from = str(Path(args.output))
    cfg.resume = False
    cfg.work_dir = str(Path(args.output).parent / 'finetune')
    meta = dict(
        cfg=cfg.pretty_text,
        model_surgery=surgery,
        dataset_meta=checkpoint.get('meta', {}).get('dataset_meta',
                                                    model.dataset_meta))
    # Do not carry over optimizer or EMA tensors with the old channel shapes.
    artifact = dict(
        state_dict={
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        meta=meta,
        pruning=dict(version=1, report=report, structure=structure))
    # Verify architecture reconstruction and all saved weights before writing.
    from mmyolo.registry import MODELS
    reloaded = MODELS.build(cfg.model).to(args.device).eval()
    reloaded.load_state_dict(artifact['state_dict'], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            RawYOLOv8(reloaded)(example),
            RawYOLOv8(model)(example),
            rtol=0,
            atol=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    cfg.dump(str(output.with_suffix('.py')))
    output.with_suffix('.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(f'Parameters: {report["parameters_before"]:,} -> '
          f'{report["parameters_after"]:,} '
          f'(-{100 * report["parameter_reduction"]:.1f}%)')
    print(
        f'Estimated MACs: {report["macs_before"]:,} -> {report["macs_after"]:,}'
    )
    print(f'Output shapes: {report["output_shapes"]}')
    print(f'Saved FP32 checkpoint: {output}')
    print(f'Matching model config: {output.with_suffix(".py")}')
    print('Accuracy and TIDL latency must be evaluated after fine-tuning.')


if __name__ == '__main__':
    main()
