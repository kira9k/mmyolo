# Copyright (c) OpenMMLab. All rights reserved.
"""Prepare a warm-start checkpoint for a modified YOLOv8 architecture."""
import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import CheckpointLoader, save_checkpoint

from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules


def transfer_weights(source_state, target_state, rgb_to_gray=False):
    """Copy matching tensors; retain initialization for changed/new layers."""
    result = target_state.copy()
    loaded = []
    skipped = {}
    for source_key, value in source_state.items():
        key = source_key
        if key.startswith('module.'):
            key = key[7:]
        if key not in target_state:
            skipped[source_key] = 'not present in target model'
        elif (rgb_to_gray and key == 'backbone.stem.conv.weight'
              and value.ndim == 4 and value.shape[1] == 3
              and target_state[key].shape
              == (value.shape[0], 1, *value.shape[2:])):
            # Preserve the RGB stem response to a replicated grayscale input.
            result[key] = value.to(
                dtype=target_state[key].dtype, device='cpu').sum(
                    dim=1, keepdim=True)
            loaded.append(key)
        elif value.shape != target_state[key].shape:
            skipped[source_key] = (f'shape {tuple(value.shape)} -> '
                                   f'{tuple(target_state[key].shape)}')
        else:
            result[key] = value.to(dtype=target_state[key].dtype, device='cpu')
            loaded.append(key)
    initialized = sorted(set(target_state) - set(loaded))
    return result, loaded, skipped, initialized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='target model config')
    parser.add_argument('checkpoint', help='source MMYOLO checkpoint')
    parser.add_argument('output', help='new warm-start checkpoint')
    parser.add_argument(
        '--rgb-to-gray',
        action='store_true',
        help='sum RGB stem kernels to initialize a one-channel input')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='target config overrides, e.g. model.backbone.widen_factor=0.25')
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')

    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint = CheckpointLoader.load_checkpoint(
        args.checkpoint, map_location='cpu')
    state_dict, loaded, skipped, initialized = transfer_weights(
        checkpoint.get('state_dict', checkpoint),
        model.state_dict(),
        rgb_to_gray=args.rgb_to_gray)
    if not loaded:
        raise ValueError(
            'No compatible tensors found in the source checkpoint')

    # Optimizer, epoch and EMA state belong to the old architecture.
    prepared = dict(
        state_dict=state_dict,
        meta=dict(
            cfg=cfg.pretty_text,
            source_checkpoint=str(args.checkpoint),
            rgb_to_gray=args.rgb_to_gray,
            transferred_tensors=len(loaded),
            initialized_tensors=initialized))
    dataset_meta = checkpoint.get('meta', {}).get('dataset_meta')
    if (dataset_meta
            and len(dataset_meta.get('classes',
                                     ())) == model.bbox_head.num_classes):
        prepared['meta']['dataset_meta'] = dataset_meta
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(prepared, str(output))
    print(f'Transferred {len(loaded)} tensors; '
          f'initialized {len(initialized)} tensors; '
          f'skipped {len(skipped)} source tensors.')
    for key, reason in skipped.items():
        print(f'  Skipped {key}: {reason}')
    for key in initialized:
        print(f'  Initialized {key}')
    print(f'Saved {output}. Use load_from, not resume, to start training.')


if __name__ == '__main__':
    main()
