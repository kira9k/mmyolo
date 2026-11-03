# Copyright (c) OpenMMLab. All rights reserved.


def configure_yolov8_training_defaults(cfg, overrides=None):
    """Apply the YOLOv8 validation/best-weight policy before runner creation.

    Call after merging CLI options. Explicit CLI settings take precedence;
    ``auto_training_defaults=False`` preserves the complete config as-is.
    Legacy flattened/pruned configs use the same policy as new configs.
    """
    if not cfg.get('auto_training_defaults', True):
        return
    head = cfg.get('model', {}).get('bbox_head', {})
    if head.get('type', '').rsplit('.', 1)[-1] != 'YOLOv8Head':
        return
    train_cfg = cfg.get('train_cfg', {})
    if train_cfg.get('type', '').rsplit('.', 1)[-1] != 'EpochBasedTrainLoop':
        return

    overrides = overrides or {}

    def explicit(path):
        return any(path == key or path.startswith(key + '.')
                   for key in overrides)

    if not explicit('train_cfg.val_interval'):
        train_cfg['val_interval'] = 10
    if not explicit('train_cfg.dynamic_intervals'):
        # MMEngine decides after incrementing epoch, using epoch + 1.
        milestone = max(2, train_cfg['max_epochs'] - 10 + 2)
        train_cfg['dynamic_intervals'] = [(milestone, 1)]

    checkpoint = cfg.get('default_hooks', {}).get('checkpoint')
    if not checkpoint:
        return
    hook_type = checkpoint.get('type', '').rsplit('.', 1)[-1]
    if hook_type == 'CheckpointHook' and not explicit(
            'default_hooks.checkpoint.type'):
        checkpoint['type'] = 'LastEpochCheckpointHook'
        hook_type = 'LastEpochCheckpointHook'
    if hook_type != 'LastEpochCheckpointHook':
        return

    for name, value in dict(
            interval=10,
            save_last_epochs=10,
            save_only_best=True,
            save_best='auto',
            max_keep_ckpts=2).items():
        if not explicit(f'default_hooks.checkpoint.{name}'):
            checkpoint[name] = value
