# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmengine.config import Config
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.utils.data import DataLoader

from mmyolo.utils.training_defaults import configure_yolov8_training_defaults


class TestTrainingDefaults(TestCase):

    def config(self, total=100):
        return Config(
            dict(
                model=dict(bbox_head=dict(type='YOLOv8Head')),
                train_cfg=dict(
                    type='EpochBasedTrainLoop',
                    max_epochs=total,
                    val_interval=10,
                    dynamic_intervals=[(490, 1)]),
                default_hooks=dict(
                    checkpoint=dict(
                        type='CheckpointHook',
                        interval=10,
                        max_keep_ckpts=2,
                        save_best='auto'))))

    def test_legacy_config(self):
        cfg = self.config()
        configure_yolov8_training_defaults(cfg)
        self.assertEqual(cfg.train_cfg.dynamic_intervals, [(92, 1)])
        self.assertEqual(cfg.default_hooks.checkpoint.type,
                         'LastEpochCheckpointHook')
        self.assertTrue(cfg.default_hooks.checkpoint.save_only_best)
        self.assertEqual(cfg.default_hooks.checkpoint.save_best, 'auto')

    def test_actual_mmengine_validation_epochs(self):
        for total in (5, 10, 23, 50, 100, 500):
            with self.subTest(total=total):
                cfg = self.config(total)
                configure_yolov8_training_defaults(cfg)
                args = dict(cfg.train_cfg)
                args.pop('type')
                loop = EpochBasedTrainLoop(Mock(), DataLoader([0]), **args)
                validated = []
                for epoch in range(1, total + 1):
                    loop._epoch = epoch
                    loop._decide_current_val_interval()
                    if epoch % loop.val_interval == 0 or epoch == total:
                        validated.append(epoch)
                expected = sorted(
                    set(range(10, total + 1, 10))
                    | set(range(max(1, total - 9), total + 1)))
                self.assertEqual(validated, expected)

    def test_cli_max_epochs_is_used(self):
        cfg = self.config(500)
        overrides = {'train_cfg.max_epochs': 100}
        cfg.merge_from_dict(overrides)
        configure_yolov8_training_defaults(cfg, overrides)
        self.assertEqual(cfg.train_cfg.dynamic_intervals, [(92, 1)])

    def test_explicit_cli_overrides(self):
        cfg = self.config()
        overrides = {
            'train_cfg.val_interval': 1,
            'train_cfg.dynamic_intervals': [],
            'default_hooks.checkpoint.save_only_best': False,
            'default_hooks.checkpoint.max_keep_ckpts': -1,
            'default_hooks.checkpoint.save_best': 'coco/bbox_mAP_50',
        }
        cfg.merge_from_dict(overrides)
        configure_yolov8_training_defaults(cfg, overrides)
        self.assertEqual(cfg.train_cfg.val_interval, 1)
        self.assertEqual(cfg.train_cfg.dynamic_intervals, [])
        self.assertFalse(cfg.default_hooks.checkpoint.save_only_best)
        self.assertEqual(cfg.default_hooks.checkpoint.max_keep_ckpts, -1)
        self.assertEqual(cfg.default_hooks.checkpoint.save_best,
                         'coco/bbox_mAP_50')

    def test_parent_cli_override(self):
        cfg = self.config()
        overrides = {
            'train_cfg':
            dict(
                type='EpochBasedTrainLoop',
                max_epochs=5,
                val_interval=1,
                dynamic_intervals=[])
        }
        cfg.merge_from_dict(overrides)
        configure_yolov8_training_defaults(cfg, overrides)
        self.assertEqual(cfg.train_cfg.dynamic_intervals, [])

    def test_opt_out(self):
        cfg = self.config()
        cfg.auto_training_defaults = False
        before = cfg.to_dict()
        configure_yolov8_training_defaults(cfg)
        self.assertEqual(cfg.to_dict(), before)

    def test_other_models_and_loops_unchanged(self):
        for head, loop in [('YOLOv5Head', 'EpochBasedTrainLoop'),
                           ('YOLOv8Head', 'IterBasedTrainLoop')]:
            cfg = self.config()
            cfg.model.bbox_head.type = head
            cfg.train_cfg.type = loop
            before = cfg.to_dict()
            configure_yolov8_training_defaults(cfg)
            self.assertEqual(cfg.to_dict(), before)

    def test_custom_checkpoint_hook_unchanged(self):
        cfg = self.config()
        cfg.default_hooks.checkpoint.type = 'CustomCheckpointHook'
        before = dict(cfg.default_hooks.checkpoint)
        configure_yolov8_training_defaults(cfg)
        self.assertEqual(cfg.default_hooks.checkpoint, before)

    def test_explicit_builtin_checkpoint_hook(self):
        cfg = self.config()
        configure_yolov8_training_defaults(
            cfg, {'default_hooks.checkpoint.type': 'CheckpointHook'})
        self.assertNotIn('save_only_best', cfg.default_hooks.checkpoint)

    def test_effective_config_can_be_serialized(self):
        cfg = self.config()
        configure_yolov8_training_defaults(cfg)
        restored = Config.fromstring(cfg.pretty_text, '.py')
        self.assertEqual(restored.to_dict(), cfg.to_dict())
