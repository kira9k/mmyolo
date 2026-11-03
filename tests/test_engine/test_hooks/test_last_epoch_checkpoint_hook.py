# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmengine.config import Config
from mmengine.hooks import CheckpointHook

from mmyolo.engine.hooks import LastEpochCheckpointHook
from mmyolo.registry import HOOKS


class TestLastEpochCheckpointHook(TestCase):

    def saved_epochs(self, max_epochs, start=0, **kwargs):
        runner = Mock(max_epochs=max_epochs)
        hook = LastEpochCheckpointHook(**kwargs)
        saved = []
        hook._save_checkpoint = Mock(
            side_effect=lambda runner: saved.append(runner.epoch + 1))
        for runner.epoch in range(start, max_epochs):
            hook.after_train_epoch(runner)
        return saved, hook

    def test_schedules_and_no_duplicate_saves(self):
        for total in (50, 100, 500):
            with self.subTest(total=total):
                saved, hook = self.saved_epochs(total, interval=10)
                self.assertEqual(
                    saved,
                    list(range(10, total - 9, 10)) +
                    list(range(total - 9, total + 1)))
                self.assertEqual(len(saved), len(set(saved)))
                self.assertEqual(hook.interval, 10)

    def test_short_run(self):
        saved, _ = self.saved_epochs(5, interval=10)
        self.assertEqual(saved, [1, 2, 3, 4, 5])

    def test_custom_tail_and_force_last(self):
        saved, _ = self.saved_epochs(
            23, interval=10, save_last_epochs=3, save_last=False)
        self.assertEqual(saved, [10, 20, 21, 22, 23])
        saved, _ = self.saved_epochs(23, interval=10, save_last_epochs=0)
        self.assertEqual(saved, [10, 20, 23])
        saved, _ = self.saved_epochs(
            23, interval=10, save_last_epochs=0, save_last=False)
        self.assertEqual(saved, [10, 20])

    def test_save_begin(self):
        saved, _ = self.saved_epochs(50, interval=10, save_begin=45)
        self.assertEqual(saved, list(range(45, 51)))

    def test_resume_inside_tail(self):
        saved, _ = self.saved_epochs(50, start=43, interval=10)
        self.assertEqual(saved, list(range(44, 51)))

    def test_iteration_based_training_unchanged(self):
        saved, _ = self.saved_epochs(50, interval=10, by_epoch=False)
        self.assertEqual(saved, [])

    def test_best_and_retention_delegated(self):
        hook = LastEpochCheckpointHook(
            interval=10, max_keep_ckpts=14, save_best='coco/bbox_mAP')
        self.assertEqual(hook.max_keep_ckpts, 14)
        self.assertEqual(hook.save_best, ['coco/bbox_mAP'])
        self.assertIs(LastEpochCheckpointHook.after_val_epoch,
                      CheckpointHook.after_val_epoch)
        hook._save_best_checkpoint = Mock()
        runner = Mock()
        metrics = {'coco/bbox_mAP': 0.8}
        hook.after_val_epoch(runner, metrics)
        hook._save_best_checkpoint.assert_called_once_with(runner, metrics)

    def test_invalid_tail(self):
        for value in (-1, 1.5, True, '10', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LastEpochCheckpointHook(save_last_epochs=value)

    def test_registry(self):
        hook = HOOKS.build(dict(type='LastEpochCheckpointHook', interval=10))
        self.assertIsInstance(hook, LastEpochCheckpointHook)

    def test_add_last_inherits_schedule(self):
        cfg = Config.fromfile('configs/yolov8/'
                              'yolov8_s_focus_add_last_spp_concat_surgery2.py')
        self.assertEqual(cfg.default_hooks.checkpoint.type,
                         'LastEpochCheckpointHook')
        self.assertEqual(cfg.default_hooks.checkpoint.interval, 10)
        self.assertEqual(cfg.default_hooks.checkpoint.save_last_epochs, 10)
        self.assertEqual(cfg.default_hooks.checkpoint.max_keep_ckpts, 2)
        self.assertTrue(cfg.default_hooks.checkpoint.save_only_best)

    def test_only_best_disables_regular_saves(self):
        saved, hook = self.saved_epochs(
            50, interval=10, save_only_best=True, save_best='auto')
        self.assertEqual(saved, [])
        hook.after_train_iter(Mock(), 0)
        hook._save_checkpoint.assert_not_called()

    def test_only_best_requires_metric(self):
        with self.assertRaises(ValueError):
            LastEpochCheckpointHook(save_only_best=True)

    def test_only_best_replaces_previous_file(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import torch
        from mmengine.logging import MessageHub

        with TemporaryDirectory() as directory:
            runner = Mock(work_dir=directory, iter=0)
            runner.message_hub = MessageHub('only-best-checkpoint-test')

            def save_checkpoint(out_dir, filename, *args, meta, **kwargs):
                torch.save({'meta': meta}, Path(out_dir) / filename)

            runner.save_checkpoint = Mock(side_effect=save_checkpoint)
            hook = LastEpochCheckpointHook(
                save_only_best=True, save_best='coco/bbox_mAP')
            hook.before_train(runner)
            for epoch, score in [(10, 0.5), (20, 0.4), (30, 0.6), (40, 0.6)]:
                runner.epoch = epoch - 1
                hook.after_train_epoch(runner)
                runner.epoch = epoch
                hook.after_val_epoch(runner, {'coco/bbox_mAP': score})
            hook.after_train(runner)
            files = sorted(p.name for p in Path(directory).iterdir())
            self.assertEqual(files, ['best_coco_bbox_mAP_epoch_30.pth'])
            self.assertEqual(runner.save_checkpoint.call_count, 2)
            for call in runner.save_checkpoint.call_args_list:
                self.assertFalse(call.kwargs['save_optimizer'])
                self.assertFalse(call.kwargs['save_param_scheduler'])
