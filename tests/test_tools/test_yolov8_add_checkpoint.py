# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from tools.model_converters.yolov8_add_checkpoint import transfer_weights


class TestYOLOv8AddCheckpoint(TestCase):

    def test_rgb_stem_conversion_matches_replicated_gray_input(self):
        source_weight = torch.randn(8, 3, 3, 3)
        target = {'backbone.stem.conv.weight': torch.zeros(8, 1, 3, 3)}
        source = {'backbone.stem.conv.weight': source_weight}
        result, loaded, skipped, initialized = transfer_weights(
            source, target, rgb_to_gray=True)
        gray = torch.randn(2, 1, 12, 16)
        rgb_output = torch.nn.functional.conv2d(
            gray.repeat(1, 3, 1, 1), source_weight)
        gray_output = torch.nn.functional.conv2d(
            gray, result['backbone.stem.conv.weight'])
        torch.testing.assert_close(
            gray_output, rgb_output, atol=1e-5, rtol=1e-5)
        self.assertEqual(loaded, ['backbone.stem.conv.weight'])
        self.assertFalse(skipped)
        self.assertFalse(initialized)

    def test_rgb_stem_conversion_is_opt_in(self):
        source = {'backbone.stem.conv.weight': torch.randn(8, 3, 3, 3)}
        target = {'backbone.stem.conv.weight': torch.zeros(8, 1, 3, 3)}
        result, loaded, skipped, initialized = transfer_weights(source, target)
        self.assertFalse(loaded)
        self.assertIn('backbone.stem.conv.weight', skipped)
        self.assertEqual(initialized, ['backbone.stem.conv.weight'])
        torch.testing.assert_close(result['backbone.stem.conv.weight'],
                                   target['backbone.stem.conv.weight'])

    def test_transfer_matching_tensors_only(self):
        source = {
            'module.same': torch.full((2, 3), 7., dtype=torch.float16),
            'changed': torch.ones(2, 6),
            'removed': torch.ones(1),
        }
        target = {
            'same': torch.zeros(2, 3),
            'changed': torch.full((2, 3), 2.),
            'new_projection': torch.full((2, 2), 3.),
        }
        result, loaded, skipped, initialized = transfer_weights(source, target)
        torch.testing.assert_close(result['same'], torch.full((2, 3), 7.))
        torch.testing.assert_close(result['changed'], target['changed'])
        torch.testing.assert_close(result['new_projection'],
                                   target['new_projection'])
        torch.testing.assert_close(target['same'], torch.zeros(2, 3))
        self.assertEqual(loaded, ['same'])
        self.assertEqual(set(skipped), {'changed', 'removed'})
        self.assertEqual(initialized, ['changed', 'new_projection'])
