# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from projects.easydeploy.tools import export_onnx


class TestExportONNXArguments(TestCase):

    def parse(self, *arguments):
        with patch.object(
                sys, 'argv',
            ['export_onnx.py', 'config.py', 'checkpoint.pth', *arguments]):
            return export_onnx.parse_args()

    def test_default_uses_config_output_format(self):
        self.assertIsNone(self.parse('--model-only').raw_output_format)
        self.assertFalse(self.parse('--model-only').maxpool_3x3)

    def test_maxpool_option(self):
        self.assertTrue(self.parse('--maxpool-3x3').maxpool_3x3)

    def test_explicit_output_formats(self):
        for value in ('separate', 'combined'):
            with self.subTest(value=value):
                self.assertEqual(
                    self.parse('--model-only', '--raw-output-format',
                               value).raw_output_format, value)

    def test_unknown_output_format_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            self.parse('--model-only', '--raw-output-format', 'unknown')

    def test_output_format_requires_model_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                work_dir=directory,
                backend='onnxruntime',
                model_only=False,
                raw_output_format='separate')
            with patch.object(export_onnx, 'parse_args', return_value=args), \
                    patch.object(export_onnx, 'build_model_from_cfg') as build, \
                    self.assertRaisesRegex(ValueError, 'requires --model-only'):
                export_onnx.main()
            build.assert_not_called()

    def test_transposed_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                work_dir=directory,
                backend='ncnn',
                model_only=True,
                raw_output_format='separate')
            with patch.object(export_onnx, 'parse_args', return_value=args), \
                    patch.object(export_onnx, 'build_model_from_cfg') as build, \
                    self.assertRaisesRegex(ValueError, 'transposed'):
                export_onnx.main()
            build.assert_not_called()

    def test_non_yolov8_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                work_dir=directory,
                backend='onnxruntime',
                model_only=True,
                raw_output_format='separate',
                config='config.py',
                checkpoint='checkpoint.pth',
                device='cpu')
            model = SimpleNamespace(
                backbone=SimpleNamespace(input_channels=3),
                bbox_head=SimpleNamespace(head_module=nn.Identity()),
                cfg={})
            with patch.object(export_onnx, 'parse_args', return_value=args), \
                    patch.object(export_onnx, 'build_model_from_cfg',
                                 return_value=model), \
                    self.assertRaisesRegex(ValueError, 'only for YOLOv8'):
                export_onnx.main()


class TestMaxPoolDecomposition(TestCase):

    def test_equivalent_pooling_including_small_images(self):
        original = nn.Sequential(nn.MaxPool2d(5, stride=1, padding=2)).eval()
        converted = nn.Sequential(nn.MaxPool2d(5, stride=1, padding=2)).eval()
        export_onnx.replace_maxpool5x5(converted)
        self.assertEqual(
            sum(isinstance(m, nn.MaxPool2d) for m in converted.modules()), 2)
        self.assertFalse(converted[0].training)
        for height, width in ((1, 1), (2, 7), (20, 20), (21, 37)):
            with self.subTest(height=height, width=width):
                sample = torch.randn(1, 4, height, width) - 5
                torch.testing.assert_close(
                    converted(sample), original(sample), rtol=0, atol=0)
        export_onnx.replace_maxpool5x5(converted)
        self.assertEqual(
            sum(isinstance(m, nn.MaxPool2d) for m in converted.modules()), 2)

    def test_other_pooling_settings_remain_unchanged(self):
        settings = [
            dict(kernel_size=3, stride=1, padding=1),
            dict(kernel_size=5, stride=2, padding=2),
            dict(kernel_size=5, stride=1, padding=0),
            dict(kernel_size=5, stride=1, padding=2, dilation=2),
            dict(kernel_size=5, stride=1, padding=2, ceil_mode=True),
            dict(kernel_size=5, stride=1, padding=2, return_indices=True)
        ]
        for kwargs in settings:
            with self.subTest(kwargs=kwargs):
                pool = nn.MaxPool2d(**kwargs)
                model = nn.Sequential(pool)
                export_onnx.replace_maxpool5x5(model)
                self.assertIs(model[0], pool)

    def test_shared_sppf_pool_exports_six_3x3_nodes(self):
        from mmyolo.models.layers import SPPFBottleneck
        model = SPPFBottleneck(16, 16).eval()
        sample = torch.randn(1, 16, 20, 20)
        with torch.no_grad():
            expected = model(sample)
        export_onnx.replace_maxpool5x5(model)
        with torch.no_grad():
            torch.testing.assert_close(model(sample), expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sppf.onnx'
            torch.onnx.export(
                model, sample, path, input_names=['images'], opset_version=11)
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            pools = [
                node for node in graph.graph.node if node.op_type == 'MaxPool'
            ]
            self.assertEqual(len(pools), 6)
            for node in pools:
                attrs = {
                    a.name: onnx.helper.get_attribute_value(a)
                    for a in node.attribute
                }
                self.assertEqual(attrs['kernel_shape'], [3, 3])
            session = ort.InferenceSession(
                str(path), providers=['CPUExecutionProvider'])
            actual = torch.from_numpy(
                session.run(None, {'images': sample.numpy()})[0])
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
