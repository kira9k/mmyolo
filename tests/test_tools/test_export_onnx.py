# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

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
