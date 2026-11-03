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
from mmengine.config import Config

from projects.easydeploy.tools import export_onnx


class TestExportONNXArguments(TestCase):

    def parse(self, *arguments):
        with patch.object(
                sys, 'argv',
            ['export_onnx.py', 'config.py', 'checkpoint.pth', *arguments]):
            return export_onnx.parse_args()

    def test_default_uses_config_output_format(self):
        self.assertIsNone(self.parse('--model-only').raw_output_format)
        self.assertIsNone(self.parse('--model-only').maxpool_kernel_size)
        self.assertFalse(self.parse('--model-only').replace_add_with_concat_conv)
        self.assertTrue(self.parse('--replace-add-with-concat-conv').replace_add_with_concat_conv)

    def test_maxpool_option(self):
        for size in (3, 5):
            self.assertEqual(
                self.parse(f'--maxpool-{size}x{size}').maxpool_kernel_size,
                size)
            self.assertEqual(
                self.parse('--maxpool-kernel-size',
                           str(size)).maxpool_kernel_size, size)

    def test_invalid_pooling_options(self):
        for args in (('--maxpool-kernel-size', '7'), ('--maxpool-3x3',
                                                      '--maxpool-5x5'),
                     ('--maxpool-kernel-size', '5', '--maxpool-3x3')):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.parse(*args)

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

    def test_rectangular_focus_export_with_surgery_and_nms(self):
        from mmyolo.registry import MODELS
        from mmyolo.utils import register_all_modules
        register_all_modules()
        root = Path(__file__).resolve().parents[2]
        cfg = Config.fromfile(
            root /
            'configs/yolov8/yolov8_n_focus_add_last_spp_concat_surgery2.py')
        model = MODELS.build(cfg.model).eval()
        sample = torch.rand(1, 16, 120, 160).numpy()
        reference = None
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config, checkpoint = directory / 'model.py', directory / 'model.pth'
            cfg.dump(config)
            torch.save(dict(state_dict=model.state_dict()), checkpoint)
            for model_only in (True, False):
                output_dir = directory / ('raw' if model_only else 'nms')
                for size, replace_add in ((5, False), (3, False), (5, True)):
                    with self.subTest(model_only=model_only, size=size):
                        argv = [
                            'export_onnx.py',
                            str(config),
                            str(checkpoint), '--model-surgery', '2',
                            '--maxpool-kernel-size',
                            str(size), '--img-size', '480', '640',
                            '--simplify', '--work-dir',
                            str(output_dir)
                        ]
                        argv += ['--model-only'] if model_only else [
                            '--export-type', 'YOLOv5'
                        ]
                        if replace_add:
                            argv += ['--replace-add-with-concat-conv']
                        with patch.object(sys, 'argv',
                                          argv), contextlib.redirect_stdout(
                                              io.StringIO()):
                            export_onnx.main()
                        stem = 'model' + ('_addconcatconv' if replace_add else '')
                        path = output_dir / f'{stem}_pool{size}x{size}.onnx'
                        self.assertTrue(path.with_suffix('.prototxt').exists())
                        graph = onnx.load(path)
                        onnx.checker.check_model(graph)
                        if replace_add:
                            replacements = [node for node in graph.graph.node
                                            if node.name.startswith('add_concat_conv_')
                                            and node.op_type == 'Conv']
                            self.assertEqual(len(replacements), 18)
                            if model_only:
                                self.assertFalse(any(node.op_type == 'Add' for node in graph.graph.node))
                        pools = [
                            node for node in graph.graph.node
                            if node.op_type == 'MaxPool'
                        ]
                        self.assertEqual(len(pools), 6 if size == 3 else 3)
                        kernels = [
                            list(
                                next(a for a in node.attribute
                                     if a.name == 'kernel_shape').ints)
                            for node in pools
                        ]
                        self.assertEqual(kernels, [[size, size]] * len(pools))
                        self.assertEqual([
                            d.dim_value for d in
                            graph.graph.input[0].type.tensor_type.shape.dim
                        ], [1, 16, 120, 160])
                        session = ort.InferenceSession(
                            str(path),
                            options,
                            providers=['CPUExecutionProvider'])
                        actual = session.run(None, {'images': sample})
                        if model_only:
                            self.assertEqual(
                                [list(value.shape) for value in actual],
                                [[1, 65, 60, 80], [1, 65, 30, 40],
                                 [1, 65, 15, 20]])
                            if reference is None:
                                reference = actual
                            for value, expected in zip(actual, reference):
                                torch.testing.assert_close(
                                    torch.from_numpy(value),
                                    torch.from_numpy(expected),
                                    rtol=1e-4,
                                    atol=2e-5)
                        else:
                            self.assertEqual(graph.graph.output[0].name,
                                             'detections')
                            self.assertEqual(len(actual), 1)
                            self.assertEqual(actual[0].shape[0], 1)
                            self.assertEqual(actual[0].shape[-1], 6)
                        self.assertTrue(
                            all(
                                torch.isfinite(torch.from_numpy(value)).all()
                                for value in actual))
                self.assertEqual(len(list(output_dir.glob('*.onnx'))), 3)

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
