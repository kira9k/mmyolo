# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn as nn
from mmengine.config import Config

from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from projects.easydeploy.tools import quantize

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]


class TestQuantizeOptions(TestCase):

    def parse(self, *arguments):
        with patch.object(sys, 'argv', ['quantize.py', *arguments]):
            return quantize.parse_args()

    def build_model(self, variant='n', num_classes=1):
        cfg = Config.fromfile(
            ROOT /
            f'configs/yolov8/yolov8_{variant}_focus_add_spp_concat_surgery2.py'
        )
        cfg.model.bbox_head.head_module.num_classes = num_classes
        return MODELS.build(cfg.model).eval()

    def test_defaults(self):
        args = self.parse()
        self.assertEqual(args.raw_output_format, 'combined')
        self.assertEqual(args.duplicate_cls_channels, 1)

    def test_explicit_options_and_paths(self):
        args = self.parse('--raw-output-format', 'separate',
                          '--duplicate-cls-channels', '16', '--config', 'n.py',
                          '--checkpoint', 'n.pth', '--calib-dir', 'images',
                          '--work-dir', 'quantized')
        self.assertEqual(args.raw_output_format, 'separate')
        self.assertEqual(args.duplicate_cls_channels, 16)
        self.assertEqual(
            (args.config, args.checkpoint, args.calib_dir, args.work_dir),
            ('n.py', 'n.pth', 'images', 'quantized'))

    def test_invalid_arguments(self):
        for arguments in (('--raw-output-format',
                           'unknown'), ('--duplicate-cls-channels', '0'),
                          ('--duplicate-cls-channels', '-16')):
            with self.subTest(arguments=arguments), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.parse(*arguments)

    def test_default_preserves_trained_class_convs(self):
        model = self.build_model()
        convs = [
            branch[-1] for branch in model.bbox_head.head_module.cls_preds
        ]
        weights = [conv.weight.detach().clone() for conv in convs]
        deploy = quantize.build_quantization_deploy_model(model).eval()
        for branch, conv, weight in zip(model.bbox_head.head_module.cls_preds,
                                        convs, weights):
            self.assertIs(branch[-1], conv)
            torch.testing.assert_close(conv.weight, weight, rtol=0, atol=0)
        with torch.no_grad():
            outputs = deploy(torch.rand(1, 16, 16, 24))
        self.assertEqual([list(o.shape) for o in outputs],
                         [[1, 65, 8, 12], [1, 65, 4, 6], [1, 65, 2, 3]])

    def test_output_modes_and_class_copies(self):
        for variant in ('n', 's'):
            for output_format in ('combined', 'separate'):
                for channels in (1, 16):
                    with self.subTest(
                            variant=variant, output_format=output_format,
                            channels=channels), torch.no_grad(), \
                            contextlib.redirect_stdout(io.StringIO()):
                        model = self.build_model(variant)
                        sample = torch.rand(1, 16, 16, 24)
                        reference = quantize.build_quantization_deploy_model(
                            model, raw_output_format='separate')(
                                sample)
                        deploy = quantize.build_quantization_deploy_model(
                            model,
                            raw_output_format=output_format,
                            duplicate_cls_channels=channels)
                        outputs = deploy(sample)
                        for output, (cls, bbox) in zip(outputs, reference):
                            expected_cls = cls.repeat(1, channels, 1, 1)
                            expected = (
                                torch.cat((expected_cls, bbox), dim=1)
                                if output_format == 'combined' else
                                (expected_cls, bbox))
                            torch.testing.assert_close(
                                output, expected, rtol=1e-5, atol=1e-6)

    def test_multiclass_default_keeps_all_classes(self):
        model = self.build_model(num_classes=3)
        deploy = quantize.build_quantization_deploy_model(model).eval()
        with torch.no_grad():
            outputs = deploy(torch.rand(1, 16, 16, 24))
        self.assertTrue(all(output.shape[1] == 67 for output in outputs))

    def test_duplication_rejects_multiclass_without_changing_weights(self):
        model = self.build_model(num_classes=3)
        convs = [
            branch[-1] for branch in model.bbox_head.head_module.cls_preds
        ]
        with self.assertRaisesRegex(ValueError, 'single-class'):
            quantize.build_quantization_deploy_model(
                model, duplicate_cls_channels=16)
        for branch, conv in zip(model.bbox_head.head_module.cls_preds, convs):
            self.assertIs(branch[-1], conv)

    def test_duplication_rejects_other_heads(self):
        model = SimpleNamespace(
            bbox_head=SimpleNamespace(head_module=nn.Identity()))
        with self.assertRaisesRegex(ValueError, 'only for YOLOv8'):
            quantize.build_quantization_deploy_model(
                model, duplicate_cls_channels=16)

    def test_quantized_layout_and_save_load(self):
        try:
            from torchao.quantization.pt2e.quantize_pt2e import (convert_pt2e,
                                                                 prepare_pt2e)
        except ImportError:
            from torch.ao.quantization.quantize_pt2e import (convert_pt2e,
                                                             prepare_pt2e)
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)

        sample = torch.rand(1, 16, 16, 24)
        for output_format in ('combined', 'separate'):
            for channels in (1, 16):
                with self.subTest(output_format=output_format, channels=channels), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        tempfile.TemporaryDirectory() as directory:
                    model = self.build_model()
                    deploy = quantize.build_quantization_deploy_model(
                        model,
                        raw_output_format=output_format,
                        duplicate_cls_channels=channels).eval()
                    exported = torch.export.export(deploy, (sample, )).module()
                    quantizer = XNNPACKQuantizer().set_global(
                        get_symmetric_quantization_config())
                    prepared = prepare_pt2e(exported, quantizer)
                    with torch.no_grad():
                        prepared(sample)
                    converted = convert_pt2e(prepared)
                    expected = converted(sample)
                    for index, output in enumerate(expected):
                        h, w = ((8, 12), (4, 6), (2, 3))[index]
                        if output_format == 'combined':
                            self.assertEqual(
                                list(output.shape), [1, 64 + channels, h, w])
                        else:
                            self.assertEqual(
                                list(output[0].shape), [1, channels, h, w])
                            self.assertEqual(
                                list(output[1].shape), [1, 64, h, w])
                    pt2 = Path(directory) / 'quantized.pt2'
                    torch.export.save(
                        torch.export.export(converted, (sample, )), pt2)
                    loaded = torch.export.load(pt2).module()
                    torch.testing.assert_close(
                        loaded(sample), expected, rtol=0, atol=0)
                    jit_path = str(Path(directory) / 'quantized.jit.pt')
                    torch.jit.trace(converted, sample).save(jit_path)
                    torch.testing.assert_close(
                        torch.jit.load(jit_path)(sample),
                        expected,
                        rtol=0,
                        atol=0)
