# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn as nn
import cv2
import numpy as np
from mmengine.config import Config

from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from mmyolo.utils.structural_pruning import describe_structure, restore_structure
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
        self.assertEqual(args.num_calib_images, 300)
        self.assertEqual(args.num_workers, 0)
        self.assertEqual(args.img_size, [640, 640])
        self.assertFalse(args.force)
        self.assertIsNone(args.maxpool_kernel_size)

    def test_explicit_pooling_options(self):
        for size in (3, 5):
            self.assertEqual(
                self.parse('--maxpool-kernel-size',
                           str(size)).maxpool_kernel_size, size)
            self.assertEqual(
                self.parse(f'--maxpool-{size}x{size}').maxpool_kernel_size,
                size)

    def test_explicit_options_and_paths(self):
        args = self.parse('--raw-output-format', 'separate',
                          '--duplicate-cls-channels', '16', '--config', 'n.py',
                          '--checkpoint', 'n.pth', '--calib-dir', 'images',
                          '--work-dir', 'quantized', '--num-calib-images', '2',
                          '--num-workers', '1', '--seed', '7', '--img-size',
                          '320', '--force')
        self.assertEqual(args.raw_output_format, 'separate')
        self.assertEqual(args.duplicate_cls_channels, 16)
        self.assertEqual(
            (args.config, args.checkpoint, args.calib_dir, args.work_dir),
            ('n.py', 'n.pth', 'images', 'quantized'))
        self.assertEqual(args.num_calib_images, 2)
        self.assertEqual(args.num_workers, 1)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.img_size, [320, 320])
        self.assertTrue(args.force)

    def test_invalid_arguments(self):
        for arguments in (('--raw-output-format',
                           'unknown'), ('--duplicate-cls-channels', '0'),
                          ('--duplicate-cls-channels', '-16'),
                          ('--num-calib-images', '0'), ('--num-workers', '-1'),
                          ('--img-size', '63'), ('--img-size', '0'),
                          ('--img-size', '64', '96',
                           '128'), ('--maxpool-kernel-size',
                                    '7'), ('--maxpool-3x3', '--maxpool-5x5'),
                          ('--maxpool-kernel-size', '5', '--maxpool-3x3')):
            with self.subTest(arguments=arguments), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.parse(*arguments)

    def test_pooling_surgery_api_compatibility(self):
        defaults = {
            'break_maxpool2d_kernel_size_ge_5': True,
            'silu_to_relu': True
        }
        flags_api = SimpleNamespace(
            get_replacement_flag_dict_default=lambda: defaults)
        mapping = {'maxpool_ge_5': object(), 'upsample': object()}
        mapping_api = SimpleNamespace(
            get_replacement_dict_default=lambda: mapping)
        for api in (flags_api, mapping_api):
            for size in (3, 5):
                with self.subTest(
                        api=api,
                        size=size), patch.object(quantize.xmodelopt, 'surgery',
                                                 SimpleNamespace(v2=api)):
                    kwargs = quantize.get_surgery_kwargs(size)
                    replacements = kwargs['replacement_dict']
                    if api is flags_api:
                        self.assertFalse(
                            replacements['break_maxpool2d_kernel_size_ge_5'])
                        self.assertTrue(replacements['silu_to_relu'])
                    else:
                        self.assertNotIn('maxpool_ge_5', replacements)
                        self.assertIs(replacements['upsample'],
                                      mapping['upsample'])
                    self.assertEqual(quantize.get_surgery_kwargs(None), {})
        self.assertTrue(defaults['break_maxpool2d_kernel_size_ge_5'])
        self.assertIn('maxpool_ge_5', mapping)

    def test_explicit_pooling_after_surgery(self):
        for size in (3, 5):
            with self.subTest(size=size), contextlib.redirect_stdout(
                    io.StringIO()):
                cfg = Config.fromfile(
                    ROOT /
                    'configs/yolov8/yolov8_n_focus_add_last_spp_concat_surgery2.py'
                )
                model = MODELS.build(cfg.model).eval()
                sample = torch.rand(1, 16, 16, 24)
                original = quantize.build_quantization_deploy_model(
                    model).eval()
                with torch.no_grad():
                    expected = original(sample)
                quantize.apply_model_surgery(model, size)
                deploy = quantize.build_quantization_deploy_model(model).eval()
                with torch.no_grad():
                    torch.testing.assert_close(
                        deploy(sample), expected, rtol=1e-4, atol=1e-5)
                graph = torch.export.export(deploy, (sample, )).graph
                self.assertEqual(
                    quantize.get_maxpool_kernels(graph),
                    [[size, size]] * (6 if size == 3 else 3))

    def test_focus_calibration_matches_data_preprocessor(self):
        cfg = Config.fromfile(
            ROOT / 'configs/yolov8/yolov8_n_focus_add_spp_concat_surgery2.py')
        transform = quantize.build_focus_calibration_transform(cfg, (64, 96))
        preprocessor = MODELS.build(cfg.model.data_preprocessor)
        for height, width in ((64, 96), (129, 37), (35, 173), (17, 19)):
            with self.subTest(height=height, width=width):
                image = np.arange(
                    height * width, dtype=np.uint8).reshape(height, width)
                results = transform.letterbox(
                    transform.resize(
                        dict(
                            img=image.copy(),
                            img_shape=image.shape,
                            ori_shape=image.shape)))
                reference = preprocessor(
                    dict(
                        inputs=[torch.from_numpy(results['img'].copy())[None]],
                        data_samples=None),
                    training=False)['inputs'][0]
                actual = transform(image)
                self.assertEqual(list(actual.shape), [16, 16, 24])
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                restored = torch.nn.functional.pixel_shuffle(actual[None],
                                                             4)[0, 0]
                torch.testing.assert_close(
                    restored,
                    torch.from_numpy(results['img']).float() / 255)
        cfg.model.data_preprocessor.type = 'YOLOv5DetDataPreprocessor'
        with self.assertRaisesRegex(ValueError, 'Focus4'):
            quantize.build_focus_calibration_transform(cfg, (64, 96))

    def test_calibration_dataset_sampling_and_small_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'nested').mkdir()
            for i in range(5):
                cv2.imwrite(
                    str(root / 'nested' / f'{i}.png'),
                    np.full((32, 32), i, np.uint8))
            transform = quantize.FocusCalibrationTransform((32, 32))
            first = quantize.CustomImageDataset(
                root, transform, max_images=3, seed=9)
            second = quantize.CustomImageDataset(
                root, transform, max_images=3, seed=9)
            self.assertEqual(first.image_paths, second.image_paths)
            self.assertEqual(len(first), 3)
            batches = list(
                torch.utils.data.DataLoader(
                    first, batch_size=1, drop_last=False))
            self.assertEqual(len(batches), 3)
            self.assertEqual(list(batches[0][0].shape), [1, 16, 8, 8])
            (root / 'empty').mkdir()
            with self.assertRaisesRegex(ValueError, 'No images'):
                quantize.CustomImageDataset(root / 'empty')

    def test_pruned_model_main_roundtrip(self):
        cfg = Config.fromfile(
            ROOT / 'configs/yolov8/yolov8_n_focus_add_spp_concat_surgery2.py')
        model = MODELS.build(cfg.model).eval()
        original_parameters = sum(p.numel() for p in model.parameters())
        structure = describe_structure(model)
        block = 'backbone.stage1.0.blocks.0'
        hidden = structure[block + '.conv1.conv']['out_channels'] // 2
        structure[block + '.conv1.conv']['out_channels'] = hidden
        structure[block + '.conv1.bn']['num_features'] = hidden
        structure[block + '.conv2.conv']['in_channels'] = hidden
        restore_structure(model, structure)
        for module in model.modules():
            if isinstance(module,
                          (nn.Conv2d, nn.modules.batchnorm._BatchNorm)):
                module.reset_parameters()
        self.assertLess(
            sum(p.numel() for p in model.parameters()), original_parameters)
        cfg.model.pruning_structure = structure
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, checkpoint_path = root / 'model.py', root / 'model.pth'
            cfg.dump(config_path)
            torch.save(dict(state_dict=model.state_dict()), checkpoint_path)
            images = root / 'images'
            images.mkdir()
            for i in range(2):
                cv2.imwrite(
                    str(images / f'{i}.png'),
                    np.full((47, 83), 80 + i * 40, np.uint8))
            output_dir = root / 'quantized'
            base_argv = [
                'quantize.py', '--config',
                str(config_path), '--checkpoint',
                str(checkpoint_path), '--calib-dir',
                str(images), '--img-size', '64', '96', '--num-calib-images',
                '2', '--work-dir',
                str(output_dir)
            ]
            sample = quantize.FocusCalibrationTransform(
                (64,
                 96))(cv2.imread(str(images / '0.png'),
                                 cv2.IMREAD_GRAYSCALE))[None]
            for size in (None, 3, 5):
                with self.subTest(size=size):
                    argv = base_argv + (['--maxpool-kernel-size',
                                         str(size)] if size else [])
                    stem = 'model_int8_combined' + (f'_pool{size}x{size}'
                                                    if size else '')
                    with patch.object(sys, 'argv',
                                      argv), contextlib.redirect_stdout(
                                          io.StringIO()):
                        quantize.main()
                    report = json.loads(
                        (output_dir / f'{stem}.json').read_text())
                    self.assertEqual(report['input_shape'], [1, 16, 16, 24])
                    self.assertEqual(
                        report['output_shapes'],
                        [[1, 65, 8, 12], [1, 65, 4, 6], [1, 65, 2, 3]])
                    self.assertEqual(len(report['calibration_images']), 2)
                    self.assertGreater(report['quantization_operator_count'],
                                       0)
                    self.assertEqual(report['duplicate_cls_channels'], 1)
                    self.assertEqual(report['maxpool_kernel_size'], size)
                    pt2_extra = {'preprocess.json': ''}
                    pt2 = torch.export.load(
                        output_dir / f'{stem}.pt2',
                        extra_files=pt2_extra).module()
                    self.assertEqual(
                        quantize.get_maxpool_kernels(pt2.graph),
                        report['maxpool_kernels'])
                    jit_extra = {'preprocess.json': ''}
                    jit = torch.jit.load(
                        str(output_dir / f'{stem}.jit.pt'),
                        _extra_files=jit_extra)
                    torch.testing.assert_close(
                        jit(sample), pt2(sample), rtol=0, atol=0)
                    if size:
                        expected_kernels = [[size, size]
                                            ] * (6 if size == 3 else 3)
                        self.assertEqual(report['maxpool_kernels'],
                                         expected_kernels)
                        frozen = torch.jit.freeze(jit.eval())
                        graph = frozen.inlined_graph
                        pools = [
                            n for n in graph.nodes()
                            if n.kind() == 'aten::max_pool2d'
                        ]
                        self.assertEqual(len(pools), len(expected_kernels))
                        self.assertEqual(
                            [list(n.inputs())[1].toIValue() for n in pools],
                            expected_kernels)
                    self.assertEqual(
                        json.loads(pt2_extra['preprocess.json']),
                        report['preprocess'])
                    self.assertEqual(
                        json.loads(jit_extra['preprocess.json']),
                        report['preprocess'])
                    with patch.object(
                            sys, 'argv',
                            argv), self.assertRaises(FileExistsError):
                        quantize.main()
            self.assertEqual(len(list(output_dir.glob('*.pt2'))), 3)

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
