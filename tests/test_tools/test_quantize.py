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
        self.assertFalse(args.keep_last_convs_fp32)
        self.assertFalse(args.keep_add_fp32)
        self.assertFalse(args.keep_add_convs_fp32)
        self.assertFalse(args.replace_add_with_concat_conv)
        self.assertEqual(args.exclude_modules, [])
        self.assertFalse(args.list_conv_modules)

    def test_selective_quantization_options(self):
        args = self.parse('--keep-last-convs-fp32', '--exclude-modules',
                          'backbone', 'neck', '--exclude-modules',
                          'bbox_head.head_module.cls_preds.*.2')
        self.assertTrue(args.keep_last_convs_fp32)
        self.assertEqual(
            args.exclude_modules,
            ['backbone', 'neck', 'bbox_head.head_module.cls_preds.*.2'])
        self.assertTrue(self.parse('--list-conv-modules').list_conv_modules)
        self.assertTrue(self.parse('--keep-add-fp32').keep_add_fp32)
        self.assertTrue(self.parse('--keep-add-fp32').keep_add_convs_fp32)
        self.assertTrue(self.parse('--add-fp32-only').keep_add_fp32)
        self.assertFalse(self.parse('--add-fp32-only').keep_add_convs_fp32)
        self.assertTrue(
            self.parse(
                '--replace-add-with-concat-conv').replace_add_with_concat_conv)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(
                io.StringIO()):
            self.parse('--keep-add-fp32', '--add-fp32-only')

    def test_add_adjacent_convolutions_follow_tensor_paths(self):

        class BranchModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d(3, 8, 1)
                self.left = nn.Conv2d(8, 8, 1)
                self.right = nn.Conv2d(8, 4, 1)
                self.post = nn.Conv2d(4, 8, 1)
                self.post_other = nn.Conv2d(4, 8, 1)
                self.far = nn.Conv2d(16, 8, 1)

            def forward(self, x):
                x = self.stem(x)
                a, _ = torch.chunk(torch.relu(self.left(x)), 2, dim=1)
                b = torch.nn.functional.max_pool2d(self.right(x), 3, 1, 1)
                z = torch.relu(a + b)
                return self.far(
                    torch.cat([torch.relu(self.post(z)),
                               self.post_other(z)],
                              dim=1))

        prepare, convert, cls, qconfig, _ = quantize.get_pt2e_quantization_api(
        )
        sample = torch.rand(1, 3, 8, 12)
        exported = torch.export.export(BranchModel().eval(),
                                       (sample, )).module()
        roots = quantize.add_adjacent_convolutions(exported)
        self.assertEqual(roots, ['left', 'post', 'post_other', 'right'])
        q = quantize.make_quantizer(cls, qconfig(), exported, roots, True)
        prepared = prepare(exported, q)
        quantize.validate_fp32_adds(prepared)
        prepared(sample)
        converted = convert(prepared)
        rows = quantize.convolution_precisions(converted)
        self.assertEqual(
            {r['module']
             for r in rows if not r['quantized_weight']}, set(roots))
        self.assertEqual({r['module']
                          for r in rows if r['quantized_weight']},
                         {'stem', 'far'})

    def test_fp32_add_patterns_and_quantized_convs(self):

        class AddModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.left = nn.Conv2d(3, 8, 3, padding=1)
                self.right = nn.Conv2d(3, 8, 1)

            def forward(self, x):
                a, b = self.left(x), self.right(x)
                return a + b, torch.relu(a + b)

        prepare, convert, cls, qconfig, _ = quantize.get_pt2e_quantization_api(
        )
        sample = torch.rand(1, 3, 8, 12)
        model = AddModel().eval()
        original_patterns = list(cls.STATIC_OPS)
        for keep in (False, True):
            with self.subTest(keep=keep):
                exported = torch.export.export(model, (sample, )).module()
                q = quantize.make_quantizer(cls, qconfig(), exported, [], keep)
                prepared = prepare(exported, q)
                adds = quantize.tensor_add_nodes(prepared.graph)
                self.assertEqual(len(adds), 2)
                if keep:
                    self.assertEqual(
                        len(quantize.validate_fp32_adds(prepared)), 2)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'annotated'):
                        quantize.validate_fp32_adds(prepared)
                prepared(sample)
                converted = convert(prepared)
                self.assertTrue(
                    all(row['quantized_weight']
                        for row in quantize.convolution_precisions(converted)))
                if keep:
                    for node in quantize.tensor_add_nodes(converted.graph):
                        self.assertFalse(
                            any('quantized_decomposed.quantize_' in str(
                                user.target) for user in node.users))
                    for value in converted(sample):
                        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(cls.STATIC_OPS, original_patterns)

    def test_fp32_add_with_yolov8_and_selective_convs(self):
        prepare, convert, cls, qconfig, _ = quantize.get_pt2e_quantization_api(
        )
        for variant, last in [('n', False), ('n', True), ('s', False)]:
            with self.subTest(variant=variant, last=last), \
                    contextlib.redirect_stdout(io.StringIO()):
                model = self.build_model(variant)
                quantize.apply_model_surgery(model, 5)
                deploy = quantize.build_quantization_deploy_model(model).eval()
                roots, expected = quantize.select_fp32_modules(
                    model, keep_last=last)
                sample = torch.rand(1, 16, 16, 24)
                exported = torch.export.export(deploy, (sample, )).module()
                adjacent = quantize.add_adjacent_convolutions(exported)
                self.assertGreater(len(adjacent), 0)
                roots = sorted(set(roots) | set(adjacent))
                expected = sorted(set(expected) | set(adjacent))
                q = quantize.make_quantizer(cls, qconfig(), exported, roots,
                                            True)
                prepared = prepare(exported, q)
                adds = quantize.validate_fp32_adds(prepared)
                self.assertGreater(len(adds), 0)
                prepared(sample)
                converted = convert(prepared)
                self.assertEqual(
                    len(quantize.validate_fp32_adds(converted)), len(adds))
                rows = quantize.convolution_precisions(converted)
                self.assertEqual(
                    {r['module']
                     for r in rows if not r['quantized_weight']},
                    set(expected))
                self.assertTrue(any(r['quantized_weight'] for r in rows))

    def test_fp32_module_selection_and_validation(self):
        model = self.build_model()
        roots, convs = quantize.select_fp32_modules(model, keep_last=True)
        self.assertEqual(len(roots), 6)
        self.assertEqual(roots, convs)
        self.assertTrue(all(name.endswith('.2') for name in convs))
        roots, convs = quantize.select_fp32_modules(model,
                                                    ['baseModel.bbox_head'])
        self.assertEqual(roots, ['bbox_head'])
        self.assertEqual(len(convs), 18)
        _, convs = quantize.select_fp32_modules(
            model, ['bbox_head.head_module.cls_preds.*.2'])
        self.assertEqual(len(convs), 3)
        for pattern in ('not_a_layer',
                        'bbox_head.head_module.cls_preds.0.0.bn'):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(
                    ValueError, 'matches'):
                quantize.select_fp32_modules(model, [pattern])

    def test_selective_quantization_preserves_float_convolutions(self):
        prepare, convert, quantizer_cls, qconfig, _ = quantize.get_pt2e_quantization_api(
        )
        cases = [
            ('n', [], True, 'combined', 1),
            ('n', ['bbox_head.head_module.cls_preds.*.2'], False, 'combined',
             1),
            ('n', ['bbox_head'], False, 'combined', 1),
            ('n', ['backbone.stage1.0'], False, 'combined', 1),
            ('n', [], True, 'separate', 16),
            ('s', [], True, 'combined', 1),
        ]
        for variant, patterns, last, output_format, channels in cases:
            with self.subTest(variant=variant, patterns=patterns, last=last,
                              output_format=output_format, channels=channels), \
                    contextlib.redirect_stdout(io.StringIO()):
                model = self.build_model(variant)
                quantize.apply_model_surgery(model, 5)
                deploy = quantize.build_quantization_deploy_model(
                    model, output_format, channels).eval()
                roots, expected = quantize.select_fp32_modules(
                    model, patterns, last)
                sample = torch.rand(1, 16, 16, 24)
                exported = torch.export.export(deploy, (sample, )).module()
                quantizer = quantize.make_quantizer(quantizer_cls, qconfig(),
                                                    exported, roots)
                prepared = prepare(exported, quantizer)
                prepared(sample)
                converted = convert(prepared)
                rows = quantize.convolution_precisions(converted)
                actual = {
                    row['module']
                    for row in rows if not row['quantized_weight']
                    and row['weight_dtype'] == 'float32'
                }
                self.assertEqual(actual, set(expected))
                self.assertTrue(any(row['quantized_weight'] for row in rows))
                if last:
                    from projects.easydeploy.model.quantization import convolution_module_name
                    for node in converted.graph.nodes:
                        if node.op == 'call_function' and str(
                                node.target) == 'aten.conv2d.default':
                            if convolution_module_name(node) in actual:
                                self.assertFalse(
                                    any('quantized_decomposed.quantize_' in
                                        str(user.target)
                                        for user in node.users))
                values, _ = torch.utils._pytree.tree_flatten(converted(sample))
                self.assertTrue(
                    all(torch.isfinite(value).all() for value in values))

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
                          ('--maxpool-kernel-size', '5',
                           '--maxpool-3x3'), ('--exclude-modules', )):
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
            with patch.object(sys, 'argv', base_argv + ['--list-conv-modules']), \
                    patch.object(quantize, 'CustomImageDataset') as dataset, \
                    patch.object(quantize, 'get_pt2e_quantization_api') as api, \
                    contextlib.redirect_stdout(io.StringIO()) as listing:
                quantize.main()
            self.assertIn('bbox_head.head_module.cls_preds.0.2',
                          listing.getvalue())
            dataset.assert_not_called()
            api.assert_not_called()
            self.assertFalse(output_dir.exists())
            variants = [
                (None, False, False, False, False),
                (3, False, False, False, False),
                (5, False, False, False, False),
                (5, True, False, False, False),
                (5, False, True, False, False),
                (5, True, True, False, False),
                (5, False, True, True, False),
                (5, False, False, False, True),
                (5, False, True, False, True),
            ]
            for size, keep_last, keep_add, add_only, replace_add in variants:
                with self.subTest(
                        size=size,
                        keep_last=keep_last,
                        keep_add=keep_add,
                        add_only=add_only):
                    argv = base_argv + (['--maxpool-kernel-size',
                                         str(size)] if size else [])
                    if replace_add:
                        argv += ['--replace-add-with-concat-conv']
                    stem = 'model_int8_combined' + (
                        '_addconcatconv' if replace_add else
                        '') + (f'_pool{size}x{size}' if size else '')
                    if keep_last:
                        argv += ['--keep-last-convs-fp32']
                        stem += '_fp32last'
                    if keep_add:
                        argv += [
                            '--add-fp32-only'
                            if add_only else '--keep-add-fp32'
                        ]
                        stem += '_fp32add'
                        if not add_only:
                            stem += '_convs'
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
                    self.assertEqual(report['keep_last_convs_fp32'], keep_last)
                    self.assertEqual(report['keep_add_fp32'], keep_add)
                    self.assertEqual(report['replace_add_with_concat_conv'],
                                     replace_add)
                    if replace_add:
                        self.assertGreater(
                            len(report['add_concat_conv_replacements']), 0)
                        self.assertEqual(report['tensor_add_count'], 0)
                    self.assertEqual(report['keep_add_convs_fp32'], keep_add
                                     and not add_only)
                    if keep_add and not replace_add:
                        self.assertGreater(len(report['fp32_add_nodes']), 0)
                        self.assertEqual(report['tensor_add_count'],
                                         len(report['fp32_add_nodes']))
                    expected_float = set(
                        report['add_adjacent_fp32_convolutions'])
                    if keep_add:
                        expected_float.update(
                            row['module']
                            for row in report['add_concat_conv_replacements'])
                    if keep_last:
                        expected_float.update(report['fp32_convolutions'])
                        self.assertTrue(
                            all(f'bbox_head.head_module.{branch}.{scale}.2' in
                                expected_float
                                for branch in ('cls_preds', 'reg_preds')
                                for scale in range(3)))
                    self.assertEqual(
                        set(report['fp32_convolutions']), expected_float)
                    if keep_add and not add_only:
                        self.assertGreater(
                            len(report['add_adjacent_fp32_convolutions']), 0)
                    pt2_extra = {'preprocess.json': ''}
                    pt2 = torch.export.load(
                        output_dir / f'{stem}.pt2',
                        extra_files=pt2_extra).module()
                    self.assertEqual(
                        quantize.get_maxpool_kernels(pt2.graph),
                        report['maxpool_kernels'])
                    precisions = quantize.convolution_precisions(pt2)
                    self.assertEqual(
                        sum(not row['quantized_weight'] for row in precisions),
                        len(expected_float))
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
            self.assertEqual(
                len(list(output_dir.glob('*.pt2'))), len(variants))

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
                            # Copying filters can select a different FP32 CPU convolution kernel.
                            torch.testing.assert_close(
                                output, expected, rtol=1e-4, atol=1e-5)

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
