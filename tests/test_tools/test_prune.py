# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config

from mmyolo.models.layers import CSPLayerWithTwoConv
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from mmyolo.utils.structural_pruning import (RawYOLOv8, describe_structure,
                                             prune_yolov8,
                                             traceable_csp_splits)
from projects.easydeploy.tools import prune

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]
RGB = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'
HAS_PRUNING = importlib.util.find_spec('torch_pruning') is not None


class TestPruneArguments(TestCase):

    def parse(self, *arguments):
        with patch.object(sys, 'argv', [
                'prune.py', 'model.py', 'source.pth', '--output', 'pruned.pth',
                *arguments
        ]):
            return prune.parse_args()

    def test_defaults_and_alias(self):
        args = self.parse()
        self.assertEqual((args.prune_ratio, args.round_to, args.scope),
                         (0.2, 16, 'backbone-neck'))
        self.assertEqual(args.device, 'cpu')
        self.assertIsNone(args.model_surgery)
        self.assertEqual(self.parse('--model-surgery', '2').model_surgery, 2)
        self.assertEqual(self.parse('--prune_ratio', '0.1').prune_ratio, 0.1)
        self.assertEqual(self.parse('--img-size', '320').img_size, [320, 320])

    def test_invalid_arguments(self):
        for arguments in (('--model-surgery', '1'), ('--model-surgery', '3'),
                          ('--prune-ratio', '0'), ('--prune-ratio', '1'),
                          ('--round-to', '0'), ('--scope', 'unknown'),
                          ('--img-size', '65'), ('--img-size', '32', '32',
                                                 '32'), ('--output',
                                                         'model.pth'),
                          ('--output', 'source.pth'), ('--output',
                                                       'wrong.pt')):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(
                    io.StringIO()), self.assertRaises(SystemExit):
                self.parse(*arguments)


@skipUnless(HAS_PRUNING, 'Install requirements/pruning.txt')
class TestStructuralPruning(TestCase):

    def build(self, config=RGB):
        cfg = Config.fromfile(ROOT / 'configs/yolov8' / config)
        return cfg, MODELS.build(cfg.model).eval()

    def example(self, model):
        return torch.randn(1, model.backbone.input_channels, 64, 96)

    def test_csp_trace_rewrite_is_temporary_and_equivalent(self):
        _, model = self.build()
        x = self.example(model)
        before_keys = set(model.state_dict())
        with torch.no_grad():
            before = RawYOLOv8(model)(x)
            with traceable_csp_splits(model):
                torch.testing.assert_close(
                    RawYOLOv8(model)(x), before, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(
                RawYOLOv8(model)(x), before, rtol=0, atol=0)
        self.assertEqual(set(model.state_dict()), before_keys)
        self.assertFalse(
            any('pruning_first' in n for n, _ in model.named_modules()))

    def test_rgb_focus_add_and_spp_concat(self):
        configs = (RGB, 'yolov8_n_focus_add_surgery2.py',
                   'yolov8_s_focus_add_surgery2.py',
                   'yolov8_n_focus_add_spp_concat_surgery2.py',
                   'yolov8_s_focus_add_spp_concat_surgery2.py')
        for config in configs:
            with self.subTest(config=config):
                cfg, model = self.build(config)
                x = self.example(model)
                first = next(m for m in model.backbone.modules()
                             if isinstance(m, nn.Conv2d))
                first_shape = tuple(first.weight.shape)
                activations = [
                    (n, type(m)) for n, m in model.named_modules()
                    if isinstance(m, (nn.SiLU, nn.ReLU, nn.Hardsigmoid))
                ]
                head_shapes = [
                    tuple(branch[-1].weight.shape)
                    for branch in model.bbox_head.head_module.cls_preds
                ]
                report = prune_yolov8(model, x)
                self.assertLess(report['parameters_after'],
                                report['parameters_before'])
                self.assertLess(report['macs_after'], report['macs_before'])
                self.assertEqual(tuple(first.weight.shape), first_shape)
                self.assertEqual([
                    tuple(branch[-1].weight.shape)
                    for branch in model.bbox_head.head_module.cls_preds
                ], head_shapes)
                self.assertEqual(
                    [(n, type(m)) for n, m in model.named_modules()
                     if isinstance(m, (nn.SiLU, nn.ReLU, nn.Hardsigmoid))],
                    activations)
                cfg.model.pruning_structure = describe_structure(model)
                restored = MODELS.build(cfg.model).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(
                        RawYOLOv8(restored)(x),
                        RawYOLOv8(model)(x),
                        rtol=0,
                        atol=0)
                for m in restored.modules():
                    if isinstance(m, CSPLayerWithTwoConv):
                        self.assertEqual(m.main_conv.conv.out_channels,
                                         2 * m.mid_channels)

    def test_scopes_and_output_head_protection(self):
        for scope in ('backbone', 'all'):
            with self.subTest(scope=scope):
                _, model = self.build()
                report = prune_yolov8(model, self.example(model), scope=scope)
                if scope == 'backbone':
                    self.assertFalse(
                        any(
                            name.startswith('neck')
                            and c['before'][1] != c['after'][1] for name, c in
                            report['changed_convolutions'].items()))
                head = model.bbox_head.head_module
                self.assertTrue(
                    all(b[-1].out_channels == head.num_classes
                        for b in head.cls_preds))
                self.assertTrue(
                    all(b[-1].out_channels == 64 for b in head.reg_preds))

    def test_pruned_model_can_train(self):
        _, model = self.build()
        prune_yolov8(model, self.example(model))
        model.train()
        data = model.data_preprocessor(
            dict(
                inputs=torch.randint(0, 256, (2, 3, 64, 96)),
                data_samples=dict(
                    bboxes_labels=torch.tensor([[0., 0., 8., 8., 40., 56.],
                                                [1., 0., 8., 8., 40., 56.]]))),
            training=True)
        losses = model(**data, mode='loss')
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters() if p.requires_grad))
        torch.optim.SGD(model.parameters(), lr=1e-3).step()

    @skipUnless(
        importlib.util.find_spec('edgeai_torchmodelopt'),
        'edgeai_torchmodelopt is not installed')
    def test_surgery2_pruning_training_and_export(self):
        from edgeai_torchmodelopt import xmodelopt
        from mmdet.apis import init_detector

        cfg, source = self.build()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'source.py'
            checkpoint = Path(directory) / 'source.pth'
            output = Path(directory) / 'pruned.pth'
            cfg.dump(str(config))
            source.backbone = xmodelopt.surgery.v2.convert_to_lite_fx(
                source.backbone)
            source.neck = xmodelopt.surgery.v2.convert_to_lite_fx(source.neck)
            source.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(
                source.bbox_head.head_module)
            torch.save(
                dict(
                    state_dict=source.state_dict(),
                    meta=dict(dataset_meta=dict(classes=('person', )))),
                checkpoint)
            args = [
                'prune.py',
                str(config),
                str(checkpoint), '--img-size', '64', '96', '--model-surgery',
                '2', '--output',
                str(output)
            ]
            with patch.object(sys, 'argv', args):
                prune.main()
            artifact = torch.load(output, weights_only=True)
            self.assertEqual(artifact['meta']['model_surgery'], 2)
            self.assertTrue(
                artifact['pruning']['report']['surgery_reference_checked'])
            updated = Config.fromfile(output.with_suffix('.py'))
            self.assertEqual(updated.pruning_model_surgery, 2)
            loaded = init_detector(
                str(output.with_suffix('.py')), str(output), device='cpu')
            self.assertFalse(
                any(isinstance(m, nn.SiLU) for m in loaded.modules()))
            self.assertTrue(
                any(isinstance(m, nn.ReLU) for m in loaded.modules()))
            x = self.example(loaded)
            with torch.no_grad():
                before = RawYOLOv8(loaded)(x)
            loaded.backbone = xmodelopt.surgery.v2.convert_to_lite_fx(
                loaded.backbone)
            loaded.neck = xmodelopt.surgery.v2.convert_to_lite_fx(loaded.neck)
            loaded.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(
                loaded.bbox_head.head_module)
            loaded.eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    RawYOLOv8(loaded)(x), before, rtol=0, atol=0)
            loaded.train()
            data = loaded.data_preprocessor(
                dict(
                    inputs=torch.randint(0, 256, (2, 3, 64, 96)),
                    data_samples=dict(
                        bboxes_labels=torch.tensor([[0., 0., 8., 8., 40., 56.],
                                                    [1., 0., 8., 8., 40., 56.]
                                                    ]))),
                training=True)
            loss = sum(loaded(**data, mode='loss').values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in loaded.parameters() if p.requires_grad))
            if (importlib.util.find_spec('onnxruntime') is not None
                    and importlib.util.find_spec('onnx') is not None):
                from projects.easydeploy.tools import export_onnx
                arguments = [
                    'export_onnx.py',
                    str(output.with_suffix('.py')),
                    str(output), '--model-only', '--model-surgery', '2',
                    '--raw-output-format', 'combined', '--img-size', '64',
                    '96', '--work-dir',
                    str(Path(directory) / 'onnx')
                ]
                with patch.object(sys, 'argv', arguments):
                    export_onnx.main()
                import onnx
                import onnxruntime as ort
                graph = onnx.load(Path(directory) / 'onnx/pruned.onnx')
                self.assertFalse(
                    any(
                        n.op_type in ('Sigmoid', 'QuantizeLinear',
                                      'DequantizeLinear')
                        for n in graph.graph.node))
                session = ort.InferenceSession(
                    str(Path(directory) / 'onnx/pruned.onnx'),
                    providers=['CPUExecutionProvider'])
                actual = session.run(None, {'images': x.numpy()})
                for expected, result in zip(before, actual):
                    np.testing.assert_allclose(
                        expected.numpy(), result, rtol=1e-4, atol=1e-5)

    @skipUnless(
        importlib.util.find_spec('edgeai_torchmodelopt'),
        'edgeai_torchmodelopt is not installed')
    def test_surgery2_reference_for_focus_variants(self):
        for config in ('yolov8_n_focus_add_surgery2.py',
                       'yolov8_s_focus_add_surgery2.py',
                       'yolov8_n_focus_add_spp_concat_surgery2.py',
                       'yolov8_s_focus_add_spp_concat_surgery2.py'):
            with self.subTest(config=config):
                cfg, model = self.build(config)
                eager = prune.prepare_surgery2_pruning_model(
                    model, cfg, self.example(model))
                self.assertFalse(
                    any(isinstance(m, nn.SiLU) for m in eager.modules()))
                self.assertTrue(
                    any(
                        isinstance(m, CSPLayerWithTwoConv)
                        for m in eager.modules()))

    def test_cli_checkpoint_config_and_fp32_onnx(self):
        from mmdet.apis import init_detector
        cfg, model = self.build()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'source.py'
            checkpoint = Path(directory) / 'source.pth'
            output = Path(directory) / 'pruned.pth'
            cfg.dump(str(config))
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    meta=dict(dataset_meta=dict(classes=('person', )))),
                checkpoint)
            arguments = [
                'prune.py',
                str(config),
                str(checkpoint), '--img-size', '64', '96', '--output',
                str(output)
            ]
            with patch.object(sys, 'argv',
                              arguments), contextlib.redirect_stdout(
                                  io.StringIO()):
                prune.main()
            artifact = torch.load(output, weights_only=True)
            self.assertFalse(artifact['pruning']['report']['quantized'])
            self.assertNotIn('ema_state_dict', artifact)
            loaded = init_detector(
                str(output.with_suffix('.py')), str(output), device='cpu')
            self.assertTrue(
                all(p.dtype == torch.float32 for p in loaded.parameters()))
            if importlib.util.find_spec('onnxruntime') is not None:
                import onnx
                import onnxruntime as ort
                x = self.example(loaded)
                wrapper = RawYOLOv8(loaded).eval()
                path = Path(directory) / 'pruned.onnx'
                torch.onnx.export(
                    wrapper,
                    x,
                    path,
                    opset_version=11,
                    input_names=['images'],
                    output_names=['s8', 's16', 's32'],
                    dynamo=False)
                graph = onnx.load(path)
                onnx.checker.check_model(graph)
                self.assertFalse(
                    any(
                        n.op_type in ('QuantizeLinear', 'DequantizeLinear')
                        for n in graph.graph.node))
                with torch.no_grad():
                    expected = wrapper(x)
                session = ort.InferenceSession(
                    str(path), providers=['CPUExecutionProvider'])
                actual = session.run(None, {'images': x.numpy()})
                for before, after in zip(expected, actual):
                    np.testing.assert_allclose(
                        before.numpy(), after, rtol=1e-4, atol=1e-5)
            with patch.object(sys, 'argv',
                              arguments), self.assertRaises(FileExistsError):
                prune.main()
