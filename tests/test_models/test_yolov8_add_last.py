# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config

from mmyolo.models.layers import CSPLayerWithTwoConv, SPPFBottleneck
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]


class TestYOLOv8AddLast(TestCase):

    def setUp(self):
        self.random_state = torch.get_rng_state()
        torch.manual_seed(0)

    def tearDown(self):
        torch.set_rng_state(self.random_state)

    def test_formula_and_gradients(self):
        for count in (0, 1, 3):
            for shortcut in (False, True):
                with self.subTest(count=count, shortcut=shortcut):
                    layer = CSPLayerWithTwoConv(
                        8,
                        8,
                        num_blocks=count,
                        add_identity=shortcut,
                        fusion_mode='add_last').eval()
                    x = torch.randn(2, 8, 8, 8, requires_grad=True)
                    first, last = layer.main_conv(x).chunk(2, dim=1)
                    for block in layer.blocks:
                        last = block(last)
                    expected = layer.final_conv(first + last)
                    actual = layer(x)
                    torch.testing.assert_close(actual, expected)
                    actual.square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertTrue(
                        all(
                            p.grad is not None
                            and torch.isfinite(p.grad).all()
                            for p in layer.parameters()))

    def test_no_direct_or_intermediate_branch_duplication(self):
        x = torch.cat((torch.ones(1, 2, 4, 4), torch.full((1, 2, 4, 4), 2.)),
                      dim=1)
        for count in (0, 1, 3):
            layers = []
            for mode in ('add', 'add_last'):
                layer = CSPLayerWithTwoConv(
                    4, 4, num_blocks=count, fusion_mode=mode)
                layer.main_conv = nn.Identity()
                layer.final_conv = nn.Identity()
                layer.blocks = nn.ModuleList(nn.Identity()
                                             for _ in range(count))
                layers.append(layer)
            torch.testing.assert_close(layers[1](x),
                                       torch.full((1, 2, 4, 4), 3.))
            torch.testing.assert_close(
                layers[0](x), torch.full((1, 2, 4, 4), 1. + 2. * (count + 1)))

    def test_full_focus_input_and_unchanged_parameter_layout(self):
        for size in ('n', 's'):
            with self.subTest(size=size):
                cfg = Config.fromfile(
                    ROOT / 'configs/yolov8' /
                    f'yolov8_{size}_focus_add_last_spp_concat_surgery2.py')
                model = MODELS.build(cfg.model).eval()
                self.assertEqual(cfg.model.backbone.csp_fusion_mode,
                                 'add_last')
                self.assertEqual(cfg.model.neck.csp_fusion_mode, 'add_last')
                self.assertEqual(cfg.model.backbone.spp_fusion_mode, 'concat')
                c2f = [
                    m for m in model.modules()
                    if isinstance(m, CSPLayerWithTwoConv)
                ]
                self.assertTrue(c2f)
                self.assertTrue(all(m.fusion_mode == 'add_last' for m in c2f))
                sppf = [
                    m for m in model.modules()
                    if isinstance(m, SPPFBottleneck)
                ]
                self.assertEqual(len(sppf), 1)
                self.assertEqual(sppf[0].fusion_mode, 'concat')
                self.assertEqual(sppf[0].conv2.conv.in_channels,
                                 4 * sppf[0].conv1.conv.out_channels)
                old_cfg = Config.fromfile(
                    ROOT / 'configs/yolov8' /
                    f'yolov8_{size}_focus_add_spp_concat_surgery2.py')
                old = MODELS.build(old_cfg.model).eval()
                model.load_state_dict(old.state_dict(), strict=True)
                self.assertEqual(
                    sum(p.numel() for p in model.parameters()),
                    sum(p.numel() for p in old.parameters()))
                with torch.no_grad():
                    cls, bbox = model(
                        torch.randn(1, 16, 160, 160), mode='tensor')
                self.assertEqual([tuple(t.shape) for t in cls],
                                 [(1, 1, 80, 80), (1, 1, 40, 40),
                                  (1, 1, 20, 20)])
                self.assertEqual([t.shape[1] for t in bbox], [64] * 3)

    @skipUnless(
        importlib.util.find_spec('onnxruntime'),
        'onnxruntime is not installed')
    def test_sppf_export_has_one_concat_and_no_add(self):
        import onnx
        import onnxruntime as ort
        from projects.easydeploy.tools.export_onnx import replace_maxpool5x5

        with tempfile.TemporaryDirectory() as directory:
            for size in ('n', 's'):
                cfg = Config.fromfile(
                    ROOT / 'configs/yolov8' /
                    f'yolov8_{size}_focus_add_last_spp_concat_surgery2.py')
                model = MODELS.build(cfg.model).eval()
                sppf, = [
                    m for m in model.modules()
                    if isinstance(m, SPPFBottleneck)
                ]
                sample = torch.randn(1, sppf.conv1.conv.in_channels, 20, 20)
                with torch.no_grad():
                    expected = sppf(sample).numpy()
                for pool3 in (False, True):
                    with self.subTest(size=size, pool3=pool3):
                        if pool3:
                            replace_maxpool5x5(sppf)
                        path = str(
                            Path(directory) / f'sppf_{size}_{pool3}.onnx')
                        torch.onnx.export(
                            sppf,
                            sample,
                            path,
                            opset_version=11,
                            input_names=['images'],
                            dynamo=False)
                        graph = onnx.load(path)
                        onnx.checker.check_model(graph)
                        concats = [
                            n for n in graph.graph.node
                            if n.op_type == 'Concat'
                        ]
                        self.assertEqual(len(concats), 1)
                        self.assertEqual(len(concats[0].input), 4)
                        self.assertFalse(
                            any(n.op_type == 'Add' for n in graph.graph.node))
                        pools = [
                            n for n in graph.graph.node
                            if n.op_type == 'MaxPool'
                        ]
                        self.assertEqual(len(pools), 6 if pool3 else 3)
                        for node in pools:
                            kernel = next(a for a in node.attribute
                                          if a.name == 'kernel_shape')
                            self.assertEqual(
                                list(kernel.ints), [3, 3] if pool3 else [5, 5])
                        options = ort.SessionOptions()
                        options.intra_op_num_threads = 1
                        session = ort.InferenceSession(
                            path, options, providers=['CPUExecutionProvider'])
                        actual = session.run(None,
                                             {'images': sample.numpy()})[0]
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-4, atol=2e-5)

    @skipUnless(
        importlib.util.find_spec('onnxruntime'),
        'onnxruntime is not installed')
    def test_onnx_add_count_and_numerical_outputs(self):
        import onnx
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as directory:
            for count in (0, 1, 3):
                for shortcut in (False, True):
                    with self.subTest(count=count, shortcut=shortcut):
                        layer = CSPLayerWithTwoConv(
                            8,
                            8,
                            num_blocks=count,
                            add_identity=shortcut,
                            fusion_mode='add_last',
                            act_cfg=dict(type='ReLU', inplace=False)).eval()
                        x = torch.randn(1, 8, 8, 8)
                        path = str(Path(directory) / 'block.onnx')
                        torch.onnx.export(
                            layer,
                            x,
                            path,
                            opset_version=11,
                            input_names=['images'],
                            dynamo=False)
                        graph = onnx.load(path)
                        onnx.checker.check_model(graph)
                        self.assertEqual(
                            sum(n.op_type == 'Add' for n in graph.graph.node),
                            1 + count * int(shortcut))
                        self.assertFalse(
                            any(n.op_type == 'Concat'
                                for n in graph.graph.node))
                        session = ort.InferenceSession(
                            path, providers=['CPUExecutionProvider'])
                        actual = session.run(None, {'images': x.numpy()})[0]
                        with torch.no_grad():
                            expected = layer(x).numpy()
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-4, atol=2e-5)
