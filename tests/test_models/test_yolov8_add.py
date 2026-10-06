# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch
import torch.nn as nn
from mmdet.structures import DetDataSample
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmyolo.models import (CSPLayerWithTwoConv, SPPFBottleneck, YOLOv8AddPAFPN,
                           YOLOv8CSPDarknet, YOLOv8PAFPN)
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]


class FeatureExtractor(nn.Module):

    def __init__(self, backbone, neck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x):
        return self.neck(self.backbone(x))


class TestYOLOv8Add(TestCase):

    def test_c2f_modes_and_backward(self):
        for mode in ('concat', 'add'):
            for num_blocks in (0, 1, 3):
                with self.subTest(mode=mode, num_blocks=num_blocks):
                    layer = CSPLayerWithTwoConv(
                        16, 24, num_blocks=num_blocks, fusion_mode=mode)
                    x = torch.randn(2, 16, 8, 12, requires_grad=True)
                    out = layer(x)
                    self.assertEqual(out.shape, (2, 24, 8, 12))
                    out.square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertTrue(
                        all(p.grad is not None for p in layer.parameters()))

    def test_c2f_add_preserves_sequential_bottlenecks(self):
        layer = CSPLayerWithTwoConv(16, 24, num_blocks=3, fusion_mode='add')
        layer.eval()
        x = torch.randn(1, 16, 8, 12)
        first, last = layer.main_conv(x).chunk(2, dim=1)
        expected = first + last
        for block in layer.blocks:
            last = block(last)
            expected = expected + last
        torch.testing.assert_close(layer(x), layer.final_conv(expected))

    def test_sppf_add_pooling_branches(self):
        for kernel_sizes in (5, [3, 5, 7], [3, 5, 7, 9]):
            for use_conv_first in (True, False):
                with self.subTest(kernels=kernel_sizes, first=use_conv_first):
                    layer = SPPFBottleneck(
                        16,
                        24,
                        kernel_sizes=kernel_sizes,
                        use_conv_first=use_conv_first,
                        fusion_mode='add')
                    layer.eval()
                    x = torch.randn(2, 16, 8, 12, requires_grad=True)
                    base = layer.conv1(x) if use_conv_first else x
                    branches = [base]
                    if isinstance(kernel_sizes, int):
                        for _ in range(3):
                            branches.append(layer.poolings(branches[-1]))
                    else:
                        branches.extend(pool(base) for pool in layer.poolings)
                    expected = layer.conv2(torch.stack(branches).sum(dim=0))
                    out = layer(x)
                    torch.testing.assert_close(out, expected)
                    self.assertEqual(out.shape, (2, 24, 8, 12))
                    out.square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())

    def test_default_layers_keep_concat_weights_and_outputs(self):
        for cls in (CSPLayerWithTwoConv, SPPFBottleneck):
            default = cls(16, 24).eval()
            explicit = cls(16, 24, fusion_mode='concat').eval()
            explicit.load_state_dict(default.state_dict(), strict=True)
            x = torch.randn(1, 16, 8, 12)
            torch.testing.assert_close(default(x), explicit(x))

    def test_invalid_fusion(self):
        for cls in (CSPLayerWithTwoConv, SPPFBottleneck):
            with self.assertRaises(ValueError):
                cls(16, 24, fusion_mode='invalid')

    def test_neck_channels_scales_and_backward(self):
        cases = [([16, 32, 64], [16, 32, 64], 1.0),
                 ([16, 32, 64], [24, 40, 80], 1.0),
                 ([32, 64, 128], [32, 64, 128], 0.5),
                 ([16, 32, 64, 128], [24, 40, 80, 144], 1.0),
                 ([16, 32], [24, 40], 1.0)]
        for in_channels, out_channels, width in cases:
            for fusion in ('concat', 'add'):
                with self.subTest(channels=in_channels, fusion=fusion):
                    neck = YOLOv8AddPAFPN(
                        in_channels,
                        out_channels,
                        widen_factor=width,
                        num_csp_blocks=1,
                        csp_fusion_mode=fusion)
                    feats = [
                        torch.randn(
                            2,
                            int(c * width),
                            16 // 2**i,
                            24 // 2**i,
                            requires_grad=True)
                        for i, c in enumerate(in_channels)
                    ]
                    outs = neck(feats)
                    for i, out in enumerate(outs):
                        self.assertEqual(out.shape,
                                         (2, int(out_channels[i] * width),
                                          16 // 2**i, 24 // 2**i))
                    sum(out.square().mean() for out in outs).backward()
                    self.assertTrue(
                        all(torch.isfinite(f.grad).all() for f in feats))
                    self.assertTrue(
                        all(p.grad is not None for p in neck.parameters()))

    def test_neck_validation_and_freeze(self):
        for in_channels, out_channels in (([16], [16]), ([16, 32], [16])):
            with self.assertRaises(ValueError):
                YOLOv8AddPAFPN(in_channels, out_channels)
        neck = YOLOv8AddPAFPN([16, 32], [16, 32], freeze_all=True)
        with self.assertRaises(ValueError):
            neck([torch.randn(1, 16, 8, 12)])
        neck.train()
        self.assertTrue(all(not p.requires_grad for p in neck.parameters()))
        self.assertTrue(
            all(not m.training for m in neck.modules()
                if isinstance(m, nn.BatchNorm2d)))

    def test_backbone_neck_has_no_cat(self):
        backbone = YOLOv8CSPDarknet(
            widen_factor=0.125,
            deepen_factor=0.33,
            csp_fusion_mode='add',
            spp_fusion_mode='add')
        neck = YOLOv8AddPAFPN([256, 512, 1024], [256, 512, 1024],
                              widen_factor=0.125,
                              deepen_factor=0.33)
        extractor = FeatureExtractor(backbone, neck)
        x = torch.randn(2, 3, 64, 96, requires_grad=True)
        with patch('torch.cat', side_effect=AssertionError('Unexpected cat')):
            outs = extractor(x)
            sum(out.square().mean() for out in outs).backward()
        self.assertEqual([tuple(out.shape) for out in outs], [(2, 32, 8, 12),
                                                              (2, 64, 4, 6),
                                                              (2, 128, 2, 3)])
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_configs_build_and_head_forward(self):
        for filename in ('yolov8_s_add.py', 'yolov8_s_add_neck_only.py',
                         'yolov8_s_add_fine_tune.py'):
            with self.subTest(config=filename):
                cfg = Config.fromfile(ROOT / 'configs/yolov8' / filename)
                self.assertIsNone(cfg.load_from)
                self.assertFalse(cfg.resume)
                model = MODELS.build(cfg.model).eval()
                with torch.no_grad():
                    outs = model(torch.randn(1, 3, 64, 96), mode='tensor')
                self.assertEqual(len(outs), 2)
                self.assertEqual([tuple(out.shape[-2:]) for out in outs[0]],
                                 [(8, 12), (4, 6), (2, 3)])
                if filename.endswith('fine_tune.py'):
                    self.assertEqual(model.backbone.frozen_stages, -1)
                    self.assertFalse(model.backbone.norm_eval)
                    self.assertIsNone(cfg.param_scheduler)
                    self.assertEqual(
                        cfg.default_hooks.param_scheduler.max_epochs,
                        cfg.train_cfg.max_epochs)
                    for hook in cfg.custom_hooks:
                        HOOKS.build(hook)
                    HOOKS.build(cfg.default_hooks.param_scheduler)

    def test_detector_training_step(self):
        cfg = Config.fromfile(ROOT /
                              'configs/yolov8/yolov8_s_add_fine_tune.py')
        model = MODELS.build(cfg.model).train()
        samples = []
        for _ in range(2):
            sample = DetDataSample(
                metainfo=dict(
                    batch_input_shape=(64, 96),
                    img_shape=(64, 96),
                    ori_shape=(64, 96),
                    scale_factor=(1., 1.)))
            sample.gt_instances = InstanceData(
                bboxes=torch.tensor([[8., 8., 40., 56.]]),
                labels=torch.zeros(1, dtype=torch.long))
            samples.append(sample)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        before = model.backbone.stem.conv.weight.detach().clone()
        losses = model.loss(torch.randn(2, 3, 64, 96), samples)
        self.assertEqual(set(losses), {'loss_cls', 'loss_bbox', 'loss_dfl'})
        self.assertGreater(float(losses['loss_bbox'].detach()), 0.)
        self.assertGreater(float(losses['loss_dfl'].detach()), 0.)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters() if p.requires_grad))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.backbone.stem.conv.weight))

    @skipUnless(importlib.util.find_spec('onnx'), 'onnx is not installed')
    def test_raw_detector_export_and_prototxt(self):
        import onnx
        from google.protobuf import text_format

        from mmyolo.utils.proto import mmyolo_meta_arch_pb2
        from mmyolo.utils.save_model import save_model_proto
        from projects.easydeploy.model import DeployModel, MMYOLOBackend

        cfg = Config.fromfile(ROOT / 'configs/yolov8/yolov8_s_add.py')
        model = MODELS.build(cfg.model).eval()
        deploy = DeployModel(
            baseModel=model, backend=MMYOLOBackend.ONNXRUNTIME).eval()
        x = torch.randn(1, 3, 64, 96)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'detector.onnx')
            torch.onnx.export(deploy, x, path, opset_version=11, dynamo=False)
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            self.assertEqual(len(graph.graph.output), 6)
            self.assertFalse(
                any(node.op_type == 'Concat' for node in graph.graph.node))
            proto_path = save_model_proto(model, graph, x, path)
            proto = text_format.Parse(
                Path(proto_path).read_text(),
                mmyolo_meta_arch_pb2.TIDLMetaArch())
            output_names = [output.name for output in graph.graph.output]
            self.assertEqual(
                list(proto.caffe_ssd[0].class_input), output_names[::2])
            self.assertEqual(
                list(proto.caffe_ssd[0].box_input), output_names[1::2])

    @skipUnless(importlib.util.find_spec('onnx'), 'onnx is not installed')
    def test_onnx_architectural_concat_counts(self):
        import onnx

        for mode, expected_count in (('concat', 13), ('neck_only', 9), ('add',
                                                                        0)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                backbone = YOLOv8CSPDarknet(
                    widen_factor=0.125,
                    deepen_factor=0.33,
                    csp_fusion_mode='add' if mode == 'add' else 'concat',
                    spp_fusion_mode='add' if mode == 'add' else 'concat')
                neck_cls = YOLOv8PAFPN if mode == 'concat' else YOLOv8AddPAFPN
                neck_args = dict(
                    in_channels=[256, 512, 1024],
                    out_channels=[256, 512, 1024],
                    widen_factor=0.125,
                    deepen_factor=0.33)
                if mode == 'neck_only':
                    neck_args['csp_fusion_mode'] = 'concat'
                extractor = FeatureExtractor(backbone, neck_cls(**neck_args))
                extractor.eval()
                path = str(Path(tmp) / 'features.onnx')
                x = torch.randn(1, 3, 64, 96)
                torch.onnx.export(
                    extractor, x, path, opset_version=11, dynamo=False)
                graph = onnx.load(path)
                onnx.checker.check_model(graph)
                count = sum(node.op_type == 'Concat'
                            for node in graph.graph.node)
                self.assertEqual(count, expected_count)
                if importlib.util.find_spec('onnxruntime'):
                    import onnxruntime as ort

                    session = ort.InferenceSession(
                        path, providers=['CPUExecutionProvider'])
                    actual = session.run(
                        None, {session.get_inputs()[0].name: x.numpy()})
                    with torch.no_grad():
                        expected = extractor(x)
                    for exported, eager in zip(actual, expected):
                        torch.testing.assert_close(
                            torch.from_numpy(exported),
                            eager,
                            rtol=1e-4,
                            atol=1e-5)
