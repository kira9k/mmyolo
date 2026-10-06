# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

import torch
import torch.nn as nn
from mmengine.config import Config

from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]


class TestYOLOv8AddSurgery2(TestCase):

    def build_model(self, filename):
        cfg = Config.fromfile(ROOT / 'configs/yolov8' / filename)
        return cfg, MODELS.build(cfg.model).eval()

    def assert_rgb_relu(self, cfg, model):
        stem = next(m for m in model.backbone.modules()
                    if isinstance(m, nn.Conv2d))
        self.assertEqual(stem.in_channels, 3)
        self.assertEqual(len(cfg.model.data_preprocessor.mean), 3)
        self.assertEqual(len(cfg.model.data_preprocessor.std), 3)
        self.assertTrue(cfg.model.data_preprocessor.bgr_to_rgb)
        self.assertIsInstance(model.bbox_head.score_activation, nn.Sigmoid)
        self.assertFalse(model.bbox_head.export_score_activation)
        for part in (model.backbone, model.neck, model.bbox_head.head_module):
            self.assertTrue(any(isinstance(m, nn.ReLU) for m in part.modules()))
            self.assertFalse(any(isinstance(m, (nn.SiLU, nn.Hardswish,
                                                nn.Hardsigmoid))
                                 for m in part.modules()))
        for name in ('train_pipeline', 'train_pipeline_stage2',
                     'test_pipeline', 'tta_pipeline'):
            self.assertNotIn('ToGrayscale', str(cfg[name]))
            self.assertNotIn("'color_type': 'grayscale'", str(cfg[name]))

    def test_configs_keep_rgb_relu_and_sigmoid_scores(self):
        for filename in ('yolov8_s_add_surgery2.py',
                         'yolov8_s_add_surgery2_fine_tune.py'):
            with self.subTest(config=filename):
                cfg, model = self.build_model(filename)
                self.assert_rgb_relu(cfg, model)
                self.assertIsNone(cfg.load_from)
                self.assertFalse(cfg.resume)
                self.assertEqual(cfg.deploy_cfg.raw_output_format, 'combined')
                self.assertEqual(cfg.deploy_cfg.output_names,
                                 ['583', '584', '585'])
                x = torch.randn(2, 3, 64, 96, requires_grad=True)
                outputs = model(x, mode='tensor')
                self.assertEqual([tuple(t.shape[-2:]) for t in outputs[0]],
                                 [(8, 12), (4, 6), (2, 3)])
                sum(t.square().mean() for group in outputs
                    for t in group).backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_neck_fx_preserves_outputs(self):
        _, model = self.build_model('yolov8_s_add_surgery2.py')
        features = model.backbone(torch.randn(1, 3, 64, 96))
        traced = torch.fx.symbolic_trace(model.neck)
        for actual, expected in zip(traced(features), model.neck(features)):
            torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            model.neck(features[:1])

    def test_combined_channel_order_and_raw_logits(self):
        from projects.easydeploy.model import DeployModel, MMYOLOBackend

        cfg, model = self.build_model('yolov8_s_add_surgery2.py')
        with torch.no_grad():
            for level in range(3):
                cls = model.bbox_head.head_module.cls_preds[level][-1]
                reg = model.bbox_head.head_module.reg_preds[level][-1]
                cls.weight.zero_()
                cls.bias.fill_(-10. - level)
                reg.weight.zero_()
                reg.bias.copy_(torch.arange(64) + 100 * level)
        deploy = DeployModel(
            baseModel=model, backend=MMYOLOBackend.ONNXRUNTIME,
            raw_output_format=cfg.deploy_cfg.raw_output_format).eval()
        with torch.no_grad():
            outputs = deploy(torch.randn(1, 3, 64, 96))
        self.assertEqual([tuple(head.shape) for head in outputs],
                         [(1, 65, 8, 12), (1, 65, 4, 6), (1, 65, 2, 3)])
        for level, head in enumerate(outputs):
            expected = torch.cat((torch.tensor([-10. - level]),
                                  torch.arange(64) + 100 * level))
            torch.testing.assert_close(
                head, expected[None, :, None, None].expand_as(head))

    @skipUnless(importlib.util.find_spec('edgeai_torchmodelopt'),
                'edgeai_torchmodelopt is not installed')
    @skipUnless(importlib.util.find_spec('onnx'), 'onnx is not installed')
    def test_surgery2_forward_backward_and_onnx(self):
        import onnx
        from edgeai_torchmodelopt import xmodelopt
        from projects.easydeploy.model import DeployModel, MMYOLOBackend
        from projects.easydeploy.tools.export_onnx import rename_outputs

        cfg, model = self.build_model('yolov8_s_add_surgery2.py')
        x = torch.randn(1, 3, 64, 96)
        with torch.no_grad():
            expected = model(x, mode='tensor')
        model.backbone = xmodelopt.surgery.v2.convert_to_lite_fx(model.backbone)
        model.neck = xmodelopt.surgery.v2.convert_to_lite_fx(model.neck)
        model.bbox_head.head_module = (
            xmodelopt.surgery.v1.convert_to_lite_model(
                model.bbox_head.head_module))
        self.assert_rgb_relu(cfg, model)
        x.requires_grad_()
        actual = model(x, mode='tensor')
        for actual_group, expected_group in zip(actual, expected):
            for actual_head, expected_head in zip(actual_group, expected_group):
                torch.testing.assert_close(actual_head, expected_head)
        sum(t.square().mean() for group in actual for t in group).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

        deploy = DeployModel(
            baseModel=model, backend=MMYOLOBackend.ONNXRUNTIME,
            raw_output_format=cfg.deploy_cfg.raw_output_format).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'detector.onnx')
            torch.onnx.export(
                deploy, x.detach(), path, opset_version=11,
                input_names=['images'],
                output_names=['raw_output_0', 'raw_output_1', 'raw_output_2'])
            graph = onnx.load(path)
            rename_outputs(graph, cfg.deploy_cfg.output_names)
            onnx.checker.check_model(graph)
            from google.protobuf import text_format
            from mmyolo.utils.proto import mmyolo_meta_arch_pb2
            from mmyolo.utils.save_model import save_model_proto
            proto_path = save_model_proto(model, graph, x.detach(), path)
            proto = text_format.Parse(
                Path(proto_path).read_text(), mmyolo_meta_arch_pb2.TIDLMetaArch())
            concat_inputs = [
                list(node.input) for output in graph.graph.output
                for node in graph.graph.node if output.name in node.output
            ]
            self.assertEqual(list(proto.caffe_ssd[0].class_input),
                             [inputs[0] for inputs in concat_inputs])
            self.assertEqual(list(proto.caffe_ssd[0].box_input),
                             [inputs[1] for inputs in concat_inputs])
        self.assertEqual(graph.graph.input[0].type.tensor_type.shape.dim[1]
                         .dim_value, 3)
        self.assertEqual([output.name for output in graph.graph.output],
                         cfg.deploy_cfg.output_names)
        self.assertEqual([
            [d.dim_value for d in output.type.tensor_type.shape.dim]
            for output in graph.graph.output
        ], [[1, 65, 8, 12], [1, 65, 4, 6], [1, 65, 2, 3]])
        operators = {node.op_type for node in graph.graph.node}
        self.assertIn('Add', operators)
        self.assertIn('Relu', operators)
        self.assertFalse(operators & {'HardSigmoid', 'HardSwish'})
        concat_nodes = [node for node in graph.graph.node
                        if node.op_type == 'Concat']
        self.assertEqual(len(concat_nodes), 3)
        self.assertEqual({node.output[0] for node in concat_nodes},
                         set(cfg.deploy_cfg.output_names))
