# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures import DetDataSample
from mmengine.config import Config

from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/yolov8/yolov8_n_focus_add_surgery2.py'


class TestYOLOv8FocusAdd(TestCase):

    config_path = CONFIG
    widen_factor = 0.25
    stem_channels = 32

    def build_model(self):
        cfg = Config.fromfile(self.config_path)
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        return cfg, model

    def test_topology_and_config(self):
        cfg, model = self.build_model()
        self.assertIsInstance(model.backbone.stem, nn.Identity)
        self.assertEqual(len(model.backbone.stage1), 1)
        first = model.backbone.stage1[0].main_conv.conv
        self.assertEqual((first.in_channels, first.out_channels),
                         (16, self.stem_channels))
        self.assertEqual(first.kernel_size, (3, 3))
        self.assertEqual(first.stride, (1, 1))
        self.assertEqual(cfg.model.backbone.widen_factor, self.widen_factor)
        self.assertEqual(cfg.model.neck.widen_factor, self.widen_factor)
        self.assertEqual(cfg.model.bbox_head.head_module.widen_factor,
                         self.widen_factor)
        self.assertIsNone(cfg.load_from)
        self.assertFalse(cfg.resume)
        for part in (model.backbone, model.neck, model.bbox_head.head_module):
            self.assertTrue(
                any(isinstance(m, nn.ReLU) for m in part.modules()))
            self.assertFalse(
                any(
                    isinstance(m, (nn.SiLU, nn.Hardsigmoid, nn.Hardswish))
                    for m in part.modules()))
        for name in ('train_pipeline', 'train_pipeline_stage2',
                     'test_pipeline', 'tta_pipeline'):
            self.assertIn('ToGrayscale', str(cfg[name]))
            self.assertNotIn('YOLOv5HSVRandomAug', str(cfg[name]))
        switch = next(hook for hook in cfg.custom_hooks
                      if hook['type'] == 'mmdet.PipelineSwitchHook')
        self.assertEqual(switch['switch_pipeline'], cfg.train_pipeline_stage2)
        model.eval()
        with torch.no_grad():
            outputs = model(torch.rand(1, 16, 16, 24), mode='tensor')
        self.assertEqual([tuple(head.shape) for head in outputs[0]],
                         [(1, 1, 8, 12), (1, 1, 4, 6), (1, 1, 2, 3)])
        self.assertEqual([tuple(head.shape) for head in outputs[1]],
                         [(1, 64, 8, 12), (1, 64, 4, 6), (1, 64, 2, 3)])

    def test_preprocessor_training_and_validation(self):
        cfg, _ = self.build_model()
        preprocessor = MODELS.build(cfg.model.data_preprocessor)
        source = torch.arange(2 * 64 * 96).reshape(2, 1, 64, 96) % 256
        boxes = torch.tensor([[0., 0., 8., 8., 40., 56.]])
        actual = preprocessor(
            dict(
                inputs=source, data_samples=dict(bboxes_labels=boxes.clone())),
            training=True)
        torch.testing.assert_close(actual['inputs'],
                                   F.pixel_unshuffle(source.float() / 255., 4))
        torch.testing.assert_close(actual['data_samples']['bboxes_labels'],
                                   boxes)
        self.assertEqual(actual['data_samples']['img_metas'],
                         [dict(batch_input_shape=(64, 96))] * 2)
        samples = [
            DetDataSample(metainfo=dict(img_shape=(64, 96))) for _ in range(2)
        ]
        actual = preprocessor(dict(inputs=list(source), data_samples=samples))
        torch.testing.assert_close(actual['inputs'],
                                   F.pixel_unshuffle(source.float() / 255., 4))
        for sample in actual['data_samples']:
            self.assertEqual(sample.batch_input_shape, (64, 96))
            self.assertEqual(sample.pad_shape, (64, 96))

    def test_invalid_preprocessor_input(self):
        cfg, _ = self.build_model()
        preprocessor = MODELS.build(cfg.model.data_preprocessor)
        for shape, error in (((1, 3, 64, 96), 'single-channel'),
                             ((1, 1, 65, 96), 'divisible by 4')):
            with self.subTest(shape=shape), self.assertRaisesRegex(
                    ValueError, error):
                preprocessor(
                    dict(
                        inputs=torch.zeros(shape),
                        data_samples=dict(bboxes_labels=torch.empty(0, 6))),
                    training=True)

    @skipUnless(
        importlib.util.find_spec('edgeai_torchmodelopt'),
        'edgeai_torchmodelopt is not installed')
    def test_surgery2_training_step(self):
        from edgeai_torchmodelopt import xmodelopt

        torch.manual_seed(0)
        _, model = self.build_model()
        with patch.object(model, '_dump_init_info'):
            model.init_weights()
        model.backbone = xmodelopt.surgery.v2.convert_to_lite_fx(
            model.backbone)
        model.neck = xmodelopt.surgery.v2.convert_to_lite_fx(model.neck)
        model.bbox_head.head_module = (
            xmodelopt.surgery.v1.convert_to_lite_model(
                model.bbox_head.head_module))
        model.train()
        data = model.data_preprocessor(
            dict(
                inputs=torch.randint(0, 256, (2, 1, 64, 96)),
                data_samples=dict(
                    bboxes_labels=torch.tensor([[0., 0., 8., 8., 40., 56.],
                                                [1., 0., 8., 8., 40., 56.]]))),
            training=True)
        first_weight = next(model.backbone.parameters())
        before = first_weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        losses = model(**data, mode='loss')
        for name in ('loss_bbox', 'loss_dfl'):
            self.assertGreater(float(losses[name].detach()), 0.)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters() if p.requires_grad))
        optimizer.step()
        self.assertFalse(torch.equal(before, first_weight))

    def export_args(self, directory):
        return SimpleNamespace(
            config=str(self.config_path),
            checkpoint=str(Path(directory) / 'epoch_1.pth'),
            work_dir=directory,
            backend='onnxruntime',
            model_only=True,
            export_type=None,
            model_surgery=2,
            device='cpu',
            batch_size=1,
            img_size=[64, 96],
            simplify=False,
            opset=11)

    @skipUnless(
        importlib.util.find_spec('edgeai_torchmodelopt'),
        'edgeai_torchmodelopt is not installed')
    @skipUnless(importlib.util.find_spec('onnx'), 'onnx is not installed')
    @skipUnless(
        importlib.util.find_spec('onnxruntime'),
        'onnxruntime is not installed')
    def test_onnx_export_topology_metadata_and_runtime(self):
        import onnx
        import onnxruntime as ort
        from google.protobuf import text_format
        from mmyolo.utils.proto import mmyolo_meta_arch_pb2
        from projects.easydeploy.model import DeployModel, MMYOLOBackend
        from projects.easydeploy.tools import export_onnx

        cfg, model = self.build_model()
        model.eval()
        x = torch.rand(1, 16, 16, 24)
        deploy = DeployModel(
            model, MMYOLOBackend.ONNXRUNTIME,
            raw_output_format='combined').eval()
        with torch.no_grad():
            expected = [head.numpy() for head in deploy(x)]
        with tempfile.TemporaryDirectory() as directory:
            args = self.export_args(directory)
            torch.save(dict(state_dict=model.state_dict()), args.checkpoint)
            with patch.object(export_onnx, 'parse_args', return_value=args), \
                    patch.object(export_onnx, 'build_model_from_cfg',
                                 return_value=model):
                export_onnx.main()
            path = Path(directory) / 'epoch_1.onnx'
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            self.assertEqual([
                d.dim_value
                for d in graph.graph.input[0].type.tensor_type.shape.dim
            ], [1, 16, 16, 24])
            self.assertEqual([output.name for output in graph.graph.output],
                             cfg.deploy_cfg.output_names)
            self.assertEqual(
                [[d.dim_value for d in output.type.tensor_type.shape.dim]
                 for output in graph.graph.output],
                [[1, 65, 8, 12], [1, 65, 4, 6], [1, 65, 2, 3]])
            producers = {
                output: node
                for node in graph.graph.node
                for output in node.output
            }
            split = next(node for node in graph.graph.node
                         if node.op_type == 'Split')
            upstream = []
            value = split.input[0]
            while value in producers:
                node = producers[value]
                upstream.append(node)
                value = node.input[0]
            self.assertEqual(
                [node.op_type for node in upstream if node.op_type == 'Conv'],
                ['Conv'])
            self.assertEqual(value, 'images')
            first_conv = next(node for node in upstream
                              if node.op_type == 'Conv')
            first_weight = next(weight for weight in graph.graph.initializer
                                if weight.name == first_conv.input[1])
            self.assertEqual(
                list(first_weight.dims), [self.stem_channels, 16, 3, 3])
            operators = {node.op_type for node in graph.graph.node}
            self.assertTrue({'Relu', 'Add'} <= operators)
            self.assertFalse(operators
                             & {'HardSigmoid', 'HardSwish', 'SpaceToDepth'})
            concat = [
                node for node in graph.graph.node if node.op_type == 'Concat'
            ]
            self.assertEqual({node.output[0]
                              for node in concat},
                             set(cfg.deploy_cfg.output_names))
            metadata = json.loads(
                dict((p.key, p.value)
                     for p in graph.metadata_props)['preprocess'])
            self.assertEqual(metadata['source_size'], [64, 96])
            self.assertEqual(metadata['packing'], 'pixel_unshuffle_4')
            proto = text_format.Parse(
                path.with_suffix('.prototxt').read_text(),
                mmyolo_meta_arch_pb2.TIDLMetaArch())
            self.assertEqual(
                (proto.caffe_ssd[0].in_height, proto.caffe_ssd[0].in_width),
                (64, 96))
            session = ort.InferenceSession(
                str(path), providers=['CPUExecutionProvider'])
            actual = session.run(None, {'images': x.numpy()})
            for actual_head, expected_head in zip(actual, expected):
                torch.testing.assert_close(
                    torch.from_numpy(actual_head),
                    torch.from_numpy(expected_head),
                    atol=5e-4,
                    rtol=1e-4)

    @skipUnless(
        importlib.util.find_spec('edgeai_torchmodelopt'),
        'edgeai_torchmodelopt is not installed')
    def test_pt_export_roundtrip(self):
        from projects.easydeploy.tools import export_pt

        _, model = self.build_model()
        model.eval()
        with torch.no_grad():
            for level in range(3):
                cls = model.bbox_head.head_module.cls_preds[level][-1]
                reg = model.bbox_head.head_module.reg_preds[level][-1]
                cls.weight.zero_()
                cls.bias.fill_(-10. - level)
                reg.weight.zero_()
                reg.bias.copy_(torch.arange(64) + 100 * level)
        with tempfile.TemporaryDirectory() as directory:
            args = self.export_args(directory)
            torch.save(dict(state_dict=model.state_dict()), args.checkpoint)
            with patch.object(export_pt, 'parse_args', return_value=args), \
                    patch.object(export_pt, 'build_model_from_cfg',
                                 return_value=model):
                export_pt.main()
            extra = {'preprocess.json': ''}
            loaded = torch.jit.load(
                str(Path(directory) / 'epoch_1.pt'), _extra_files=extra)
            self.assertEqual(
                json.loads(extra['preprocess.json'])['source_size'], [64, 96])
            with torch.no_grad():
                outputs = loaded(torch.rand(1, 16, 16, 24))
        self.assertEqual([tuple(head.shape) for head in outputs],
                         [(1, 65, 8, 12), (1, 65, 4, 6), (1, 65, 2, 3)])
        for level, head in enumerate(outputs):
            expected = torch.cat(
                (torch.tensor([-10. - level]), torch.arange(64) + 100 * level))
            torch.testing.assert_close(
                head, expected[None, :, None, None].expand_as(head))


class TestYOLOv8SFocusAdd(TestYOLOv8FocusAdd):

    config_path = ROOT / 'configs/yolov8/yolov8_s_focus_add_surgery2.py'
    widen_factor = 0.5
    stem_channels = 64
