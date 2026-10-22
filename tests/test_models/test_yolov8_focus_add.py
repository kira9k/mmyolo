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

from mmyolo.models.layers import CSPLayerWithTwoConv, SPPFBottleneck
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/yolov8/yolov8_n_focus_add_surgery2.py'


class TestYOLOv8FocusAdd(TestCase):

    config_path = CONFIG
    widen_factor = 0.25
    stem_channels = 32
    spp_fusion_mode = 'add'
    csp_fusion_mode = 'add'

    def build_model(self):
        cfg = Config.fromfile(self.config_path)
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        return cfg, model

    def assert_concat_layout(self, graph, output_names=()):
        concats = [
            node for node in graph.graph.node if node.op_type == 'Concat'
        ]
        terminal = [node for node in concats if node.output[0] in output_names]
        internal = [
            node for node in concats if node.output[0] not in output_names
        ]
        self.assertEqual({node.output[0]
                          for node in terminal}, set(output_names))
        self.assertEqual(len(internal), int(self.spp_fusion_mode == 'concat'))
        if internal:
            pool_concat = internal[0]
            self.assertEqual(len(pool_concat.input), 4)
            self.assertEqual(len(set(pool_concat.input)), 4)
            self.assertEqual(
                next(a.i for a in pool_concat.attribute if a.name == 'axis'),
                1)
            producers = {
                value: node
                for node in graph.graph.node
                for value in node.output
            }
            self.assertEqual(
                [producers[value].op_type for value in pool_concat.input],
                ['Relu', 'MaxPool', 'MaxPool', 'MaxPool'])

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
        spp_layers = [
            layer for layer in model.backbone.modules()
            if isinstance(layer, SPPFBottleneck)
        ]
        self.assertEqual(len(spp_layers), 1)
        spp = spp_layers[0]
        self.assertEqual(spp.fusion_mode, self.spp_fusion_mode)
        factor = 4 if self.spp_fusion_mode == 'concat' else 1
        self.assertEqual(spp.conv2.conv.in_channels,
                         factor * spp.conv1.conv.out_channels)
        for layer in (*model.backbone.modules(), *model.neck.modules()):
            if isinstance(layer, CSPLayerWithTwoConv):
                self.assertEqual(layer.fusion_mode, self.csp_fusion_mode)
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
            self.assert_concat_layout(graph, cfg.deploy_cfg.output_names)
            metadata = json.loads(
                dict((p.key, p.value)
                     for p in graph.metadata_props)['preprocess'])
            self.assertEqual(metadata['source_size'], [64, 96])
            self.assertEqual(metadata['packing'], 'pixel_unshuffle_4')
            proto = text_format.Parse(
                path.with_suffix('.prototxt').read_text(),
                mmyolo_meta_arch_pb2.TIDLMetaArch())
            head_inputs = [
                producers[name].input for name in cfg.deploy_cfg.output_names
            ]
            self.assertEqual(
                list(proto.caffe_ssd[0].class_input),
                [inputs[0] for inputs in head_inputs])
            self.assertEqual(
                list(proto.caffe_ssd[0].box_input),
                [inputs[1] for inputs in head_inputs])
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
    @skipUnless(importlib.util.find_spec('onnx'), 'onnx is not installed')
    @skipUnless(
        importlib.util.find_spec('onnxruntime'),
        'onnxruntime is not installed')
    def test_separate_onnx_export_matches_combined(self):
        import onnx
        import onnxruntime as ort
        from google.protobuf import text_format
        from mmyolo.utils.proto import mmyolo_meta_arch_pb2
        from projects.easydeploy.model import DeployModel, MMYOLOBackend
        from projects.easydeploy.tools import export_onnx

        _, model = self.build_model()
        model.eval()
        x = torch.rand(1, 16, 16, 24)
        deploy = DeployModel(
            model, MMYOLOBackend.ONNXRUNTIME,
            raw_output_format='combined').eval()
        with torch.no_grad():
            merged = deploy(x)
        expected = [
            branch for head in merged for branch in (head[:, :1], head[:, 1:])
        ]
        names = [
            'cls_s8', 'bbox_s8', 'cls_s16', 'bbox_s16', 'cls_s32', 'bbox_s32'
        ]
        with tempfile.TemporaryDirectory() as directory:
            args = self.export_args(directory)
            args.raw_output_format = 'separate'
            torch.save(dict(state_dict=model.state_dict()), args.checkpoint)
            with patch.object(export_onnx, 'parse_args', return_value=args), \
                    patch.object(export_onnx, 'build_model_from_cfg',
                                 return_value=model):
                export_onnx.main()
            path = Path(directory) / 'epoch_1.onnx'
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            self.assertEqual([output.name for output in graph.graph.output],
                             names)
            self.assertEqual(
                [[d.dim_value for d in output.type.tensor_type.shape.dim]
                 for output in graph.graph.output],
                [list(head.shape) for head in expected])
            self.assert_concat_layout(graph)
            self.assertFalse(
                any(node.op_type == 'Sigmoid' for node in graph.graph.node))
            proto = text_format.Parse(
                path.with_suffix('.prototxt').read_text(),
                mmyolo_meta_arch_pb2.TIDLMetaArch())
            self.assertEqual(list(proto.caffe_ssd[0].class_input), names[::2])
            self.assertEqual(list(proto.caffe_ssd[0].box_input), names[1::2])
            self.assertEqual(
                (proto.caffe_ssd[0].in_height, proto.caffe_ssd[0].in_width),
                (64, 96))
            metadata = json.loads(
                dict((p.key, p.value)
                     for p in graph.metadata_props)['preprocess'])
            self.assertEqual(metadata['source_size'], [64, 96])
            session = ort.InferenceSession(
                str(path), providers=['CPUExecutionProvider'])
            actual = session.run(None, {'images': x.numpy()})
            for actual_head, expected_head in zip(actual, expected):
                torch.testing.assert_close(
                    torch.from_numpy(actual_head),
                    expected_head,
                    atol=5e-4,
                    rtol=1e-4)

    @skipUnless(
        importlib.util.find_spec('edgeai_torchmodelopt'),
        'edgeai_torchmodelopt is not installed')
    @skipUnless(importlib.util.find_spec('onnx'), 'onnx is not installed')
    def test_explicit_combined_overrides_separate_config_names(self):
        import onnx
        from projects.easydeploy.tools import export_onnx

        cfg, model = self.build_model()
        cfg.deploy_cfg.raw_output_format = 'separate'
        cfg.deploy_cfg.output_names = [
            'cls_s8', 'bbox_s8', 'cls_s16', 'bbox_s16', 'cls_s32', 'bbox_s32'
        ]
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            args = self.export_args(directory)
            args.raw_output_format = 'combined'
            torch.save(dict(state_dict=model.state_dict()), args.checkpoint)
            with patch.object(export_onnx, 'parse_args', return_value=args), \
                    patch.object(export_onnx, 'build_model_from_cfg',
                                 return_value=model):
                export_onnx.main()
            graph = onnx.load(str(Path(directory) / 'epoch_1.onnx'))
            onnx.checker.check_model(graph)
        self.assertEqual([output.name for output in graph.graph.output],
                         ['head_s8', 'head_s16', 'head_s32'])
        self.assertEqual(
            [[d.dim_value for d in output.type.tensor_type.shape.dim]
             for output in graph.graph.output],
            [[1, 65, 8, 12], [1, 65, 4, 6], [1, 65, 2, 3]])
        self.assert_concat_layout(graph, ['head_s8', 'head_s16', 'head_s32'])

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


class TestYOLOv8NFocusSPPConcat(TestYOLOv8FocusAdd):

    config_path = ROOT / 'configs/yolov8/yolov8_n_focus_add_spp_concat_surgery2.py'
    spp_fusion_mode = 'concat'


class TestYOLOv8SFocusSPPConcat(TestYOLOv8SFocusAdd):

    config_path = ROOT / 'configs/yolov8/yolov8_s_focus_add_spp_concat_surgery2.py'
    spp_fusion_mode = 'concat'


class TestYOLOv8NFocusAddLastSPPConcat(TestYOLOv8NFocusSPPConcat):

    config_path = (
        ROOT / 'configs/yolov8/'
        'yolov8_n_focus_add_last_spp_concat_surgery2.py')
    csp_fusion_mode = 'add_last'


class TestYOLOv8SFocusAddLastSPPConcat(TestYOLOv8SFocusSPPConcat):

    config_path = (
        ROOT / 'configs/yolov8/'
        'yolov8_s_focus_add_last_spp_concat_surgery2.py')
    csp_fusion_mode = 'add_last'
