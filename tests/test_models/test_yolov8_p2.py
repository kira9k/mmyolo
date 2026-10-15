# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config

from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/yolov8/yolov8_s_p2_syncbn_fast_8xb16-500e_coco.py'
BASE_CONFIG = ROOT / 'configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'


class TestYOLOv8P2(TestCase):

    def setUp(self):
        self.random_state = torch.get_rng_state()
        torch.manual_seed(0)

    def tearDown(self):
        torch.set_rng_state(self.random_state)

    def build(self):
        cfg = Config.fromfile(CONFIG)
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        return cfg, model

    def surgery(self, model):
        from edgeai_torchmodelopt import xmodelopt
        model.backbone = xmodelopt.surgery.v2.convert_to_lite_fx(
            model.backbone)
        model.neck = xmodelopt.surgery.v2.convert_to_lite_fx(model.neck)
        model.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(
            model.bbox_head.head_module)
        return model

    def test_config_and_inherited_pipelines(self):
        cfg, model = self.build()
        base = Config.fromfile(BASE_CONFIG)
        self.assertEqual(model.backbone.out_indices, (1, 2, 3, 4))
        self.assertEqual(model.neck.in_channels, [128, 256, 512, 1024])
        head = model.bbox_head.head_module
        self.assertEqual(head.in_channels, [64, 128, 256, 512])
        self.assertEqual(list(head.featmap_strides), [4, 8, 16, 32])
        self.assertEqual(head.num_levels, 4)
        self.assertEqual(head.cls_out_channels, 128)
        for branch in head.cls_preds:
            self.assertEqual(branch[0].conv.out_channels, 128)
            self.assertEqual(branch[1].conv.in_channels, 128)
            self.assertEqual(branch[1].conv.out_channels, 128)
            self.assertEqual(branch[2].in_channels, 128)
        self.assertTrue(
            all(branch[0].conv.out_channels == 64
                for branch in head.reg_preds))
        self.assertEqual(
            list(model.bbox_head.prior_generator.strides), [(4, 4), (8, 8),
                                                            (16, 16),
                                                            (32, 32)])
        self.assertIsNone(cfg.load_from)
        self.assertFalse(cfg.resume)
        for field in ('train_dataloader', 'val_dataloader', 'test_dataloader',
                      'train_pipeline', 'train_pipeline_stage2',
                      'custom_hooks', 'optim_wrapper', 'default_hooks'):
            self.assertEqual(cfg[field], base[field])

    def test_default_width_preserves_existing_heads(self):
        base = Config.fromfile(BASE_CONFIG)
        head = MODELS.build(base.model.bbox_head.head_module)
        self.assertEqual(head.cls_out_channels, 128)
        cfg = Config.fromfile(CONFIG)
        cfg.model.bbox_head.head_module.cls_out_channels = None
        legacy = MODELS.build(cfg.model.bbox_head.head_module)
        self.assertEqual(legacy.cls_out_channels, 64)
        explicit_cfg = cfg.model.bbox_head.head_module.copy()
        explicit_cfg.cls_out_channels = 64
        explicit = MODELS.build(explicit_cfg)
        explicit.load_state_dict(legacy.state_dict(), strict=True)

    def test_explicit_width_is_not_scaled(self):
        cfg = Config.fromfile(CONFIG)
        cfg.model.bbox_head.head_module.cls_out_channels = 96
        head = MODELS.build(cfg.model.bbox_head.head_module)
        self.assertEqual(head.cls_out_channels, 96)
        self.assertTrue(
            all(branch[0].conv.out_channels == 96
                for branch in head.cls_preds))

    def test_invalid_classification_width(self):
        cfg = Config.fromfile(CONFIG)
        for width in (0, -1, 1.5, True):
            with self.subTest(width=width):
                cfg.model.bbox_head.head_module.cls_out_channels = width
                with self.assertRaisesRegex(ValueError, 'cls_out_channels'):
                    MODELS.build(cfg.model.bbox_head.head_module)

    def test_full_size_feature_maps_and_predictions(self):
        _, model = self.build()
        model.eval()
        with torch.no_grad():
            features = model.extract_feat(torch.randn(1, 3, 640, 640))
            cls, bbox = model.bbox_head.head_module(features)
        self.assertEqual([list(feature.shape) for feature in features],
                         [[1, 64, 160, 160], [1, 128, 80, 80],
                          [1, 256, 40, 40], [1, 512, 20, 20]])
        self.assertEqual(
            [list(value.shape) for value in cls],
            [[1, 1, 160, 160], [1, 1, 80, 80], [1, 1, 40, 40], [1, 1, 20, 20]])
        self.assertTrue(all(value.shape[1] == 64 for value in bbox))

    def test_four_level_decoding_and_nms(self):
        from mmdet.structures import DetDataSample

        _, model = self.build()
        model.eval()
        sample = DetDataSample(
            metainfo=dict(
                img_shape=(64, 96),
                ori_shape=(64, 96),
                scale_factor=(1., 1.),
                pad_param=(0., 0., 0., 0.)))
        with torch.no_grad():
            predictions = model.predict(torch.randn(1, 3, 64, 96), [sample])
        self.assertEqual(len(predictions), 1)
        instances = predictions[0].pred_instances
        self.assertEqual(instances.bboxes.shape[1], 4)
        self.assertTrue(torch.isfinite(instances.bboxes).all())
        self.assertTrue(torch.isfinite(instances.scores).all())

    @skipUnless(
        importlib.util.find_spec('edgeai_torchmodelopt'),
        'edgeai_torchmodelopt is not installed')
    def test_surgery2_train_step(self):
        _, model = self.build()
        self.surgery(model)
        self.assertFalse(any(isinstance(m, nn.SiLU) for m in model.modules()))
        model.train()
        data = model.data_preprocessor(
            dict(
                inputs=torch.randint(0, 256, (2, 3, 64, 96)),
                data_samples=dict(
                    bboxes_labels=torch.tensor([[0., 0., 8., 8., 12., 12.],
                                                [1., 0., 8., 8., 40., 56.]]))),
            training=True)
        before = next(model.parameters()).detach().clone()
        losses = model(**data, mode='loss')
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(losses['loss_bbox'].detach()), 0)
        self.assertGreater(float(losses['loss_dfl'].detach()), 0)
        total.backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters() if p.requires_grad))
        torch.optim.SGD(model.parameters(), lr=1e-3).step()
        self.assertFalse(torch.equal(next(model.parameters()), before))

    @skipUnless(
        importlib.util.find_spec('edgeai_torchmodelopt'),
        'edgeai_torchmodelopt is not installed')
    @skipUnless(
        importlib.util.find_spec('onnxruntime'),
        'onnxruntime is not installed')
    def test_surgery2_onnx_four_outputs_and_prototxt(self):
        import onnx
        import onnxruntime as ort
        from google.protobuf import text_format
        from mmyolo.utils.proto import mmyolo_meta_arch_pb2
        from projects.easydeploy.model import DeployModel, MMYOLOBackend
        from projects.easydeploy.tools import export_onnx

        cfg, source = self.build()
        source.eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'epoch_1.pth'
            # Training surgery leaves the learned parameter layout unchanged.
            self.surgery(source)
            torch.save(dict(state_dict=source.state_dict()), checkpoint)
            eager = MODELS.build(cfg.model)
            eager.cfg = cfg
            eager.load_state_dict(source.state_dict(), strict=True)
            args = [
                'export_onnx.py',
                str(CONFIG),
                str(checkpoint), '--model-only', '--model-surgery', '2',
                '--img-size', '64', '96', '--work-dir', directory
            ]
            with patch.object(
                    export_onnx, 'build_model_from_cfg',
                    return_value=eager), patch('sys.argv', args):
                export_onnx.main()
            graph = onnx.load(Path(directory) / 'epoch_1.onnx')
            self.assertEqual([output.name for output in graph.graph.output],
                             ['head_s4', 'head_s8', 'head_s16', 'head_s32'])
            self.assertEqual(
                [[dim.dim_value for dim in o.type.tensor_type.shape.dim]
                 for o in graph.graph.output],
                [[1, 65, 16, 24], [1, 65, 8, 12], [1, 65, 4, 6], [1, 65, 2, 3]
                 ])
            self.assertFalse(
                any(
                    node.op_type in ('Sigmoid', 'QuantizeLinear',
                                     'DequantizeLinear')
                    for node in graph.graph.node))
            self.assertTrue(
                any(node.op_type == 'Relu' for node in graph.graph.node))
            proto = text_format.Parse(
                (Path(directory) / 'epoch_1.prototxt').read_text(),
                mmyolo_meta_arch_pb2.TIDLMetaArch())
            self.assertEqual(len(proto.caffe_ssd[0].class_input), 4)
            self.assertEqual(len(proto.caffe_ssd[0].box_input), 4)
            x = torch.randn(1, 3, 64, 96)
            deploy = DeployModel(
                source,
                MMYOLOBackend.ONNXRUNTIME,
                raw_output_format='combined').eval()
            with torch.no_grad():
                expected = deploy(x)
            session = ort.InferenceSession(
                str(Path(directory) / 'epoch_1.onnx'),
                providers=['CPUExecutionProvider'])
            actual = session.run(None, {'images': x.numpy()})
            for reference, value in zip(expected, actual):
                np.testing.assert_allclose(
                    reference.numpy(), value, rtol=1e-4, atol=2e-5)
