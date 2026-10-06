# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import Compose
from mmdet.structures import DetDataSample
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData

from mmyolo.registry import MODELS, TRANSFORMS
from mmyolo.utils import register_all_modules

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/yolov8/yolov8_s_add_gray_hard_fine_tune.py'


class TestYOLOv8GrayHard(TestCase):

    def test_500_epoch_config_matches_original_recipe(self):
        original = Config.fromfile(
            ROOT / 'configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py')
        cfg = Config.fromfile(ROOT /
                              'configs/yolov8/yolov8_s_add_gray_hard.py')
        self.assertEqual(cfg.train_cfg, original.train_cfg)
        self.assertEqual(cfg.optim_wrapper, original.optim_wrapper)
        self.assertEqual(cfg.default_hooks, original.default_hooks)
        self.assertEqual(cfg.model.test_cfg, original.model.test_cfg)
        self.assertEqual(cfg.model.train_cfg, original.model.train_cfg)
        self.assertEqual(cfg.train_dataloader.batch_size, 16)
        self.assertEqual(cfg.train_dataloader.dataset.data_root,
                         original.train_dataloader.dataset.data_root)
        self.assertIsNone(cfg.load_from)
        self.assertFalse(cfg.resume)
        switches = [
            hook for hook in cfg.custom_hooks
            if hook['type'] == 'mmdet.PipelineSwitchHook'
        ]
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0]['switch_epoch'], 490)
        self.assertEqual(switches[0]['switch_pipeline'],
                         cfg.train_pipeline_stage2)
        self.assertEqual(cfg.model.data_preprocessor.std, [255.])
        model = MODELS.build(cfg.model).eval()
        self.assertEqual(model.backbone.stem.conv.in_channels, 1)
        sample = DetDataSample(metainfo=dict(img_shape=(64, 96)))
        preprocessed = model.data_preprocessor(
            dict(
                inputs=[torch.full((1, 64, 96), 255, dtype=torch.uint8)],
                data_samples=[sample]))
        torch.testing.assert_close(preprocessed['inputs'],
                                   torch.ones(1, 1, 64, 96))
        with torch.no_grad():
            outs = model(preprocessed['inputs'], mode='tensor')
        self.assertEqual([tuple(out.shape[-2:]) for out in outs[0]], [(8, 12),
                                                                      (4, 6),
                                                                      (2, 3)])

    def test_default_head_keeps_sigmoid(self):
        cfg = Config.fromfile(ROOT / 'configs/yolov8/yolov8_s_add.py')
        head = MODELS.build(cfg.model).bbox_head
        logits = torch.tensor([-10., -1.5, 0., 1.5, 10.])
        torch.testing.assert_close(
            head._activate_cls_scores(logits), logits.sigmoid())
        self.assertFalse(head.export_score_activation)

    def test_config_has_one_channel_and_hard_activations(self):
        cfg = Config.fromfile(CONFIG)
        model = MODELS.build(cfg.model)
        self.assertEqual(model.backbone.stem.conv.in_channels, 1)
        self.assertEqual(tuple(model.data_preprocessor.mean.shape), (1, 1, 1))
        self.assertFalse(model.data_preprocessor._channel_conversion)
        self.assertFalse(any(isinstance(m, nn.SiLU) for m in model.modules()))
        self.assertTrue(
            any(isinstance(m, nn.Hardswish) for m in model.modules()))
        logits = torch.tensor([-10., -3., -1.5, 0., 1.5, 3., 10.])
        torch.testing.assert_close(
            model.bbox_head._activate_cls_scores(logits),
            torch.tensor([0., 0., .25, .5, .75, 1., 1.]))
        self.assertTrue(model.bbox_head.export_score_activation)

        def check_transforms(value):
            if isinstance(value, dict):
                self.assertNotIn(
                    value.get('type'),
                    ('YOLOv5HSVRandomAug', 'HueSaturationValue'))
                if value.get('type', '').split('.')[-1] == 'LoadImageFromFile':
                    self.assertEqual(value['color_type'], 'grayscale')
                for child in value.values():
                    check_transforms(child)
            elif isinstance(value, list):
                for child in value:
                    check_transforms(child)

        for pipeline in (cfg.train_pipeline, cfg.train_pipeline_stage2,
                         cfg.test_pipeline, cfg.tta_pipeline):
            check_transforms(pipeline)

    def test_to_grayscale(self):
        transform = TRANSFORMS.build(dict(type='ToGrayscale'))
        bgr = np.zeros((8, 12, 3), dtype=np.uint8)
        bgr[..., 2] = 255
        expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for image in (bgr, expected, expected[..., None]):
            result = transform(dict(img=image))
            np.testing.assert_array_equal(result['img'], expected)
            self.assertEqual(result['img_shape'], (8, 12))
        with self.assertRaises(ValueError):
            transform(dict(img=np.zeros((8, 12, 4), dtype=np.uint8)))

    def test_file_and_array_preprocessing(self):
        cfg = Config.fromfile(CONFIG)
        model = MODELS.build(cfg.model).eval()
        for from_array in (False, True):
            pipeline = deepcopy(cfg.test_pipeline)
            pipeline = [
                t for t in pipeline
                if t['type'].split('.')[-1] != 'LoadAnnotations'
            ]
            if from_array:
                pipeline[0]['type'] = 'mmdet.LoadImageFromNDArray'
            with tempfile.TemporaryDirectory() as tmp:
                bgr = np.full((32, 48, 3), (10, 80, 200), dtype=np.uint8)
                path = str(Path(tmp) / 'input.png')
                self.assertTrue(cv2.imwrite(path, bgr))
                data = dict(
                    img=bgr, img_id=0) if from_array else dict(
                        img_path=path, img_id=0)
                packed = Compose(pipeline)(data)
                self.assertEqual(packed['inputs'].shape[0], 1)
                preprocessed = model.data_preprocessor(
                    dict(
                        inputs=[packed['inputs']],
                        data_samples=[packed['data_samples']]))
                self.assertEqual(preprocessed['inputs'].shape,
                                 (1, 1, 640, 640))
                self.assertEqual(preprocessed['inputs'].dtype, torch.float32)
                # Confirm there is no accidental /255 normalization.
                self.assertGreater(float(preprocessed['inputs'].max()), 100.)

    def test_grayscale_training_and_score_assignment(self):
        torch.manual_seed(0)
        cfg = Config.fromfile(CONFIG)
        model = MODELS.build(cfg.model).train()
        samples = []
        for _ in range(2):
            sample = DetDataSample(metainfo=dict(batch_input_shape=(64, 96)))
            sample.gt_instances = InstanceData(
                bboxes=torch.tensor([[8., 8., 40., 56.]]),
                labels=torch.zeros(1, dtype=torch.long))
            samples.append(sample)
        x = torch.randn(2, 1, 64, 96)
        with patch.object(
                model.bbox_head,
                '_activate_cls_scores',
                wraps=model.bbox_head._activate_cls_scores) as gate:
            losses = model.loss(x, samples)
            gate.assert_not_called()
        self.assertGreater(float(losses['loss_bbox'].detach()), 0.)
        self.assertGreater(float(losses['loss_dfl'].detach()), 0.)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters() if p.requires_grad))

    def test_scratch_assignment_survives_negative_initial_logits(self):
        torch.manual_seed(0)
        cfg = Config.fromfile(ROOT /
                              'configs/yolov8/yolov8_s_add_gray_hard.py')
        model = MODELS.build(cfg.model).train()
        with patch.object(model, '_dump_init_info'):
            model.init_weights()
        head = model.bbox_head
        inputs = torch.rand(2, 1, 64, 96)
        targets = [
            InstanceData(
                bboxes=torch.tensor([[8., 8., 40., 56.]]),
                labels=torch.zeros(1, dtype=torch.long)) for _ in range(2)
        ]
        metadata = [dict(batch_input_shape=(64, 96)) for _ in range(2)]
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        assignments = []
        assign = head.assigner.forward

        def capture_assignment(*args, **kwargs):
            result = assign(*args, **kwargs)
            assignments.append(result)
            return result

        with patch.object(
                head.assigner, 'forward',
                side_effect=capture_assignment) as assigner, patch.object(
                    head, '_activate_cls_scores',
                    wraps=head._activate_cls_scores) as gate:
            for step in range(3):
                optimizer.zero_grad()
                outputs = model(inputs, mode='tensor')
                if step == 0:
                    # Real YOLOv8 bias initialization is below hard sigmoid's
                    # zero cutoff, but must still yield weighted positives.
                    self.assertTrue(all((x < -3.).all() for x in outputs[0]))
                losses = head.loss_by_feat(
                    *outputs,
                    batch_gt_instances=targets,
                    batch_img_metas=metadata)
                expected_scores = torch.cat([
                    x.permute(0, 2, 3, 1).reshape(2, -1, head.num_classes)
                    for x in outputs[0]
                ], dim=1).detach().sigmoid()
                torch.testing.assert_close(assigner.call_args.args[1],
                                           expected_scores)
                self.assertGreater(
                    float(assignments[-1]['assigned_scores'].sum()), 0.)
                for name in ('loss_bbox', 'loss_dfl'):
                    self.assertGreater(float(losses[name].detach()), 0.)
                loss = sum(losses.values())
                self.assertTrue(torch.isfinite(loss).all())
                loss.backward()
                for branch in (head.head_module.cls_preds,
                               head.head_module.reg_preds):
                    grads = [p.grad for p in branch.parameters()]
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(float(g.abs().sum()) for g in grads),
                                       0.)
                optimizer.step()
            gate.assert_not_called()

    def test_inference_uses_hard_sigmoid(self):
        cfg = Config.fromfile(CONFIG)
        head = MODELS.build(cfg.model).bbox_head.eval()
        logits = [
            torch.full((1, 1, h, w), -1.5)
            for h, w in ((8, 12), (4, 6), (2, 3))
        ]
        boxes = [
            torch.ones((1, 64, h, w)) for h, w in ((8, 12), (4, 6), (2, 3))
        ]
        test_cfg = ConfigDict(
            multi_label=False, nms_pre=-1, score_thr=.2, max_per_img=100)
        with torch.no_grad():
            predictions = head.predict_by_feat(
                logits,
                boxes,
                batch_img_metas=[
                    dict(
                        img_shape=(64, 96),
                        ori_shape=(64, 96),
                        scale_factor=(1., 1.))
                ],
                cfg=test_cfg,
                rescale=False,
                with_nms=False)
        self.assertGreater(len(predictions[0].scores), 0)
        torch.testing.assert_close(predictions[0].scores,
                                   torch.full_like(predictions[0].scores, .25))

    def test_deploy_postprocessing_uses_hard_sigmoid(self):
        from projects.easydeploy.model import DeployModel, MMYOLOBackend

        cfg = Config.fromfile(CONFIG)
        model = MODELS.build(cfg.model).eval()
        deploy = DeployModel(
            baseModel=model,
            backend=MMYOLOBackend.ONNXRUNTIME,
            postprocess_cfg=ConfigDict(
                pre_top_k=10, keep_top_k=5, export_type='MMYOLO'))
        logits = [
            torch.full((1, 1, h, w), -1.5)
            for h, w in ((8, 12), (4, 6), (2, 3))
        ]
        boxes = [
            torch.ones((1, 64, h, w)) for h, w in ((8, 12), (4, 6), (2, 3))
        ]
        with patch.object(deploy, 'select_nms') as select_nms:
            deploy.pred_by_feat(logits, boxes)
        passed_scores = select_nms.return_value.call_args.args[1]
        torch.testing.assert_close(passed_scores,
                                   torch.full_like(passed_scores, .25))

    @skipUnless(importlib.util.find_spec('onnx'), 'onnx is not installed')
    def test_gray_hard_onnx_and_prototxt(self):
        import onnx
        from google.protobuf import text_format

        from mmyolo.utils.proto import mmyolo_meta_arch_pb2
        from mmyolo.utils.save_model import save_model_proto
        from projects.easydeploy.model import DeployModel, MMYOLOBackend

        model = MODELS.build(Config.fromfile(CONFIG).model).eval()
        x = torch.randn(1, 1, 64, 96)
        with torch.no_grad():
            raw_scores, _ = model(x, mode='tensor')
        deploy = DeployModel(
            baseModel=model, backend=MMYOLOBackend.ONNXRUNTIME).eval()
        with torch.no_grad():
            expected = [tensor for pair in deploy(x) for tensor in pair]
        for raw, score in zip(raw_scores, expected[::2]):
            torch.testing.assert_close(score, ((raw + 3.) / 6.).clamp(0., 1.))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'gray_hard.onnx')
            torch.onnx.export(deploy, x, path, opset_version=11, dynamo=False)
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            operators = {node.op_type for node in graph.graph.node}
            self.assertNotIn('Sigmoid', operators)
            self.assertNotIn('Concat', operators)
            self.assertIn('HardSigmoid', operators)
            input_shape = graph.graph.input[0].type.tensor_type.shape.dim
            self.assertEqual([d.dim_value for d in input_shape],
                             [1, 1, 64, 96])
            proto_path = save_model_proto(model, graph, x, path)
            proto = text_format.Parse(
                Path(proto_path).read_text(),
                mmyolo_meta_arch_pb2.TIDLMetaArch())
            self.assertEqual(proto.caffe_ssd[0].score_converter,
                             mmyolo_meta_arch_pb2.IDENTITY)
            output_names = [output.name for output in graph.graph.output]
            self.assertEqual(
                list(proto.caffe_ssd[0].class_input), output_names[::2])
            self.assertEqual(
                list(proto.caffe_ssd[0].box_input), output_names[1::2])
            if importlib.util.find_spec('onnxruntime'):
                import onnxruntime as ort

                session = ort.InferenceSession(
                    path, providers=['CPUExecutionProvider'])
                actual = session.run(None,
                                     {session.get_inputs()[0].name: x.numpy()})
                for exported, eager in zip(actual, expected):
                    torch.testing.assert_close(
                        torch.from_numpy(exported),
                        eager,
                        rtol=1e-4,
                        atol=1e-5)
