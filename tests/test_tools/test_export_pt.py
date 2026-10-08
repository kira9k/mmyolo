# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from mmengine.config import Config

from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from projects.easydeploy.tools import export_pt

register_all_modules()
ROOT = Path(__file__).resolve().parents[2]


class TestExportPT(TestCase):

    def test_model_only_surgery2_roundtrip(self):
        cfg = Config.fromfile(ROOT /
                              'configs/yolov8/yolov8_s_add_surgery2.py')
        model = MODELS.build(cfg.model).eval()
        model.cfg = cfg
        with torch.no_grad():
            for level in range(3):
                cls = model.bbox_head.head_module.cls_preds[level][-1]
                reg = model.bbox_head.head_module.reg_preds[level][-1]
                cls.weight.zero_()
                cls.bias.fill_(-10. - level)
                reg.weight.zero_()
                reg.bias.copy_(torch.arange(64) + 100 * level)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'epoch_1.pth'
            torch.save(dict(state_dict=model.state_dict()), checkpoint)
            args = SimpleNamespace(
                config=str(ROOT / 'configs/yolov8/yolov8_s_add_surgery2.py'),
                checkpoint=str(checkpoint), work_dir=directory,
                backend='onnxruntime', model_only=True, export_type=None,
                model_surgery=2, device='cpu', batch_size=1, img_size=[64, 96])
            with patch.object(export_pt, 'parse_args', return_value=args), \
                    patch.object(export_pt, 'build_model_from_cfg',
                                 return_value=model):
                export_pt.main()
            loaded = torch.jit.load(str(Path(directory) / 'epoch_1.pt'))
            with torch.no_grad():
                outputs = loaded(torch.randn(1, 3, 64, 96))
        self.assertEqual([tuple(head.shape) for head in outputs],
                         [(1, 65, 8, 12), (1, 65, 4, 6), (1, 65, 2, 3)])
        for level, head in enumerate(outputs):
            expected = torch.cat((torch.tensor([-10. - level]),
                                  torch.arange(64) + 100 * level))
            torch.testing.assert_close(
                head, expected[None, :, None, None].expand_as(head))
