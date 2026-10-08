# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from ..layers import CSPLayerWithTwoConv
from ..utils import make_divisible, make_round
from .csp_darknet import YOLOv8CSPDarknet


@MODELS.register_module()
class YOLOv8FocusCSPDarknet(YOLOv8CSPDarknet):
    """External grayscale Focus4 input with one convolution before C2f Split."""

    def __init__(self, input_channels=16, **kwargs):
        if input_channels != 16:
            raise ValueError('Expected 16 packed grayscale Focus4 channels')
        super().__init__(input_channels=input_channels, **kwargs)

    def build_stem_layer(self):
        return nn.Identity()

    def build_stage_layer(self, stage_idx, setting):
        if stage_idx != 0:
            return super().build_stage_layer(stage_idx, setting)
        _, out_channels, num_blocks, add_identity, _ = setting
        out_channels = make_divisible(out_channels, self.widen_factor)
        csp = CSPLayerWithTwoConv(
            self.input_channels,
            out_channels,
            num_blocks=make_round(num_blocks, self.deepen_factor),
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            fusion_mode=self.csp_fusion_mode)
        # This is the only projection between the packed input and first Split.
        csp.main_conv = ConvModule(
            self.input_channels,
            2 * csp.mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        return [csp]
