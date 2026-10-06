# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import CSPLayerWithTwoConv
from ..utils import make_divisible, make_round
from .yolov8_pafpn import YOLOv8PAFPN


@MODELS.register_module()
class YOLOv8AddPAFPN(YOLOv8PAFPN):
    """YOLOv8 neck with projected addition between feature scales.

    Channel projections run before upsampling. Downsampling convolutions
    directly produce the channels needed for the next addition.

    Args:
        in_channels (List[int]): Backbone channels before width scaling.
        out_channels (List[int]): Output channels before width scaling.
        deepen_factor (float): Depth multiplier. Defaults to 1.0.
        widen_factor (float): Width multiplier. Defaults to 1.0.
        num_csp_blocks (int): Base C2f depth. Defaults to 3.
        freeze_all (bool): Freeze the neck. Defaults to False.
        norm_cfg (dict): Normalization configuration.
        act_cfg (dict): Activation configuration.
        init_cfg (dict, optional): Initialization configuration.
        csp_fusion_mode (str): Internal C2f fusion, 'concat' or 'add'.
            Defaults to 'add'.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 csp_fusion_mode: str = 'add'):
        if len(in_channels) < 2 or len(in_channels) != len(out_channels):
            raise ValueError('Expected matching channel lists with >=2 scales')
        self.csp_fusion_mode = csp_fusion_mode
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_upsample_layer(self, idx: int) -> nn.Module:
        high_channels = (
            self.in_channels[idx] if idx == self.num_in_channels -
            1 else self.out_channels[idx])
        in_channels = make_divisible(high_channels, self.widen_factor)
        out_channels = make_divisible(self.in_channels[idx - 1],
                                      self.widen_factor)
        projection = nn.Identity()
        if in_channels != out_channels:
            projection = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        return nn.Sequential(projection,
                             nn.Upsample(scale_factor=2, mode='nearest'))

    def _build_csp_layer(self, in_channels: int,
                         out_channels: int) -> nn.Module:
        return CSPLayerWithTwoConv(
            make_divisible(in_channels, self.widen_factor),
            make_divisible(out_channels, self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            fusion_mode=self.csp_fusion_mode)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        return self._build_csp_layer(self.in_channels[idx - 1],
                                     self.out_channels[idx - 1])

    def _bottom_up_channels(self, idx: int) -> int:
        if idx == self.num_in_channels - 2:
            return self.in_channels[idx + 1]
        return self.out_channels[idx + 1]

    def build_downsample_layer(self, idx: int) -> nn.Module:
        return ConvModule(
            make_divisible(self.out_channels[idx], self.widen_factor),
            make_divisible(self._bottom_up_channels(idx), self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        return self._build_csp_layer(
            self._bottom_up_channels(idx), self.out_channels[idx + 1])

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        if len(inputs) != self.num_in_channels:
            raise ValueError('Input feature count must match in_channels')
        reduce_outs = [
            layer(feat) for layer, feat in zip(self.reduce_layers, inputs)
        ]
        inner_outs = [reduce_outs[-1]]
        for layer_idx, idx in enumerate(
                range(self.num_in_channels - 1, 0, -1)):
            upsample_feat = self.upsample_layers[layer_idx](inner_outs[0])
            merged = upsample_feat + reduce_outs[idx - 1]
            inner_outs.insert(0, self.top_down_layers[layer_idx](merged))

        outs = [inner_outs[0]]
        for idx in range(self.num_in_channels - 1):
            downsample_feat = self.downsample_layers[idx](outs[-1])
            merged = downsample_feat + inner_outs[idx + 1]
            outs.append(self.bottom_up_layers[idx](merged))
        return tuple(layer(feat) for layer, feat in zip(self.out_layers, outs))
