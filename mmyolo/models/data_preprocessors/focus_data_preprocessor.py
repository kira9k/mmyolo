# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmyolo.registry import MODELS
from .data_preprocessor import YOLOv5DetDataPreprocessor


@MODELS.register_module()
class YOLOv5FocusDetDataPreprocessor(YOLOv5DetDataPreprocessor):
    """Pack normalized grayscale images while retaining full-size box metadata."""

    def forward(self, data, training=False):
        result = super().forward(data, training=training)
        inputs = result['inputs']
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError('Expected a single-channel grayscale image batch')
        if inputs.shape[-2] % 4 or inputs.shape[-1] % 4:
            raise ValueError('Focus4 source dimensions must be divisible by 4')
        result['inputs'] = F.pixel_unshuffle(inputs, 4).contiguous()
        return result
