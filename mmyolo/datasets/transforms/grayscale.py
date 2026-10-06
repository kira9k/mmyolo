# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import cv2
from mmcv.transforms import BaseTransform

from mmyolo.registry import TRANSFORMS


def make_grayscale_pipeline(value):
    """Copy a pipeline, adapting loaders and removing color-only transforms."""
    if isinstance(value, list):
        result = []
        for item in value:
            if (isinstance(item, dict) and item.get('type')
                    in ('YOLOv5HSVRandomAug', 'HueSaturationValue')):
                continue
            result.append(make_grayscale_pipeline(item))
            if (isinstance(item, dict) and item.get('type', '').split('.')[-1]
                    == 'LoadImageFromFile'):
                result.append(dict(type='ToGrayscale'))
        return result
    if isinstance(value, dict):
        result = {
            key: make_grayscale_pipeline(item)
            for key, item in value.items()
        }
        transform_type = result.get('type', '')
        if transform_type.split('.')[-1] == 'LoadImageFromFile':
            result['color_type'] = 'grayscale'
        if transform_type == 'YOLOv5RandomAffine':
            result['border_val'] = (114, )
        return result
    return deepcopy(value)


@TRANSFORMS.register_module()
class ToGrayscale(BaseTransform):
    """Ensure a single-channel image, including inputs loaded from arrays."""

    def transform(self, results: dict) -> dict:
        image = results['img']
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[..., 0]
            else:
                raise ValueError(
                    'Expected grayscale or three-channel BGR image')
        elif image.ndim != 2:
            raise ValueError('Expected a 2D or HWC image')
        results['img'] = image
        results['img_shape'] = image.shape[:2]
        return results
