# Keep _base_ before third-party imports for legacy MMEngine parsing.
_base_ = './yolov8_s_add.py'

from copy import deepcopy
from mmyolo.datasets.transforms.grayscale import make_grayscale_pipeline

# Preserve the original 500-epoch training recipe and 0..1 normalization.
model = dict(
    data_preprocessor=dict(mean=[0.], std=[255.], bgr_to_rgb=False),
    backbone=dict(input_channels=1, act_cfg=dict(type='HSwish', inplace=True)),
    neck=dict(act_cfg=dict(type='HSwish', inplace=True)),
    bbox_head=dict(
        score_act_cfg=dict(type='HSigmoid', bias=3., divisor=6.),
        head_module=dict(act_cfg=dict(type='HSwish', inplace=True))))

train_pipeline = make_grayscale_pipeline(_base_.train_pipeline)
train_pipeline_stage2 = make_grayscale_pipeline(_base_.train_pipeline_stage2)
test_pipeline = make_grayscale_pipeline(_base_.test_pipeline)
tta_pipeline = make_grayscale_pipeline(_base_.tta_pipeline)
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

custom_hooks = deepcopy(_base_.custom_hooks)
for hook in custom_hooks:
    if hook['type'] == 'mmdet.PipelineSwitchHook':
        hook['switch_pipeline'] = train_pipeline_stage2

load_from = None
resume = False
