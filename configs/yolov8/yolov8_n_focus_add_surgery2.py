_base_ = './yolov8_n_syncbn_fast_8xb16-500e_coco.py'

from copy import deepcopy
from mmyolo.datasets.transforms.grayscale import make_grayscale_pipeline

model = dict(
    data_preprocessor=dict(
        type='YOLOv5FocusDetDataPreprocessor',
        mean=[0.],
        std=[255.],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='YOLOv8FocusCSPDarknet',
        input_channels=16,
        csp_fusion_mode='add',
        spp_fusion_mode='add',
        act_cfg=dict(type='ReLU', inplace=False)),
    neck=dict(
        type='YOLOv8AddPAFPN',
        csp_fusion_mode='add',
        act_cfg=dict(type='ReLU', inplace=False)),
    bbox_head=dict(head_module=dict(act_cfg=dict(type='ReLU', inplace=False))))

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

deploy_cfg = dict(
    raw_output_format='combined',
    output_names=['583', '584', '585'],
    input_spatial_divisor=4,
    preprocess=dict(
        color_order='grayscale',
        packing='pixel_unshuffle_4',
        channel_order='channel = (row % 4) * 4 + col % 4',
        packing_location='external_cpu',
        input_scale=1. / 255.,
        resize_mode='letterbox',
        pad=114))

# Removing nonlinear stem layers changes the model; train this variant anew.
load_from = None
resume = False
