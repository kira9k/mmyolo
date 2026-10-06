_base_ = './yolov8_s_fine_tune.py'

model = dict(
    backbone=dict(
        csp_fusion_mode='add',
        spp_fusion_mode='add',
        frozen_stages=-1,
        norm_eval=False),
    neck=dict(type='YOLOv8AddPAFPN', csp_fusion_mode='add'))

# Use the dataset available in the main YOLOv8-s config.
data_root = 'data/small_people_v3/'
data_root_test = data_root
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
val_evaluator = dict(ann_file=data_root + 'val/ann.json')
test_evaluator = dict(ann_file=data_root + 'val/ann.json')

# Train BN and use one scheduler for all trainable layers.
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
optim_wrapper = dict(
    paramwise_cfg=dict(_delete_=True, base_total_batch_size=32))
param_scheduler = None
default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        max_epochs=100,
        lr_factor=0.1,
        warmup_bias_lr=0.001))

# The original fine-tune checkpoint has incompatible convolution shapes.
load_from = None
resume = False
