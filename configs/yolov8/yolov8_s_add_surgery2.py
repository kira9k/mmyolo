_base_ = './yolov8_s_add.py'

# RGB Add architecture for --model-surgery 2; keep ReLU in compute layers.
model = dict(
    backbone=dict(act_cfg=dict(type='ReLU', inplace=False)),
    neck=dict(act_cfg=dict(type='ReLU', inplace=False)),
    bbox_head=dict(
        head_module=dict(act_cfg=dict(type='ReLU', inplace=False))))

# Match the R5 model-only interface: class logit followed by 64 DFL logits.
deploy_cfg = dict(
    raw_output_format='combined', output_names=['583', '584', '585'])
