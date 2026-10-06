_base_ = './yolov8_s_add_fine_tune.py'

# Keep the RGB training pipelines and the original sigmoid class scores.
model = dict(
    backbone=dict(act_cfg=dict(type='ReLU', inplace=False)),
    neck=dict(act_cfg=dict(type='ReLU', inplace=False)),
    bbox_head=dict(
        head_module=dict(act_cfg=dict(type='ReLU', inplace=False))))

deploy_cfg = dict(
    raw_output_format='combined', output_names=['583', '584', '585'])
