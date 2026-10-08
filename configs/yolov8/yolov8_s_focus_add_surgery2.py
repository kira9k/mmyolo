_base_ = './yolov8_n_focus_add_surgery2.py'

# Keep the external Focus4 contract and scale all compute layers to YOLOv8s.
widen_factor = 0.5

model = dict(
    backbone=dict(widen_factor=widen_factor),
    neck=dict(widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
