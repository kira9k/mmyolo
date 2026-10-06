_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

model = dict(
    backbone=dict(csp_fusion_mode='add', spp_fusion_mode='add'),
    neck=dict(type='YOLOv8AddPAFPN', csp_fusion_mode='add'))

# Warm-start only from a checkpoint prepared for this architecture.
load_from = None
resume = False
