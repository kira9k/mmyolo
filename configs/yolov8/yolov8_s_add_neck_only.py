_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# Replace only the four inter-scale concatenations; retain C2f and SPPF.
model = dict(neck=dict(type='YOLOv8AddPAFPN', csp_fusion_mode='concat'))
load_from = None
resume = False
