_base_ = './yolov8_s_focus_add_surgery2.py'

# Concatenate the four SPPF branches; keep C2f and neck fusion as Add.
model = dict(backbone=dict(spp_fusion_mode='concat'))
