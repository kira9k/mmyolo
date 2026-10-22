_base_ = './yolov8_n_focus_add_spp_concat_surgery2.py'

# Fuse only the first split and final bottleneck output in every C2f.
# Keep one four-branch Concat after SPPF pooling, with no pooling-branch Add.
model = dict(
    backbone=dict(csp_fusion_mode='add_last', spp_fusion_mode='concat'),
    neck=dict(csp_fusion_mode='add_last'))

load_from = None
resume = False
