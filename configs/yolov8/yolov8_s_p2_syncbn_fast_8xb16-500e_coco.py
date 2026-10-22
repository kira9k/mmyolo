_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# P2-P5 feature maps: 160, 80, 40 and 20 pixels for a 640-pixel input.
strides = [4, 8, 16, 32]
num_det_layers = 4
feature_channels = [128, 256, 512, _base_.last_stage_out_channels]

model = dict(
    backbone=dict(out_indices=(1, 2, 3, 4)),
    neck=dict(in_channels=feature_channels, out_channels=feature_channels),
    bbox_head=dict(
        head_module=dict(
            in_channels=feature_channels,
            featmap_strides=strides,
            # Keep the standard YOLOv8s classification width despite P2.
            cls_out_channels=128),
        prior_generator=dict(strides=strides)))

deploy_cfg = dict(
    raw_output_format='combined',
    output_names=['head_s4', 'head_s8', 'head_s16', 'head_s32'])

# P2 changes neck/head shapes; use a separate training run.
#load_from = "/mmyolo/work_dirs/yolov8_s_p2_cls128_small_dataset_v3/best_coco_bbox_mAP_epoch_20.pth"
resume = True
