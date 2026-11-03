
#CONFIG_FILE="configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"
#CHECKPOINT_FILE="/home/kira9k/ieos/mmyolo/work_dirs/yolov8_21_02_2025_multi_label_false_back_norm_people_255/best_coco_bbox_mAP_epoch_180.pth"
#WORK_DIR="work_dirs/onnx_export/yolov8_v3"

CONFIG_FILE="work_dirs/yolov8_s_focus_add_last_spp_concat_surgery2_480/yolov8_s_focus_add_last_spp_concat_surgery2.py"
#CHECKPOINT_FILE="work_dirs/yolov8_s_small_dataset_v3/best_coco_bbox_mAP_epoch_499.pth"
CHECKPOINT_FILE="work_dirs/yolov8_s_focus_add_last_spp_concat_surgery2_480/best_coco_bbox_mAP_epoch_497.pth"
WORK_DIR="work_dirs/yolov8_s_focus_add_last_spp_concat_surgery2_480/onnx"

python projects/easydeploy/tools/export_onnx.py  ${CONFIG_FILE}  ${CHECKPOINT_FILE} \
    --work-dir ${WORK_DIR}   \
    --img-size 480 640   \
    --batch 1    \
    --device cpu    \
    --simplify  \
    --opset 11      \
    --pre-topk 1000     \
    --keep-topk 100      \
    --iou-threshold 0.65    \
    --score-threshold 0.001 \
    --export-type YOLOv5 \
    --model-surgery 2 \
    --model-only
