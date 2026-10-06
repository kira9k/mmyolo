
#CONFIG_FILE="configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"
#CHECKPOINT_FILE="/home/kira9k/ieos/mmyolo/work_dirs/yolov8_21_02_2025_multi_label_false_back_norm_people_255/best_coco_bbox_mAP_epoch_180.pth"
#WORK_DIR="work_dirs/onnx_export/yolov8_v3"

CONFIG_FILE="configs/yolov8/yolov8_s_add_gray_hard.py"
#CHECKPOINT_FILE="work_dirs/yolov8_s_small_dataset_v3/best_coco_bbox_mAP_epoch_499.pth"
CHECKPOINT_FILE="work_dirs/yolov8_s_add_gray_hard_scratch_v2/best_coco_bbox_mAP_epoch_20.pth"
WORK_DIR="work_dirs/onnx_export/yolov8_s_add_gray_hard_scratch_v2"

python projects/easydeploy/tools/export_onnx.py  ${CONFIG_FILE}  ${CHECKPOINT_FILE} \
    --model-only \
    --work-dir ${WORK_DIR}   \
    --img-size 640 640   \
    --batch 1    \
    --device cpu    \
    --simplify  \
    --opset 11      \
    --export-type MMYOLO \
    --model-surgery 0