
CONFIG_FILE="/home/kira9k/ieos/mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"
CHECKPOINT_FILE="/home/kira9k/ieos/mmyolo/work_dirs/last_train_norm_people_25_12/yolov8_s_norm_people_big_data_send/best_coco_bbox_mAP_epoch_200.pth"
WORK_DIR="/home/kira9k/ieos/mmyolo/work_dirs/onnx_export/yolov8_s_norm_people_big_data"

python projects/easydeploy/tools/export_onnx.py  ${CONFIG_FILE}  ${CHECKPOINT_FILE} \
    --work-dir ${WORK_DIR}   \
    --img-size 640 640   \
    --batch 1    \
    --device cpu    \
    --simplify  \
    --opset 11      \
    --pre-topk 1000     \
    --keep-topk 100      \
    --iou-threshold 0.65    \
    --score-threshold 0.25 \
    --export-type YOLOv5 \
    --model-surgery 2
