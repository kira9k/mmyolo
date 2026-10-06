
CONFIG_FILE="configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"
#CHECKPOINT_FILE="/home/kira9k/ieos/mmyolo/work_dirs/yolov8_21_02_2025_multi_label_false_back_norm_people_255/best_coco_bbox_mAP_epoch_180.pth"
#WORK_DIR="work_dirs/onnx_export/yolov8_v3"

#CONFIG_FILE="/home/kira9k/ieos/mmyolo/configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py"
#CHECKPOINT_FILE="/home/kira9k/ieos/mmyolo/work_dirs/yolov8_s_big_dataset/best_coco_bbox_mAP_epoch_495.pth"
CHECKPOINT_FILE="work_dirs/yolov8_21_02_2025_multi_label_false_back_norm_people/best_coco_bbox_mAP_epoch_140.pth"
WORK_DIR="work_dirs/onnx_export/yolov8_s_r5_640_not_surgery"

python projects/easydeploy/tools/export_pt.py  ${CONFIG_FILE}  ${CHECKPOINT_FILE} \
    --work-dir ${WORK_DIR}   \
    --img-size 640 640   \
    --batch 1    \
    --device cpu    \
    --pre-topk 500 \
    --keep-topk 100 \
    --iou-threshold 0.65    \
    --score-threshold 0.25 \
    --export-type YOLOv5 \
    --opset 11 \
    --model-only \
    --model-surgery 2 \
    # --simplify  \
    
    
    
    