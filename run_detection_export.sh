
CONFIG_FILE="configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py"
CHECKPOINT_FILE="work_dirs/fully_training_small+norn_people/yolov5s_norm_small_people/best_coco_bbox_mAP_epoch_300.pth"
WORK_DIR="work_dirs/onnx_export/test_docker"

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