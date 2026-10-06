CONFIG_FILE="configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"


CHECKPOINT_FILE="work_dirs/yolov8_s_new_dataset_29_08/best_coco_bbox_mAP_epoch_496.pth"
#CHECKPOINT_FILE="/home/kira9k/ieos/mmyolo/work_dirs/yolov8_21_02_2025_multi_label_false_back_norm_people_255/best_coco_bbox_mAP_epoch_180.pth"
WORK_DIR="work_dirs/test/yolovs8_ieos_496"

IMAGE_DIR=${WORK_DIR}/images
METRICS_DIR=${WORK_DIR}/metrics

python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir ${WORK_DIR} --model-surgery 2 --show-dir ${IMAGE_DIR}
