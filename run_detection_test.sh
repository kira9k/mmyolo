CONFIG_FILE="/home/kira9k/ieos/mmyolo/configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py"
CHECKPOINT_FILE="/home/kira9k/ieos/mmyolo/work_dirs/last_train_norm_people_25_12/yolov5_s_norm_people_last_train_25_12/best_coco_bbox_mAP_epoch_260.pth"
WORK_DIR="/home/kira9k/ieos/mmyolo/work_dirs/test/norm_people_last_test_25_12/yolov5_s_norm_people_big_dataset_last_train_25_12_showing_test"

IMAGE_DIR=${WORK_DIR}/images
METRICS_DIR=${WORK_DIR}/metrics

python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir ${METRICS_DIR} --model-surgery 2 --show-dir ${IMAGE_DIR}