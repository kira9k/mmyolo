CONFIG_FILE="/home/kira9k/ieos/mmyolo/configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py"
WORK_DIR="/home/kira9k/ieos/mmyolo/work_dirs/last_train_norm_people_25_12/yolov5_s_norm_people_last_train_25_12_showing"

python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --model-surgery 2 --resume
