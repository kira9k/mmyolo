CONFIG_FILE="configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py"
WORK_DIR="work_dirs/last_train_norm_people_25_12/test"

python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --model-surgery 2 
