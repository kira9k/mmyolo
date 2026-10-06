CONFIG_FILE="configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"
#CONFIG_FILE="configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py"
WORK_DIR="work_dirs/yolov8_s_small_dataset_v3"

#WORK_DIR="work_dirs/yolov8_21_02_2025_multi_label_false_back_norm_people" # ← для эксперимента с multi_label=False с norm 1.1.1

python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --model-surgery 2 #--resume
