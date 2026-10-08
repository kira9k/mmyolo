#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="configs/yolov8/yolov8_s_add_surgery2.py"
CHECKPOINT_FILE="$(cat work_dirs/yolov8_s_add_surgery2/last_checkpoint)"
WORK_DIR="work_dirs/yolov8_s_add_surgery2/pt"

python projects/easydeploy/tools/export_pt.py "$CONFIG_FILE" "$CHECKPOINT_FILE" \
    --work-dir "$WORK_DIR" \
    --model-surgery 2 \
    --model-only \
    --img-size 640 640 \
    --batch-size 1 \
    --opset 11 \
    --backend onnxruntime \
    --device cpu
