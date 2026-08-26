#!/bin/bash
set -euo pipefail

TASK_NAME="task_v2_minicpm_o"
MODEL_NAME="minicpm_o"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/${TASK_NAME}"
PRETRAINED="${MINICPM_O_PRETRAINED:-openbmb/MiniCPM-o-4_5}"
MODEL_ARGS="backend=local,pretrained=${PRETRAINED},max_video_frames=${MINICPM_O_MAX_VIDEO_FRAMES:-4},use_video_audio=${MINICPM_O_USE_VIDEO_AUDIO:-True}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "Launching ${TASK_NAME}..."
echo "  MODEL_ARGS: ${MODEL_ARGS}"
nohup conda run --no-capture-output -n omnigui python -m lmms_eval \
  --model "${MODEL_NAME}" \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASK_NAME}" \
  --batch_size 1 \
  --log_samples \
  --output_path "logs/${TASK_NAME}" \
  > "${LOG_DIR}/run.log" 2>&1 &
echo "  PID: $!"
echo "Logs: ${LOG_DIR}/run.log"
