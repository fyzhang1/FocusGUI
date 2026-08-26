#!/bin/bash
set -euo pipefail

TASK_NAME="task_v2_qwen2_5_omni"
MODEL_NAME="qwen2_5_omni"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/${TASK_NAME}"

PRETRAINED="${QWEN2_5_OMNI_PRETRAINED:-Qwen/Qwen2.5-Omni-7B}"
ATTN_IMPLEMENTATION="${QWEN2_5_OMNI_ATTN_IMPLEMENTATION:-eager}"
MAX_NUM_FRAMES="${QWEN2_5_OMNI_MAX_NUM_FRAMES:-768}"

# Memory / input controls. Defaults align with qwen3_omni (4 video frames,
# audio handled as a separate element, capped per-image/per-frame pixels).
IMAGE_MAX_PIXELS="${QWEN2_5_OMNI_IMAGE_MAX_PIXELS:-1003520}"
VIDEO_MAX_FRAMES="${QWEN2_5_OMNI_VIDEO_MAX_FRAMES:-4}"
VIDEO_MAX_PIXELS="${QWEN2_5_OMNI_VIDEO_MAX_PIXELS:-200704}"
USE_AUDIO_IN_VIDEO="${QWEN2_5_OMNI_USE_AUDIO_IN_VIDEO:-False}"
RESPONSE_CACHE_PATH="${QWEN2_5_OMNI_RESPONSE_CACHE_PATH:-logs/${TASK_NAME}/response_cache.jsonl}"

MODEL_ARGS="pretrained=${PRETRAINED},attn_implementation=${ATTN_IMPLEMENTATION},max_num_frames=${MAX_NUM_FRAMES},image_max_pixels=${IMAGE_MAX_PIXELS},video_max_frames=${VIDEO_MAX_FRAMES},video_max_pixels=${VIDEO_MAX_PIXELS},use_audio_in_video=${USE_AUDIO_IN_VIDEO},response_cache_path=${RESPONSE_CACHE_PATH}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
