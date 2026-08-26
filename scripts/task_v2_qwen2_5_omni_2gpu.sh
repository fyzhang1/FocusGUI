#!/bin/bash
set -euo pipefail

TASK_BASE="task_v2_qwen2_5_omni"
MODEL_NAME="qwen2_5_omni"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHUNK_DIR="${REPO_ROOT}/data/qwen2_5_omni_chunks"

PRETRAINED="${QWEN2_5_OMNI_PRETRAINED:-Qwen/Qwen2.5-Omni-7B}"
ATTN_IMPLEMENTATION="${QWEN2_5_OMNI_ATTN_IMPLEMENTATION:-eager}"
MAX_NUM_FRAMES="${QWEN2_5_OMNI_MAX_NUM_FRAMES:-768}"
IMAGE_MAX_PIXELS="${QWEN2_5_OMNI_IMAGE_MAX_PIXELS:-1003520}"
VIDEO_MAX_FRAMES="${QWEN2_5_OMNI_VIDEO_MAX_FRAMES:-4}"
VIDEO_MAX_PIXELS="${QWEN2_5_OMNI_VIDEO_MAX_PIXELS:-200704}"
USE_AUDIO_IN_VIDEO="${QWEN2_5_OMNI_USE_AUDIO_IN_VIDEO:-False}"

cd "${REPO_ROOT}"
mkdir -p "${CHUNK_DIR}"
python scripts/prepare_omnigui_chunks.py

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launch_chunk() {
  local gpu="$1"
  local idx="$2"
  local task_name="${TASK_BASE}_chunk${idx}"
  local log_dir="${REPO_ROOT}/logs/${task_name}"
  local response_cache_path="logs/${task_name}/response_cache.jsonl"
  local model_args

  mkdir -p "${log_dir}"
  model_args="pretrained=${PRETRAINED},attn_implementation=${ATTN_IMPLEMENTATION},max_num_frames=${MAX_NUM_FRAMES},image_max_pixels=${IMAGE_MAX_PIXELS},video_max_frames=${VIDEO_MAX_FRAMES},video_max_pixels=${VIDEO_MAX_PIXELS},use_audio_in_video=${USE_AUDIO_IN_VIDEO},response_cache_path=${response_cache_path}"

  echo "Launching ${task_name} on GPU ${gpu}..."
  echo "  MODEL_ARGS: ${model_args}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup conda run --no-capture-output -n omnigui python -m lmms_eval \
    --model "${MODEL_NAME}" \
    --model_args "${model_args}" \
    --tasks "${task_name}" \
    --batch_size 1 \
    --log_samples \
    --output_path "logs/${task_name}" \
    > "${log_dir}/run.log" 2>&1 &
  echo "  PID: $!"
  echo "  Logs: ${log_dir}/run.log"
}

launch_chunk 0 0
launch_chunk 1 1

echo "Both chunks launched."
echo "Monitor:"
echo "  tail -f logs/${TASK_BASE}_chunk0/run.log"
echo "  tail -f logs/${TASK_BASE}_chunk1/run.log"
