#!/bin/bash
set -euo pipefail

TASK_NAME="task_v2_vita"
MODEL_NAME="vita"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/${TASK_NAME}"
PRETRAINED="${VITA_PRETRAINED:-VITA-MLLM/VITA-1.5}"
MODEL_ARGS="pretrained=${PRETRAINED}"
if [[ -n "${VITA_CODE_PATH:-}" ]]; then
  export PYTHONPATH="${VITA_CODE_PATH}:${PYTHONPATH:-}"
fi

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
