#!/bin/bash
set -euo pipefail

TASK_NAME="task_v2_baichuan_omni"
MODEL_NAME="baichuan_omni"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/${TASK_NAME}"
PRETRAINED="${BAICHUAN_OMNI_PRETRAINED:-baichuan-inc/Baichuan-Omni-1d5}"
MODEL_ARGS="backend=local,pretrained=${PRETRAINED}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

if [[ -n "${BAICHUAN_COMPAT_PATH:-}" ]]; then
  export PYTHONPATH="${BAICHUAN_COMPAT_PATH}:${PYTHONPATH:-}"
fi

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
