#!/bin/bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
ZERO_RESIDUAL_CHECK="${ZERO_RESIDUAL_CHECK:-3}"
RESUME_COMPLETED="${RESUME_COMPLETED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/logs/focusgui_lm_adapter_other_models_${RUN_ID}}"

if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${REPO_ROOT}/${RUN_ROOT}"
fi

mkdir -p "${RUN_ROOT}"
LOG_FILE="${RUN_ROOT}/sequence.log"
STATUS_FILE="${RUN_ROOT}/status.tsv"
PID_FILE="${RUN_ROOT}/runner.pid"
CURRENT_MODEL="initializing"
TERMINATION_SIGNAL=""
STARTED_AT="$(date --iso-8601=seconds)"

printf '%s\n' "$$" > "${PID_FILE}"
printf 'status\tRUNNING\ncurrent_model\t%s\nstarted_at\t%s\ngpu\t%s\n' \
  "${CURRENT_MODEL}" "${STARTED_AT}" "${GPU}" > "${STATUS_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

finish() {
  local exit_code=$?
  local final_status="COMPLETED"
  if [[ ${exit_code} -ne 0 ]]; then
    final_status="FAILED"
  fi
  printf 'status\t%s\ncurrent_model\t%s\nstarted_at\t%s\nfinished_at\t%s\ngpu\t%s\nexit_code\t%s\n' \
    "${final_status}" "${CURRENT_MODEL}" "${STARTED_AT}" "$(date --iso-8601=seconds)" \
    "${GPU}" "${exit_code}" > "${STATUS_FILE}"
  printf '[%s] Sequence %s with exit code %s.\n' \
    "$(date --iso-8601=seconds)" "${final_status}" "${exit_code}"
}
trap finish EXIT

terminate() {
  TERMINATION_SIGNAL="$1"
  printf '[%s] Received %s; terminating sequence.\n' \
    "$(date --iso-8601=seconds)" "${TERMINATION_SIGNAL}"
  if [[ "${TERMINATION_SIGNAL}" == "INT" ]]; then
    exit 130
  fi
  exit 143
}
trap 'terminate INT' INT
trap 'terminate TERM' TERM

run_model() {
  local model_name="$1"
  local launcher="$2"
  local output_dir="$3"
  shift 3

  if [[ "${RESUME_COMPLETED}" == "1" && -f "${output_dir}/SEQUENCE_STAGE_COMPLETE" ]]; then
    printf '[%s] Skipping completed stage %s.\n' \
      "$(date --iso-8601=seconds)" "${model_name}"
    return
  fi

  CURRENT_MODEL="${model_name}"
  printf 'status\tRUNNING\ncurrent_model\t%s\nstarted_at\t%s\ngpu\t%s\n' \
    "${CURRENT_MODEL}" "${STARTED_AT}" "${GPU}" > "${STATUS_FILE}"
  printf '\n[%s] Starting %s on physical GPU %s.\n' \
    "$(date --iso-8601=seconds)" "${model_name}" "${GPU}"

  GPU="${GPU}" "${launcher}" \
    --out-dir "${output_dir}" \
    --zero-residual-check "${ZERO_RESIDUAL_CHECK}" \
    "$@"

  touch "${output_dir}/SEQUENCE_STAGE_COMPLETE"
  printf '[%s] Finished %s successfully.\n' \
    "$(date --iso-8601=seconds)" "${model_name}"
}

cd "${REPO_ROOT}"
printf '[%s] Corrected FocusGUI sequential run started.\n' "${STARTED_AT}"
printf 'Repository: %s\nRun root: %s\nGPU: %s\n' "${REPO_ROOT}" "${RUN_ROOT}" "${GPU}"
printf 'Resume completed stages: %s\n' "${RESUME_COMPLETED}"
printf 'Git commit: %s\n' "$(git rev-parse --verify HEAD 2>/dev/null || printf 'unavailable')"
printf 'Shared arguments:'
printf ' %q' "$@"
printf '\n'

run_model \
  "MiniCPM-o-4.5" \
  "${REPO_ROOT}/scripts/train_focusgui_lm_adapter_minicpm.sh" \
  "${RUN_ROOT}/minicpm" \
  "$@"

run_model \
  "VITA-1.5" \
  "${REPO_ROOT}/scripts/train_focusgui_lm_adapter_vita.sh" \
  "${RUN_ROOT}/vita" \
  "$@"

run_model \
  "Baichuan-Omni-1d5" \
  "${REPO_ROOT}/scripts/train_focusgui_lm_adapter_baichuan.sh" \
  "${RUN_ROOT}/baichuan" \
  "$@"

CURRENT_MODEL="all_models"
