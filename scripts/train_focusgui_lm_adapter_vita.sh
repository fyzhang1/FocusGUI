#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
PRETRAINED="${VITA_PRETRAINED:-VITA-MLLM/VITA-1.5}"
if [[ -n "${VITA_CODE_PATH:-}" ]]; then
  export PYTHONPATH="${VITA_CODE_PATH}:${PYTHONPATH:-}"
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON:-python}" scripts/train_focusgui_lm_adapter_multibackbone.py \
  --backbone vita \
  --pretrained "${PRETRAINED}" \
  --hidden-cache logs/focusgui_evidence/vita_1_5_hidden_p256_strict.npz \
  --out-dir logs/focusgui_lm_adapter_vita_strict \
  "$@"
