#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
PRETRAINED="${MINICPM_O_PRETRAINED:-openbmb/MiniCPM-o-4_5}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON:-python}" scripts/train_focusgui_lm_adapter_multibackbone.py \
  --backbone minicpm \
  --pretrained "${PRETRAINED}" \
  --hidden-cache logs/focusgui_evidence/minicpm_o_4_5_hidden_p256_strict.npz \
  --out-dir logs/focusgui_lm_adapter_minicpm_strict \
  "$@"
