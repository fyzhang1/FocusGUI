#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
PRETRAINED="${BAICHUAN_OMNI_PRETRAINED:-baichuan-inc/Baichuan-Omni-1d5}"

cd "${REPO_ROOT}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON:-python}" scripts/train_focusgui_lm_adapter_multibackbone.py \
  --backbone baichuan \
  --pretrained "${PRETRAINED}" \
  --hidden-cache logs/focusgui_evidence/baichuan_omni_1d5_hidden_p256_strict.npz \
  --out-dir logs/focusgui_lm_adapter_baichuan_strict \
  --max-video-frames 4 \
  --image-max-pixels 200704 \
  --video-max-pixels 200704 \
  --activation-offload-threshold 4096 \
  "$@"
