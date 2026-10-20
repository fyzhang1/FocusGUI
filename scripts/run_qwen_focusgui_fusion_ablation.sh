#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
OUT_ROOT="${OUT_ROOT:-logs/focusgui_lm_adapter_qwen_fusion_ablation_20260802}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_ROOT"

run_mode() {
    local mode="$1"
    local out_dir="$OUT_ROOT/$mode"
    mkdir -p "$out_dir"
    printf '%s\n' "$(date --iso-8601=seconds) START $mode" | tee -a "$OUT_ROOT/run_status.log"
    "$TORCHRUN" --standalone --nproc_per_node=2 \
        scripts/train_focusgui_lm_adapter_qwen.py \
        --fusion-mode "$mode" \
        --generation-split test \
        --epochs 3 \
        --patience 2 \
        --grad-accum 8 \
        --lr 2e-4 \
        --weight-decay 1e-3 \
        --residual-scale 1.0 \
        --activation-offload-threshold 5500 \
        --no-vib \
        --beta-kl 0 \
        --lambda-weight 0 \
        --out-dir "$out_dir" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$out_dir/run.log"
    printf '%s\n' "$(date --iso-8601=seconds) DONE $mode" | tee -a "$OUT_ROOT/run_status.log"
}

EXTRA_ARGS=("$@")
# Train every variant under the same protocol; do not assume cached results.
for mode in dynamic fullmodal_only uniform static naive_concat; do
    run_mode "$mode"
done

"$PYTHON" - "$OUT_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
missing = [mode for mode in ("dynamic", "fullmodal_only", "uniform", "static", "naive_concat") if not (root / mode / "results.json").exists()]
if missing:
    raise SystemExit(f"Missing formal results: {missing}")
print("All formal fusion-control results are present.")
PY
