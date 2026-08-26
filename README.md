# FOG: Focused Omni-modal GUI Agents

FOG is a lightweight decoder-conditioning method for sequential omni-modal
GUI grounding. It learns which evidence channels should influence the next
action while preserving the pretrained model's original autoregressive
decoder and LM head.

At each interaction step, FOG:

1. extracts complementary evidence from the frozen omni-modal backbone and
   label-free screen, audio, video, text/history, and state descriptors;
2. computes step-adaptive channel priorities from the current full-modal
   query;
3. compresses the selected evidence into action-sufficient latent tokens;
4. injects a zero-initialized residual at the action-generation boundary; and
5. generates the structured action through the frozen decoder and original
   LM head.

Only the evidence projectors, prioritization network, compression module, and
latent-token projection are trained. The backbone, Transformer decoder, and
LM head remain frozen.

## Release Scope

This compact release supports the four backbones used in our experiments:

- Qwen2.5-Omni-7B
- MiniCPM-o-4.5
- VITA-1.5
- Baichuan-Omni-1.5

The repository contains FOG training, frozen-evidence extraction, OmniGUI
evaluation utilities, component ablations, and analysis code. It intentionally
does **not** contain:

- OmniGUI annotations or media;
- model weights or hidden-state caches;
- checkpoints, predictions, or experiment logs;
- screenshot-only GUI grounding baseline implementations.

See [DATA.md](DATA.md) for the separate data setup.

## Repository Layout

```text
scripts/
  fog_common.py                         shared split/projection utilities
  prepare_omnigui_chunks.py             episode-preserving data preparation
  extract_observation_descriptors.py    label-free modality descriptors
  extract_qwen_ares_evidence.py         Qwen channel evidence extraction
  extract_*_hidden_states.py             other-backbone hidden extraction
  train_ares_lm_adapter_qwen.py          Qwen FOG training and generation
  train_ares_lm_adapter_multibackbone.py portable three-backbone runner
  run_qwen_ares_fusion_ablation.sh       fusion/component controls
  analyze_fog_trajectory_depth.py        trajectory-depth analysis
  plot_ares_channel_prioritization.py    channel-priority analysis
lmms_eval/
  models/                                four supported backbone wrappers
  tasks/agentcpm_gui/                    OmniGUI task and evaluator
reports/                                 curated paper tables and figures
```

Some implementation filenames retain the historical internal prefix `ares_`
for checkpoint and command compatibility. They implement the final FOG method
described above; no legacy EIB/EAF classification head is included.

## Installation

```bash
conda create -n fog python=3.10 -y
conda activate fog
pip install -e ./lmms_eval
pip install librosa qwen-omni-utils
```

Install each backbone's official runtime dependencies before using its local
wrapper. VITA may additionally require its source tree through
`VITA_CODE_PATH`; Baichuan installations that need a compatibility package can
set `BAICHUAN_COMPAT_PATH`. The supported environment variables are listed in
[.env.example](.env.example).

## Data Preparation

After downloading OmniGUI into `data/` as described in [DATA.md](DATA.md), run:

```bash
python scripts/prepare_omnigui_chunks.py
```

The deterministic paper split is created later by episode, with seed
`20260626`, using 70%/10%/20% train/dev/test proportions. Episodes never cross
splits, and test labels are used only for final evaluation.

## Qwen Workflow

Extract frozen, label-free Qwen evidence in two shards:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_qwen_ares_evidence.py \
  --shard-index 0 --num-shards 2 \
  --out logs/fog_evidence/qwen_shard0.npz

CUDA_VISIBLE_DEVICES=1 python scripts/extract_qwen_ares_evidence.py \
  --shard-index 1 --num-shards 2 \
  --out logs/fog_evidence/qwen_shard1.npz
```

Train the formal deterministic FOG adapter and generate held-out actions:

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/train_ares_lm_adapter_qwen.py \
  --semantic-evidence-cache \
    logs/fog_evidence/qwen_shard0.npz,logs/fog_evidence/qwen_shard1.npz \
  --fusion-mode dynamic \
  --no-vib --beta-kl 0 --lambda-weight 0 \
  --generation-split test \
  --out-dir logs/fog_qwen
```

This command trains only FOG and then performs fresh autoregressive generation
through Qwen's frozen decoder and original LM head.

## Other Backbones

First extract the shared label-free observation descriptors:

```bash
python scripts/extract_observation_descriptors.py
```

Then extract the selected backbone's frozen hidden evidence and launch FOG:

```bash
python scripts/extract_minicpm_hidden_states.py
bash scripts/train_ares_lm_adapter_minicpm.sh

python scripts/extract_vita_hidden_states.py
bash scripts/train_ares_lm_adapter_vita.sh

python scripts/extract_baichuan_hidden_states.py
bash scripts/train_ares_lm_adapter_baichuan.sh
```

The three stages can also be run sequentially with:

```bash
bash scripts/run_corrected_ares_other_models_sequential.sh
```

All generated artifacts are written under `logs/` and excluded from Git.

## Analyses

The release includes scripts for:

- dynamic channel-priority statistics across task categories and phases;
- trajectory-depth performance analysis with episode-clustered bootstrap CIs;
- sequential leave-one-modality-out sensitivity;
- observation redundancy statistics;
- controlled Qwen fusion ablations.

Curated, non-log paper artifacts are retained under `reports/`.

## Acknowledgment

FOG is evaluated on [OmniGUI](https://github.com/omni-gui/OmniGUI) and uses a
compact subset of its evaluation framework. Please follow the upstream
dataset, model, and evaluator licenses and cite OmniGUI when using this code.
