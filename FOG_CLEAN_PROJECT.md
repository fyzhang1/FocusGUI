# FOG Release Audit

This directory is the compact, public-code release of **FOG (Focused
Omni-modal GUI Agents)**.

## Canonical Pipeline

1. Frozen-backbone evidence extraction.
2. Step-adaptive channel prioritization.
3. Deterministic action-sufficient projection.
4. Zero-initialized residual conditioning at the generation boundary.
5. Autoregressive action generation through the original frozen decoder and
   LM head.

The formal Qwen configuration uses deterministic compression (`vib=false`,
`beta_kl=0`). Legacy EIB/EAF classification heads, coordinate-grid prediction,
and replacement LM heads are absent.

## Retained Method Code

| Purpose | File |
|---|---|
| Shared data split and projections | `scripts/fog_common.py` |
| Label-free observation descriptors | `scripts/extract_observation_descriptors.py` |
| Qwen FOG training/generation | `scripts/train_ares_lm_adapter_qwen.py` |
| Qwen evidence extraction | `scripts/extract_qwen_ares_evidence.py` |
| Portable MiniCPM/VITA/Baichuan FOG | `scripts/train_ares_lm_adapter_multibackbone.py` |
| Other-backbone hidden extraction | `scripts/extract_{minicpm,vita,baichuan}_hidden_states.py` |
| Fusion ablation | `scripts/run_qwen_ares_fusion_ablation.sh` |
| Channel and trajectory analyses | `scripts/plot_ares_channel_prioritization.py`, `scripts/analyze_fog_trajectory_depth.py` |

Historical `ares_` names are retained only for command/checkpoint compatibility.

## Intentionally Excluded

- all `data/` content, including annotations, media, and generated splits;
- all `logs/`, caches, checkpoints, and generated predictions;
- GUI grounding baseline implementations and their task configurations;
- unrelated benchmark tasks and model wrappers;
- local absolute paths, compatibility stubs, and Python caches.

These exclusions are enforced by `.gitignore` and validated before release.
