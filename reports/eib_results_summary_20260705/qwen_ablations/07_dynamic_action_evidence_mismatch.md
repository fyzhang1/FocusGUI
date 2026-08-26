# Dynamic Action-Evidence Mismatch Diagnostic

## Question

Does a multi-step omni-modal GUI trajectory require different evidence at different steps, and does each complete observation contain evidence that is unnecessary or interfering for the next action?

## Experimental Design

We run a training-free leave-one-modality-out sensitivity probe on the frozen local Qwen2.5-Omni-7B backbone. The sample is fixed before inference using seed `20260716`: 20 multi-step episodes from each of the five OmniGUI task categories, giving 100 episodes, 352 steps, and 252 adjacent-step transitions. For every step, the model performs five teacher-forced forward passes: full observation, no current screen, no video, no audio, and no history. All 1,760 planned condition rows completed; no inference failure was recorded.

For official ground-truth action tokens $y_t$, define modality sensitivity as

$$
\Delta_t^m = \bar{\ell}(y_t\mid\mathcal O_t)-\bar{\ell}(y_t\mid\mathcal O_t\setminus m),
$$

where $\bar{\ell}$ is average token log-likelihood. The effect threshold is fixed at $\epsilon=0.01$ nat/token:

- $\Delta_t^m>\epsilon$: materially useful under the probe;
- $\Delta_t^m<-\epsilon$: potentially interfering, because removing it improves target likelihood;
- $|\Delta_t^m|\leq\epsilon$: low sensitivity, hence locally removable under the probe.

The modality role-change rate is the fraction of comparable modality instances whose role changes between adjacent steps. Dominant-evidence switching is computed only when both adjacent steps have at least one materially useful modality.

## Main Diagnostic Results

| Category | Steps | Role change across steps | Dominant switch | Episodes with a state switch | Useful modality instances | Interfering modality instances | Steps with interference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 352 | 45.25% | 53.42% | 94.00% | 41.55% | 53.94% | 90.06% |
| Localization | 56 | 50.00% | 75.00% | 95.00% | 44.12% | 53.43% | 94.64% |
| Semantic | 99 | 42.57% | 45.83% | 100.00% | 42.82% | 51.86% | 89.90% |
| Cross-modal | 58 | 41.54% | 47.62% | 80.00% | 40.76% | 56.40% | 91.38% |
| Temporal | 76 | 50.00% | 48.57% | 95.00% | 40.85% | 54.23% | 86.84% |
| Instant | 63 | 43.42% | 55.17% | 100.00% | 38.79% | 55.17% | 88.89% |

Across all steps, an average observation exposes `3.71` available evidence modalities, but only `1.54` are materially useful for the official next action under this probe. A mean of `2.00` modalities per step are potentially interfering. The strict low-sensitivity share is `4.51%`; it should not be conflated with the interfering share.

## Mean Sensitivity by Modality

Positive values indicate that retaining the modality improves official-action likelihood on average; negative values indicate that omission improves it.

| Category | Screen | Video | Audio | History |
|---|---:|---:|---:|---:|
| Overall | -0.1289 | -0.1029 | +0.0537 | +0.1756 |
| Localization | -0.1471 | -0.0839 | +0.0821 | +0.2342 |
| Semantic | -0.1196 | -0.0826 | +0.0729 | +0.0720 |
| Cross-modal | -0.1250 | -0.1533 | +0.0678 | +0.2543 |
| Temporal | -0.1314 | -0.1092 | +0.0307 | +0.1143 |
| Instant | -0.1279 | -0.0979 | +0.0129 | +0.3290 |

These averages do not mean that screen or video is universally irrelevant. Their role is step-dependent: the aggregate role-change result shows that a modality can move between useful, low-sensitivity, and interfering states along a trajectory.

## Intra-Step Redundancy Proxies

Two label-free descriptive statistics are computed on the same fixed manifest.

| Statistic | Valid steps | Mean | Median | Q1 | Q3 | Additional result |
|---|---:|---:|---:|---:|---:|---|
| First-to-last video-frame grayscale MAD | 242 | 12.59% | 7.41% | 1.14% | 18.57% | 31.0% below 2%; 43.0% below 5% |
| Target-box area / screenshot area | 168 | 4.01% | 0.87% | 0.39% | 3.79% | Median 99.13% of screen lies outside the target box |

The video statistic uses the first and last decoded frames resized to `16x16`; it is a conservative endpoint-staticity proxy, not a complete measure of repeated intermediate frames. The target-area statistic measures spatial support sparsity, not proof that every pixel outside the target is useless.

## Supported Conclusion

The diagnostic supports the existence of a **dynamic action-evidence mismatch** in the frozen backbone:

1. Evidence relevance is non-stationary: 45.25% of modality roles change across adjacent steps, and 94% of sampled episodes exhibit at least one evidence-state switch.
2. Complete observations frequently contain potentially interfering evidence: at least one modality hurts official-action likelihood in 90.06% of steps under leave-one-modality-out intervention.
3. Observations contain measurable temporal repetition and sparse spatial action support: 31.0% of valid clips are near-static even at the strict 2% endpoint threshold, while the median target occupies 0.87% of the screen.

This motivates sequential action-sufficient grounding: selecting evidence again at every action step and retaining only a compact action-sufficient representation.

## Scientific Scope

- This is a model-behavior diagnostic, not a human-annotated oracle of modality relevance.
- Modality omission changes the input distribution, so negative sensitivity is evidence of potential interference, not definitive causal proof of semantic noise.
- The probe uses ground-truth actions only as fixed scoring targets; it does not train, tune, rerank, or alter benchmark predictions.
- Results are from Qwen2.5-Omni-7B and should be described as the observation that motivated FOG, not automatically generalized to every omni-modal backbone.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/probe_sequential_evidence_sensitivity.py \
  --out-dir logs/sequential_evidence_probe \
  --manifest logs/sequential_evidence_probe/manifest.json \
  --rank 0 --world-size 2

CUDA_VISIBLE_DEVICES=1 python scripts/probe_sequential_evidence_sensitivity.py \
  --out-dir logs/sequential_evidence_probe \
  --manifest logs/sequential_evidence_probe/manifest.json \
  --rank 1 --world-size 2

python scripts/probe_sequential_evidence_sensitivity.py \
  --out-dir logs/sequential_evidence_probe \
  --manifest logs/sequential_evidence_probe/manifest.json \
  --aggregate-only

python scripts/probe_observation_redundancy.py
```

Raw outputs: `logs/sequential_evidence_probe/scores_rank0.jsonl`, `scores_rank1.jsonl`, `results.json`, and `redundancy_statistics.json`.
