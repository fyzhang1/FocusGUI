# Qwen Ablation: Raw Qwen Same-Test Baseline

Raw cached Qwen outputs filtered to the exact EIB held-out test split. This is the clean baseline for strict Qwen comparison.

## Protocol
- Raw model outputs filtered to the same held-out test episodes used by final FOG split.


## Overall Metrics

| Setting | Steps | Episodes | Type | Exact | Success | Progress |
|---|---:|---:|---:|---:|---:|---:|
| Raw Qwen Same-Test Baseline | 508 | 142 | 0.4961 | 0.1260 | 0.0000 | 0.0550 |

## Category Breakdown

| Category | Steps | Episodes | Type | Exact | Success | Progress |
|---|---:|---:|---:|---:|---:|---:|
| cross_modal_discrimination | 109 | 31 | 0.4220 | 0.0917 | 0.0000 | 0.0317 |
| instant_response | 82 | 23 | 0.5854 | 0.1220 | 0.0000 | 0.0507 |
| localization | 82 | 27 | 0.5732 | 0.1220 | 0.0000 | 0.0494 |
| semantic_understanding | 111 | 29 | 0.4865 | 0.1712 | 0.0000 | 0.0675 |
| temporal_reasoning | 124 | 32 | 0.4597 | 0.1210 | 0.0000 | 0.0740 |


## Gate Means

_No gate statistics._


## Interpretation

This baseline establishes the raw Qwen performance on the exact held-out test episodes. It should be used for the cleanest Qwen-vs-EIB comparison, rather than comparing EIB test results to the full 2572-step raw average only.

## Source File

- `logs/eaf_eib_strict/raw_qwen_cached_same_test.json`
