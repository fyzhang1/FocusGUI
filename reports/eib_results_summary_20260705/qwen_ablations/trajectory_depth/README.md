# FOG Trajectory-Depth Analysis

This analysis pairs the real frozen-Qwen and FOG generations on the exact same episode-disjoint held-out split. Test labels are used only to score predictions that had already been generated.

## Per-Step Results

| Step | Steps | Raw Type | FOG Type | Gain | Raw Exact | FOG Exact | Gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 142 | 0.6479 | 0.8028 | +0.1549 | 0.1620 | 0.3873 | +0.2254 |
| 1 | 141 | 0.5319 | 0.6454 | +0.1135 | 0.1064 | 0.3404 | +0.2340 |
| 2 | 103 | 0.4369 | 0.7184 | +0.2816 | 0.1165 | 0.4272 | +0.3107 |
| 3 | 62 | 0.3387 | 0.6935 | +0.3548 | 0.0968 | 0.4839 | +0.3871 |
| 4 | 33 | 0.3030 | 0.8182 | +0.5152 | 0.0909 | 0.5758 | +0.4848 |
| 5+ | 27 | 0.3333 | 0.5556 | +0.2222 | 0.1852 | 0.2963 | +0.1111 |

## Task-Category Grouped Curves

For category-level visualization, steps 3 and later are pooled into a common `3+` bin using the same rule for every category. This avoids category-specific binning and keeps at least 14 actions in every plotted cell.

| Category | Metric | Step-0 gain | Step-1 gain | Step-2 gain | Step-3+ gain |
|---|---|---:|---:|---:|---:|
| Overall | Action Type Match | +0.1549 | +0.1135 | +0.2816 | +0.3689 |
| Overall | Exact Action Match | +0.2254 | +0.2340 | +0.3107 | +0.3525 |
| Localization | Action Type Match | +0.0741 | +0.2963 | +0.2143 | +0.2857 |
| Localization | Exact Action Match | +0.3333 | +0.4074 | +0.2857 | +0.2143 |
| Semantic | Action Type Match | +0.3103 | +0.1786 | +0.0500 | +0.2059 |
| Semantic | Exact Action Match | +0.2414 | +0.2143 | +0.2000 | +0.2353 |
| Cross-modal | Action Type Match | +0.1613 | +0.1290 | +0.3810 | +0.4231 |
| Cross-modal | Exact Action Match | +0.1613 | +0.1613 | +0.1905 | +0.4615 |
| Temporal | Action Type Match | +0.2500 | +0.0938 | +0.4074 | +0.4848 |
| Temporal | Exact Action Match | +0.2812 | +0.2500 | +0.4815 | +0.3939 |
| Instant | Action Type Match | -0.0870 | -0.1739 | +0.2857 | +0.4667 |
| Instant | Exact Action Match | +0.0870 | +0.1304 | +0.3333 | +0.4667 |

All error bars are 95% episode-clustered bootstrap confidence intervals. The overall fine-grained figure pools steps 5 and later, while the category figures pool steps 3 and later according to the fixed rules stated above.

## Depth-Dependent Gain

The primary trend test is restricted a priori to steps 0--4, for which every point has at least 33 actions. It regresses the paired correctness gain (FOG minus raw) on step index and bootstraps complete episodes.

| Metric | Gain slope per step | 95% episode-clustered CI |
|---|---:|---:|
| Action Type Match | +8.54 pp | [+4.61, +12.67] pp |
| Exact Action Match | +6.06 pp | [+2.22, +10.01] pp |

A same-trajectory contrast further compares the FOG-minus-raw gain at step 4 against the gain at step 0 using only the 33 episodes that contain both actions.

| Metric | Matched episodes | Step-4 minus step-0 gain | 95% CI |
|---|---:|---:|---:|
| Action Type Match | 33 | +33.33 pp | [+9.09, +57.58] pp |
| Exact Action Match | 33 | +27.27 pp | [+3.03, +51.52] pp |

The matched raw/FOG accuracies underlying that contrast are:

| Metric | Method | Step 0 | Step 4 | Change |
|---|---|---:|---:|---:|
| Action Type Match | Frozen Qwen | 0.6667 | 0.3030 | -0.3636 |
| Action Type Match | Qwen + FOG | 0.8485 | 0.8182 | -0.0303 |
| Exact Action Match | Frozen Qwen | 0.2424 | 0.0909 | -0.1515 |
| Exact Action Match | Qwen + FOG | 0.4545 | 0.5758 | +0.1212 |

## Interpretation

Raw Qwen degrades markedly over the first five interaction steps, whereas FOG remains substantially more stable. Consequently, FOG's paired advantage grows with interaction depth over steps 0--4. This supports the claim that the frozen backbone is increasingly susceptible to sequential distraction and that focused perception mitigates this depth-dependent degradation.

The `5+` bin remains favorable to FOG but is noisy and does not continue the monotonic trend; it contains only 27 actions from 12 episodes and must be shown rather than hidden.

## Claim Boundary

OmniGUI evaluates each logged step against the recorded trajectory context rather than executing each model prediction and feeding the resulting environment state into the next step. Therefore, this experiment demonstrates depth-dependent performance degradation and its mitigation, but it does not by itself causally establish a live closed-loop error cascade or irreversible trajectory drift. Those stronger claims require environment rollout evaluation.

## Artifacts

- `fog_performance_vs_step.png` / `.pdf` / `.svg`
- `fog_category_depth_exact.png` / `.pdf` / `.svg`
- `fog_category_depth_type.png` / `.pdf` / `.svg`
- `trajectory_depth_statistics.json`
- Reproduction: `python scripts/analyze_fog_trajectory_depth.py`
