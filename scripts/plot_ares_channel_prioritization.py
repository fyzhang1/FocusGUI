#!/usr/bin/env python
"""Plot held-out channel priorities for the corrected generative FOG model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = (
    "text_history",
    "screen",
    "audio",
    "video",
    "state",
    "qwen_fullmodal",
    "screen_spatial",
)
CHANNEL_LABELS = (
    "Text / history",
    "Screen",
    "Audio",
    "Video",
    "Interaction state",
    "Full-modal hidden",
    "Screen spatial",
)
CATEGORIES = (
    ("localization", "Localization"),
    ("semantic_understanding", "Semantic"),
    ("cross_modal_discrimination", "Cross-modal"),
    ("temporal_reasoning", "Temporal"),
    ("instant_response", "Instant"),
)
PHASES = (
    ("initial", "Initial", lambda step: step == 0),
    ("early", "Early", lambda step: 1 <= step <= 2),
    ("middle", "Middle", lambda step: 3 <= step <= 4),
    ("late", "Late", lambda step: step >= 5),
)


def mean_matrix(
    grouped: Dict[str, List[np.ndarray]], keys: Sequence[str]
) -> Tuple[np.ndarray, List[int]]:
    matrix, counts = [], []
    for key in keys:
        rows = np.asarray(grouped[key], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(CHANNELS):
            raise ValueError(f"Group {key!r} has invalid evidence shape {rows.shape}")
        matrix.append(rows.mean(axis=0))
        counts.append(int(rows.shape[0]))
    return np.asarray(matrix).T, counts


def annotate_heatmap(ax, matrix: np.ndarray) -> None:
    threshold = float(np.nanmean(matrix))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            ax.text(
                col,
                row,
                f"{100.0 * value:.1f}",
                ha="center",
                va="center",
                color="white" if value > threshold * 1.16 else "#1f2933",
                fontsize=8.2,
                fontweight="semibold" if value > threshold * 1.16 else "normal",
            )


def draw_panel(
    ax,
    matrix: np.ndarray,
    column_labels: Sequence[str],
    counts: Sequence[int],
    title: str,
) -> None:
    image = ax.imshow(matrix, cmap="Blues", vmin=0.05, vmax=0.31, aspect="auto")
    annotate_heatmap(ax, matrix)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(
        [f"{label}\n($n$={count})" for label, count in zip(column_labels, counts)],
        fontsize=8.4,
    )
    ax.set_yticks(np.arange(len(CHANNEL_LABELS)))
    ax.set_yticklabels(CHANNEL_LABELS, fontsize=8.8)
    ax.set_title(title, fontsize=11.2, pad=11, fontweight="semibold")
    ax.tick_params(axis="both", length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CHANNELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def named_rows(matrix: np.ndarray, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    return {
        channel: {column: float(matrix[i, j]) for j, column in enumerate(columns)}
        for i, channel in enumerate(CHANNELS)
    }


def markdown_table(matrix: np.ndarray, labels: Sequence[str], counts: Sequence[int]) -> str:
    header = "| Channel | " + " | ".join(f"{x} ({n})" for x, n in zip(labels, counts)) + " |"
    align = "|---|" + "---:|" * len(labels)
    rows = [header, align]
    for i, label in enumerate(CHANNEL_LABELS):
        rows.append("| " + label + " | " + " | ".join(f"{100*x:.2f}" for x in matrix[i]) + " |")
    return "\n".join(rows)


def between_group_stat(values: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    """Weighted between-group variance, independently for each channel."""
    overall = values.mean(axis=0)
    statistic = np.zeros(values.shape[1], dtype=np.float64)
    for key in keys:
        mask = labels == key
        if mask.any():
            statistic += int(mask.sum()) * np.square(values[mask].mean(axis=0) - overall)
    return statistic


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def clustered_permutation_tests(
    weights: np.ndarray,
    category_labels: np.ndarray,
    phase_labels: np.ndarray,
    episode_labels: np.ndarray,
    category_keys: Sequence[str],
    phase_keys: Sequence[str],
    seed: int = 20260826,
    permutations: int = 5000,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Test category and phase association without treating steps as independent."""
    rng = np.random.default_rng(seed)
    category_observed = between_group_stat(weights, category_labels, category_keys)
    phase_observed = between_group_stat(weights, phase_labels, phase_keys)
    category_exceed = np.zeros(weights.shape[1], dtype=np.int64)
    phase_exceed = np.zeros(weights.shape[1], dtype=np.int64)

    episodes = np.asarray(sorted(set(episode_labels.tolist())))
    episode_categories = np.asarray(
        [category_labels[np.flatnonzero(episode_labels == episode)[0]] for episode in episodes]
    )
    episode_indices = [np.flatnonzero(episode_labels == episode) for episode in episodes]
    for _ in range(permutations):
        shuffled_episode_categories = rng.permutation(episode_categories)
        shuffled_categories = category_labels.copy()
        shuffled_phases = phase_labels.copy()
        for indices, category in zip(episode_indices, shuffled_episode_categories):
            shuffled_categories[indices] = category
        for indices in episode_indices:
            shuffled_phases[indices] = rng.permutation(shuffled_phases[indices])
        category_exceed += between_group_stat(weights, shuffled_categories, category_keys) >= category_observed
        phase_exceed += between_group_stat(weights, shuffled_phases, phase_keys) >= phase_observed

    category_p = (category_exceed + 1.0) / (permutations + 1.0)
    phase_p = (phase_exceed + 1.0) / (permutations + 1.0)
    category_q = bh_adjust(category_p)
    phase_q = bh_adjust(phase_p)
    return {
        "category": {
            channel: {
                "statistic": float(category_observed[i]),
                "p_value": float(category_p[i]),
                "bh_q_value": float(category_q[i]),
            }
            for i, channel in enumerate(CHANNELS)
        },
        "phase": {
            channel: {
                "statistic": float(phase_observed[i]),
                "p_value": float(phase_p[i]),
                "bh_q_value": float(phase_q[i]),
            }
            for i, channel in enumerate(CHANNELS)
        },
        "metadata": {
            "permutations": permutations,
            "seed": seed,
            "category_null": "Episode-level category labels are permuted across episodes.",
            "phase_null": "Phase labels are permuted among steps within each episode.",
            "multiple_testing": "Benjamini-Hochberg correction across seven channels, separately per analysis.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="logs/ares_lm_adapter_qwen_semantic_strict_gating/results.json",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/eib_results_summary_20260705/qwen_ablations/channel_prioritization",
    )
    args = parser.parse_args()

    source = REPO_ROOT / args.results
    raw = source.read_bytes()
    report = json.loads(raw)
    if report.get("decoder_frozen") is not True or report.get("lm_head_frozen") is not True:
        raise ValueError("The source is not a corrected frozen-decoder/frozen-LM-head FOG result.")
    if report.get("legacy_classification_head_used") is not False:
        raise ValueError("Legacy classification-head output is not valid for this analysis.")
    if report.get("gate_type") != "frozen-fullmodal-query-conditioned additive evidence posterior":
        raise ValueError("The source does not contain the dynamic FOG evidence posterior.")

    records = report.get("records", [])
    expected = int(report["split"]["test_steps"])
    if len(records) != expected or expected != 508:
        raise ValueError(f"Expected 508 held-out records, found {len(records)}")

    weights, category_groups, phase_groups = [], defaultdict(list), defaultdict(list)
    record_categories, record_phases, record_episodes = [], [], []
    dominant = Counter()
    for record in records:
        evidence = record.get("evidence_weight")
        if not isinstance(evidence, dict) or set(evidence) != set(CHANNELS):
            raise ValueError(f"Incomplete evidence posterior for global_id={record.get('global_id')}")
        row = np.asarray([float(evidence[channel]) for channel in CHANNELS], dtype=np.float64)
        if not np.isfinite(row).all() or not np.isclose(row.sum(), 1.0, atol=2e-5):
            raise ValueError(f"Invalid posterior for global_id={record.get('global_id')}: sum={row.sum()}")
        weights.append(row)
        category_groups[str(record["task_category_full"])].append(row)
        step = int(record["step_index"])
        matched = [key for key, _label, predicate in PHASES if predicate(step)]
        if len(matched) != 1:
            raise ValueError(f"Step {step} did not map to exactly one phase")
        phase_groups[matched[0]].append(row)
        record_categories.append(str(record["task_category_full"]))
        record_phases.append(matched[0])
        record_episodes.append(f"{record['app']}-{record['episode_id']}")
        dominant[CHANNELS[int(np.argmax(row))]] += 1

    weights_array = np.asarray(weights)
    category_keys = [key for key, _ in CATEGORIES]
    category_labels = [label for _, label in CATEGORIES]
    phase_keys = [key for key, _label, _predicate in PHASES]
    phase_labels = [label for _key, label, _predicate in PHASES]
    category_matrix, category_counts = mean_matrix(category_groups, category_keys)
    phase_matrix, phase_counts = mean_matrix(phase_groups, phase_keys)
    overall = weights_array.mean(axis=0)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.1, 4.7), constrained_layout=True)
    image = draw_panel(
        axes[0], category_matrix, category_labels, category_counts,
        "(a) Priority by task category",
    )
    draw_panel(
        axes[1], phase_matrix, phase_labels, phase_counts,
        "(b) Priority by interaction phase",
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label(r"Mean evidence weight $\alpha_t^{(m)}$", fontsize=9.2)
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "FOG channel prioritization on the held-out OmniGUI test split",
        fontsize=12.4,
        fontweight="semibold",
    )
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"ares_channel_prioritization.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    category_shift = category_matrix - overall[:, None]
    phase_shift = phase_matrix - overall[:, None]
    total_variation = 0.5 * np.abs(weights_array - overall[None, :]).sum(axis=1)
    normalized_entropy = -np.sum(weights_array * np.log(np.maximum(weights_array, 1e-12)), axis=1) / np.log(len(CHANNELS))
    permutation_tests = clustered_permutation_tests(
        weights_array,
        np.asarray(record_categories),
        np.asarray(record_phases),
        np.asarray(record_episodes),
        category_keys,
        phase_keys,
    )
    audit = {
        "source": str(source.relative_to(REPO_ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "method": report["method"],
        "gate_type": report["gate_type"],
        "held_out_steps": len(records),
        "held_out_episodes": int(report["split"]["test_episodes"]),
        "channels": list(CHANNELS),
        "phase_definition": {
            "initial": "step_index = 0",
            "early": "step_index in {1, 2}",
            "middle": "step_index in {3, 4}",
            "late": "step_index >= 5",
        },
        "category_counts": dict(zip(category_keys, category_counts)),
        "phase_counts": dict(zip(phase_keys, phase_counts)),
        "overall_mean": dict(zip(CHANNELS, map(float, overall))),
        "category_mean": named_rows(category_matrix, category_keys),
        "phase_mean": named_rows(phase_matrix, phase_keys),
        "category_shift_from_overall": named_rows(category_shift, category_keys),
        "phase_shift_from_overall": named_rows(phase_shift, phase_keys),
        "per_sample_adaptation": {
            "mean_total_variation_from_global_mean": float(total_variation.mean()),
            "median_total_variation_from_global_mean": float(np.median(total_variation)),
            "mean_normalized_entropy": float(normalized_entropy.mean()),
            "dominant_channel_counts": dict(dominant),
        },
        "clustered_permutation_tests": permutation_tests,
    }
    (out_dir / "channel_prioritization_statistics.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    category_range = 100.0 * (category_matrix.max(axis=1) - category_matrix.min(axis=1))
    phase_range = 100.0 * (phase_matrix.max(axis=1) - phase_matrix.min(axis=1))
    report_md = f"""# Corrected FOG Channel Prioritization

This analysis uses the corrected generative FOG result, in which the original omni-modal decoder and LM head are frozen and no legacy classification head is used. All statistics come from the `508` held-out test actions (`142` episodes). Task labels and step indices are used only to aggregate post-inference weights; they are never inputs to the gate.

![FOG channel prioritization](ares_channel_prioritization.png)

## Task-Category Priority (%)

{markdown_table(category_matrix, category_labels, category_counts)}

## Interaction-Phase Priority (%)

Phase definitions are: Initial = step 0; Early = steps 1--2; Middle = steps 3--4; Late = steps 5 and later.

{markdown_table(phase_matrix, phase_labels, phase_counts)}

## Variation Summary

| Channel | Category range (pp) | Phase range (pp) |
|---|---:|---:|
"""
    for label, cat_value, phase_value in zip(CHANNEL_LABELS, category_range, phase_range):
        report_md += f"| {label} | {cat_value:.2f} | {phase_value:.2f} |\n"
    report_md += """

## Cluster-Aware Association Tests

Category labels are permuted across episodes; phase labels are permuted only among steps of the same episode. Reported $q$ values use Benjamini--Hochberg correction over the seven channels separately for each analysis.

| Channel | Category $p$ | Category $q$ | Phase $p$ | Phase $q$ |
|---|---:|---:|---:|---:|
"""
    for channel, label in zip(CHANNELS, CHANNEL_LABELS):
        category_test = permutation_tests["category"][channel]
        phase_test = permutation_tests["phase"][channel]
        report_md += (
            f"| {label} | {category_test['p_value']:.4f} | {category_test['bh_q_value']:.4f} "
            f"| {phase_test['p_value']:.4f} | {phase_test['bh_q_value']:.4f} |\n"
        )
    report_md += f"""

The mean per-action total-variation distance from the global channel distribution is `{audit['per_sample_adaptation']['mean_total_variation_from_global_mean']:.4f}` (median `{audit['per_sample_adaptation']['median_total_variation_from_global_mean']:.4f}`). This confirms that the dynamic gate does not apply one identical weight vector to every action.

## Honest Interpretation

- Localization places the largest mean weight on the integrated full-modal channel, followed by interaction state. This is consistent with relying on the current integrated GUI state, but it is not a pure screen-only routing pattern.
- Instant Response assigns the highest Audio weight among the five categories. This is the clearest task-category-aligned shift in the current result.
- Temporal Reasoning does **not** prioritize the standalone Video channel: its mean Video weight is below the overall Video mean. Cross-modal Discrimination likewise does **not** assign its highest weight to standalone Audio. These intuitively attractive patterns are not supported by the current checkpoint and must not be claimed.
- Interaction phase produces a stronger routing shift than task category. Initial actions strongly prioritize interaction state; later actions reduce state weight and place more mass on audio, video, and integrated full-modal evidence.
- Cluster-aware permutation tests confirm this phase dependence for Text/History, Audio, Video, Interaction State, and Full-modal Hidden after multiple-testing correction ($q<0.001$). In contrast, no task-category association remains significant after correcting across seven channels. Category differences in this figure are therefore descriptive tendencies, not confirmatory evidence.
- The integrated full-modal channel already contains mixed audio-visual information. Therefore, a low standalone Audio or Video weight can reflect redundancy with the integrated channel rather than rejection of that modality.

## Scientific Scope

The heatmap visualizes the gate's learned routing posterior, not a causal oracle of modality importance. The current evidence establishes adaptive channel prioritization across individual actions and interaction phases. It does not establish a statistically robust task-category effect, although several descriptive category tendencies are visible. Causal claims about whether a channel is necessary still require modality ablation or intervention.

Source SHA-256: `{audit['source_sha256']}`
"""
    (out_dir / "README.md").write_text(report_md, encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("source", "source_sha256", "held_out_steps", "held_out_episodes", "category_counts", "phase_counts", "per_sample_adaptation")}, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
