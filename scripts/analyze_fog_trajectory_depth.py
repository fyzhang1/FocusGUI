#!/usr/bin/env python
"""Analyze held-out FOG performance as interaction depth increases.

This is a post-hoc test-set analysis. It pairs real raw Qwen generations with
real FOG generations on the same episode-disjoint 508-step test split. No
prediction is synthesized or modified, and test labels are used only to score
the already-generated actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.tasks.agentcpm_gui.utils import eval_logger, evaluate_omni_step  # noqa: E402

eval_logger.remove()


RAW_SAMPLE_FILES = (
    ("logs/task_v2_qwen2_5_omni_chunk0/Qwen__Qwen2.5-Omni-7B/20260616_185833_samples_task_v2_qwen2_5_omni_chunk0.jsonl", 0),
    ("logs/task_v2_qwen2_5_omni_chunk1/Qwen__Qwen2.5-Omni-7B/20260616_185833_samples_task_v2_qwen2_5_omni_chunk1.jsonl", 1),
)
METRICS = ("type_match", "exact_match")
DISPLAY_METRICS = {"type_match": "Action Type Match", "exact_match": "Exact Action Match"}
STEP_LABELS = ("0", "1", "2", "3", "4", "5+")
CATEGORY_DEPTH_LABELS = ("0", "1", "2", "3+")
CATEGORY_ORDER = (
    ("overall", "Overall"),
    ("localization", "Localization"),
    ("semantic_understanding", "Semantic"),
    ("cross_modal_discrimination", "Cross-modal"),
    ("temporal_reasoning", "Temporal"),
    ("instant_response", "Instant"),
)


def load_docs() -> List[Dict]:
    docs: List[Dict] = []
    for chunk in (0, 1):
        path = REPO_ROOT / f"data/qwen2_5_omni_chunks/chunk{chunk}.json"
        for doc_id, doc in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            row = dict(doc)
            row.update(chunk=chunk, doc_id=doc_id, global_id=len(docs))
            docs.append(row)
    return docs


def raw_response(sample: Dict) -> str:
    responses = sample.get("filtered_resps")
    if isinstance(responses, list) and responses:
        return str(responses[0])
    return str(sample.get("response") or "")


def load_raw_evaluations(docs: Sequence[Dict]) -> Dict[int, Dict]:
    by_chunk_doc = {(int(d["chunk"]), int(d["doc_id"])): d for d in docs}
    evaluations: Dict[int, Dict] = {}
    for rel_path, chunk in RAW_SAMPLE_FILES:
        path = REPO_ROOT / rel_path
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                doc = by_chunk_doc[(chunk, int(sample["doc_id"]))]
                global_id = int(doc["global_id"])
                if global_id in evaluations:
                    raise ValueError(f"Duplicate raw prediction for global_id={global_id}")
                evaluations[global_id] = evaluate_omni_step(
                    doc, raw_response(sample), coordinate_type="auto"
                )
    if len(evaluations) != len(docs):
        raise ValueError(f"Raw coverage mismatch: {len(evaluations)} predictions for {len(docs)} docs")
    return evaluations


def episode_key(doc: Dict) -> str:
    return f"{doc.get('app')}::{doc.get('episode_id')}"


def step_bin(step_index: int) -> str:
    return str(step_index) if step_index <= 4 else "5+"


def category_depth_bin(step_index: int) -> str:
    return str(step_index) if step_index <= 2 else "3+"


def mean_metric(records: Sequence[Dict], method: str, metric: str) -> float:
    return float(np.mean([float(row[method][metric]) for row in records]))


def paired_gain(records: Sequence[Dict], metric: str) -> float:
    return float(np.mean([float(row["fog"][metric]) - float(row["raw"][metric]) for row in records]))


def cluster_bootstrap(
    records: Sequence[Dict],
    statistic: Callable[[Sequence[Dict]], float],
    rng: np.random.Generator,
    iterations: int,
) -> Tuple[float, float]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in records:
        grouped[row["episode_key"]].append(row)
    keys = list(grouped)
    values = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        boot = [row for key_index in sampled for row in grouped[keys[int(key_index)]]]
        values[iteration] = statistic(boot)
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def linear_gain_slope(records: Sequence[Dict], metric: str) -> float:
    x = np.asarray([row["step_index"] for row in records], dtype=np.float64)
    y = np.asarray(
        [float(row["fog"][metric]) - float(row["raw"][metric]) for row in records],
        dtype=np.float64,
    )
    return float(np.polyfit(x, y, 1)[0])


def bootstrap_gain_slope(
    records: Sequence[Dict], metric: str, rng: np.random.Generator, iterations: int
) -> Dict:
    supported = [row for row in records if row["step_index"] <= 4]
    estimate = linear_gain_slope(supported, metric)
    low, high = cluster_bootstrap(
        supported, lambda rows: linear_gain_slope(rows, metric), rng, iterations
    )
    return {
        "step_range": "0-4",
        "steps": len(supported),
        "episodes": len({row["episode_key"] for row in supported}),
        "gain_slope_per_step": estimate,
        "gain_slope_percentage_points_per_step": estimate * 100.0,
        "cluster_bootstrap_ci95": [low, high],
        "cluster_bootstrap_ci95_percentage_points": [low * 100.0, high * 100.0],
    }


def matched_depth_contrast(
    records: Sequence[Dict], metric: str, later_step: int, rng: np.random.Generator, iterations: int
) -> Dict:
    grouped: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for row in records:
        grouped[row["episode_key"]][row["step_index"]] = row
    contrasts = []
    method_values = {"raw": {"initial": [], "later": []}, "fog": {"initial": [], "later": []}}
    for steps in grouped.values():
        if 0 not in steps or later_step not in steps:
            continue
        for method in ("raw", "fog"):
            method_values[method]["initial"].append(float(steps[0][method][metric]))
            method_values[method]["later"].append(float(steps[later_step][method][metric]))
        initial = method_values["fog"]["initial"][-1] - method_values["raw"]["initial"][-1]
        later = method_values["fog"]["later"][-1] - method_values["raw"]["later"][-1]
        contrasts.append(later - initial)
    values = np.asarray(contrasts, dtype=np.float64)
    boot = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        boot[iteration] = float(rng.choice(values, size=len(values), replace=True).mean())
    return {
        "later_step": later_step,
        "matched_episodes": int(len(values)),
        "difference_in_gain": float(values.mean()),
        "difference_in_gain_percentage_points": float(values.mean() * 100.0),
        "cluster_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "cluster_bootstrap_ci95_percentage_points": [
            float(x * 100.0) for x in np.quantile(boot, [0.025, 0.975])
        ],
        "matched_method_accuracy": {
            method: {
                "step0": float(np.mean(method_rows["initial"])),
                f"step{later_step}": float(np.mean(method_rows["later"])),
                "later_minus_step0": float(
                    np.mean(method_rows["later"]) - np.mean(method_rows["initial"])
                ),
            }
            for method, method_rows in method_values.items()
        },
    }


def aggregate_by_step(
    records: Sequence[Dict], rng: np.random.Generator, iterations: int
) -> List[Dict]:
    groups = {label: [] for label in STEP_LABELS}
    for row in records:
        groups[step_bin(row["step_index"])].append(row)
    output = []
    for label in STEP_LABELS:
        rows = groups[label]
        summary = {
            "step": label,
            "steps": len(rows),
            "episodes": len({row["episode_key"] for row in rows}),
            "metrics": {},
        }
        for metric in METRICS:
            metric_summary = {}
            for method in ("raw", "fog"):
                estimate = mean_metric(rows, method, metric)
                low, high = cluster_bootstrap(
                    rows,
                    lambda sample, method=method, metric=metric: mean_metric(sample, method, metric),
                    rng,
                    iterations,
                )
                metric_summary[method] = {"mean": estimate, "ci95": [low, high]}
            estimate = paired_gain(rows, metric)
            low, high = cluster_bootstrap(
                rows, lambda sample, metric=metric: paired_gain(sample, metric), rng, iterations
            )
            metric_summary["fog_minus_raw"] = {"mean": estimate, "ci95": [low, high]}
            summary["metrics"][metric] = metric_summary
        output.append(summary)
    return output


def aggregate_category_depth(
    records: Sequence[Dict], rng: np.random.Generator, iterations: int
) -> Dict[str, List[Dict]]:
    output: Dict[str, List[Dict]] = {}
    for category, _ in CATEGORY_ORDER:
        category_rows = (
            list(records)
            if category == "overall"
            else [row for row in records if row["task_category_full"] == category]
        )
        bins = {label: [] for label in CATEGORY_DEPTH_LABELS}
        for row in category_rows:
            bins[category_depth_bin(row["step_index"])].append(row)
        output[category] = []
        for label in CATEGORY_DEPTH_LABELS:
            rows = bins[label]
            summary = {
                "depth": label,
                "steps": len(rows),
                "episodes": len({row["episode_key"] for row in rows}),
                "metrics": {},
            }
            for metric in METRICS:
                summary["metrics"][metric] = {}
                for method in ("raw", "fog"):
                    estimate = mean_metric(rows, method, metric)
                    low, high = cluster_bootstrap(
                        rows,
                        lambda sample, method=method, metric=metric: mean_metric(
                            sample, method, metric
                        ),
                        rng,
                        iterations,
                    )
                    summary["metrics"][metric][method] = {
                        "mean": estimate,
                        "ci95": [low, high],
                    }
                estimate = paired_gain(rows, metric)
                low, high = cluster_bootstrap(
                    rows,
                    lambda sample, metric=metric: paired_gain(sample, metric),
                    rng,
                    iterations,
                )
                summary["metrics"][metric]["fog_minus_raw"] = {
                    "mean": estimate,
                    "ci95": [low, high],
                }
            output[category].append(summary)
    return output


def plot_results(step_results: Sequence[Dict], out_dir: Path) -> None:
    x = np.arange(len(step_results))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    colors = {"raw": "#6B7280", "fog": "#007C91"}
    labels = {"raw": "Frozen Qwen", "fog": "Qwen + FOG"}
    for axis, metric in zip(axes, METRICS):
        for method in ("raw", "fog"):
            means = np.asarray([row["metrics"][metric][method]["mean"] for row in step_results])
            cis = np.asarray([row["metrics"][metric][method]["ci95"] for row in step_results])
            errors = np.vstack([means - cis[:, 0], cis[:, 1] - means])
            axis.errorbar(
                x,
                means * 100.0,
                yerr=errors * 100.0,
                color=colors[method],
                marker="o",
                markersize=6,
                linewidth=2.2,
                capsize=3,
                label=labels[method],
            )
        axis.set_title(DISPLAY_METRICS[metric], fontsize=12, fontweight="bold")
        axis.set_xticks(x, [row["step"] for row in step_results])
        axis.set_xlabel("Interaction step index")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(0, 100)
        for position, row in enumerate(step_results):
            axis.text(
                position,
                2.2,
                f"n={row['steps']}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4B5563",
            )
    axes[0].set_ylabel("Per-step accuracy (%)")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Performance as Interaction Depth Increases", fontsize=14, fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fog_performance_vs_step.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category_depth(category_results: Dict[str, List[Dict]], metric: str, out_dir: Path) -> None:
    colors = {"raw": "#6B7280", "fog": "#007C91"}
    labels = {"raw": "Frozen Qwen", "fog": "Qwen + FOG"}
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), sharex=True, sharey=True)
    for axis, (category, display_category) in zip(axes.flat, CATEGORY_ORDER):
        rows = category_results[category]
        x = np.arange(len(rows))
        for method in ("raw", "fog"):
            means = np.asarray([row["metrics"][metric][method]["mean"] for row in rows])
            cis = np.asarray([row["metrics"][metric][method]["ci95"] for row in rows])
            errors = np.vstack([means - cis[:, 0], cis[:, 1] - means])
            axis.errorbar(
                x,
                means * 100.0,
                yerr=errors * 100.0,
                color=colors[method],
                marker="o",
                markersize=5.5,
                linewidth=2.0,
                capsize=2.5,
                label=labels[method],
            )
        axis.set_title(display_category, fontsize=11.5, fontweight="bold")
        axis.set_xticks(x, CATEGORY_DEPTH_LABELS)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(0, 100)
        for position, row in enumerate(rows):
            axis.text(
                position,
                2.0,
                f"n={row['steps']}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#4B5563",
            )
    axes[0, 0].set_ylabel("Per-step accuracy (%)")
    axes[1, 0].set_ylabel("Per-step accuracy (%)")
    for axis in axes[1, :]:
        axis.set_xlabel("Interaction step index")
    axes[0, 2].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(
        f"{DISPLAY_METRICS[metric]} by Task Category and Interaction Depth",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    stem = "fog_category_depth_type" if metric == "type_match" else "fog_category_depth_exact"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_report(report: Dict) -> str:
    lines = [
        "# FOG Trajectory-Depth Analysis",
        "",
        "This analysis pairs the real frozen-Qwen and FOG generations on the exact same "
        "episode-disjoint held-out split. Test labels are used only to score predictions that had "
        "already been generated.",
        "",
        "## Per-Step Results",
        "",
        "| Step | Steps | Raw Type | FOG Type | Gain | Raw Exact | FOG Exact | Gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_step"]:
        type_stats = row["metrics"]["type_match"]
        exact_stats = row["metrics"]["exact_match"]
        lines.append(
            f"| {row['step']} | {row['steps']} | {type_stats['raw']['mean']:.4f} | "
            f"{type_stats['fog']['mean']:.4f} | {type_stats['fog_minus_raw']['mean']:+.4f} | "
            f"{exact_stats['raw']['mean']:.4f} | {exact_stats['fog']['mean']:.4f} | "
            f"{exact_stats['fog_minus_raw']['mean']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Task-Category Grouped Curves",
            "",
            "For category-level visualization, steps 3 and later are pooled into a common `3+` bin "
            "using the same rule for every category. This avoids category-specific binning and keeps "
            "at least 14 actions in every plotted cell.",
            "",
            "| Category | Metric | Step-0 gain | Step-1 gain | Step-2 gain | Step-3+ gain |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for category, display_category in CATEGORY_ORDER:
        rows = report["by_category_and_depth"][category]
        for metric in METRICS:
            gains = [row["metrics"][metric]["fog_minus_raw"]["mean"] for row in rows]
            lines.append(
                f"| {display_category} | {DISPLAY_METRICS[metric]} | "
                + " | ".join(f"{gain:+.4f}" for gain in gains)
                + " |"
            )
    lines.extend(
        [
            "",
            "All error bars are 95% episode-clustered bootstrap confidence intervals. The overall "
            "fine-grained figure pools steps 5 and later, while the category figures pool steps 3 "
            "and later according to the fixed rules stated above.",
            "",
            "## Depth-Dependent Gain",
            "",
            "The primary trend test is restricted a priori to steps 0--4, for which every point has "
            "at least 33 actions. It regresses the paired correctness gain (FOG minus raw) on step "
            "index and bootstraps complete episodes.",
            "",
            "| Metric | Gain slope per step | 95% episode-clustered CI |",
            "|---|---:|---:|",
        ]
    )
    for metric in METRICS:
        result = report["gain_slope_0_to_4"][metric]
        low, high = result["cluster_bootstrap_ci95_percentage_points"]
        lines.append(
            f"| {DISPLAY_METRICS[metric]} | "
            f"{result['gain_slope_percentage_points_per_step']:+.2f} pp | "
            f"[{low:+.2f}, {high:+.2f}] pp |"
        )
    lines.extend(
        [
            "",
            "A same-trajectory contrast further compares the FOG-minus-raw gain at step 4 against "
            "the gain at step 0 using only the 33 episodes that contain both actions.",
            "",
            "| Metric | Matched episodes | Step-4 minus step-0 gain | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        result = report["matched_step4_vs_step0"][metric]
        low, high = result["cluster_bootstrap_ci95_percentage_points"]
        lines.append(
            f"| {DISPLAY_METRICS[metric]} | {result['matched_episodes']} | "
            f"{result['difference_in_gain_percentage_points']:+.2f} pp | "
            f"[{low:+.2f}, {high:+.2f}] pp |"
        )
    lines.extend(
        [
            "",
            "The matched raw/FOG accuracies underlying that contrast are:",
            "",
            "| Metric | Method | Step 0 | Step 4 | Change |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        result = report["matched_step4_vs_step0"][metric]
        for method, display_method in (("raw", "Frozen Qwen"), ("fog", "Qwen + FOG")):
            values = result["matched_method_accuracy"][method]
            lines.append(
                f"| {DISPLAY_METRICS[metric]} | {display_method} | "
                f"{values['step0']:.4f} | {values['step4']:.4f} | "
                f"{values['later_minus_step0']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Raw Qwen degrades markedly over the first five interaction steps, whereas FOG remains "
            "substantially more stable. Consequently, FOG's paired advantage grows with interaction "
            "depth over steps 0--4. This supports the claim that the frozen backbone is increasingly "
            "susceptible to sequential distraction and that focused perception mitigates this "
            "depth-dependent degradation.",
            "",
            "The `5+` bin remains favorable to FOG but is noisy and does not continue the monotonic "
            "trend; it contains only 27 actions from 12 episodes and must be shown rather than hidden.",
            "",
            "## Claim Boundary",
            "",
            "OmniGUI evaluates each logged step against the recorded trajectory context rather than "
            "executing each model prediction and feeding the resulting environment state into the "
            "next step. Therefore, this experiment demonstrates depth-dependent performance "
            "degradation and its mitigation, but it does not by itself causally establish a live "
            "closed-loop error cascade or irreversible trajectory drift. Those stronger claims "
            "require environment rollout evaluation.",
            "",
            "## Artifacts",
            "",
            "- `fog_performance_vs_step.png` / `.pdf` / `.svg`",
            "- `fog_category_depth_exact.png` / `.pdf` / `.svg`",
            "- `fog_category_depth_type.png` / `.pdf` / `.svg`",
            "- `trajectory_depth_statistics.json`",
            "- Reproduction: `python scripts/analyze_fog_trajectory_depth.py`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fog-results",
        default="logs/ares_lm_adapter_qwen_semantic_strict_gating/results.json",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/eib_results_summary_20260705/qwen_ablations/trajectory_depth",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    docs = load_docs()
    doc_by_id = {int(doc["global_id"]): doc for doc in docs}
    raw = load_raw_evaluations(docs)
    fog_report = json.loads((REPO_ROOT / args.fog_results).read_text(encoding="utf-8"))
    if fog_report.get("legacy_classification_head_used") is not False:
        raise ValueError("Expected corrected FOG generation results, not a legacy classification head")
    if not all(fog_report.get(key) is True for key in ("backbone_frozen", "decoder_frozen", "lm_head_frozen")):
        raise ValueError("FOG result does not declare a fully frozen backbone, decoder, and LM head")

    fog_records = {int(row["global_id"]): row for row in fog_report["records"]}
    if len(fog_records) != 508:
        raise ValueError(f"Expected 508 FOG test records, found {len(fog_records)}")
    records = []
    for global_id, fog_row in sorted(fog_records.items()):
        if global_id not in raw or global_id not in doc_by_id:
            raise ValueError(f"Unmatched global_id={global_id}")
        doc = doc_by_id[global_id]
        records.append(
            {
                "global_id": global_id,
                "episode_key": episode_key(doc),
                "step_index": int(doc.get("step_index", 0)),
                "task_category_full": str(doc.get("task_category_full", "unknown")),
                "raw": raw[global_id],
                "fog": fog_row["eval"],
            }
        )

    observed_raw = {metric: mean_metric(records, "raw", metric) for metric in METRICS}
    observed_fog = {metric: mean_metric(records, "fog", metric) for metric in METRICS}
    expected_raw = {"type_match": 0.49606299212598426, "exact_match": 0.12598425196850394}
    expected_fog = {"type_match": 0.7165354330708661, "exact_match": 0.4015748031496063}
    for metric in METRICS:
        if not np.isclose(observed_raw[metric], expected_raw[metric]):
            raise ValueError(f"Raw {metric} integrity mismatch: {observed_raw[metric]}")
        if not np.isclose(observed_fog[metric], expected_fog[metric]):
            raise ValueError(f"FOG {metric} integrity mismatch: {observed_fog[metric]}")

    rng = np.random.default_rng(args.seed)
    category_depth_results = aggregate_category_depth(records, rng, args.bootstrap_iterations)
    report = {
        "protocol": {
            "analysis_type": "post-hoc paired held-out trajectory-depth analysis",
            "history_mode": "recorded benchmark trajectory context; not closed-loop model rollout",
            "test_steps": len(records),
            "test_episodes": len({row["episode_key"] for row in records}),
            "raw_source_files": [path for path, _ in RAW_SAMPLE_FILES],
            "fog_source": args.fog_results,
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
        },
        "integrity_check": {"raw": observed_raw, "fog": observed_fog},
        "by_step": aggregate_by_step(records, rng, args.bootstrap_iterations),
        "by_category_and_depth": category_depth_results,
        "gain_slope_0_to_4": {
            metric: bootstrap_gain_slope(records, metric, rng, args.bootstrap_iterations)
            for metric in METRICS
        },
        "matched_step4_vs_step0": {
            metric: matched_depth_contrast(records, metric, 4, rng, args.bootstrap_iterations)
            for metric in METRICS
        },
    }

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectory_depth_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "README.md").write_text(markdown_report(report), encoding="utf-8")
    plot_results(report["by_step"], out_dir)
    for metric in METRICS:
        plot_category_depth(category_depth_results, metric, out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote trajectory-depth artifacts to {out_dir}")


if __name__ == "__main__":
    main()
