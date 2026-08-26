#!/usr/bin/env python
"""Descriptive redundancy statistics for a fixed OmniGUI probe manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COORD_ACTIONS = {0, 1, 2}


def load_docs() -> List[Dict]:
    docs: List[Dict] = []
    for chunk in (0, 1):
        path = REPO_ROOT / f"data/qwen2_5_omni_chunks/chunk{chunk}.json"
        for raw in json.loads(path.read_text(encoding="utf-8")):
            row = dict(raw)
            row["global_id"] = len(docs)
            docs.append(row)
    return docs


def target_area_fraction(doc: Dict) -> float | None:
    if int(doc.get("result_action_type", -1)) not in COORD_ACTIONS:
        return None
    try:
        box = json.loads(doc["result_touch_xy"])
        (x1, y1), (x2, y2) = box
        width = float(doc["image_width"])
        height = float(doc["image_height"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    x1, x2 = np.clip([x1, x2], 0.0, width)
    y1, y2 = np.clip([y1, y2], 0.0, height)
    return float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / max(width * height, 1.0))


def describe(values: np.ndarray) -> Dict:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="logs/sequential_evidence_probe/manifest.json")
    parser.add_argument("--dense-cache", default="logs/fog_evidence/observation_descriptors.npz")
    parser.add_argument("--out", default="logs/sequential_evidence_probe/redundancy_statistics.json")
    args = parser.parse_args()

    manifest = json.loads((REPO_ROOT / args.manifest).read_text(encoding="utf-8"))
    docs = load_docs()
    selected_ids = np.array([int(x["global_id"]) for x in manifest["entries"]], dtype=np.int64)
    selected = [docs[i] for i in selected_ids]

    dense = np.load(REPO_ROOT / args.dense_cache)["dense"].astype(np.float32)
    if dense.shape[0] != len(docs):
        raise ValueError(f"Dense cache has {dense.shape[0]} rows but benchmark has {len(docs)} documents")
    # video_features is 256 endpoint-difference pixels followed by five metadata values.
    endpoint_mad = dense[selected_ids, 333 + 256 + 3]
    decoded_frames = dense[selected_ids, 333 + 256]
    valid_video = endpoint_mad[decoded_frames >= 2]

    target_areas = np.array([x for doc in selected if (x := target_area_fraction(doc)) is not None], dtype=np.float64)
    report = {
        "manifest": {
            "seed": manifest["seed"],
            "steps": manifest["num_steps"],
            "episodes": manifest["num_episodes"],
        },
        "video_endpoint_mean_absolute_difference": {
            **describe(valid_video),
            "definition": "Mean absolute grayscale pixel difference between the first and last decoded video frames, resized to 16x16 and scaled to [0,1].",
            "fraction_below_0_02": float(np.mean(valid_video < 0.02)),
            "fraction_below_0_05": float(np.mean(valid_video < 0.05)),
        },
        "coordinate_target_area_fraction": {
            **describe(target_areas),
            "definition": "Clipped ground-truth interaction rectangle area divided by screenshot area, for coordinate-bearing actions.",
            "mean_outside_target": float(1.0 - np.mean(target_areas)),
            "median_outside_target": float(1.0 - np.median(target_areas)),
        },
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
