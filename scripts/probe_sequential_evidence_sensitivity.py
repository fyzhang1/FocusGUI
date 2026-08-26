#!/usr/bin/env python
"""Training-free diagnostic for dynamic action-evidence mismatch.

The probe measures how the frozen Qwen2.5-Omni likelihood of the official
ground-truth action changes when one currently available evidence source is
omitted.  It is an analysis experiment only: no probe is trained, and no
ablation result is used to alter predictions or labels.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.models.simple.qwen2_5_omni import Qwen2_5_Omni  # noqa: E402
from lmms_eval.tasks.agentcpm_gui.utils import omni_doc_to_target, unified_v2_doc_to_messages  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402


CONDITIONS = ("full", "no_screen", "no_video", "no_audio", "no_history")
MODALITIES = ("screen", "video", "audio", "history")
CATEGORY_LABELS = {
    "localization": "Localization",
    "semantic_understanding": "Semantic",
    "cross_modal_discrimination": "Cross-modal",
    "temporal_reasoning": "Temporal",
    "instant_response": "Instant",
}


def load_docs() -> List[Dict]:
    docs: List[Dict] = []
    for chunk in (0, 1):
        path = REPO_ROOT / f"data/qwen2_5_omni_chunks/chunk{chunk}.json"
        for doc_id, raw in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            doc = dict(raw)
            doc["chunk"] = chunk
            doc["doc_id"] = doc_id
            doc["global_id"] = len(docs)
            doc["episode_key"] = f"{doc.get('app')}-{doc.get('episode_id')}"
            doc["category_label"] = CATEGORY_LABELS.get(
                doc.get("task_category_full"), doc.get("task_category_full", "unknown")
            )
            docs.append(doc)
    return docs


def build_manifest(docs: List[Dict], episodes_per_category: int, seed: int) -> Dict:
    by_episode: Dict[str, List[Dict]] = defaultdict(list)
    for doc in docs:
        by_episode[doc["episode_key"]].append(doc)

    by_category: Dict[str, List[Tuple[str, List[Dict]]]] = defaultdict(list)
    for episode_key, rows in by_episode.items():
        rows.sort(key=lambda d: int(d.get("step_index", d.get("step_id", 0))))
        if len(rows) < 2:
            continue
        by_category[rows[0]["category_label"]].append((episode_key, rows))

    rng = random.Random(seed)
    selected_episodes = []
    for category in CATEGORY_LABELS.values():
        candidates = sorted(by_category[category], key=lambda item: item[0])
        rng.shuffle(candidates)
        if len(candidates) < episodes_per_category:
            raise ValueError(f"Only {len(candidates)} multi-step episodes for {category}")
        selected_episodes.extend(candidates[:episodes_per_category])

    selected_episodes.sort(key=lambda item: item[0])
    entries = []
    for episode_order, (episode_key, rows) in enumerate(selected_episodes):
        for doc in rows:
            entries.append(
                {
                    "episode_order": episode_order,
                    "episode_key": episode_key,
                    "category": doc["category_label"],
                    "chunk": int(doc["chunk"]),
                    "doc_id": int(doc["doc_id"]),
                    "global_id": int(doc["global_id"]),
                    "step_index": int(doc.get("step_index", doc.get("step_id", 0))),
                }
            )
    return {
        "seed": seed,
        "episodes_per_category": episodes_per_category,
        "num_episodes": len(selected_episodes),
        "num_steps": len(entries),
        "category_episode_counts": dict(Counter(x[1][0]["category_label"] for x in selected_episodes)),
        "entries": entries,
    }


def evidence_available(doc: Dict, modality: str) -> bool:
    if modality == "screen":
        return bool(doc.get("current_image") or doc.get("image") or doc.get("image_path"))
    if modality == "video":
        return bool(doc.get("current_video") or doc.get("video") or doc.get("video_path"))
    if modality == "audio":
        return bool(doc.get("current_audio") or doc.get("audio") or doc.get("audio_path"))
    if modality == "history":
        return bool(doc.get("action_history") or doc.get("history_screenshot_path"))
    raise ValueError(modality)


def ablated_doc(doc: Dict, condition: str) -> Dict:
    row = copy.deepcopy(doc)
    if condition == "full":
        return row
    if condition == "no_screen":
        for key in ("current_image", "image", "image_path"):
            row[key] = ""
    elif condition == "no_video":
        for key in ("current_video", "video", "video_path"):
            row[key] = ""
    elif condition == "no_audio":
        for key in ("current_audio", "audio", "audio_path"):
            row[key] = ""
    elif condition == "no_history":
        row["action_history"] = []
        row["history_screenshot_path"] = ""
    else:
        raise ValueError(condition)
    return row


def score_target(model: Qwen2_5_Omni, doc: Dict, condition: str) -> Dict:
    conditioned = ablated_doc(doc, condition)
    raw_messages = unified_v2_doc_to_messages(conditioned, {"media_dir": "data/benchmark"})
    messages, use_audio_in_video = model._convert_messages_for_qwen(raw_messages)
    prompt_text = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    target = omni_doc_to_target(doc)
    full_text = prompt_text + target

    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    prompt_inputs = model.processor(
        text=prompt_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    full_inputs = model.processor(
        text=full_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    device = "cuda" if model.device_map == "auto" else model.model.device
    prompt_inputs = prompt_inputs.to(device).to(model.model.dtype)
    full_inputs = full_inputs.to(device).to(model.model.dtype)
    prompt_len = int(prompt_inputs.input_ids.shape[1])
    labels = full_inputs.input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.inference_mode():
        try:
            outputs = model.model.thinker(
                **full_inputs,
                labels=labels,
                use_audio_in_video=use_audio_in_video,
                return_dict=True,
                use_cache=False,
            )
        except TypeError:
            full_inputs.pop("feature_attention_mask", None)
            outputs = model.model.thinker(
                **full_inputs,
                labels=labels,
                use_audio_in_video=use_audio_in_video,
                return_dict=True,
                use_cache=False,
            )

    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    selected = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    total_logp = float(selected.masked_select(mask).sum().item())
    token_count = int(mask.sum().item())
    avg_logp = total_logp / max(token_count, 1)

    del prompt_inputs, full_inputs, labels, outputs, shift_logits, shift_labels, safe_labels, selected
    gc.collect()
    torch.cuda.empty_cache()
    return {"avg_logp": avg_logp, "total_logp": total_logp, "target_tokens": token_count}


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed(path: Path) -> set:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add((int(row["global_id"]), row["condition"]))
    return completed


def run_probe(args: argparse.Namespace, manifest: Dict) -> None:
    docs = {doc["global_id"]: doc for doc in load_docs()}
    entries = [e for e in manifest["entries"] if int(e["episode_order"]) % args.world_size == args.rank]
    out_path = Path(args.out_dir) / f"scores_rank{args.rank}.jsonl"
    completed = load_completed(out_path)

    model = Qwen2_5_Omni(
        pretrained=args.pretrained,
        attn_implementation=args.attn_implementation,
        image_max_pixels=args.image_max_pixels,
        video_max_frames=args.video_max_frames,
        video_max_pixels=args.video_max_pixels,
        use_audio_in_video=False,
        device_map="auto",
    )

    failures_path = Path(args.out_dir) / f"failures_rank{args.rank}.jsonl"
    total = len(entries) * len(CONDITIONS)
    with tqdm(total=total, desc=f"Evidence probe rank {args.rank}") as pbar:
        for entry in entries:
            doc = docs[int(entry["global_id"])]
            availability = {m: evidence_available(doc, m) for m in MODALITIES}
            for condition in CONDITIONS:
                key = (int(entry["global_id"]), condition)
                if key in completed:
                    pbar.update(1)
                    continue
                if condition != "full" and not availability[condition.removeprefix("no_")]:
                    row = {**entry, "condition": condition, "available": False, "score": None}
                    append_jsonl(out_path, row)
                    completed.add(key)
                    pbar.update(1)
                    continue
                try:
                    score = score_target(model, doc, condition)
                    row = {
                        **entry,
                        "condition": condition,
                        "available": True,
                        "target": omni_doc_to_target(doc),
                        "result_action_type": int(doc.get("result_action_type")),
                        "score": score,
                    }
                    append_jsonl(out_path, row)
                    completed.add(key)
                except Exception as exc:
                    append_jsonl(
                        failures_path,
                        {**entry, "condition": condition, "error_type": type(exc).__name__, "error": str(exc)},
                    )
                    raise
                finally:
                    pbar.update(1)


def percentile_ci(values: List[float], groups: List[str], statistic, seed: int, samples: int = 2000) -> List[float]:
    by_group: Dict[str, List[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(value)
    keys = sorted(by_group)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled = []
        for key in rng.choice(keys, size=len(keys), replace=True):
            sampled.extend(by_group[str(key)])
        estimates.append(float(statistic(sampled)))
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def aggregate(args: argparse.Namespace, manifest: Dict) -> Dict:
    rows = []
    for path in sorted(Path(args.out_dir).glob("scores_rank*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    expected = len(manifest["entries"]) * len(CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} score rows, found {len(rows)}")

    by_step: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for row in rows:
        by_step[int(row["global_id"])][row["condition"]] = row

    epsilon = args.effect_epsilon

    def evidence_role(delta: float | None) -> str:
        if delta is None:
            return "unavailable"
        if delta > epsilon:
            return "useful"
        if delta < -epsilon:
            return "harmful"
        return "low_sensitivity"

    step_records = []
    for entry in manifest["entries"]:
        conds = by_step[int(entry["global_id"])]
        full = float(conds["full"]["score"]["avg_logp"])
        deltas = {}
        for modality in MODALITIES:
            row = conds[f"no_{modality}"]
            deltas[modality] = None if not row["available"] else full - float(row["score"]["avg_logp"])
        material = {m: v for m, v in deltas.items() if v is not None and v > epsilon}
        dominant = max(material, key=material.get) if material else "none"
        harmful = [m for m, v in deltas.items() if v is not None and v < -epsilon]
        roles = {m: evidence_role(v) for m, v in deltas.items()}
        step_records.append(
            {
                **entry,
                "full_avg_logp": full,
                "delta": deltas,
                "role": roles,
                "dominant": dominant,
                "harmful": harmful,
            }
        )

    by_episode: Dict[str, List[Dict]] = defaultdict(list)
    for row in step_records:
        by_episode[row["episode_key"]].append(row)
    transitions = []
    episode_switch = {}
    for episode_key, episode_rows in by_episode.items():
        episode_rows.sort(key=lambda r: r["step_index"])
        switches = []
        for prev, cur in zip(episode_rows, episode_rows[1:]):
            common_modalities = [
                m
                for m in MODALITIES
                if prev["role"][m] != "unavailable" and cur["role"][m] != "unavailable"
            ]
            role_changes = [prev["role"][m] != cur["role"][m] for m in common_modalities]
            switches.append(
                {
                    "episode_key": episode_key,
                    "category": cur["category"],
                    "prev_step": prev["step_index"],
                    "step": cur["step_index"],
                    "prev_dominant": prev["dominant"],
                    "dominant": cur["dominant"],
                    "state_switch": prev["dominant"] != cur["dominant"],
                    "valid_pair": prev["dominant"] != "none" and cur["dominant"] != "none",
                    "dominant_switch": prev["dominant"] != cur["dominant"]
                    if prev["dominant"] != "none" and cur["dominant"] != "none"
                    else None,
                    "common_modalities": len(common_modalities),
                    "modality_role_changes": int(sum(role_changes)),
                }
            )
        transitions.extend(switches)
        episode_switch[episode_key] = any(x["state_switch"] for x in switches)

    def category_summary(category: str) -> Dict:
        steps = step_records if category == "Overall" else [x for x in step_records if x["category"] == category]
        trans = transitions if category == "Overall" else [x for x in transitions if x["category"] == category]
        valid = [x for x in trans if x["valid_pair"]]
        episodes = sorted(set(x["episode_key"] for x in steps))
        harmful_flags = [float(bool(x["harmful"])) for x in steps]
        state_flags = [float(x["state_switch"]) for x in trans]
        dominant_flags = [float(x["dominant_switch"]) for x in valid]
        available_roles = [
            x["role"][m]
            for x in steps
            for m in MODALITIES
            if x["role"][m] != "unavailable"
        ]
        role_change_count = sum(x["modality_role_changes"] for x in trans)
        common_modality_count = sum(x["common_modalities"] for x in trans)
        return {
            "steps": len(steps),
            "episodes": len(episodes),
            "transitions": len(trans),
            "valid_dominant_transitions": len(valid),
            "dominant_switch_rate": float(np.mean(dominant_flags)) if dominant_flags else None,
            "evidence_state_switch_rate": float(np.mean(state_flags)) if state_flags else None,
            "episodes_with_state_switch": float(np.mean([episode_switch[e] for e in episodes])),
            "modality_role_change_rate": role_change_count / max(common_modality_count, 1),
            "available_modality_instances": len(available_roles),
            "materially_useful_share": available_roles.count("useful") / max(len(available_roles), 1),
            "low_sensitivity_share": available_roles.count("low_sensitivity") / max(len(available_roles), 1),
            "harmful_modality_share": available_roles.count("harmful") / max(len(available_roles), 1),
            "avg_available_modalities_per_step": len(available_roles) / max(len(steps), 1),
            "avg_useful_modalities_per_step": available_roles.count("useful") / max(len(steps), 1),
            "avg_low_sensitivity_modalities_per_step": available_roles.count("low_sensitivity") / max(len(steps), 1),
            "steps_with_harmful_modality": float(np.mean(harmful_flags)),
            "avg_harmful_modalities": float(np.mean([len(x["harmful"]) for x in steps])),
            "dominant_distribution": dict(Counter(x["dominant"] for x in steps)),
            "mean_delta": {
                m: float(np.mean([x["delta"][m] for x in steps if x["delta"][m] is not None])) for m in MODALITIES
            },
        }

    categories = ["Overall", *CATEGORY_LABELS.values()]
    report = {
        "experiment": "Step-wise modality sensitivity of frozen Qwen2.5-Omni",
        "definition": "delta(t,m) = avg_logp(full) - avg_logp(with modality m omitted)",
        "effect_epsilon_nats_per_token": epsilon,
        "manifest": {k: v for k, v in manifest.items() if k != "entries"},
        "categories": {category: category_summary(category) for category in categories},
        "step_records": step_records,
        "transitions": transitions,
    }
    out_path = Path(args.out_dir) / "results.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="logs/sequential_evidence_probe")
    parser.add_argument("--manifest", default="logs/sequential_evidence_probe/manifest.json")
    parser.add_argument("--episodes-per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--effect-epsilon", type=float, default=0.01)
    parser.add_argument(
        "--pretrained",
        default=os.environ.get(
            "QWEN2_5_OMNI_PRETRAINED",
            "Qwen/Qwen2.5-Omni-7B",
        ),
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--video-max-frames", type=int, default=4)
    parser.add_argument("--video-max-pixels", type=int, default=200704)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(load_docs(), args.episodes_per_category, args.seed)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, ensure_ascii=False, indent=2), flush=True)

    if args.prepare_only:
        return
    if args.aggregate_only:
        report = aggregate(args, manifest)
        print(json.dumps(report["categories"], ensure_ascii=False, indent=2), flush=True)
        return
    run_probe(args, manifest)


if __name__ == "__main__":
    main()
