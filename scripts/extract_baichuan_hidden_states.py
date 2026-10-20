#!/usr/bin/env python
"""Extract frozen Baichuan-Omni hidden states for the FocusGUI pipeline.

This follows the local Baichuan-Omni preprocessing/generation path used by the
raw evaluation.  It captures the LLM prefill hidden state with a forward hook
and never uses labels or generated responses to construct features.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.models.simple.baichuan_omni import BaichuanOmni  # noqa: E402
from lmms_eval.tasks.agentcpm_gui.utils import unified_v2_doc_to_messages  # noqa: E402
from scripts.focusgui_common import load_docs, projection_matrix  # noqa: E402

logger.remove()


def cache_key(doc: Dict) -> str:
    return f"{doc['chunk']}:{doc['doc_id']}"


def load_existing(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    keys = [str(x) for x in data["keys"].tolist()]
    return {
        key: {
            "global_feature": data["global_features"][i],
            "query_feature": data["query_features"][i],
            "meta": data["meta"].tolist()[i],
        }
        for i, key in enumerate(keys)
    }


def save_cache(path: Path, records: Dict[str, Dict], proj_dim: int, projection_seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(records, key=lambda x: tuple(map(int, x.split(":"))))
    global_features = np.stack([records[k]["global_feature"] for k in keys]).astype(np.float16)
    query_features = np.stack([records[k]["query_feature"] for k in keys]).astype(np.float16)
    meta = np.array([records[k]["meta"] for k in keys], dtype=object)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        keys=np.array(keys),
        global_features=global_features,
        query_features=query_features,
        meta=meta,
        proj_dim=np.array([proj_dim], dtype=np.int16),
        projection_seed=np.array([projection_seed], dtype=np.int64),
    )
    tmp.replace(path)


def _hidden_from_output(output) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    return None


def _find_llm_norm(model: BaichuanOmni) -> torch.nn.Module:
    root = model._model
    candidates = [
        "model.norm",
        "model.model.norm",
        "language_model.model.norm",
        "llm.model.norm",
    ]
    for path in candidates:
        cur = root
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    matches = [(name, module) for name, module in root.named_modules() if name.endswith("model.norm") or name.endswith(".norm")]
    if not matches:
        raise RuntimeError("Could not locate Baichuan LLM final norm module for hidden hook.")
    return matches[-1][1]


def extract_one(model: BaichuanOmni, doc: Dict, proj_dim: int, projection_seed: int, max_new_tokens: int) -> Dict:
    raw_messages = unified_v2_doc_to_messages(doc, {"media_dir": "data/benchmark"})
    system_prompt, user_text, visuals = model._parse_messages(raw_messages)
    captured: List[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = _hidden_from_output(output)
        if hidden is not None and hidden.ndim == 3 and hidden.shape[1] > 1:
            if not captured or hidden.shape[1] > captured[0].shape[1]:
                captured[:] = [hidden.detach().float().cpu()]

    handle = _find_llm_norm(model).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            _ = model._local_generate(
                system_prompt,
                user_text,
                visuals,
                {"max_new_tokens": max_new_tokens, "temperature": 0.0},
            )
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Baichuan prefill hidden state was not captured.")
    hidden = captured[0].squeeze(0)
    device = model.device
    mean_hidden = hidden.mean(dim=0).to(device)
    query_hidden = hidden[-1].to(device)
    proj = projection_matrix(int(hidden.shape[-1]), proj_dim, projection_seed, device)
    global_feature = mean_hidden @ proj
    query_feature = query_hidden @ proj
    result = {
        "global_feature": global_feature.detach().cpu().numpy().astype(np.float16),
        "query_feature": query_feature.detach().cpu().numpy().astype(np.float16),
    }
    del hidden, mean_hidden, query_hidden, proj, global_feature, query_feature
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="logs/focusgui_evidence/baichuan_omni_1d5_hidden_p256_strict.npz")
    parser.add_argument("--pretrained", default=os.environ.get("BAICHUAN_OMNI_PRETRAINED", "baichuan-inc/Baichuan-Omni-1d5"))
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=20260706)
    parser.add_argument("--max-video-frames", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    parser.add_argument("--video-max-pixels", type=int, default=200704)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    out_path = REPO_ROOT / args.out
    docs = load_docs()
    selected = docs[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    records = load_existing(out_path)
    model = BaichuanOmni(
        backend="local",
        pretrained=args.pretrained,
        local_max_new_tokens=args.max_new_tokens,
        max_video_frames=args.max_video_frames,
        image_max_pixels=args.image_max_pixels,
        video_max_pixels=args.video_max_pixels,
        fail_fast=False,
    )
    failures = []
    completed_since_save = 0
    for doc in tqdm(selected, desc="Extract Baichuan hidden states"):
        key = cache_key(doc)
        if key in records:
            continue
        try:
            result = extract_one(model, doc, args.proj_dim, args.projection_seed, args.max_new_tokens)
            records[key] = {
                **result,
                "meta": {
                    "chunk": int(doc["chunk"]),
                    "doc_id": int(doc["doc_id"]),
                    "global_id": int(doc["global_id"]),
                    "episode_id": doc.get("episode_id"),
                    "step_index": doc.get("step_index"),
                },
            }
            completed_since_save += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"key": key, "error": error})
            print(f"failed {key}: {error}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        if completed_since_save >= args.save_every:
            save_cache(out_path, records, args.proj_dim, args.projection_seed)
            completed_since_save = 0
    save_cache(out_path, records, args.proj_dim, args.projection_seed)
    if failures:
        out_path.with_suffix(".failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out_path), "records": len(records), "failures": len(failures)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
