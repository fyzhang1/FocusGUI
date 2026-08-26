#!/usr/bin/env python
"""Extract label-free, modality-aligned frozen Qwen evidence for FOG.

The extractor performs one ordinary multimodal prompt forward per step. It
never reads action targets or benchmark task categories. Token positions are
partitioned by Qwen's native modality placeholder IDs; the final image span is
used as the current GUI screenshot because the V2 prompt places it after any
history screenshot.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.models.simple.qwen2_5_omni import Qwen2_5_Omni  # noqa: E402
from scripts.train_ares_lm_adapter_qwen import model_device, prepare_prompt  # noqa: E402
from scripts.fog_common import load_docs  # noqa: E402


def cache_key(doc: Dict) -> str:
    return f"{doc['chunk']}:{doc['doc_id']}"


def projection_matrix(input_dim: int, output_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    matrix /= np.sqrt(float(output_dim))
    return matrix.to(device)


def summarize(tokens: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros(projection.shape[1] * 2, device=projection.device)
    projected = tokens.float() @ projection
    return torch.cat([projected.mean(dim=0), projected.amax(dim=0)], dim=0)


def current_screen_tokens(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    merge_size: int,
) -> Tuple[torch.Tensor, int, int]:
    positions = (input_ids == image_token_id).nonzero(as_tuple=False).flatten()
    grids = image_grid_thw.detach().cpu().long().tolist()
    if not grids or positions.numel() == 0:
        return hidden.new_zeros((0, hidden.shape[-1])), 1, 1
    counts = [int(t * h * w // (merge_size * merge_size)) for t, h, w in grids]
    current_count = counts[-1]
    if positions.numel() < sum(counts):
        raise ValueError(f"Image placeholders={positions.numel()} but grid spans require {sum(counts)}")
    current_positions = positions[sum(counts[:-1]) : sum(counts)]
    t, h_raw, w_raw = grids[-1]
    h, w = max(h_raw // merge_size, 1), max(w_raw // merge_size, 1)
    tokens = hidden[current_positions]
    if t > 1:
        tokens = tokens.reshape(t, h * w, -1).mean(dim=0)
    elif tokens.shape[0] != h * w:
        tokens = tokens[-(h * w) :]
    if tokens.shape[0] != h * w or current_count < h * w:
        raise ValueError(f"Current screen token count={tokens.shape[0]}, expected={h}x{w}")
    return tokens, h, w


def spatial_field(tokens: torch.Tensor, h: int, w: int, grid_size: int, projection: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros(grid_size * grid_size, projection.shape[1], device=projection.device)
    projected = tokens.float() @ projection
    field = projected.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(field, (grid_size, grid_size))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(grid_size * grid_size, -1)


def extract_one(
    model: Qwen2_5_Omni,
    doc: Dict,
    grid_size: int,
    proj_dim: int,
    projection_seed: int,
) -> Dict:
    inputs, use_audio_in_video = prepare_prompt(model, doc)
    thinker = model.model.thinker
    with torch.inference_mode():
        outputs = thinker(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            use_audio_in_video=use_audio_in_video,
        )
    hidden = outputs.hidden_states[-1][0].float()
    input_ids = inputs["input_ids"][0]
    attention = inputs.get("attention_mask", torch.ones_like(input_ids)).bool()[0] if inputs.get("attention_mask", None) is not None and inputs["attention_mask"].ndim == 2 else torch.ones_like(input_ids, dtype=torch.bool)
    config = thinker.config
    projection = projection_matrix(hidden.shape[-1], proj_dim, projection_seed, hidden.device)

    screen_tokens, screen_h, screen_w = current_screen_tokens(
        hidden,
        input_ids,
        inputs["image_grid_thw"],
        int(config.image_token_index),
        int(config.vision_config.spatial_merge_size),
    )
    audio_tokens = hidden[input_ids == int(config.audio_token_index)]
    video_tokens = hidden[input_ids == int(config.video_token_index)]
    modality_ids = torch.tensor(
        [
            int(config.image_token_index),
            int(config.video_token_index),
            int(config.audio_token_index),
        ],
        device=input_ids.device,
    )
    text_mask = attention & ~torch.isin(input_ids, modality_ids)
    text_tokens = hidden[text_mask]
    valid_tokens = hidden[attention]

    result = {
        "field": spatial_field(screen_tokens, screen_h, screen_w, grid_size, projection),
        "screen": summarize(screen_tokens, projection),
        "audio": summarize(audio_tokens, projection),
        "video": summarize(video_tokens, projection),
        "text": summarize(text_tokens, projection),
        "full": torch.cat([(valid_tokens @ projection).mean(dim=0), hidden[-1] @ projection], dim=0),
        "grid": np.asarray([screen_h, screen_w], dtype=np.int16),
        "counts": np.asarray(
            [screen_tokens.shape[0], audio_tokens.shape[0], video_tokens.shape[0], text_tokens.shape[0]],
            dtype=np.int32,
        ),
    }
    converted = {
        key: value.detach().cpu().numpy().astype(np.float16) if torch.is_tensor(value) else value
        for key, value in result.items()
    }
    del inputs, outputs, hidden, projection, screen_tokens, audio_tokens, video_tokens, text_tokens, valid_tokens
    gc.collect()
    torch.cuda.empty_cache()
    return converted


def load_existing(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    keys = [str(key) for key in data["keys"].tolist()]
    return {
        key: {
            "field": data["fields"][i],
            "screen": data["screen_features"][i],
            "audio": data["audio_features"][i],
            "video": data["video_features"][i],
            "text": data["text_features"][i],
            "full": data["full_features"][i],
            "grid": data["grids"][i],
            "counts": data["token_counts"][i],
        }
        for i, key in enumerate(keys)
    }


def save_records(path: Path, records: Dict[str, Dict], metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(records)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        tmp,
        keys=np.asarray(keys, dtype=object),
        fields=np.stack([records[key]["field"] for key in keys]),
        screen_features=np.stack([records[key]["screen"] for key in keys]),
        audio_features=np.stack([records[key]["audio"] for key in keys]),
        video_features=np.stack([records[key]["video"] for key in keys]),
        text_features=np.stack([records[key]["text"] for key in keys]),
        full_features=np.stack([records[key]["full"] for key in keys]),
        grids=np.stack([records[key]["grid"] for key in keys]),
        token_counts=np.stack([records[key]["counts"] for key in keys]),
        metadata=np.asarray([json.dumps(metadata, ensure_ascii=True)], dtype=object),
    )
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=20260617)
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--video-max-frames", type=int, default=4)
    parser.add_argument("--video-max-pixels", type=int, default=200704)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--pretrained", default=os.environ.get(
        "QWEN2_5_OMNI_PRETRAINED",
        "Qwen/Qwen2.5-Omni-7B",
    ))
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")

    docs = load_docs()
    selected = docs[args.shard_index :: args.num_shards]
    if args.limit is not None:
        selected = selected[: args.limit]
    out_path = REPO_ROOT / args.out
    records = load_existing(out_path)
    model = Qwen2_5_Omni(
        pretrained=args.pretrained,
        attn_implementation=args.attn_implementation,
        image_max_pixels=args.image_max_pixels,
        video_max_frames=args.video_max_frames,
        video_max_pixels=args.video_max_pixels,
        use_audio_in_video=False,
        device_map="auto",
    )
    metadata = {
        "schema": "qwen_ares_modality_evidence_v1",
        "label_free": True,
        "current_screen_rule": "last image span in V2 prompt",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "grid_size": args.grid_size,
        "proj_dim": args.proj_dim,
        "projection_seed": args.projection_seed,
        "image_max_pixels": args.image_max_pixels,
        "video_max_frames": args.video_max_frames,
        "video_max_pixels": args.video_max_pixels,
    }
    completed = 0
    failures: List[Dict] = []
    for doc in tqdm(selected, desc=f"Qwen FOG evidence shard {args.shard_index}"):
        key = cache_key(doc)
        if key in records:
            continue
        try:
            records[key] = extract_one(model, doc, args.grid_size, args.proj_dim, args.projection_seed)
            completed += 1
        except Exception as exc:
            failures.append({"key": key, "error": repr(exc)})
        if completed and completed % args.save_every == 0:
            save_records(out_path, records, metadata)
    if records:
        save_records(out_path, records, metadata)
    failure_path = out_path.with_suffix(".failures.json")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "records": len(records), "failures": len(failures)}))


if __name__ == "__main__":
    main()
