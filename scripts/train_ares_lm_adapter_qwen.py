#!/usr/bin/env python
"""Train FOG as latent-token conditioning for the frozen Qwen LM decoder.

Unlike the legacy EIB action head, this implementation never predicts an
action class or coordinate grid directly. FOG produces continuous prefix
tokens; the original frozen Qwen thinker and LM head remain responsible for
token-level action modeling and autoregressive generation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.models.simple.qwen2_5_omni import Qwen2_5_Omni  # noqa: E402
from lmms_eval.tasks.agentcpm_gui.utils import (  # noqa: E402
    ENGLISH_APPS,
    eval_logger,
    evaluate_omni_step,
    omni_doc_to_target,
    unified_v2_doc_to_messages,
)
from qwen_omni_utils import process_mm_info  # noqa: E402
from scripts.fog_common import load_docs, split_by_episode  # noqa: E402

eval_logger.remove()
logging.getLogger().setLevel(logging.ERROR)

EVIDENCE_CONTRACT = {
    "text_history": "Prompt-visible goal/history plus frozen text-token hidden evidence",
    "screen": "Frozen hidden evidence from the final image span (current screenshot)",
    "audio": "Frozen hidden evidence at native Qwen audio-token positions",
    "video": "Frozen hidden evidence at native Qwen video-token positions",
    "state": "Current step index and observed action-history length only",
    "qwen_fullmodal": "Frozen Qwen full-observation global/query hidden evidence",
    "screen_spatial": "Frozen spatial field from the final current-screen image span",
}

FUSION_MODES = (
    "dynamic",
    "static",
    "uniform",
    "naive_concat",
    "fullmodal_only",
)

FUSION_DESCRIPTIONS = {
    "dynamic": "Current-step full-modal-query-conditioned evidence selection",
    "static": "Sample-independent learned evidence weights",
    "uniform": "Fixed equal-weight evidence averaging",
    "naive_concat": "Indiscriminate concatenation followed by a learned compression layer",
    "fullmodal_only": "Single frozen Qwen full-modal evidence stream",
}


def observable_text(doc: Dict) -> str:
    """Reproduce prompt-visible text without paths, taxonomy, or future state."""
    app = str(doc.get("app", ""))
    if app in ENGLISH_APPS:
        goal = str(doc.get("instruction_en") or doc.get("instruction") or "")
    else:
        goal = str(doc.get("instruction") or "")
    step = int(doc.get("step_index", doc.get("step_id", 0)) or 0)
    history_lines = []
    for item in doc.get("action_history") or []:
        if not isinstance(item, dict):
            continue
        history_lines.append(
            f"Step {int(item.get('step_index', 0) or 0)}: {str(item.get('action_text', ''))}"
        )
    parts = [f"Goal: {goal}", f"Step: {step}"]
    if history_lines:
        parts.append("Action History:\n" + "\n".join(history_lines))
    return "\n\n".join(parts)


def standardize_group(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train).astype(np.float32), scaler.transform(test).astype(np.float32)


def build_observable_groups(
    docs: Sequence[Dict],
    semantic_evidence: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    text_dim: int,
    lexical_features: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Build train-fitted evidence streams from inference-time observables only."""
    text_all = semantic_evidence["text"]
    text_train = text_all[train_idx]
    text_test = text_all[test_idx]
    if lexical_features:
        texts = [observable_text(doc) for doc in docs]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), max_features=5000, min_df=2
        )
        text_train_sparse = vectorizer.fit_transform([texts[i] for i in train_idx])
        text_test_sparse = vectorizer.transform([texts[i] for i in test_idx])
        svd_dim = min(text_dim, text_train_sparse.shape[1] - 1)
        if svd_dim > 8:
            svd = TruncatedSVD(n_components=svd_dim, random_state=13)
            lexical_train = svd.fit_transform(text_train_sparse).astype(np.float32)
            lexical_test = svd.transform(text_test_sparse).astype(np.float32)
        else:
            lexical_train = text_train_sparse.toarray().astype(np.float32)
            lexical_test = text_test_sparse.toarray().astype(np.float32)
        text_train = np.concatenate([text_train, lexical_train], axis=1)
        text_test = np.concatenate([text_test, lexical_test], axis=1)

    observable_state = np.asarray(
        [
            [
                float(doc.get("step_index", doc.get("step_id", 0)) or 0) / 30.0,
                len(doc.get("action_history") or []) / 30.0,
            ]
            for doc in docs
        ],
        dtype=np.float32,
    )
    raw = {
        "text_history": (text_train, text_test),
        "screen": (semantic_evidence["screen"][train_idx], semantic_evidence["screen"][test_idx]),
        "audio": (semantic_evidence["audio"][train_idx], semantic_evidence["audio"][test_idx]),
        "video": (semantic_evidence["video"][train_idx], semantic_evidence["video"][test_idx]),
        "state": (observable_state[train_idx], observable_state[test_idx]),
        "qwen_fullmodal": (semantic_evidence["full"][train_idx], semantic_evidence["full"][test_idx]),
        "screen_spatial": (semantic_evidence["spatial"][train_idx], semantic_evidence["spatial"][test_idx]),
    }
    names, train_groups, test_groups = [], [], []
    for name, (train, test) in raw.items():
        train_scaled, test_scaled = standardize_group(train, test)
        names.append(name)
        train_groups.append(train_scaled)
        test_groups.append(test_scaled)
    return train_groups, test_groups, names


def load_semantic_evidence(docs: Sequence[Dict], cache_paths: Sequence[Path]) -> Tuple[Dict[str, np.ndarray], Dict]:
    records: Dict[str, Dict[str, np.ndarray]] = {}
    cache_metadata = []
    for path in cache_paths:
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata"][0]))
        if metadata.get("label_free") is not True:
            raise ValueError(f"Semantic evidence cache is not marked label-free: {path}")
        cache_metadata.append({"path": str(path), **metadata})
        fields = data["fields"].astype(np.float32)
        full_features = data["full_features"].astype(np.float32)
        query = full_features[:, full_features.shape[1] // 2 :]
        field_unit = fields / np.maximum(np.linalg.norm(fields, axis=2, keepdims=True), 1e-6)
        query_unit = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-6)
        scores = np.einsum("nkd,nd->nk", field_unit, query_unit) / 0.1
        scores -= scores.max(axis=1, keepdims=True)
        posterior = np.exp(scores)
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-8)
        side = int(round(np.sqrt(fields.shape[1])))
        if side * side != fields.shape[1]:
            raise ValueError(f"Spatial fields in {path} are not square: {fields.shape}")
        yy, xx = np.meshgrid(
            (np.arange(side, dtype=np.float32) + 0.5) / side,
            (np.arange(side, dtype=np.float32) + 0.5) / side,
            indexing="ij",
        )
        spatial_statistics = np.stack(
            [
                posterior @ xx.reshape(-1),
                posterior @ yy.reshape(-1),
                -np.sum(posterior * np.log(np.maximum(posterior, 1e-8)), axis=1)
                / np.log(fields.shape[1]),
                posterior.max(axis=1),
            ],
            axis=1,
        ).astype(np.float32)
        spatial_features = np.concatenate(
            [
                fields.mean(axis=1),
                fields.max(axis=1),
                np.einsum("nk,nkd->nd", posterior, fields),
                spatial_statistics,
            ],
            axis=1,
        ).astype(np.float32)
        screen_features = data["screen_features"].astype(np.float32)
        audio_features = data["audio_features"].astype(np.float32)
        video_features = data["video_features"].astype(np.float32)
        text_features = data["text_features"].astype(np.float32)
        token_counts = data["token_counts"].astype(np.int64)
        for i, key in enumerate(data["keys"].tolist()):
            key = str(key)
            if key in records:
                raise ValueError(f"Duplicate semantic evidence key {key} across caches")
            records[key] = {
                "screen": screen_features[i],
                "audio": audio_features[i],
                "video": video_features[i],
                "text": text_features[i],
                "full": full_features[i],
                "spatial": spatial_features[i],
                "counts": token_counts[i],
            }
    ordered = []
    missing = []
    for doc in docs:
        key = f"{doc['chunk']}:{doc['doc_id']}"
        if key not in records:
            missing.append(key)
        else:
            ordered.append(records[key])
    if missing:
        raise ValueError(f"Semantic evidence caches miss {len(missing)} documents, e.g. {missing[:5]}")
    names = ["screen", "audio", "video", "text", "full", "spatial", "counts"]
    evidence = {name: np.stack([record[name] for record in ordered]) for name in names}
    audit = {
        "caches": cache_metadata,
        "documents": len(ordered),
        "token_count_mean": {
            name: float(evidence["counts"][:, i].mean())
            for i, name in enumerate(["screen", "audio", "video", "text"])
        },
        "zero_token_documents": {
            name: int(np.sum(evidence["counts"][:, i] == 0))
            for i, name in enumerate(["screen", "audio", "video", "text"])
        },
        "spatial_pooling": {
            "method": "label-free frozen-query cosine posterior over the current-screen 12x12 field",
            "temperature": 0.1,
            "statistics": ["posterior_mean", "posterior_max", "posterior_attended", "expected_x", "expected_y", "normalized_entropy", "peak"],
        },
    }
    return evidence, audit


def _box_center(box: Sequence[Sequence[float]]) -> Tuple[float, float]:
    (x1, y1), (x2, y2) = box
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def _normalized_point(doc: Dict, point: Tuple[float, float]) -> List[int]:
    width = max(float(doc.get("image_width") or 1.0), 1.0)
    height = max(float(doc.get("image_height") or 1.0), 1.0)
    return [
        int(round(np.clip(point[0] / width * 1000.0, 0.0, 1000.0))),
        int(round(np.clip(point[1] / height * 1000.0, 0.0, 1000.0))),
    ]


def canonical_action_target(doc: Dict) -> str:
    """Serialize GT in the same normalized JSON protocol requested from Qwen."""
    action_type = int(doc["result_action_type"])
    action: Dict = {"action_type": action_type}
    if action_type in {0, 1, 2}:
        action["coordinate"] = _normalized_point(doc, _box_center(json.loads(doc["result_touch_xy"])))
    elif action_type in {3, 4, 5, 6}:
        start_box, end_box = json.loads(doc["result_lift_xy"])
        action["start_coordinate"] = _normalized_point(doc, _box_center(start_box))
        action["end_coordinate"] = _normalized_point(doc, _box_center(end_box))
    elif action_type == 7:
        action["text"] = str(doc.get("result_action_text", ""))
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


class FOGLatentTokenAdapter(nn.Module):
    """Fuse observable evidence and condition the frozen action decoder."""

    def __init__(
        self,
        group_dims: Sequence[int],
        token_dim: int,
        z_dim: int,
        num_latent_tokens: int,
        model_dim: int,
        dropout: float,
        temperature: float,
        query_group_index: int,
        use_vib: bool,
        initial_logvar: float,
        fusion_mode: str = "dynamic",
    ) -> None:
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"Unknown fusion mode {fusion_mode!r}; expected one of {FUSION_MODES}")
        self.num_groups = len(group_dims)
        self.z_dim = z_dim
        self.num_latent_tokens = num_latent_tokens
        self.model_dim = model_dim
        self.temperature = temperature
        self.query_group_index = query_group_index
        self.use_vib = use_vib
        self.fusion_mode = fusion_mode
        self.group_proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, token_dim), nn.GELU(), nn.LayerNorm(token_dim)) for dim in group_dims]
        )
        if fusion_mode == "dynamic":
            self.group_embedding = nn.Parameter(torch.empty(self.num_groups, token_dim))
            self.query_proj = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.LayerNorm(token_dim))
            self.weight_stream = nn.Linear(token_dim, token_dim)
            self.weight_query = nn.Linear(token_dim, token_dim, bias=False)
            self.weight_score = nn.Linear(token_dim, 1, bias=False)
            nn.init.normal_(self.group_embedding, mean=0.0, std=0.02)
        elif fusion_mode == "static":
            self.static_logits = nn.Parameter(torch.zeros(self.num_groups))
        elif fusion_mode == "naive_concat":
            self.concat_fuser = nn.Sequential(
                nn.Linear(self.num_groups * token_dim, token_dim),
                nn.GELU(),
                nn.LayerNorm(token_dim),
            )
        self.vib_encoder = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(token_dim, z_dim)
        self.logvar = nn.Linear(token_dim, z_dim)
        self.latent_projection = nn.Linear(z_dim, num_latent_tokens * model_dim)
        # Starting from unit posterior variance injects noise as large as the
        # signal before the zero-initialized decoder projection has learned a
        # useful direction. A low-variance initialization keeps early updates
        # stable while leaving both posterior moments fully trainable.
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, initial_logvar)
        # Identity-preserving initialization: before learning, FOG leaves the
        # frozen decoder exactly unchanged while gradients still reach W_p.
        nn.init.zeros_(self.latent_projection.weight)
        nn.init.zeros_(self.latent_projection.bias)

    def forward(
        self, groups: Sequence[torch.Tensor], sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        streams = torch.stack([proj(group) for proj, group in zip(self.group_proj, groups)], dim=1)
        if self.fusion_mode == "dynamic":
            query = self.query_proj(streams[:, self.query_group_index])
            scoring_streams = streams + self.group_embedding.unsqueeze(0)
            weight_hidden = torch.tanh(
                self.weight_stream(scoring_streams) + self.weight_query(query).unsqueeze(1)
            )
            weight_logits = self.weight_score(weight_hidden).squeeze(-1)
            weights = torch.softmax(weight_logits / self.temperature, dim=-1)
            fused = torch.sum(streams * weights.unsqueeze(-1), dim=1)
        elif self.fusion_mode == "static":
            weights = torch.softmax(self.static_logits / self.temperature, dim=0)
            weights = weights.unsqueeze(0).expand(streams.shape[0], -1)
            fused = torch.sum(streams * weights.unsqueeze(-1), dim=1)
        elif self.fusion_mode == "uniform":
            weights = streams.new_full(
                (streams.shape[0], self.num_groups), 1.0 / self.num_groups
            )
            fused = torch.mean(streams, dim=1)
        elif self.fusion_mode == "fullmodal_only":
            weights = streams.new_zeros((streams.shape[0], self.num_groups))
            weights[:, self.query_group_index] = 1.0
            fused = streams[:, self.query_group_index]
        else:
            # The placeholder is returned only to keep the training/evaluation
            # interface uniform. It is never reported as an evidence posterior.
            weights = streams.new_full(
                (streams.shape[0], self.num_groups), 1.0 / self.num_groups
            )
            fused = self.concat_fuser(streams.flatten(start_dim=1))
        hidden = self.vib_encoder(fused)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-8.0, 6.0)
        if self.training and sample and self.use_vib:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        latent_tokens = self.latent_projection(z).reshape(-1, self.num_latent_tokens, self.model_dim)
        return latent_tokens, mu, logvar, weights


def aggregate(records: List[Dict]) -> Dict:
    episode_results = defaultdict(list)
    for record in records:
        key = f"{record['doc'].get('app')}-{record['doc'].get('episode_id')}"
        episode_results[key].append(record["eval"])
    successes, progresses = [], []
    for rows in episode_results.values():
        rows.sort(key=lambda x: x.get("step_id", 0))
        successes.append(all(x.get("exact_match", False) for x in rows))
        prefix = 0
        for row in rows:
            if not row.get("exact_match", False):
                break
            prefix += 1
        progresses.append(prefix / len(rows) if rows else 0.0)
    return {
        "num_steps": len(records),
        "num_episodes": len(episode_results),
        "hit_format": float(np.mean([r["eval"]["hit_format"] for r in records])),
        "type_match": float(np.mean([r["eval"]["type_match"] for r in records])),
        "exact_match": float(np.mean([r["eval"]["exact_match"] for r in records])),
        "success_rate": float(np.mean(successes)),
        "goal_progress": float(np.mean(progresses)),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_info() -> Tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def unwrap_adapter(adapter: nn.Module) -> FOGLatentTokenAdapter:
    return adapter.module if isinstance(adapter, DDP) else adapter


def shard_positions(length: int, rank: int, world_size: int) -> np.ndarray:
    return np.arange(length, dtype=np.int64)[rank::world_size]


def select_group_rows(groups: Sequence[np.ndarray], positions: np.ndarray) -> List[np.ndarray]:
    return [group[positions] for group in groups]


def model_device(model: Qwen2_5_Omni) -> torch.device:
    return model.model.thinker.get_input_embeddings().weight.device


def move_processor_inputs(inputs, device: torch.device, dtype: torch.dtype):
    inputs = inputs.to(device)
    for key, value in list(inputs.items()):
        if torch.is_tensor(value) and value.is_floating_point():
            inputs[key] = value.to(dtype=dtype)
    return inputs


def prepare_prompt(model: Qwen2_5_Omni, doc: Dict):
    raw_messages = unified_v2_doc_to_messages(doc, {"media_dir": "data/benchmark"})
    messages, use_audio_in_video = model._convert_messages_for_qwen(raw_messages)
    prompt_text = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = model.processor(
        text=prompt_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    return move_processor_inputs(inputs, model_device(model), model.model.dtype), use_audio_in_video


def prepare_conditioned_inputs(
    model: Qwen2_5_Omni,
    prompt_inputs,
    target_ids: torch.Tensor | None,
) -> Tuple[Dict, torch.Tensor, int, int]:
    """Build standard Qwen inputs without bypassing multimodal token injection."""
    device = model_device(model)
    prompt_ids = prompt_inputs["input_ids"].to(device)
    if prompt_ids.shape[0] != 1:
        raise ValueError("Current Qwen backend intentionally supports batch size 1 only.")
    im_end_token_id = model.tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_end_positions = (prompt_ids[0] == im_end_token_id).nonzero(as_tuple=False).flatten()
    if im_end_positions.numel() == 0:
        raise ValueError("Could not locate the final user <|im_end|> token in the Qwen prompt.")
    # The generation template already appends the assistant-role prefix. FOG
    # conditions the final K prompt embeddings at this action boundary, rather
    # than overwriting arbitrary words inside the user instruction.
    anchor_end = int(prompt_ids.shape[1])
    model_inputs = {key: value for key, value in prompt_inputs.items()}
    if target_ids is not None:
        target_ids = target_ids.to(device).reshape(1, -1)
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        prompt_mask = prompt_inputs.get("attention_mask", torch.ones_like(prompt_ids)).to(device)
        model_inputs["attention_mask"] = torch.cat([prompt_mask, torch.ones_like(target_ids)], dim=1)
        model_inputs["input_ids"] = input_ids
        labels = torch.full_like(input_ids, -100)
        labels[:, -target_ids.shape[1] :] = target_ids
    else:
        input_ids = prompt_ids
        model_inputs["input_ids"] = input_ids
        labels = torch.full_like(input_ids, -100)
    return model_inputs, labels, anchor_end, int(input_ids.shape[1])


@contextmanager
def action_cue_residual(
    model: Qwen2_5_Omni,
    anchor_end: int,
    latent_tokens: torch.Tensor,
    residual_scale: float,
):
    """Add FOG residuals while retaining Qwen's native multimodal forward path.

    A forward hook is necessary here: supplying ``inputs_embeds`` directly to
    Qwen skips its internal replacement of audio/image/video placeholder
    embeddings with encoder features. The hook acts on ordinary text-token
    embeddings first; Qwen then performs its original multimodal scatter.
    """
    if latent_tokens.shape[0] != 1:
        raise ValueError("Current Qwen backend intentionally supports batch size 1 only.")
    anchor_start = anchor_end - latent_tokens.shape[1]
    if anchor_start < 0:
        raise ValueError("The Qwen prompt is shorter than the requested FOG latent-token span.")
    if residual_scale == 0.0:
        yield
        return

    embedder = model.model.thinker.get_input_embeddings()
    residual = (residual_scale * latent_tokens).to(device=model_device(model), dtype=embedder.weight.dtype)

    def add_residual(_module, _args, output):
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < anchor_end:
            return output
        return torch.cat(
            [
                output[:, :anchor_start],
                output[:, anchor_start:anchor_end] + residual,
                output[:, anchor_end:],
            ],
            dim=1,
        )

    handle = embedder.register_forward_hook(add_residual)
    try:
        yield
    finally:
        handle.remove()


def thinker_forward(model: Qwen2_5_Omni, kwargs: Dict, use_audio_in_video: bool):
    try:
        return model.model.thinker(
            **kwargs,
            use_audio_in_video=use_audio_in_video,
            use_cache=False,
            return_dict=True,
        )
    except TypeError:
        kwargs = dict(kwargs)
        kwargs.pop("feature_attention_mask", None)
        return model.model.thinker(
            **kwargs,
            use_audio_in_video=use_audio_in_video,
            use_cache=False,
            return_dict=True,
        )


@contextmanager
def supervised_lm_head_window(model: Qwen2_5_Omni, start: int, end: int):
    """Apply the frozen LM head only to hidden states supervised by action CE.

    Hugging Face normally materializes vocabulary logits for every multimodal
    prompt token even though all prompt labels are ignored. Slicing the input
    to the original LM head to ``[prompt_len - 1, sequence_len - 1)`` is
    mathematically identical to causal LM loss on the action suffix and avoids
    a large ``sequence_length x vocabulary_size`` allocation.
    """
    lm_head = model.model.thinker.lm_head

    def select_hidden(_module, args):
        if not args or args[0].ndim != 3:
            return args
        return (args[0][:, start:end], *args[1:])

    handle = lm_head.register_forward_pre_hook(select_hidden)
    try:
        yield
    finally:
        handle.remove()


def token_loss(
    model: Qwen2_5_Omni,
    doc: Dict,
    latent_tokens: torch.Tensor,
    residual_scale: float,
    activation_offload_threshold: int | None = None,
) -> Tuple[torch.Tensor, int, int, bool]:
    prompt_inputs, use_audio_in_video = prepare_prompt(model, doc)
    target_text = canonical_action_target(doc)
    target_list = model.tokenizer.encode(target_text, add_special_tokens=False) + [model.tokenizer.eos_token_id]
    target_ids = torch.tensor(target_list, dtype=torch.long, device=model_device(model))
    model_inputs, _, anchor_end, sequence_length = prepare_conditioned_inputs(model, prompt_inputs, target_ids)
    prompt_length = sequence_length - target_ids.numel()
    use_offload = bool(
        torch.is_grad_enabled()
        and activation_offload_threshold is not None
        and activation_offload_threshold > 0
        and sequence_length >= activation_offload_threshold
    )
    offload_context = torch.autograd.graph.save_on_cpu(pin_memory=True) if use_offload else nullcontext()
    with offload_context:
        with action_cue_residual(model, anchor_end, latent_tokens, residual_scale):
            with supervised_lm_head_window(model, prompt_length - 1, sequence_length - 1):
                outputs = thinker_forward(model, model_inputs, use_audio_in_video)
    logits = outputs.logits.float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
    del prompt_inputs, model_inputs, outputs, logits, target_ids
    return loss, len(target_list), sequence_length, use_offload


@torch.inference_mode()
def generate_action(
    model: Qwen2_5_Omni,
    doc: Dict,
    latent_tokens: torch.Tensor,
    residual_scale: float,
    max_new_tokens: int,
) -> str:
    prompt_inputs, use_audio_in_video = prepare_prompt(model, doc)
    model_inputs, _, anchor_end, input_length = prepare_conditioned_inputs(model, prompt_inputs, None)
    generate_kwargs = {
        **model_inputs,
        "return_audio": False,
        "eos_token_id": model.tokenizer.eos_token_id,
        "pad_token_id": model.tokenizer.pad_token_id,
        "do_sample": False,
        "num_beams": 1,
        "thinker_max_new_tokens": max_new_tokens,
        "use_cache": True,
        "use_audio_in_video": use_audio_in_video,
        "thinker_do_sample": False,
    }
    with action_cue_residual(model, anchor_end, latent_tokens, residual_scale):
        try:
            generated = model.model.generate(**generate_kwargs)
        except TypeError:
            generate_kwargs.pop("feature_attention_mask", None)
            generated = model.model.generate(**generate_kwargs)
    generated_ids = generated[0, input_length:]
    response = model.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del prompt_inputs, model_inputs, generate_kwargs, generated
    return response.strip()


@torch.inference_mode()
def generate_raw_action(model: Qwen2_5_Omni, doc: Dict, max_new_tokens: int) -> str:
    """Run the untouched frozen Qwen generation path for a matched baseline."""
    prompt_inputs, use_audio_in_video = prepare_prompt(model, doc)
    input_length = int(prompt_inputs["input_ids"].shape[1])
    generate_kwargs = {
        **prompt_inputs,
        "return_audio": False,
        "eos_token_id": model.tokenizer.eos_token_id,
        "pad_token_id": model.tokenizer.pad_token_id,
        "do_sample": False,
        "num_beams": 1,
        "thinker_max_new_tokens": max_new_tokens,
        "use_cache": True,
        "use_audio_in_video": use_audio_in_video,
        "thinker_do_sample": False,
    }
    try:
        generated = model.model.generate(**generate_kwargs)
    except TypeError:
        generate_kwargs.pop("feature_attention_mask", None)
        generated = model.model.generate(**generate_kwargs)
    generated_ids = generated[0, input_length:]
    response = model.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del prompt_inputs, generate_kwargs, generated
    return response.strip()


def evidence_batch(groups: Sequence[np.ndarray], index: int, device: torch.device) -> List[torch.Tensor]:
    return [torch.from_numpy(group[index : index + 1]).float().to(device) for group in groups]


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean() / mu.shape[-1]


def weight_loss(weights: torch.Tensor) -> torch.Tensor:
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy.mean() / math.log(weights.shape[-1])


def evaluate_teacher_forcing(
    adapter: nn.Module,
    model: Qwen2_5_Omni,
    groups: Sequence[np.ndarray],
    docs: Sequence[Dict],
    indices: Sequence[int],
    residual_scale: float,
    beta_kl: float,
    lambda_weight: float,
    limit: int | None,
    show_progress: bool = True,
) -> Dict:
    adapter.eval()
    selected = list(indices[:limit] if limit is not None else indices)
    sample_loss_sum = token_loss_sum = kl_sum = weight_sum = 0.0
    token_count = 0
    for local_index, doc_index in enumerate(
        tqdm(selected, desc="FOG dev teacher forcing", disable=not show_progress)
    ):
        with torch.no_grad():
            latent, mu, logvar, weights = adapter(
                evidence_batch(groups, local_index, model_device(model)), sample=False
            )
        loss, tokens, _, _ = token_loss(model, docs[doc_index], latent, residual_scale)
        sample_loss_sum += float(loss.detach().cpu())
        token_loss_sum += float(loss.detach().cpu()) * tokens
        kl_sum += float(kl_loss(mu, logvar).detach().cpu())
        weight_sum += float(weight_loss(weights).detach().cpu())
        token_count += tokens
        del latent, mu, logvar, weights, loss
        gc.collect()
        torch.cuda.empty_cache()
    totals = torch.tensor(
        [sample_loss_sum, token_loss_sum, kl_sum, weight_sum, len(selected), token_count],
        dtype=torch.float64,
        device=model_device(model),
    )
    if distributed_ready():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    sample_loss_sum, token_loss_sum, kl_sum, weight_sum, sample_count, token_count = totals.cpu().tolist()
    n = max(int(sample_count), 1)
    token_count = int(token_count)
    mean_token = token_loss_sum / max(token_count, 1)
    mean_sample = sample_loss_sum / n
    mean_kl = kl_sum / n
    mean_weight = weight_sum / n
    return {
        "action_nll": mean_sample,
        "token_nll": mean_token,
        "kl": mean_kl,
        "weight_entropy": mean_weight,
        # Training gives every GUI action equal weight. Use the matching
        # per-action validation objective so longer serialized actions do not
        # dominate checkpoint selection merely because they contain more tokens.
        "objective": mean_sample + beta_kl * mean_kl + lambda_weight * mean_weight,
        "samples": int(sample_count),
        "target_tokens": token_count,
    }


def save_checkpoint(path: Path, adapter: nn.Module, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": unwrap_adapter(adapter).state_dict(), "metadata": metadata}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=os.environ.get(
        "QWEN2_5_OMNI_PRETRAINED",
        "Qwen/Qwen2.5-Omni-7B",
    ))
    parser.add_argument(
        "--semantic-evidence-cache",
        default=(
            "logs/ares_evidence/qwen_modality_semantic_v1_shard0.npz,"
            "logs/ares_evidence/qwen_modality_semantic_v1_shard1.npz"
        ),
    )
    parser.add_argument("--out-dir", default="logs/ares_lm_adapter_qwen_strict")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260626, help="Episode split seed")
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Adapter initialization and training-order seed; defaults to --seed",
    )
    parser.add_argument("--text-dim", type=int, default=192)
    parser.add_argument(
        "--lexical-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally append train-fitted TF-IDF/SVD features to the frozen text hidden stream.",
    )
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--num-latent-tokens", type=int, default=4)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help=(
            "Scale of the learned action-boundary residual. The projection is zero-initialized, "
            "so 1.0 preserves the exact frozen-model initialization without attenuating gradients."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gate-query-group", default="qwen_fullmodal")
    parser.add_argument(
        "--fusion-mode",
        choices=FUSION_MODES,
        default="dynamic",
        help="Evidence fusion ablation. All modes retain the same frozen decoder/LM-head path.",
    )
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--initial-logvar", type=float, default=-4.0)
    parser.add_argument("--lambda-weight", type=float, default=1e-3)
    parser.add_argument("--vib", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-split", choices=["dev", "test"], default="test")
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--video-max-frames", type=int, default=4)
    parser.add_argument("--video-max-pixels", type=int, default=200704)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--activation-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload saved frozen-decoder activations for long sequences during FOG training.",
    )
    parser.add_argument(
        "--activation-offload-threshold",
        type=int,
        default=6000,
        help="Minimum full sequence length that activates CPU offload; <=0 disables it.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--anchor-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.fusion_mode == "naive_concat" and args.lambda_weight != 0.0:
        raise ValueError("--lambda-weight must be 0 for naive_concat because it has no evidence posterior.")

    rank, local_rank, world_size = distributed_info()
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed FOG training requires CUDA.")
        torch.cuda.set_device(local_rank)
        if args.raw_only or args.anchor_only:
            raise ValueError("Use a single process for --raw-only and --anchor-only controls.")

    init_seed = args.seed if args.init_seed is None else args.init_seed
    set_seed(init_seed)
    out_dir = REPO_ROOT / args.out_dir
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best_adapter.pt"

    docs = load_docs()
    train_idx, dev_idx, test_idx = split_by_episode(docs, args.test_ratio, args.dev_ratio, args.seed)
    evidence_paths = [REPO_ROOT / value.strip() for value in args.semantic_evidence_cache.split(",") if value.strip()]
    if not evidence_paths:
        raise ValueError("--semantic-evidence-cache must name at least one cache")
    semantic_evidence, evidence_audit = load_semantic_evidence(docs, evidence_paths)
    train_groups, dev_groups, group_names = build_observable_groups(
        docs, semantic_evidence, train_idx, dev_idx, args.text_dim, args.lexical_features
    )
    _, test_groups, _ = build_observable_groups(
        docs, semantic_evidence, train_idx, test_idx, args.text_dim, args.lexical_features
    )
    if args.gate_query_group not in group_names:
        raise ValueError(f"Unknown --gate-query-group={args.gate_query_group!r}; available: {group_names}")
    query_group_index = group_names.index(args.gate_query_group)

    if args.smoke:
        args.train_limit = max(world_size, 1)
        args.dev_limit = max(world_size, 1)
        args.test_limit = max(world_size, 1)
        args.epochs = 1
        args.grad_accum = max(world_size, 1)

    split_metadata = {
        "seed": args.seed,
        "train_steps": int(len(train_idx)),
        "dev_steps": int(len(dev_idx)),
        "test_steps": int(len(test_idx)),
        "train_episodes": len({f"{docs[i].get('app')}-{docs[i].get('episode_id')}" for i in train_idx}),
        "dev_episodes": len({f"{docs[i].get('app')}-{docs[i].get('episode_id')}" for i in dev_idx}),
        "test_episodes": len({f"{docs[i].get('app')}-{docs[i].get('episode_id')}" for i in test_idx}),
    }

    model = Qwen2_5_Omni(
        pretrained=args.pretrained,
        attn_implementation=args.attn_implementation,
        image_max_pixels=args.image_max_pixels,
        video_max_frames=args.video_max_frames,
        video_max_pixels=args.video_max_pixels,
        use_audio_in_video=False,
        device_map="auto",
    )
    if world_size > 1 and not distributed_ready():
        raise RuntimeError("Qwen initialization did not establish the torch.distributed process group.")
    if world_size > 1:
        dist.barrier()

    generation_idx = dev_idx if args.generation_split == "dev" else test_idx
    generation_groups = dev_groups if args.generation_split == "dev" else test_groups

    if args.raw_only:
        selected_test = list(generation_idx[: args.test_limit] if args.test_limit is not None else generation_idx)
        raw_records = []
        for doc_index in tqdm(selected_test, desc="Raw Qwen matched held-out generation"):
            response = generate_raw_action(model, docs[int(doc_index)], args.max_new_tokens)
            raw_records.append(
                {
                    "doc": docs[int(doc_index)],
                    "prediction": response,
                    "eval": evaluate_omni_step(docs[int(doc_index)], response, coordinate_type="auto"),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
        raw_report = {
            "method": "Raw frozen Qwen decoder and LM head",
            "split": split_metadata,
            "evaluation_scope": {
                "split": args.generation_split,
                "test_limit": args.test_limit,
                "evaluated_steps": len(selected_test),
                "is_full_split": args.test_limit is None and len(selected_test) == len(generation_idx),
            },
            args.generation_split: aggregate(raw_records),
            "records": [
                {
                    "global_id": int(row["doc"]["global_id"]),
                    "app": row["doc"].get("app"),
                    "episode_id": row["doc"].get("episode_id"),
                    "step_index": row["doc"].get("step_index"),
                    "prediction": row["prediction"],
                    "target": canonical_action_target(row["doc"]),
                    "benchmark_target": omni_doc_to_target(row["doc"]),
                    "eval": row["eval"],
                }
                for row in raw_records
            ],
        }
        raw_path = out_dir / ("raw_dev_results.json" if args.generation_split == "dev" else "raw_results.json")
        raw_path.write_text(json.dumps(raw_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in raw_report.items() if key != "records"}, ensure_ascii=False, indent=2))
        return

    if args.anchor_only:
        thinker = model.model.thinker
        model_dim = int(thinker.config.get_text_config().hidden_size)
        zero_latent = torch.zeros(
            1,
            args.num_latent_tokens,
            model_dim,
            device=model_device(model),
            dtype=model.model.dtype,
        )
        selected_test = list(generation_idx[: args.test_limit] if args.test_limit is not None else generation_idx)
        anchor_records = []
        for doc_index in tqdm(selected_test, desc="Anchor-only matched held-out generation"):
            response = generate_action(
                model,
                docs[int(doc_index)],
                zero_latent,
                args.residual_scale,
                args.max_new_tokens,
            )
            anchor_records.append(
                {
                    "doc": docs[int(doc_index)],
                    "prediction": response,
                    "eval": evaluate_omni_step(docs[int(doc_index)], response, coordinate_type="auto"),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
        anchor_report = {
            "method": "Frozen Qwen with zero residual at the assistant action boundary",
            "num_latent_tokens": args.num_latent_tokens,
            "latent_insertion": "zero residual over the final K assistant action-boundary prompt embeddings",
            "split": split_metadata,
            "evaluation_scope": {
                "split": args.generation_split,
                "test_limit": args.test_limit,
                "evaluated_steps": len(selected_test),
                "is_full_split": args.test_limit is None and len(selected_test) == len(generation_idx),
            },
            args.generation_split: aggregate(anchor_records),
            "records": [
                {
                    "global_id": int(row["doc"]["global_id"]),
                    "prediction": row["prediction"],
                    "target": canonical_action_target(row["doc"]),
                    "eval": row["eval"],
                }
                for row in anchor_records
            ],
        }
        anchor_path = out_dir / (
            "anchor_only_dev_results.json" if args.generation_split == "dev" else "anchor_only_results.json"
        )
        anchor_path.write_text(json.dumps(anchor_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in anchor_report.items() if key != "records"}, ensure_ascii=False, indent=2))
        return

    thinker = model.model.thinker
    thinker.requires_grad_(False)
    thinker.eval()
    if args.gradient_checkpointing:
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model_dim = int(thinker.config.get_text_config().hidden_size)
    num_latent_tokens = args.num_latent_tokens
    adapter_core = FOGLatentTokenAdapter(
        [x.shape[1] for x in train_groups],
        args.token_dim,
        args.z_dim,
        num_latent_tokens,
        model_dim,
        args.dropout,
        args.temperature,
        query_group_index,
        args.vib,
        args.initial_logvar,
        args.fusion_mode,
    ).to(model_device(model))
    if checkpoint_path.exists() and args.eval_only:
        checkpoint = torch.load(checkpoint_path, map_location=model_device(model), weights_only=False)
        adapter_core.load_state_dict(checkpoint["adapter"])
    elif args.eval_only:
        raise FileNotFoundError(f"FOG checkpoint does not exist: {checkpoint_path}")

    adapter: nn.Module
    if world_size > 1:
        adapter = DDP(
            adapter_core,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        # DDP broadcasts rank 0's initialized adapter; rank-specific RNG then
        # prevents identical dropout masks and VIB samples across GPUs.
        set_seed(init_seed + rank)
    else:
        adapter = adapter_core
    metadata = {
        "method": "FOG latent-token conditioning through frozen Qwen decoder and LM head",
        "split": split_metadata,
        "init_seed": init_seed,
        "group_names": group_names,
        "group_dims": [int(x.shape[1]) for x in train_groups],
        "evidence_contract": EVIDENCE_CONTRACT,
        "lexical_features": args.lexical_features,
        "evidence_audit": evidence_audit,
        "forbidden_evidence": [
            "ground-truth action",
            "task category",
            "episode length",
            "future state",
            "media paths or episode IDs",
        ],
        "gate_query_group": args.gate_query_group,
        "fusion_mode": args.fusion_mode,
        "fusion_description": FUSION_DESCRIPTIONS[args.fusion_mode],
        "evidence_weights_interpretable": args.fusion_mode != "naive_concat",
        "gate_type": (
            "frozen-fullmodal-query-conditioned additive evidence posterior"
            if args.fusion_mode == "dynamic"
            else None
        ),
        "model_dim": model_dim,
        "num_latent_tokens": num_latent_tokens,
        "latent_insertion": "zero-initialized residual over the final K assistant action-boundary prompt embeddings",
        "residual_scale": args.residual_scale,
        "token_dim": args.token_dim,
        "z_dim": args.z_dim,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "beta_kl": args.beta_kl,
        "initial_logvar": args.initial_logvar,
        "lambda_weight": args.lambda_weight,
        "vib": args.vib,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "max_grad_norm": args.max_grad_norm,
        "backbone_frozen": True,
        "decoder_frozen": True,
        "lm_head_frozen": True,
        "legacy_classification_head_used": False,
        "target_serialization": "canonical normalized action JSON requested by UNIFIED_V2_SYSTEM_PROMPT",
        "checkpoint_selection": "minimum dev per-action teacher-forced objective; test is never consulted",
        "adapter_parameters": int(sum(parameter.numel() for parameter in adapter_core.parameters())),
        "thinker_parameters": int(sum(parameter.numel() for parameter in thinker.parameters())),
        "trainable_thinker_parameters": int(sum(parameter.numel() for parameter in thinker.parameters() if parameter.requires_grad)),
        "distributed_training": world_size > 1,
        "world_size": world_size,
        "effective_global_grad_accum": args.grad_accum,
        "gradient_checkpointing": args.gradient_checkpointing,
        "activation_offload": args.activation_offload,
        "activation_offload_threshold": args.activation_offload_threshold,
    }

    if not args.eval_only:
        if args.grad_accum < world_size or args.grad_accum % world_size != 0:
            raise ValueError(
                f"--grad-accum ({args.grad_accum}) must be divisible by WORLD_SIZE ({world_size})."
            )
        local_grad_accum = args.grad_accum // world_size
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        selected_train = list(train_idx[: args.train_limit] if args.train_limit is not None else train_idx)
        selected_dev = np.asarray(dev_idx[: args.dev_limit] if args.dev_limit is not None else dev_idx)
        dev_positions = shard_positions(len(selected_dev), rank, world_size)
        local_dev_indices = selected_dev[dev_positions]
        local_dev_groups = select_group_rows(dev_groups, dev_positions)
        best_objective = float("inf")
        best_state = None
        patience = args.patience
        training_log = []
        generator = np.random.default_rng(init_seed)
        for epoch in range(args.epochs):
            order = generator.permutation(len(selected_train)).tolist()
            padding = (-len(order)) % world_size
            if padding:
                order.extend(order[:padding])
            local_order = order[rank::world_size]
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            running = {
                "action": 0.0,
                "token": 0.0,
                "kl": 0.0,
                "weight": 0.0,
                "samples": 0,
                "target_tokens": 0,
                "offloaded_samples": 0,
                "max_sequence_length": 0,
            }
            started = time.time()
            iterator = tqdm(
                local_order,
                desc=f"FOG train epoch {epoch} rank {rank}",
                disable=rank != 0,
            )
            for step, local_index in enumerate(iterator, start=1):
                sync_gradients = step % local_grad_accum == 0 or step == len(local_order)
                group_start = ((step - 1) // local_grad_accum) * local_grad_accum
                accumulation_size = min(local_grad_accum, len(local_order) - group_start)
                sync_context = nullcontext() if sync_gradients or world_size == 1 else adapter.no_sync()
                with sync_context:
                    doc_index = int(selected_train[local_index])
                    groups = evidence_batch(train_groups, local_index, model_device(model))
                    latent, mu, logvar, weights = adapter(groups, sample=True)
                    offload_threshold = args.activation_offload_threshold if args.activation_offload else None
                    lm_loss, target_tokens, sequence_length, used_offload = token_loss(
                        model,
                        docs[doc_index],
                        latent,
                        args.residual_scale,
                        activation_offload_threshold=offload_threshold,
                    )
                    kl = kl_loss(mu, logvar)
                    weight = weight_loss(weights)
                    objective = lm_loss + args.beta_kl * kl + args.lambda_weight * weight
                    (objective / accumulation_size).backward()
                if sync_gradients:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running["token"] += float(lm_loss.detach().cpu()) * target_tokens
                running["action"] += float(lm_loss.detach().cpu())
                running["kl"] += float(kl.detach().cpu())
                running["weight"] += float(weight.detach().cpu())
                running["samples"] += 1
                running["target_tokens"] += target_tokens
                running["offloaded_samples"] += int(used_offload)
                running["max_sequence_length"] = max(running["max_sequence_length"], sequence_length)
                del groups, latent, mu, logvar, weights, lm_loss, kl, weight, objective
                gc.collect()
                torch.cuda.empty_cache()

            dev = evaluate_teacher_forcing(
                unwrap_adapter(adapter),
                model,
                local_dev_groups,
                docs,
                local_dev_indices,
                args.residual_scale,
                args.beta_kl,
                args.lambda_weight,
                None,
                show_progress=rank == 0,
            )
            running_totals = torch.tensor(
                [
                    running["action"],
                    running["token"],
                    running["kl"],
                    running["weight"],
                    running["samples"],
                    running["target_tokens"],
                    running["offloaded_samples"],
                ],
                dtype=torch.float64,
                device=model_device(model),
            )
            if distributed_ready():
                dist.all_reduce(running_totals, op=dist.ReduceOp.SUM)
            (
                train_action,
                train_token,
                train_kl,
                train_weight,
                train_samples,
                train_target_tokens,
                offloaded_samples,
            ) = running_totals.cpu().tolist()
            max_length_tensor = torch.tensor(
                running["max_sequence_length"], dtype=torch.int64, device=model_device(model)
            )
            if distributed_ready():
                dist.all_reduce(max_length_tensor, op=dist.ReduceOp.MAX)
            epoch_record = {
                "epoch": epoch,
                "train_action_nll": train_action / max(int(train_samples), 1),
                "train_token_nll": train_token / max(int(train_target_tokens), 1),
                "train_kl": train_kl / max(int(train_samples), 1),
                "train_weight_entropy": (
                    train_weight / max(int(train_samples), 1)
                    if args.fusion_mode != "naive_concat"
                    else None
                ),
                "offloaded_samples": int(offloaded_samples),
                "max_sequence_length": int(max_length_tensor.item()),
                "distributed_padding_samples": padding,
                "seconds": time.time() - started,
                "dev": dev,
            }
            training_log.append(epoch_record)
            if rank == 0:
                print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
            if dev["objective"] < best_objective - 1e-5:
                best_objective = dev["objective"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in unwrap_adapter(adapter).state_dict().items()
                }
                patience = args.patience
            else:
                patience -= 1
                if patience <= 0:
                    break
        if best_state is None:
            raise RuntimeError("Training completed without a valid adapter state")
        unwrap_adapter(adapter).load_state_dict(best_state)
        if rank == 0:
            save_checkpoint(checkpoint_path, adapter, {**metadata, "training_log": training_log})
        if distributed_ready():
            dist.barrier()

    adapter_core = unwrap_adapter(adapter)
    adapter_core.eval()
    selected_test = np.asarray(
        generation_idx[: args.test_limit] if args.test_limit is not None else generation_idx
    )
    test_positions = shard_positions(len(selected_test), rank, world_size)
    local_test_indices = selected_test[test_positions]
    local_test_groups = select_group_rows(generation_groups, test_positions)
    local_records = []
    iterator = tqdm(
        local_test_indices,
        desc=f"FOG held-out generation rank {rank}",
        disable=rank != 0,
    )
    for local_index, doc_index in enumerate(iterator):
        groups = evidence_batch(local_test_groups, local_index, model_device(model))
        with torch.inference_mode():
            latent, _, _, weights = adapter_core(groups, sample=False)
            response = generate_action(
                model,
                docs[int(doc_index)],
                latent,
                args.residual_scale,
                args.max_new_tokens,
            )
        evaluation = evaluate_omni_step(docs[int(doc_index)], response, coordinate_type="auto")
        local_records.append(
            {
                "doc_index": int(doc_index),
                "prediction": response,
                "eval": evaluation,
                "weights": weights.squeeze(0).detach().cpu().float().tolist(),
            }
        )
        del groups, latent, weights
        gc.collect()
        torch.cuda.empty_cache()

    if distributed_ready():
        gathered_records: List[List[Dict] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_records, local_records)
        flat_records = [row for rank_rows in gathered_records for row in (rank_rows or [])]
    else:
        flat_records = local_records

    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return

    flat_records.sort(key=lambda row: int(docs[row["doc_index"]]["global_id"]))
    records = [
        {
            "doc": docs[row["doc_index"]],
            "prediction": row["prediction"],
            "eval": row["eval"],
            "weights": np.asarray(row["weights"], dtype=np.float32),
        }
        for row in flat_records
    ]
    weights_interpretable = args.fusion_mode != "naive_concat"
    weight_rows = (
        np.asarray([row["weights"] for row in flat_records], dtype=np.float32)
        if weights_interpretable
        else None
    )
    records_by_category = defaultdict(list)
    weights_by_category = defaultdict(list)
    weights_by_step = defaultdict(list)
    for row in records:
        category = str(row["doc"].get("task_category_full", "unknown"))
        step_index = int(row["doc"].get("step_index", 0) or 0)
        records_by_category[category].append(row)
        if weights_interpretable:
            weights_by_category[category].append(row["weights"])
            weights_by_step[step_index].append(row["weights"])

    def named_weight_mean(rows: Sequence[np.ndarray]) -> Dict[str, float]:
        values = np.mean(np.stack(rows), axis=0)
        return {name: float(value) for name, value in zip(group_names, values.tolist())}

    report = {
        **metadata,
        "checkpoint": str(checkpoint_path),
        "evaluation_scope": {
            "split": args.generation_split,
            "test_limit": args.test_limit,
            "evaluated_steps": len(records),
            "is_full_split": args.test_limit is None and len(selected_test) == len(generation_idx),
        },
        args.generation_split: aggregate(records),
        "mean_evidence_weight": (
            {
                name: float(value)
                for name, value in zip(group_names, np.mean(weight_rows, axis=0).tolist())
            }
            if weights_interpretable
            else None
        ),
        "breakdown_by_task_category": {
            category: aggregate(rows) for category, rows in sorted(records_by_category.items())
        },
        "evidence_weight_by_task_category": (
            {
                category: named_weight_mean(rows)
                for category, rows in sorted(weights_by_category.items())
            }
            if weights_interpretable
            else None
        ),
        "evidence_weight_by_step_index": (
            {
                str(step): named_weight_mean(rows)
                for step, rows in sorted(weights_by_step.items())
            }
            if weights_interpretable
            else None
        ),
        "records": [
            {
                "global_id": int(row["doc"]["global_id"]),
                "app": row["doc"].get("app"),
                "episode_id": row["doc"].get("episode_id"),
                "step_index": row["doc"].get("step_index"),
                "task_category_full": row["doc"].get("task_category_full"),
                "prediction": row["prediction"],
                "target": canonical_action_target(row["doc"]),
                "benchmark_target": omni_doc_to_target(row["doc"]),
                "eval": row["eval"],
                "evidence_weight": (
                    {
                        name: float(value)
                        for name, value in zip(group_names, row["weights"].tolist())
                    }
                    if weights_interpretable
                    else None
                ),
            }
            for row in records
        ],
    }
    if args.smoke:
        report_path = out_dir / "smoke_results.json"
    elif args.generation_split == "dev":
        report_path = out_dir / "dev_results.json"
    else:
        report_path = out_dir / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    if distributed_ready():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
