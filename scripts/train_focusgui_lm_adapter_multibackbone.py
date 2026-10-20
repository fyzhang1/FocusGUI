#!/usr/bin/env python
"""Train corrected FocusGUI LM-conditioning adapters on non-Qwen backbones.

Supported frozen local backbones:
  * openbmb/MiniCPM-o-4_5
  * VITA-MLLM/VITA-1.5
  * baichuan-inc/Baichuan-Omni-1d5

This is deliberately not the standalone classification-head path. FocusGUI maps
label-free evidence streams to residual prompt embeddings. The original
backbone decoder and LM head remain frozen and autoregressively generate the
action JSON. Only FocusGUI parameters receive gradients.

The portable evidence contract uses frozen model prefill hidden states plus
label-free screen/audio/video descriptors. It excludes benchmark taxonomy,
ground-truth actions, future states, episode length, and IDs from features.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.tasks.agentcpm_gui.utils import (  # noqa: E402
    eval_logger,
    evaluate_omni_step,
    omni_doc_to_text,
    unified_v2_doc_to_messages,
)
from scripts.focusgui_common import load_docs  # noqa: E402

eval_logger.remove()


BACKBONE_DEFAULTS = {
    "minicpm": {
        "pretrained": "openbmb/MiniCPM-o-4_5",
        "hidden_cache": "logs/focusgui_evidence/minicpm_o_4_5_hidden_p256_strict.npz",
        "out_dir": "logs/focusgui_lm_adapter_minicpm_strict",
    },
    "vita": {
        "pretrained": "VITA-MLLM/VITA-1.5",
        "hidden_cache": "logs/focusgui_evidence/vita_1_5_hidden_p256_strict.npz",
        "out_dir": "logs/focusgui_lm_adapter_vita_strict",
    },
    "baichuan": {
        "pretrained": "baichuan-inc/Baichuan-Omni-1d5",
        "hidden_cache": "logs/focusgui_evidence/baichuan_omni_1d5_hidden_p256_strict.npz",
        "out_dir": "logs/focusgui_lm_adapter_baichuan_strict",
    },
}

GROUP_NAMES = [
    "screen_descriptor",
    "audio_descriptor",
    "video_descriptor",
    "observable_state",
    "model_global_hidden",
    "model_query_hidden",
]

EVIDENCE_CONTRACT = {
    "screen_descriptor": "Label-free current-screen appearance descriptor",
    "audio_descriptor": "Label-free acoustic descriptor from the current audio",
    "video_descriptor": "Label-free temporal descriptor from the current video",
    "observable_state": "Current step index and observed action-history length only",
    "model_global_hidden": "Frozen backbone mean prefill hidden evidence",
    "model_query_hidden": "Frozen backbone final prefill/query hidden evidence",
}

FORBIDDEN_EVIDENCE = [
    "ground-truth action",
    "task category",
    "episode length",
    "future state",
    "media paths as lexical features",
    "app, episode, document, or global identifiers",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_episode(
    docs: Sequence[Dict], test_ratio: float, dev_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray([f"{d.get('app')}-{d.get('episode_id')}" for d in docs])
    unique = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_ratio)))
    n_dev = max(1, int(round(len(unique) * dev_ratio)))
    test_groups = set(unique[:n_test])
    dev_groups = set(unique[n_test : n_test + n_dev])
    train_groups = set(unique[n_test + n_dev :])
    train_idx = np.asarray([i for i, g in enumerate(groups) if g in train_groups], dtype=np.int64)
    dev_idx = np.asarray([i for i, g in enumerate(groups) if g in dev_groups], dtype=np.int64)
    test_idx = np.asarray([i for i, g in enumerate(groups) if g in test_groups], dtype=np.int64)
    return train_idx, dev_idx, test_idx


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


def aggregate(records: Sequence[Dict]) -> Dict:
    episode_results = defaultdict(list)
    for record in records:
        key = f"{record['doc'].get('app')}-{record['doc'].get('episode_id')}"
        episode_results[key].append(record["eval"])
    successes, progresses = [], []
    for rows in episode_results.values():
        rows.sort(key=lambda row: row.get("step_id", 0))
        successes.append(all(row.get("exact_match", False) for row in rows))
        prefix = 0
        for row in rows:
            if not row.get("exact_match", False):
                break
            prefix += 1
        progresses.append(prefix / len(rows) if rows else 0.0)
    if not records:
        return {
            "num_steps": 0,
            "num_episodes": 0,
            "hit_format": 0.0,
            "type_match": 0.0,
            "exact_match": 0.0,
            "success_rate": 0.0,
            "goal_progress": 0.0,
        }
    return {
        "num_steps": len(records),
        "num_episodes": len(episode_results),
        "hit_format": float(np.mean([r["eval"]["hit_format"] for r in records])),
        "type_match": float(np.mean([r["eval"]["type_match"] for r in records])),
        "exact_match": float(np.mean([r["eval"]["exact_match"] for r in records])),
        "success_rate": float(np.mean(successes)),
        "goal_progress": float(np.mean(progresses)),
    }


def load_hidden_evidence(docs: Sequence[Dict], path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    data = np.load(path, allow_pickle=True)
    required = {"keys", "global_features", "query_features"}
    missing_arrays = required.difference(data.files)
    if missing_arrays:
        raise ValueError(f"Hidden cache {path} is missing arrays: {sorted(missing_arrays)}")
    forbidden_arrays = {"labels", "targets", "actions", "task_categories"}.intersection(data.files)
    if forbidden_arrays:
        raise ValueError(f"Hidden cache contains forbidden supervision arrays: {sorted(forbidden_arrays)}")
    key_to_idx = {str(key): i for i, key in enumerate(data["keys"].tolist())}
    positions, missing = [], []
    for doc in docs:
        key = f"{doc['chunk']}:{doc['doc_id']}"
        if key not in key_to_idx:
            missing.append(key)
        else:
            positions.append(key_to_idx[key])
    if missing:
        raise ValueError(f"Hidden cache is missing {len(missing)} documents, e.g. {missing[:5]}")
    order = np.asarray(positions, dtype=np.int64)
    global_hidden = data["global_features"][order].astype(np.float32)
    query_hidden = data["query_features"][order].astype(np.float32)
    audit = {
        "path": str(path),
        "arrays": sorted(data.files),
        "documents": int(len(order)),
        "global_dim": int(global_hidden.shape[1]),
        "query_dim": int(query_hidden.shape[1]),
        "label_free_contract": True,
    }
    return global_hidden, query_hidden, audit


class EvidenceTransform:
    """Train-fitted standardization for portable, inference-time evidence."""

    def __init__(self, names: Sequence[str], means: Sequence[np.ndarray], scales: Sequence[np.ndarray]):
        self.names = list(names)
        self.means = [np.asarray(x, dtype=np.float32) for x in means]
        self.scales = [np.asarray(x, dtype=np.float32) for x in scales]

    @classmethod
    def fit(cls, raw_groups: Sequence[np.ndarray], train_idx: np.ndarray) -> "EvidenceTransform":
        means, scales = [], []
        for group in raw_groups:
            train = group[train_idx].astype(np.float32)
            mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
            scale[scale < 1e-6] = 1.0
            means.append(mean)
            scales.append(scale)
        return cls(GROUP_NAMES, means, scales)

    def transform(self, raw_groups: Sequence[np.ndarray], idx: np.ndarray) -> List[np.ndarray]:
        return [
            ((group[idx].astype(np.float32) - mean) / scale).astype(np.float32)
            for group, mean, scale in zip(raw_groups, self.means, self.scales)
        ]

    def state_dict(self) -> Dict:
        return {
            "names": self.names,
            "means": [value.tolist() for value in self.means],
            "scales": [value.tolist() for value in self.scales],
        }


def build_raw_evidence(
    docs: Sequence[Dict], dense_cache: Path, global_hidden: np.ndarray, query_hidden: np.ndarray
) -> List[np.ndarray]:
    dense = np.load(dense_cache)["dense"].astype(np.float32)
    if dense.shape[0] != len(docs) or dense.shape[1] < 594:
        raise ValueError(f"Unexpected dense cache shape: {dense.shape}")
    # Only label-free media blocks are retained. The old final four metadata
    # fields are intentionally ignored because one of them is episode length.
    screen = dense[:, :299]
    audio = dense[:, 299:333]
    video = dense[:, 333:594]
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
    return [screen, audio, video, observable_state, global_hidden, query_hidden]


class DeterministicFocusGUIAdapter(nn.Module):
    """Query-conditioned evidence posterior and deterministic action bottleneck."""

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
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.model_dim = model_dim
        self.temperature = temperature
        self.query_group_index = query_group_index
        self.group_proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, token_dim), nn.GELU(), nn.LayerNorm(token_dim)) for dim in group_dims]
        )
        self.group_embedding = nn.Parameter(torch.empty(len(group_dims), token_dim))
        self.query_proj = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.LayerNorm(token_dim))
        self.weight_stream = nn.Linear(token_dim, token_dim)
        self.weight_query = nn.Linear(token_dim, token_dim, bias=False)
        self.weight_score = nn.Linear(token_dim, 1, bias=False)
        self.bottleneck = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, z_dim),
        )
        self.latent_projection = nn.Linear(z_dim, num_latent_tokens * model_dim)
        nn.init.normal_(self.group_embedding, mean=0.0, std=0.02)
        # An untrained adapter must exactly preserve the frozen raw model.
        nn.init.zeros_(self.latent_projection.weight)
        nn.init.zeros_(self.latent_projection.bias)

    def forward(self, groups: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        streams = torch.stack([proj(group) for proj, group in zip(self.group_proj, groups)], dim=1)
        query = self.query_proj(streams[:, self.query_group_index])
        score_hidden = torch.tanh(
            self.weight_stream(streams + self.group_embedding.unsqueeze(0))
            + self.weight_query(query).unsqueeze(1)
        )
        logits = self.weight_score(score_hidden).squeeze(-1)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        weighted = torch.sum(streams * weights.unsqueeze(-1), dim=1)
        z = self.bottleneck(weighted)
        latent = self.latent_projection(z).reshape(-1, self.num_latent_tokens, self.model_dim)
        return latent, z, weights


@dataclass
class PromptState:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None


@contextmanager
def supervised_lm_head_window(lm_head: nn.Module, start: int, end: int):
    """Restrict the frozen LM head to causal positions supervising the action."""

    def select_hidden(_module, args):
        if not args or not torch.is_tensor(args[0]) or args[0].ndim != 3:
            return args
        return (args[0][:, start:end], *args[1:])

    handle = lm_head.register_forward_pre_hook(select_hidden)
    try:
        yield
    finally:
        handle.remove()


def add_action_residual(prompt: torch.Tensor, latent: torch.Tensor, scale: float) -> torch.Tensor:
    if prompt.ndim != 3 or prompt.shape[0] != 1 or latent.shape[0] != 1:
        raise ValueError("FocusGUI currently supports one sample per frozen-backbone forward.")
    if prompt.shape[1] < latent.shape[1]:
        raise ValueError("Prompt is shorter than the requested FocusGUI residual span.")
    start = prompt.shape[1] - latent.shape[1]
    residual = (scale * latent).to(device=prompt.device, dtype=prompt.dtype)
    return torch.cat([prompt[:, :start], prompt[:, start:] + residual], dim=1)


def extend_attention(mask: torch.Tensor | None, prompt_len: int, target_len: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        mask = torch.ones((1, prompt_len), dtype=torch.long, device=device)
    else:
        mask = mask.to(device=device)
    return torch.cat([mask, torch.ones((1, target_len), dtype=mask.dtype, device=device)], dim=1)


def extend_positions(
    position_ids: torch.Tensor | None, prompt_len: int, target_len: int, device: torch.device
) -> torch.Tensor | None:
    if position_ids is None:
        return None
    position_ids = position_ids.to(device=device, dtype=torch.long)
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    start = int(position_ids[0, -1].item()) + 1 if prompt_len else 0
    suffix = torch.arange(start, start + target_len, device=device, dtype=torch.long).unsqueeze(0)
    return torch.cat([position_ids, suffix], dim=1)


class FrozenLMBackend:
    name: str
    modality_contract: str

    def __init__(self) -> None:
        self.model_dim = 0
        self.device = torch.device("cuda")
        self.tokenizer = None
        self.lm_head: nn.Module | None = None

    def freeze(self) -> None:
        trainable = 0
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            trainable += int(parameter.requires_grad)
        if trainable:
            raise RuntimeError(f"{self.name} still has trainable backbone parameters.")

    def parameters(self):
        raise NotImplementedError

    def prepare_prompt(self, doc: Dict) -> PromptState:
        raise NotImplementedError

    def embed_target(self, target_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
    ):
        raise NotImplementedError

    def generate_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        max_new_tokens: int,
    ) -> str:
        raise NotImplementedError

    def native_generate(self, doc: Dict, max_new_tokens: int) -> str:
        raise NotImplementedError

    def encode_target(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids.append(int(eos))
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def token_loss(
        self,
        doc: Dict,
        latent: torch.Tensor,
        residual_scale: float,
        activation_offload_threshold: int | None,
    ) -> Tuple[torch.Tensor, int, int, bool]:
        state = self.prepare_prompt(doc)
        prompt = add_action_residual(state.embeddings, latent, residual_scale)
        target_ids = self.encode_target(canonical_action_target(doc))
        with torch.no_grad():
            target_embeddings = self.embed_target(target_ids.unsqueeze(0)).detach()
        full = torch.cat([prompt, target_embeddings], dim=1)
        prompt_len = int(prompt.shape[1])
        target_len = int(target_ids.numel())
        attention_mask = extend_attention(state.attention_mask, prompt_len, target_len, self.device)
        position_ids = extend_positions(state.position_ids, prompt_len, target_len, self.device)
        use_offload = bool(
            torch.is_grad_enabled()
            and activation_offload_threshold
            and full.shape[1] >= activation_offload_threshold
        )
        offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if use_offload else nullcontext()
        with offload:
            with supervised_lm_head_window(self.lm_head, prompt_len - 1, full.shape[1] - 1):
                outputs = self.forward_embeddings(full, attention_mask, position_ids)
        logits = outputs.logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
        return loss, target_len, int(full.shape[1]), use_offload

    @torch.inference_mode()
    def generate_action(
        self, doc: Dict, latent: torch.Tensor, residual_scale: float, max_new_tokens: int
    ) -> str:
        state = self.prepare_prompt(doc)
        prompt = add_action_residual(state.embeddings, latent, residual_scale)
        mask = state.attention_mask
        if mask is None:
            mask = torch.ones((1, prompt.shape[1]), dtype=torch.long, device=self.device)
        return self.generate_embeddings(prompt, mask, state.position_ids, max_new_tokens).strip()


class MiniCPMBackend(FrozenLMBackend):
    name = "MiniCPM-o-4.5"
    modality_contract = "Native MiniCPM local prompt with screenshot, audio, sampled video frames/audio, and text"

    def __init__(self, pretrained: str, max_video_frames: int) -> None:
        super().__init__()
        from PIL import Image
        from lmms_eval.models.simple.minicpm_o import MiniCPMo

        self.Image = Image
        self.wrapper = MiniCPMo(
            backend="local",
            pretrained=pretrained,
            max_video_frames=max_video_frames,
            use_video_audio=True,
        )
        self.core = self.wrapper._model
        self.core.prepare_processor()
        self.tokenizer = self.core.processor.tokenizer
        self.device = self.core.device
        self.model_dim = int(self.core.llm.config.hidden_size)
        self.lm_head = self.core.llm.lm_head
        self.freeze()

    def parameters(self):
        return self.core.parameters()

    def _processor_inputs(self, doc: Dict):
        raw = unified_v2_doc_to_messages(doc, {"media_dir": "data/benchmark"})
        system_prompt, user_text, visuals = self.wrapper._parse_messages(raw)
        msgs = self.wrapper._build_local_msgs(system_prompt, user_text, visuals)
        copy_msgs = deepcopy(msgs)
        images, audios, audio_parts = [], [], []
        use_tts_template = False
        for message_index, message in enumerate(copy_msgs):
            content = message["content"]
            if not isinstance(content, list):
                content = [content]
            pieces = []
            for item in content:
                if isinstance(item, self.Image.Image):
                    images.append(item)
                    pieces.append("<image>./</image>")
                elif isinstance(item, np.ndarray):
                    audios.append(item)
                    audio_parts.append(message_index)
                    pieces.append("<audio>./</audio>")
                    use_tts_template = True
                elif isinstance(item, str):
                    pieces.append(item)
            message["content"] = "".join(pieces)
        prompt = self.tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=True,
            use_tts_template=use_tts_template,
            enable_thinking=False,
        )
        inputs = self.core.processor(
            [prompt],
            [images],
            [audios],
            [audio_parts],
            max_slice_nums=1,
            stream_input=False,
            return_tensors="pt",
            max_length=8192,
        ).to(self.device)
        inputs.pop("image_sizes", None)
        return inputs, system_prompt, user_text, visuals

    def prepare_prompt(self, doc: Dict) -> PromptState:
        inputs, _, _, _ = self._processor_inputs(doc)
        with torch.no_grad():
            embeddings, _ = self.core.get_vllm_embedding(inputs)
            embeddings = self.core.get_omni_embedding(
                inputs,
                input_embeddings=embeddings,
                chunk_length=self.core.config.audio_chunk_length,
            ).detach()
        mask = inputs.get("attention_mask")
        position_ids = inputs.get("position_ids")
        return PromptState(embeddings, mask, position_ids)

    def embed_target(self, target_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.core.llm.model.embed_tokens(target_ids)
        if hasattr(self.core.llm.config, "scale_emb"):
            embeddings = embeddings * self.core.llm.config.scale_emb
        return embeddings

    def forward_embeddings(self, embeddings, attention_mask, position_ids):
        return self.core.llm(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

    def generate_embeddings(self, embeddings, attention_mask, position_ids, max_new_tokens):
        terminators = [self.tokenizer.convert_tokens_to_ids(token) for token in self.core.terminators]
        outputs = self.core.llm.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            pad_token_id=0,
            eos_token_id=terminators,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.02,
            use_cache=True,
            return_dict_in_generate=True,
        )
        return self.core._decode_text(outputs.sequences, self.tokenizer)[0]

    def native_generate(self, doc: Dict, max_new_tokens: int) -> str:
        raw = unified_v2_doc_to_messages(doc, {"media_dir": "data/benchmark"})
        system_prompt, user_text, visuals = self.wrapper._parse_messages(raw)
        return self.wrapper._local_generate(
            system_prompt,
            user_text,
            visuals,
            {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        )


class VITABackend(FrozenLMBackend):
    name = "VITA-1.5"
    modality_contract = "Native VITA local prompt with video frames, audio, current screenshot, and text"

    def __init__(self, pretrained: str, max_video_frames: int) -> None:
        super().__init__()
        vita_code_path = os.environ.get("VITA_CODE_PATH")
        if vita_code_path and vita_code_path not in sys.path:
            sys.path.insert(0, vita_code_path)
        from lmms_eval.models.simple.vita import VITA
        from scripts.extract_vita_hidden_states import _prepare_visual_inputs

        self.prepare_visual_inputs = _prepare_visual_inputs
        self.wrapper = VITA(pretrained=pretrained, max_frames=max_video_frames, batch_size=1)
        self.core = self.wrapper.model
        self.tokenizer = self.wrapper.tokenizer
        self.device = next(self.core.parameters()).device
        self.model_dim = int(self.core.config.hidden_size)
        self.lm_head = self.core.lm_head
        self.freeze()

    def parameters(self):
        return self.core.parameters()

    def _native_inputs(self, doc: Dict):
        return self.prepare_visual_inputs(self.wrapper, doc)

    def prepare_prompt(self, doc: Dict) -> PromptState:
        input_ids, images, audios, _ = self._native_inputs(doc)
        with torch.no_grad():
            _, position_ids, attention_mask, _, embeddings, _ = self.core.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                images,
                audios,
                None,
            )
        if embeddings is None:
            embeddings = self.core.get_model().embed_tokens(input_ids)
        return PromptState(embeddings.detach(), attention_mask, position_ids)

    def embed_target(self, target_ids: torch.Tensor) -> torch.Tensor:
        return self.core.get_model().embed_tokens(target_ids)

    def forward_embeddings(self, embeddings, attention_mask, position_ids):
        return self.core(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

    @staticmethod
    def _clean_control_prefix(text: str) -> str:
        text = text.strip()
        while text.startswith(("☞", "☜")):
            text = text[1:].lstrip()
        return text

    def generate_embeddings(self, embeddings, attention_mask, position_ids, max_new_tokens):
        from transformers import Qwen2ForCausalLM

        outputs = Qwen2ForCausalLM.generate(
            self.core,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        return self._clean_control_prefix(
            self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        )

    def native_generate(self, doc: Dict, max_new_tokens: int) -> str:
        input_ids, images, audios, stopping = self._native_inputs(doc)
        outputs = self.core.generate(
            input_ids,
            images=images,
            audios=audios,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
            shared_v_pid_stride=None,
            return_dict_in_generate=True,
        )
        return self._clean_control_prefix(
            self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        )


class _CapturedMultimodalEmbeddings(RuntimeError):
    pass


class BaichuanBackend(FrozenLMBackend):
    name = "Baichuan-Omni-1d5"
    modality_contract = (
        "Native Baichuan generation sees one primary medium (video > screenshot > audio); "
        "FocusGUI separately receives label-free descriptors for all observed media"
    )

    def __init__(
        self,
        pretrained: str,
        max_video_frames: int,
        image_max_pixels: int,
        video_max_pixels: int,
    ) -> None:
        super().__init__()
        # Baichuan's remote audio module imports DeepSpeed solely to register
        # VQ codebooks as external ZeRO parameters.  ZeRO is not used by this
        # single-process frozen-backbone path, so provide the inference-safe
        # no-op registration API when the optional dependency is absent.
        try:
            import deepspeed  # noqa: F401
        except ModuleNotFoundError:
            import types

            deepspeed_compat = types.ModuleType("deepspeed")
            deepspeed_compat.zero = types.SimpleNamespace(
                register_external_parameter=lambda _module, _parameter: None
            )
            sys.modules["deepspeed"] = deepspeed_compat
        from lmms_eval.models.simple.baichuan_omni import BaichuanOmni

        original_package_version = importlib.metadata.version

        def package_version_with_flash_compat(distribution_name: str) -> str:
            if distribution_name.replace("-", "_").lower() == "flash_attn":
                return "2.5.8"
            return original_package_version(distribution_name)

        importlib.metadata.version = package_version_with_flash_compat
        try:
            self.wrapper = BaichuanOmni(
                backend="local",
                pretrained=pretrained,
                max_video_frames=max_video_frames,
                image_max_pixels=image_max_pixels,
                video_max_pixels=video_max_pixels,
                fail_fast=True,
            )
        finally:
            importlib.metadata.version = original_package_version
        self.core = self.wrapper._model
        self.tokenizer = self.wrapper._tokenizer
        self.device = self.wrapper.device
        self.model_dim = int(self.core.config.hidden_size)
        self.lm_head = self.core.lm_head
        self.freeze()

    def parameters(self):
        return self.core.parameters()

    def _model_inputs(self, doc: Dict):
        raw = unified_v2_doc_to_messages(doc, {"media_dir": "data/benchmark"})
        system_prompt, user_text, visuals = self.wrapper._parse_messages(raw)
        prompt = self.wrapper._local_prompt(system_prompt, user_text, visuals)
        inputs = self.wrapper._processor([prompt])
        unused = {
            "raw_text",
            "index",
            "images_size",
            "videos_size",
            "crop_size",
            "videos_crop_size",
        }
        model_inputs = {
            key: self.wrapper._move_to_device(value)
            for key, value in inputs.items()
            if value is not None and key not in unused
        }
        return model_inputs, system_prompt, user_text, visuals

    def prepare_prompt(self, doc: Dict) -> PromptState:
        model_inputs, _, _, _ = self._model_inputs(doc)
        captured: List[torch.Tensor] = []

        def capture(_module, args):
            if args and torch.is_tensor(args[0]):
                captured.append(args[0].detach())
                raise _CapturedMultimodalEmbeddings()

        handle = self.core.model.layers[0].register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                try:
                    self.core(**model_inputs, use_cache=False, return_dict=True)
                except _CapturedMultimodalEmbeddings:
                    pass
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Failed to capture Baichuan multimodal prompt embeddings.")
        embeddings = captured[0]
        mask = model_inputs.get("attention_mask")
        if mask is None or mask.shape[-1] != embeddings.shape[1]:
            mask = torch.ones((1, embeddings.shape[1]), dtype=torch.long, device=self.device)
        position_ids = model_inputs.get("position_ids")
        return PromptState(embeddings, mask, position_ids)

    def embed_target(self, target_ids: torch.Tensor) -> torch.Tensor:
        return self.core.get_input_embeddings()(target_ids)

    def _single_sequence_lengths(self, length: int) -> torch.Tensor:
        return torch.tensor([0, length], dtype=torch.int32, device=self.device)

    def forward_embeddings(self, embeddings, attention_mask, position_ids):
        return self.core(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seqlens=self._single_sequence_lengths(embeddings.shape[1]),
            use_cache=False,
            return_dict=True,
        )

    def generate_embeddings(self, embeddings, attention_mask, position_ids, max_new_tokens):
        # This checkpoint's custom GenerationMixin emits a tuple of None
        # cache entries even with use_cache=False and then dereferences it on
        # the next step. Reproduce native greedy, no-cache decoding directly
        # through the unchanged Transformer and LM head.
        current = embeddings
        current_mask = attention_mask
        current_positions = position_ids
        generated: List[int] = []
        eos = self.core.generation_config.eos_token_id
        eos_ids = {int(x) for x in eos} if isinstance(eos, (list, tuple)) else {int(eos)}
        for _ in range(max_new_tokens):
            outputs = self.core(
                input_ids=None,
                inputs_embeds=current,
                attention_mask=current_mask,
                position_ids=current_positions,
                seqlens=self._single_sequence_lengths(current.shape[1]),
                use_cache=False,
                return_dict=True,
            )
            next_id = int(outputs.logits[:, -1].argmax(dim=-1).item())
            generated.append(next_id)
            if next_id in eos_ids:
                break
            next_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            current = torch.cat([current, self.embed_target(next_ids)], dim=1)
            current_mask = torch.cat(
                [current_mask, torch.ones((1, 1), dtype=current_mask.dtype, device=self.device)],
                dim=1,
            )
            if current_positions is not None:
                next_position = current_positions[:, -1:] + 1
                current_positions = torch.cat([current_positions, next_position], dim=1)
        return self.wrapper._clean_response(
            self.tokenizer.decode(generated, skip_special_tokens=True)
        )

    def native_generate(self, doc: Dict, max_new_tokens: int) -> str:
        _, system_prompt, user_text, visuals = self._model_inputs(doc)
        return self.wrapper._local_generate(
            system_prompt,
            user_text,
            visuals,
            {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        )


def build_backend(args) -> FrozenLMBackend:
    if args.backbone == "minicpm":
        return MiniCPMBackend(args.pretrained, args.max_video_frames)
    if args.backbone == "vita":
        return VITABackend(args.pretrained, args.max_video_frames)
    if args.backbone == "baichuan":
        return BaichuanBackend(
            args.pretrained,
            args.max_video_frames,
            args.image_max_pixels,
            args.video_max_pixels,
        )
    raise ValueError(args.backbone)


def row_groups(groups: Sequence[np.ndarray], position: int, device: torch.device) -> List[torch.Tensor]:
    return [torch.from_numpy(group[position : position + 1]).to(device=device) for group in groups]


def evaluate_teacher_nll(
    backend: FrozenLMBackend,
    adapter: DeterministicFocusGUIAdapter,
    docs: Sequence[Dict],
    indices: np.ndarray,
    groups: Sequence[np.ndarray],
    residual_scale: float,
    activation_offload_threshold: int | None,
) -> Dict:
    adapter.eval()
    action_losses, token_loss_sum, target_tokens = [], 0.0, 0
    with torch.inference_mode():
        for position, doc_index in enumerate(indices):
            latent, _, _ = adapter(row_groups(groups, position, backend.device))
            loss, tokens, _, _ = backend.token_loss(
                docs[int(doc_index)], latent, residual_scale, activation_offload_threshold
            )
            value = float(loss.cpu())
            action_losses.append(value)
            token_loss_sum += value * tokens
            target_tokens += tokens
    return {
        "action_nll": float(np.mean(action_losses)),
        "token_nll": token_loss_sum / max(target_tokens, 1),
        "samples": int(len(indices)),
        "target_tokens": int(target_tokens),
    }


def generate_split(
    backend: FrozenLMBackend,
    adapter: DeterministicFocusGUIAdapter,
    docs: Sequence[Dict],
    indices: np.ndarray,
    groups: Sequence[np.ndarray],
    residual_scale: float,
    max_new_tokens: int,
) -> Tuple[List[Dict], np.ndarray]:
    adapter.eval()
    records, all_weights = [], []
    with torch.inference_mode():
        for position, doc_index in enumerate(indices):
            latent, _, weights = adapter(row_groups(groups, position, backend.device))
            response = backend.generate_action(
                docs[int(doc_index)], latent, residual_scale, max_new_tokens
            )
            evaluation = evaluate_omni_step(docs[int(doc_index)], response, coordinate_type="auto")
            records.append(
                {
                    "doc": docs[int(doc_index)],
                    "prediction": response,
                    "target": canonical_action_target(docs[int(doc_index)]),
                    "eval": evaluation,
                    "weights": weights[0].float().cpu().numpy(),
                }
            )
            all_weights.append(weights[0].float().cpu().numpy())
    return records, np.stack(all_weights)


def zero_residual_check(
    backend: FrozenLMBackend,
    docs: Sequence[Dict],
    indices: np.ndarray,
    num_latent_tokens: int,
    max_new_tokens: int,
    limit: int,
) -> List[Dict]:
    checks = []
    zero = torch.zeros(
        (1, num_latent_tokens, backend.model_dim),
        dtype=next(backend.parameters()).dtype,
        device=backend.device,
    )
    for doc_index in indices[:limit]:
        conditioned = backend.generate_action(docs[int(doc_index)], zero, 1.0, max_new_tokens)
        native = backend.native_generate(docs[int(doc_index)], max_new_tokens)
        checks.append(
            {
                "global_id": int(docs[int(doc_index)]["global_id"]),
                "native": native,
                "zero_residual": conditioned,
                "exact_text_match": native.strip() == conditioned.strip(),
                "native_eval": evaluate_omni_step(docs[int(doc_index)], native, coordinate_type="auto"),
                "zero_eval": evaluate_omni_step(docs[int(doc_index)], conditioned, coordinate_type="auto"),
            }
        )
    return checks


def checkpoint_metadata(
    args,
    backend: FrozenLMBackend,
    split: Dict,
    evidence_audit: Dict,
    transform: EvidenceTransform,
    adapter: DeterministicFocusGUIAdapter,
    training_log: List[Dict],
) -> Dict:
    return {
        "method": "Corrected deterministic FocusGUI latent residual conditioning through frozen decoder and LM head",
        "backbone": backend.name,
        "backbone_pretrained": args.pretrained,
        "modality_contract": backend.modality_contract,
        "split": split,
        "evidence_contract": EVIDENCE_CONTRACT,
        "forbidden_evidence": FORBIDDEN_EVIDENCE,
        "evidence_audit": evidence_audit,
        "group_names": GROUP_NAMES,
        "group_dims": [int(mean.shape[0]) for mean in transform.means],
        "query_group": "model_query_hidden",
        "model_dim": backend.model_dim,
        "token_dim": args.token_dim,
        "z_dim": args.z_dim,
        "num_latent_tokens": args.num_latent_tokens,
        "residual_scale": args.residual_scale,
        "vib": False,
        "kl_loss": False,
        "legacy_classification_head_used": False,
        "decoder_frozen": True,
        "lm_head_frozen": True,
        "checkpoint_selection": "minimum dev per-action teacher-forced token NLL; test never consulted",
        "adapter_parameters": int(sum(p.numel() for p in adapter.parameters())),
        "trainable_backbone_parameters": int(sum(p.numel() for p in backend.parameters() if p.requires_grad)),
        "training_log": training_log,
        "transform": transform.state_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(BACKBONE_DEFAULTS), required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--hidden-cache", default=None)
    parser.add_argument("--dense-cache", default="logs/focusgui_evidence/observation_descriptors.npz")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--num-latent-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--activation-offload-threshold", type=int, default=5500)
    parser.add_argument("--max-video-frames", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    parser.add_argument("--video-max-pixels", type=int, default=200704)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--zero-residual-check", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()
    defaults = BACKBONE_DEFAULTS[args.backbone]
    args.pretrained = args.pretrained or defaults["pretrained"]
    args.hidden_cache = args.hidden_cache or defaults["hidden_cache"]
    args.out_dir = args.out_dir or defaults["out_dir"]
    return args


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "This corrected multi-backbone runner is single-process by design. "
            "Use CUDA_VISIBLE_DEVICES to select one GPU; do not launch it with torchrun."
        )
    set_seed(args.seed)
    docs = load_docs()
    train_idx, dev_idx, test_idx = split_by_episode(docs, args.test_ratio, args.dev_ratio, args.seed)
    if args.train_limit is not None:
        train_idx = train_idx[: args.train_limit]
    if args.dev_limit is not None:
        dev_idx = dev_idx[: args.dev_limit]
    if args.test_limit is not None:
        test_idx = test_idx[: args.test_limit]

    global_hidden, query_hidden, hidden_audit = load_hidden_evidence(
        docs, REPO_ROOT / args.hidden_cache
    )
    raw_groups = build_raw_evidence(
        docs, REPO_ROOT / args.dense_cache, global_hidden, query_hidden
    )
    transform = EvidenceTransform.fit(raw_groups, train_idx)
    train_groups = transform.transform(raw_groups, train_idx)
    dev_groups = transform.transform(raw_groups, dev_idx)
    test_groups = transform.transform(raw_groups, test_idx)

    backend = build_backend(args)
    query_index = GROUP_NAMES.index("model_query_hidden")
    adapter = DeterministicFocusGUIAdapter(
        [group.shape[1] for group in train_groups],
        args.token_dim,
        args.z_dim,
        args.num_latent_tokens,
        backend.model_dim,
        args.dropout,
        args.temperature,
        query_index,
    ).to(backend.device)
    if any(parameter.requires_grad for parameter in backend.parameters()):
        raise RuntimeError("Frozen backbone audit failed before training.")

    split = {
        "seed": args.seed,
        "train_steps": int(len(train_idx)),
        "dev_steps": int(len(dev_idx)),
        "test_steps": int(len(test_idx)),
        "train_episodes": len({f"{docs[i].get('app')}-{docs[i].get('episode_id')}" for i in train_idx}),
        "dev_episodes": len({f"{docs[i].get('app')}-{docs[i].get('episode_id')}" for i in dev_idx}),
        "test_episodes": len({f"{docs[i].get('app')}-{docs[i].get('episode_id')}" for i in test_idx}),
    }
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best_adapter.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    if args.zero_residual_check:
        checks = zero_residual_check(
            backend,
            docs,
            dev_idx,
            args.num_latent_tokens,
            args.max_new_tokens,
            args.zero_residual_check,
        )
        (out_dir / "zero_residual_check.json").write_text(
            json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "zero_residual_checks": len(checks),
                    "exact_text_matches": sum(x["exact_text_match"] for x in checks),
                    "metric_matches": sum(
                        x["native_eval"]["exact_match"] == x["zero_eval"]["exact_match"]
                        and x["native_eval"]["type_match"] == x["zero_eval"]["type_match"]
                        for x in checks
                    ),
                },
                indent=2,
            ),
            flush=True,
        )

    training_log: List[Dict] = []
    if not args.skip_train:
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        best_objective = float("inf")
        remaining_patience = args.patience
        generator = torch.Generator().manual_seed(args.seed)
        for epoch in range(args.epochs):
            start_time = time.time()
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            permutation = torch.randperm(len(train_idx), generator=generator).tolist()
            action_loss_sum, token_loss_sum, target_token_sum = 0.0, 0.0, 0
            max_sequence_length, offloaded_samples = 0, 0
            for train_step, position in enumerate(permutation):
                doc_index = int(train_idx[position])
                latent, _, _ = adapter(row_groups(train_groups, position, backend.device))
                loss, target_tokens, sequence_length, used_offload = backend.token_loss(
                    docs[doc_index],
                    latent,
                    args.residual_scale,
                    args.activation_offload_threshold,
                )
                (loss / args.grad_accum).backward()
                if (train_step + 1) % args.grad_accum == 0 or train_step + 1 == len(permutation):
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 2.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                value = float(loss.detach().cpu())
                action_loss_sum += value
                token_loss_sum += value * target_tokens
                target_token_sum += target_tokens
                max_sequence_length = max(max_sequence_length, sequence_length)
                offloaded_samples += int(used_offload)
                del latent, loss
                if (train_step + 1) % 50 == 0:
                    print(
                        f"epoch={epoch} step={train_step + 1}/{len(permutation)} "
                        f"action_nll={action_loss_sum / (train_step + 1):.6f}",
                        flush=True,
                    )
            dev = evaluate_teacher_nll(
                backend,
                adapter,
                docs,
                dev_idx,
                dev_groups,
                args.residual_scale,
                args.activation_offload_threshold,
            )
            epoch_log = {
                "epoch": epoch,
                "train_action_nll": action_loss_sum / max(len(train_idx), 1),
                "train_token_nll": token_loss_sum / max(target_token_sum, 1),
                "dev": dev,
                "seconds": time.time() - start_time,
                "offloaded_samples": offloaded_samples,
                "max_sequence_length": max_sequence_length,
            }
            training_log.append(epoch_log)
            print(json.dumps(epoch_log, indent=2), flush=True)
            if dev["action_nll"] + 1e-6 < best_objective:
                best_objective = dev["action_nll"]
                remaining_patience = args.patience
                metadata = checkpoint_metadata(
                    args,
                    backend,
                    split,
                    hidden_audit,
                    transform,
                    adapter,
                    training_log,
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                        "metadata": metadata,
                    },
                    checkpoint_path,
                )
            else:
                remaining_patience -= 1
                if remaining_patience <= 0:
                    break
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.to(backend.device).eval()

    if args.skip_test:
        print(f"Checkpoint ready: {checkpoint_path}")
        return

    records, weights = generate_split(
        backend,
        adapter,
        docs,
        test_idx,
        test_groups,
        args.residual_scale,
        args.max_new_tokens,
    )
    by_category = defaultdict(list)
    by_category_weights = defaultdict(list)
    for record in records:
        category = record["doc"].get("task_category_full", "unknown")
        by_category[category].append(record)
        by_category_weights[category].append(record["weights"])
    report_metadata = {
        key: value
        for key, value in checkpoint["metadata"].items()
        if key != "transform"
    }
    report = {
        **report_metadata,
        "evaluation_scope": {
            "split": "test",
            "evaluated_steps": int(len(test_idx)),
            "is_full_split": args.test_limit is None,
        },
        "test": aggregate(records),
        "breakdown_by_task_category": {
            key: aggregate(value) for key, value in sorted(by_category.items())
        },
        "mean_evidence_weight": {
            name: float(value) for name, value in zip(GROUP_NAMES, weights.mean(axis=0))
        },
        "evidence_weight_by_task_category": {
            key: {
                name: float(value)
                for name, value in zip(GROUP_NAMES, np.mean(np.stack(group), axis=0))
            }
            for key, group in sorted(by_category_weights.items())
        },
        "records": [
            {
                "global_id": int(record["doc"]["global_id"]),
                "app": record["doc"].get("app"),
                "episode_id": record["doc"].get("episode_id"),
                "step_index": record["doc"].get("step_index"),
                "task_category_full": record["doc"].get("task_category_full"),
                "prediction": record["prediction"],
                "target": record["target"],
                "eval": record["eval"],
                "evidence_weight": {
                    name: float(value) for name, value in zip(GROUP_NAMES, record["weights"])
                },
            }
            for record in records
        ],
    }
    result_path = out_dir / "results.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, default=str, indent=2))
    print(f"Wrote {result_path}")
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
