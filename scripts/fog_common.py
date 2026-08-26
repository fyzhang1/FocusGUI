#!/usr/bin/env python
"""Shared data and projection utilities for the final FOG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_docs() -> List[Dict]:
    """Load both OmniGUI chunks with stable local and global identifiers."""
    docs: List[Dict] = []
    for chunk in (0, 1):
        path = REPO_ROOT / f"data/qwen2_5_omni_chunks/chunk{chunk}.json"
        for doc_id, doc in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            row = dict(doc)
            row.update(chunk=chunk, doc_id=doc_id, global_id=len(docs))
            docs.append(row)
    return docs


def split_by_episode(
    docs: Sequence[Dict], test_ratio: float, dev_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic train/dev/test splits with disjoint episodes."""
    groups = np.asarray([f"{doc.get('app')}-{doc.get('episode_id')}" for doc in docs])
    unique = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_ratio)))
    n_dev = max(1, int(round(len(unique) * dev_ratio)))
    test_groups = set(unique[:n_test])
    dev_groups = set(unique[n_test : n_test + n_dev])
    train_groups = set(unique[n_test + n_dev :])
    train_idx = np.asarray(
        [index for index, group in enumerate(groups) if group in train_groups], dtype=np.int64
    )
    dev_idx = np.asarray(
        [index for index, group in enumerate(groups) if group in dev_groups], dtype=np.int64
    )
    test_idx = np.asarray(
        [index for index, group in enumerate(groups) if group in test_groups], dtype=np.int64
    )
    return train_idx, dev_idx, test_idx


def projection_matrix(
    hidden_dim: int, proj_dim: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Build the fixed signed random projection used by frozen evidence extractors."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(
        0, 2, (hidden_dim, proj_dim), generator=generator, dtype=torch.float32
    )
    signs = signs.mul_(2).sub_(1).div_(proj_dim**0.5)
    return signs.to(device=device)
