#!/usr/bin/env python
"""Extract label-free screen, audio, video, and state descriptors for FOG."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import librosa
import numpy as np

from scripts.fog_common import REPO_ROOT, load_docs


def resolve_media(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if (REPO_ROOT / candidate).exists():
        return REPO_ROOT / candidate
    return REPO_ROOT / "data" / "benchmark" / candidate


def image_features(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros(299, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    pixels = image.reshape(-1, 3)
    color = np.concatenate((pixels.mean(axis=0), pixels.std(axis=0))) / 255.0
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 80, 160)
    height, width = image.shape[:2]
    metadata = np.asarray(
        [
            edges.mean() / 255.0,
            width / max(height, 1),
            height / max(width, 1),
            width / 1000.0,
            height / 2000.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        (gray.reshape(-1), color, metadata, gray.mean(axis=1), gray.mean(axis=0))
    ).astype(np.float32)


def audio_features(path: Path) -> np.ndarray:
    try:
        waveform, sample_rate = librosa.load(
            str(path), sr=16000, mono=True, duration=8.0
        )
    except Exception:
        return np.zeros(34, dtype=np.float32)
    if waveform.size == 0:
        return np.zeros(34, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    rms = librosa.feature.rms(y=waveform)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    return np.concatenate(
        (
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            [zcr.mean(), zcr.std(), rms.mean(), rms.std()],
            [centroid.mean(), centroid.std(), len(waveform) / sample_rate, np.abs(waveform).mean()],
        )
    ).astype(np.float32)


def video_features(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return np.zeros(261, dtype=np.float32)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = []
    for position in ([0, max(frame_count - 1, 0)] if frame_count else [0]):
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = capture.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(
                cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
                / 255.0
            )
    capture.release()
    if not frames:
        return np.zeros(261, dtype=np.float32)
    difference = (
        np.abs(frames[-1] - frames[0])
        if len(frames) > 1
        else np.zeros_like(frames[0])
    )
    metadata = np.asarray(
        [len(frames), frame_count / 100.0, fps / 30.0, difference.mean(), difference.std()],
        dtype=np.float32,
    )
    return np.concatenate((difference.reshape(-1), metadata)).astype(np.float32)


def observation_features(doc: dict) -> np.ndarray:
    state = np.asarray(
        [
            float(doc.get("step_index", doc.get("step_id", 0))) / 30.0,
            float(doc.get("episode_length") or 0) / 30.0,
            len(doc.get("action_history") or []) / 30.0,
            float(doc.get("duration") or 0),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        (
            image_features(resolve_media(doc["image_path"])),
            audio_features(resolve_media(doc["audio_path"])),
            video_features(resolve_media(doc["video_path"])),
            state,
        )
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="logs/fog_evidence/observation_descriptors.npz"
    )
    args = parser.parse_args()

    docs = load_docs()
    rows = []
    for index, doc in enumerate(docs):
        if index % 100 == 0:
            print(f"descriptor {index}/{len(docs)}", flush=True)
        rows.append(observation_features(doc))
    dense = np.vstack(rows).astype(np.float32)
    output = REPO_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, dense=dense)
    print(f"Wrote {dense.shape} label-free descriptors to {output}")


if __name__ == "__main__":
    main()
