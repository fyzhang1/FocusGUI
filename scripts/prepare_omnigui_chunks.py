#!/usr/bin/env python
"""Create deterministic, episode-preserving OmniGUI data chunks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/merged_sorted.json")
    parser.add_argument("--out-dir", default="data/qwen2_5_omni_chunks")
    parser.add_argument("--num-chunks", type=int, default=2)
    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be positive")

    source = Path(args.input)
    rows = json.loads(source.read_text(encoding="utf-8"))
    episodes = defaultdict(list)
    for row in rows:
        key = (str(row.get("app", "unknown")), str(row.get("episode_id", "unknown")))
        episodes[key].append(row)

    chunks = [[] for _ in range(args.num_chunks)]
    sizes = [0 for _ in range(args.num_chunks)]
    for _, episode_rows in sorted(
        episodes.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        chunk_index = min(range(args.num_chunks), key=lambda index: (sizes[index], index))
        chunks[chunk_index].extend(episode_rows)
        sizes[chunk_index] += len(episode_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, chunk in enumerate(chunks):
        output = out_dir / f"chunk{index}.json"
        output.write_text(
            json.dumps(chunk, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{output}: {len(chunk)} steps")
    print(f"total: {sum(sizes)} steps across {len(episodes)} episodes")


if __name__ == "__main__":
    main()
