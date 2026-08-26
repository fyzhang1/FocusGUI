#!/usr/bin/env python
import json
from collections import defaultdict
from pathlib import Path


def latest_sample_file(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("**/*_samples_task_v2_qwen2_5_omni_chunk*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No sample jsonl found under {log_dir}")
    return files[-1]


def load_metric_records(sample_file: Path):
    records = []
    with sample_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(row["exact_match"])
    return records


def aggregate(records):
    total = len(records)
    type_correct = sum(1 for r in records if r.get("type_match", False))
    exact_correct = sum(1 for r in records if r.get("exact_match", False))

    episode_results = defaultdict(list)
    for r in records:
        key = f"{r.get('subset', 'unknown')}-{r.get('episode_id', 'unknown')}"
        episode_results[key].append(r)

    successes = []
    progresses = []
    for rows in episode_results.values():
        rows.sort(key=lambda x: x.get("step_id", 0))
        successes.append(all(r.get("exact_match", False) for r in rows))

        correct_prefix = 0
        for r in rows:
            if r.get("exact_match", False):
                correct_prefix += 1
            else:
                break
        progresses.append(correct_prefix / len(rows) if rows else 0.0)

    return {
        "num_steps": total,
        "num_episodes": len(episode_results),
        "type_match": type_correct / total if total else 0.0,
        "exact_match": exact_correct / total if total else 0.0,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "goal_progress": sum(progresses) / len(progresses) if progresses else 0.0,
    }


def main():
    root = Path("logs")
    all_records = []
    sample_files = []

    for idx in (0, 1):
        sample_file = latest_sample_file(root / f"task_v2_qwen2_5_omni_chunk{idx}")
        sample_files.append(str(sample_file))
        all_records.extend(load_metric_records(sample_file))

    result = aggregate(all_records)
    output_path = root / "task_v2_qwen2_5_omni_2gpu_aggregate.json"
    output_path.write_text(json.dumps({"sample_files": sample_files, "metrics": result}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"sample_files": sample_files, "metrics": result}, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
