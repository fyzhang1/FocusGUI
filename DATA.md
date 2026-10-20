# Data Setup

This repository does not contain OmniGUI annotations, screenshots, audio,
video, generated splits, model predictions, hidden-state caches, or model
weights.

Download OmniGUI from its official distribution:

```bash
huggingface-cli download XIAOCHENLIN00zz/OmniGUI \
  --repo-type dataset \
  --local-dir data
```

The FocusGUI pipeline expects at least:

```text
data/
├── merged_sorted.json
└── benchmark/
```

Create the two episode-preserving chunk files used by the training and
analysis scripts:

```bash
python scripts/prepare_omnigui_chunks.py
```

This writes `data/qwen2_5_omni_chunks/chunk0.json` and `chunk1.json`. The
entire `data/` directory is ignored by Git.

Frozen evidence caches and all method outputs are written beneath `logs/`,
which is also ignored by Git. They must be regenerated locally from the
downloaded benchmark and the selected backbone weights.
