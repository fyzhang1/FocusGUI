# Corrected FOG for the Other Omni-Modal Backbones

## Scope

The corrected implementation supports the three additional local backbones:

| Backbone | Native decoder retained | Native LM head retained | Trainable backbone parameters | Trainable FOG parameters |
|---|---:|---:|---:|---:|
| MiniCPM-o-4.5 | yes | yes | 0 | 1,284,416 |
| VITA-1.5 | yes | yes | 0 | 1,151,296 |
| Baichuan-Omni-1d5 | yes | yes | 0 | 1,151,296 |

This is not the legacy action-classification head. The action is generated as a token sequence by each model's original frozen Transformer decoder and LM head.

## Shared FOG Path

All backbones use the same evidence contract and adapter:

1. Load label-free frozen prefill evidence (`model_global_hidden` and `model_query_hidden`).
2. Build explicit label-free screen, audio, video, and observable interaction-state descriptors.
3. Fit evidence standardization on the train split only.
4. Project each stream to a shared 128-dimensional space.
5. Use the current full-modal query hidden state to predict a step-dependent softmax weight over the six streams.
6. Form a weighted representation and pass it through a deterministic 64-dimensional action bottleneck.
7. Project the bottleneck to four model-width residual embeddings at the assistant/action boundary.
8. Train only the FOG projectors, weighting network, bottleneck, and residual projection with teacher-forced action-token cross entropy.
9. Select the checkpoint using dev per-action token NLL and generate the held-out test actions autoregressively.

The corrected deterministic version does not use VIB or KL loss. Ground-truth actions supervise only the generated action suffix during training. Task category, episode length, future state, media paths, app/episode/document IDs, and test labels are not input evidence.

## Model-Specific Paths

- **MiniCPM-o-4.5:** uses the native MiniCPM processor, multimodal embedding functions, decoder, LM head, and generation terminators.
- **VITA-1.5:** uses VITA's native visual/audio preparation and multimodal prompt construction, followed by its original Qwen decoder and LM head. Known VITA generation-control prefixes are removed consistently from both native and conditioned outputs.
- **Baichuan-Omni-1d5:** captures the native multimodal prompt embedding before the first decoder block, then performs greedy no-cache decoding through the unchanged Baichuan Transformer and LM head. This is required because the checkpoint's custom cached generation path dereferences invalid cache entries.

Baichuan has an important native limitation: its processor exposes one primary medium to the decoder (`video > screenshot > audio`). FOG still receives label-free descriptors for all observed media, but this does not make the frozen Baichuan decoder natively omni-input. This limitation must be disclosed in any comparison.

## Full Runs

Each launcher is a single-process, one-GPU run. Two different backbones can run concurrently on the two GPUs.

```bash
cd /home/ecs-user/OmniGUI

GPU=0 scripts/train_ares_lm_adapter_minicpm.sh --zero-residual-check 3
GPU=1 scripts/train_ares_lm_adapter_vita.sh --zero-residual-check 3
GPU=0 scripts/train_ares_lm_adapter_baichuan.sh --zero-residual-check 3
```

Default full-run settings are three epochs, patience two, gradient accumulation eight, seed `20260626`, and the strict episode-disjoint `1807/257/508` train/dev/test step split. Outputs are written under:

```text
logs/ares_lm_adapter_minicpm_strict/
logs/ares_lm_adapter_vita_strict/
logs/ares_lm_adapter_baichuan_strict/
```

## Executed Structural Tests

Real local-model smoke tests were executed for all three backbones. For each model, the following checks passed:

- zero FOG residual reproduces the native structured action text and evaluation outcome;
- the teacher-forced action-token loss is finite and backpropagates into FOG;
- the original decoder and LM head remain frozen;
- the backbone has zero trainable parameters;
- test generation goes through the original decoder and LM head;
- no legacy action-classification head is used.

The smoke runs use one train, one dev, and one test step only. Their task scores are not benchmark results and must not be added to the main-results table. Full benchmark scores require completion of the commands above.
