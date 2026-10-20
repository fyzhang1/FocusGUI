#!/usr/bin/env python
"""Extract frozen VITA multimodal hidden states for the FocusGUI pipeline.

This extractor follows the same local VITA preprocessing/generation path used
by the OmniGUI raw evaluation: ``omni_doc_to_text`` provides the prompt and
``omni_doc_to_visual`` provides video/audio/image inputs.  It captures the LLM
prefill hidden state with a forward hook and never reads labels or predictions
to construct features.
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
import soundfile as sf
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if os.environ.get("VITA_CODE_PATH"):
    sys.path.insert(0, os.environ["VITA_CODE_PATH"])

from lmms_eval.models.simple.vita import VITA  # noqa: E402
from lmms_eval.tasks.agentcpm_gui.utils import omni_doc_to_text, omni_doc_to_visual  # noqa: E402
from scripts.focusgui_common import load_docs, projection_matrix  # noqa: E402

from vita.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from vita.conversation import SeparatorStyle, conv_templates  # noqa: E402
from vita.util.mm_utils import KeywordsStoppingCriteria, tokenizer_image_audio_token, tokenizer_image_token  # noqa: E402


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


def _find_llm_norm(model: VITA) -> torch.nn.Module:
    candidates = [
        "model.model.norm",
        "model.norm",
        "language_model.model.norm",
        "llm.model.norm",
    ]
    root = model.model
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
        raise RuntimeError("Could not locate VITA LLM final norm module for hidden hook.")
    return matches[-1][1]


def _prepare_visual_inputs(model: VITA, doc: Dict) -> tuple[str, torch.Tensor, Dict]:
    prompt = omni_doc_to_text(doc, {"media_dir": "data/benchmark"})
    visuals = omni_doc_to_visual(doc, {"media_dir": "data/benchmark"})
    visuals = model.flatten([visuals])

    audios = None
    image_tensors = []
    modality = "lang"
    for visual in visuals:
        if model._is_video_path(visual):
            video_frames, slice_len = model._get_rawvideo_dec(
                os.path.abspath(visual),
                model._image_processor,
                max_frames=model.max_frames,
                video_framerate=1,
                image_aspect_ratio=getattr(model._model.config, "image_aspect_ratio", None),
            )
            image_tensors.append(video_frames.half().cuda())
            prompt = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + prompt
            modality = "video"
        elif model._is_image_path(visual):
            image = Image.open(os.path.abspath(visual)).convert("RGB")
            image_tensor_piece, p_num = model._prepare_image_tensor(image)
            image_tensors.append(image_tensor_piece)
            prompt = DEFAULT_IMAGE_TOKEN * p_num + "\n" + prompt
            if modality != "video":
                modality = "image"
        elif model._is_audio_path(visual):
            audios = model._prepare_audio(visual)
            prompt = prompt + DEFAULT_AUDIO_TOKEN
        elif isinstance(visual, Image.Image):
            image_tensor_piece, p_num = model._prepare_image_tensor(visual.convert("RGB"))
            image_tensors.append(image_tensor_piece)
            prompt = DEFAULT_IMAGE_TOKEN * p_num + "\n" + prompt
            if modality != "video":
                modality = "image"
        elif isinstance(visual, dict) and "array" in visual:
            temp_file_name = f"temp_vita_hidden_{model.rank}.wav"
            sf.write(temp_file_name, visual["array"], visual["sampling_rate"])
            audios = model._prepare_audio(temp_file_name)
            prompt = prompt + DEFAULT_AUDIO_TOKEN
            os.remove(temp_file_name)
        elif isinstance(visual, dict) and "path" in visual and model._is_audio_path(visual["path"]):
            audios = model._prepare_audio(visual["path"])
            prompt = prompt + DEFAULT_AUDIO_TOKEN

    has_real_audio = audios is not None

    if image_tensors:
        image_tensor = torch.cat(image_tensors, dim=0)
    else:
        image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.model.dtype, device="cuda")

    if not audios:
        audio = torch.zeros(400, 80)
        audio_length = torch.unsqueeze(torch.tensor(audio.shape[0]), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(60), dim=0)
        audio = torch.unsqueeze(audio, dim=0)
        audios = {
            "audios": audio.half().cuda(),
            "lengths": audio_length.half().cuda(),
            "lengths_for_llm": audio_for_llm_lens.cuda(),
        }

    conv = conv_templates[model.conv_template].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt(modality)
    if has_real_audio:
        input_ids = tokenizer_image_audio_token(full_prompt, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    else:
        input_ids = tokenizer_image_token(full_prompt, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], model.tokenizer, input_ids)
    return input_ids, image_tensor, audios, stopping_criteria


def extract_one(model: VITA, doc: Dict, proj_dim: int, projection_seed: int, max_new_tokens: int) -> Dict:
    input_ids, image_tensor, audios, stopping_criteria = _prepare_visual_inputs(model, doc)
    captured: List[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = _hidden_from_output(output)
        if hidden is not None and hidden.ndim == 3 and hidden.shape[1] > 1:
            if not captured or hidden.shape[1] > captured[0].shape[1]:
                captured[:] = [hidden.detach().float().cpu()]

    handle = _find_llm_norm(model).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            _ = model.model.generate(
                input_ids,
                images=image_tensor,
                audios=audios,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                output_scores=False,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                shared_v_pid_stride=None,
            )
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("VITA prefill hidden state was not captured.")
    hidden = captured[0].squeeze(0)
    device = next(model.model.parameters()).device
    mean_hidden = hidden.mean(dim=0).to(device)
    query_hidden = hidden[-1].to(device)
    proj = projection_matrix(int(hidden.shape[-1]), proj_dim, projection_seed, device)
    global_feature = mean_hidden @ proj
    query_feature = query_hidden @ proj
    result = {
        "global_feature": global_feature.detach().cpu().numpy().astype(np.float16),
        "query_feature": query_feature.detach().cpu().numpy().astype(np.float16),
    }
    del input_ids, image_tensor, audios, hidden, mean_hidden, query_hidden, proj, global_feature, query_feature
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="logs/focusgui_evidence/vita_1_5_hidden_p256_strict.npz")
    parser.add_argument("--pretrained", default=os.environ.get("VITA_PRETRAINED", "VITA-MLLM/VITA-1.5"))
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=20260705)
    parser.add_argument("--max-frames", type=int, default=32)
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
    model = VITA(pretrained=args.pretrained, max_frames=args.max_frames, batch_size=1)
    failures = []
    completed_since_save = 0
    for doc in tqdm(selected, desc="Extract VITA hidden states"):
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
