"""
MiniCPM-o 4.5 API Model Adapter for lmms-eval

This module provides integration with MiniCPM-o 4.5 model via its custom FastAPI server.
The server exposes multiple endpoints:
  - /v1/chat/completions (JSON with base64 media - unified multimodal)
  - /v1/image/chat (multipart form - image only)
  - /v1/video/chat (multipart form - video only)
  - /v1/audio/chat (multipart form - audio only)

We use the unified JSON endpoint (/v1/chat/completions) for evaluation since it
supports sending image + video + audio simultaneously via base64 encoding.

Usage:
    python -m lmms_eval \
        --model minicpm_o \
        --model_args api_url=$MINICPM_O_API_URL \
        --tasks task_v2_minicpm_o \
        --batch_size 1
"""

import base64
import json
import os
import time
import types
from typing import Dict, List, Tuple, Union

import librosa
import requests as http_requests
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.env import get_env_value

try:
    from minicpmo.utils import get_video_frame_audio_segments
except Exception:
    get_video_frame_audio_segments = None

NUM_SECONDS_TO_SLEEP = 5


@register_model("minicpm_o")
class MiniCPMo(lmms):
    """
    MiniCPM-o 4.5 model adapter using custom FastAPI HTTP endpoint.

    Uses the /v1/chat/completions JSON endpoint which accepts:
    - messages: list of {role, content}
    - images: list of base64-encoded images
    - video_base64: base64-encoded video
    - audio_base64: base64-encoded audio
    - max_new_tokens, temperature, do_sample

    Returns JSON: {"id": str, "text": str, "usage": {"inference_time_s": float}}
    """

    is_simple: bool = False  # Use ConfigurableMessagesTask to receive doc_to_messages

    def __init__(
        self,
        api_url: str = None,
        backend: str = "local",
        pretrained: str = None,
        attn_implementation: str = "sdpa",
        torch_dtype: str = "bfloat16",
        max_video_frames: int = 4,
        use_video_audio: bool = True,
        timeout: int = 300,
        max_retries: int = 5,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.backend = backend
        self.api_url = api_url or get_env_value(
            ["MINICPM_O_API_URL", "MINICPM_O_BASE_URL"],
            default=None,
        )
        if self.api_url:
            self.api_url = self.api_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.continual_mode = continual_mode
        self.max_video_frames = int(max_video_frames)
        self.use_video_audio = str(use_video_audio).lower() != "false"

        # Distributed setup
        accelerator = Accelerator()
        self.accelerator = accelerator
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

        self._model = None
        if self.backend == "local":
            self._patch_dynamic_cache_seen_tokens()
            pretrained = pretrained or get_env_value(
                ["MINICPM_O_PRETRAINED", "MINICPM_O_MODEL_PATH"],
                default="openbmb/MiniCPM-o-4_5",
            )
            dtype = torch.bfloat16 if torch_dtype in ("bf16", "bfloat16") else torch.float16
            eval_logger.info(f"Loading local MiniCPM-o from: {pretrained}")
            self._model = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                init_vision=True,
                init_audio=True,
                init_tts=False,
            )
            self._patch_whisper_encoder_layers()
            self._model.eval().to(self.device)
        else:
            if not self.api_url:
                self.api_url = get_env_value(
                    ["MINICPM_O_API_URL", "MINICPM_O_BASE_URL"],
                    required=True,
                    description="MiniCPM-o API URL",
                ).rstrip("/")
            eval_logger.info(f"Initialized MiniCPMo with endpoint: {self.api_url}")

        # Response caching setup
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/minicpm_o_persistent_folder"
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder,
                "minicpm_o_response.json",
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

    def _patch_dynamic_cache_seen_tokens(self) -> None:
        if DynamicCache is None or hasattr(DynamicCache, "seen_tokens"):
            return

        def seen_tokens(cache):
            try:
                return cache.get_seq_length()
            except Exception:
                return 0

        DynamicCache.seen_tokens = property(seen_tokens)
        eval_logger.info("Patched transformers DynamicCache.seen_tokens for MiniCPM-o compatibility.")

    def _patch_whisper_encoder_layers(self) -> None:
        """Compat patch for MiniCPM-o remote code with newer Transformers.

        Some Transformers WhisperAttention implementations return two values
        while MiniCPM-o 4.5 remote code expects three.  Preserve the original
        layer math and fill the absent cache value with None.
        """

        def patched_forward(
            layer,
            hidden_states,
            attention_mask,
            layer_head_mask,
            output_attentions=False,
            past_key_values=None,
            use_cache=False,
        ):
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            attn_out = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_values,
            )
            if len(attn_out) == 2:
                hidden_states, attn_weights = attn_out
                past_key_values = None
            else:
                hidden_states, attn_weights, past_key_values = attn_out
            hidden_states = torch.nn.functional.dropout(hidden_states, p=layer.dropout, training=layer.training)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.activation_fn(layer.fc1(hidden_states))
            hidden_states = torch.nn.functional.dropout(hidden_states, p=layer.activation_dropout, training=layer.training)
            hidden_states = layer.fc2(hidden_states)
            hidden_states = torch.nn.functional.dropout(hidden_states, p=layer.dropout, training=layer.training)
            hidden_states = residual + hidden_states

            if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (attn_weights,)
            if use_cache:
                outputs += (past_key_values,)
            return outputs

        patched = 0
        for module in self._model.modules():
            if module.__class__.__name__ == "MiniCPMWhisperEncoderLayer":
                module.forward = types.MethodType(patched_forward, module)
                patched += 1
        if patched:
            eval_logger.info(f"Patched {patched} MiniCPM Whisper encoder layers for attention output compatibility.")

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------
    def _parse_messages(self, messages: List[Dict]) -> Tuple[str, str, List]:
        """
        Parse messages format to extract system prompt, user text, and visuals.

        Returns:
            Tuple of (system_text, combined_user_text, all_visuals)
            where all_visuals is a list of (type, path) tuples:
              type is one of: "image", "video", "audio"
        """
        system_texts = []
        all_texts = []
        all_visuals = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", [])

            if isinstance(content, str):
                if role == "system":
                    system_texts.append(content)
                else:
                    all_texts.append(f"[{role.capitalize()}]: {content}")
                continue

            if not isinstance(content, list):
                continue

            message_texts = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")

                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        message_texts.append(text)

                elif item_type == "image":
                    url = item.get("url", "")
                    if url and isinstance(url, str):
                        all_visuals.append(("image", url))

                elif item_type == "video":
                    url = item.get("url", "")
                    if url and isinstance(url, str):
                        all_visuals.append(("video", url))

                elif item_type == "audio":
                    url = item.get("url", "")
                    if url:
                        if isinstance(url, str):
                            all_visuals.append(("audio", url))
                        elif isinstance(url, dict) and "path" in url:
                            all_visuals.append(("audio", url["path"]))

            if message_texts:
                if role == "system":
                    system_texts.append(" ".join(message_texts))
                elif role == "user":
                    all_texts.append(f"[User]: {' '.join(message_texts)}")
                elif role == "assistant":
                    all_texts.append(f"[Assistant]: {' '.join(message_texts)}")
                else:
                    all_texts.append(" ".join(message_texts))

        system_text = "\n\n".join(system_texts) if system_texts else ""
        combined_text = "\n\n".join(all_texts)
        return system_text, combined_text, all_visuals

    # ------------------------------------------------------------------
    # File encoding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_file_base64(file_path: str) -> str:
        """Encode a file to base64 string."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _is_image(path: str) -> bool:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        return ext in {"jpg", "jpeg", "png", "gif", "bmp", "webp"}

    @staticmethod
    def _is_video(path: str) -> bool:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        return ext in {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}

    @staticmethod
    def _is_audio(path: str) -> bool:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        return ext in {"wav", "mp3", "m4a", "flac", "ogg"}

    # ------------------------------------------------------------------
    # Build JSON request payload
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        question: str,
        visuals: List[Tuple[str, str]],
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict:
        """
        Build JSON payload for the /v1/chat/completions endpoint.

        The MiniCPM-o server supports:
        - messages: [{"role": "user", "content": "..."}]
        - images: [base64_str, ...]  (list of base64 encoded images)
        - video_base64: base64_str   (single video)
        - audio_base64: base64_str   (single audio)
        """
        payload = {
            "messages": [{"role": "user", "content": question}],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }

        images_b64 = []
        video_b64 = None
        audio_b64 = None

        for vtype, vpath in visuals:
            if not os.path.exists(vpath):
                eval_logger.warning(f"File not found: {vpath}")
                continue

            if vtype == "image":
                images_b64.append(self._encode_file_base64(vpath))
            elif vtype == "video":
                # Only use the last video (server supports one)
                video_b64 = self._encode_file_base64(vpath)
            elif vtype == "audio":
                # Only use the last audio
                audio_b64 = self._encode_file_base64(vpath)

        if images_b64:
            payload["images"] = images_b64
        if video_b64:
            payload["video_base64"] = video_b64
        if audio_b64:
            payload["audio_base64"] = audio_b64

        return payload

    def _sample_evenly(self, items: List, limit: int) -> List:
        if limit <= 0 or len(items) <= limit:
            return items
        if limit == 1:
            return [items[len(items) // 2]]
        idxs = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
        return [items[i] for i in idxs]

    def _build_local_msgs(self, system_prompt: str, user_text: str, visuals: List[Tuple[str, str]]):
        content = []
        for vtype, vpath in visuals:
            if not os.path.exists(vpath):
                eval_logger.warning(f"File not found: {vpath}")
                continue
            try:
                if vtype == "image":
                    content.append(Image.open(vpath).convert("RGB"))
                elif vtype == "audio":
                    wav, _ = librosa.load(vpath, sr=16000, mono=True)
                    content.append(wav)
                elif vtype == "video":
                    if get_video_frame_audio_segments is None:
                        eval_logger.warning("minicpmo-utils is unavailable; skipping video")
                        continue
                    video_items = get_video_frame_audio_segments(
                        vpath,
                        stack_frames=1,
                        use_ffmpeg=True,
                        adjust_audio_length=True,
                        use_audio=self.use_video_audio,
                    )
                    if isinstance(video_items, tuple) and len(video_items) == 3:
                        frames, audio_segments, stacked_frames = video_items
                    elif isinstance(video_items, tuple) and len(video_items) == 2:
                        frames, audio_segments = video_items
                        stacked_frames = None
                    else:
                        frames, audio_segments, stacked_frames = video_items, None, None
                    frame_items = []
                    for i, frame in enumerate(frames):
                        frame_items.append(frame)
                        if stacked_frames is not None and i < len(stacked_frames) and stacked_frames[i] is not None:
                            frame_items.append(stacked_frames[i])
                        if self.use_video_audio and audio_segments is not None and i < len(audio_segments):
                            frame_items.append(audio_segments[i])
                    content.extend(self._sample_evenly(frame_items, self.max_video_frames * 2))
            except Exception as e:
                eval_logger.warning(f"Failed to load {vtype} {vpath}: {e}")
        if user_text:
            content.append(user_text)
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": content})
        return msgs

    def _local_generate(self, system_prompt: str, user_text: str, visuals: List[Tuple[str, str]], gen_kwargs: dict) -> str:
        msgs = self._build_local_msgs(system_prompt, user_text, visuals)
        max_new_tokens = gen_kwargs.get("max_new_tokens", 4096)
        temperature = gen_kwargs.get("temperature", 0)
        with torch.inference_mode():
            response = self._model.chat(
                msgs=msgs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 0.0,
                num_beams=1,
                enable_thinking=False,
                omni_mode=True,
                generate_audio=False,
                max_slice_nums=1,
            )
        if isinstance(response, dict):
            return response.get("text", "") or str(response)
        return str(response)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    def generate_until(self, requests) -> List[str]:
        """Generate responses for a list of requests."""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="MiniCPM-o Responding",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for request in requests:
            args = request.args

            # Parse request arguments
            if len(args) >= 6 and callable(args[1]):
                # Messages mode
                ctx, doc_to_messages_func, gen_kwargs, doc_id, task, split = args
                messages = doc_to_messages_func(self.task_dict[task][split][doc_id])
                system_prompt, user_text, visuals = self._parse_messages(messages)
            else:
                eval_logger.error(f"Unexpected args format: {args}")
                res.append("")
                pbar.update(1)
                continue

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            # Combine system prompt and user text as the question
            question_parts = []
            if system_prompt:
                question_parts.append(
                    f"[System Instructions]\n{system_prompt}\n[End System Instructions]"
                )
            if user_text:
                question_parts.append(user_text)
            question = "\n\n".join(question_parts)

            max_new_tokens = gen_kwargs.get("max_new_tokens", 4096)
            temperature = gen_kwargs.get("temperature", 0)

            if self.backend == "local":
                try:
                    response_text = self._local_generate(system_prompt, user_text, visuals, gen_kwargs)
                except Exception as e:
                    eval_logger.exception(f"MiniCPM-o local generation failed: {e}")
                    response_text = ""
            else:
                payload = self._build_payload(question, visuals, max_new_tokens, temperature)
                response_text = ""
                for attempt in range(self.max_retries):
                    try:
                        resp = http_requests.post(
                            f"{self.api_url}/v1/chat/completions",
                            json=payload,
                            timeout=self.timeout,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        response_text = result.get("text", "")
                        if not response_text and "choices" in result:
                            choices = result["choices"]
                            if choices and len(choices) > 0:
                                response_text = choices[0].get("message", {}).get("content", "")
                        if response_text:
                            break
                        if attempt < self.max_retries - 1:
                            time.sleep(NUM_SECONDS_TO_SLEEP)
                    except Exception as e:
                        eval_logger.info(f"Attempt {attempt+1}/{self.max_retries} failed: {str(e)}")
                        if attempt < self.max_retries - 1:
                            time.sleep(NUM_SECONDS_TO_SLEEP)
                        else:
                            eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {str(e)}")
                            response_text = ""

            res.append(response_text)
            pbar.update(1)

            # Cache response
            if self.continual_mode:
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not supported for MiniCPM-o")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("MiniCPM-o API does not support loglikelihood")
