"""
Baichuan-Omni-1.5 API Model Adapter for lmms-eval

This module provides integration with Baichuan-Omni-1.5 model via its custom FastAPI server.
The server exposes a /chat endpoint that accepts multipart form data
with instruction text, input_type, and an optional file upload.

Note: Baichuan server only accepts ONE file per request.
When multiple media types are present, we prioritize: video > image > audio,
since video contains the most information (visual + temporal).

Usage:
    python -m lmms_eval \
        --model baichuan_omni \
        --model_args api_url=$BAICHUAN_OMNI_API_URL \
        --tasks task_v2_baichuan_omni \
        --batch_size 1
"""

import io
import json
import os
import sys
import time
import types
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import requests as http_requests
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.env import get_env_value

NUM_SECONDS_TO_SLEEP = 5


@register_model("baichuan_omni")
class BaichuanOmni(lmms):
    """
    Baichuan-Omni-1.5 model adapter using custom FastAPI HTTP endpoint.

    The server accepts multipart form data:
    - instruction: str (text prompt)
    - input_type: str (one of: "text", "audio", "image", "video")
    - file: UploadFile (optional, single file)

    Returns JSON: {"response": str, "audio_base64": str|null}
    Response may end with "<|endoftext|>" which needs to be stripped.

    Important: Only ONE file is supported per request. When the message
    contains multiple media types, priority is: video > image > audio.
    """

    is_simple: bool = False  # Use ConfigurableMessagesTask to receive doc_to_messages

    def __init__(
        self,
        api_url: str = None,
        backend: str = "local",
        pretrained: str = None,
        torch_dtype: str = "bfloat16",
        timeout: int = 180,
        max_retries: int = 5,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        fail_fast: Union[bool, str] = True,
        max_video_frames: Union[int, str, None] = None,
        image_max_pixels: Union[int, str, None] = None,
        video_max_pixels: Union[int, str, None] = None,
        local_max_new_tokens: Union[int, str, None] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.backend = backend
        self.api_url = api_url or get_env_value(
            ["BAICHUAN_OMNI_API_URL", "BAICHUAN_OMNI_BASE_URL"],
            default=None,
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.continual_mode = continual_mode
        self.fail_fast = str(fail_fast).lower() not in ("0", "false", "no")
        self.max_video_frames = int(max_video_frames) if max_video_frames not in (None, "", "none") else None
        self.image_max_pixels = int(image_max_pixels) if image_max_pixels not in (None, "", "none") else None
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels not in (None, "", "none") else None
        self.local_max_new_tokens = int(local_max_new_tokens) if local_max_new_tokens not in (None, "", "none") else None

        # Distributed setup
        accelerator = Accelerator()
        self.accelerator = accelerator
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

        self._tokenizer = None
        self._model = None
        self._processor = None
        if self.backend == "local":
            compat_path = os.environ.get("BAICHUAN_COMPAT_PATH", "")
            if os.path.isdir(compat_path) and compat_path not in sys.path:
                sys.path.insert(0, compat_path)
            pretrained = pretrained or get_env_value(
                ["BAICHUAN_OMNI_PRETRAINED", "BAICHUAN_OMNI_MODEL_PATH"],
                default="baichuan-inc/Baichuan-Omni-1d5",
            )
            dtype = torch.bfloat16 if torch_dtype in ("bf16", "bfloat16") else torch.float16
            eval_logger.info(f"Loading local Baichuan-Omni from: {pretrained}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            self._model.eval().to(self.device)
            self._patch_visual_encoder_position_embeddings()
            self._install_torchaudio_info_compat()
            self._processor = self._model.bind_processor(self._tokenizer, training=False)
            self._apply_local_resource_limits()
            self._ensure_chat_token_ids()
        else:
            if not self.api_url:
                self.api_url = get_env_value(
                    ["BAICHUAN_OMNI_API_URL", "BAICHUAN_OMNI_BASE_URL"],
                    required=True,
                    description="Baichuan-Omni API URL",
                )
            eval_logger.info(f"Initialized BaichuanOmni with endpoint: {self.api_url}")

        # Response caching setup
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/baichuan_omni_persistent_folder"
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder,
                "baichuan_omni_response.json",
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

    # ------------------------------------------------------------------
    # Message parsing (same logic as gemini adapter)
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
    # Clean response
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip Baichuan-specific artifacts from response."""
        if not text:
            return ""
        # Strip trailing <|endoftext|>
        if text.endswith("<|endoftext|>"):
            text = text[: -len("<|endoftext|>")].strip()
        return text.strip()

    # ------------------------------------------------------------------
    # Determine mime type from extension
    # ------------------------------------------------------------------
    @staticmethod
    def _get_mime_type(path: str) -> str:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "webp": "image/webp",
            "mp4": "video/mp4",
            "avi": "video/avi",
            "mov": "video/quicktime",
            "mkv": "video/x-matroska",
            "webm": "video/webm",
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "m4a": "audio/mp4",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
        }
        return mime_map.get(ext, "application/octet-stream")

    # ------------------------------------------------------------------
    # Select primary media (Baichuan only supports ONE file per request)
    # ------------------------------------------------------------------
    @staticmethod
    def _select_primary_media(visuals: List[Tuple[str, str]]) -> Tuple[str, str]:
        """
        Select the most informative single media file.

        Priority: video > image > audio
        For each type, take the LAST occurrence (most recent/primary).

        Returns:
            Tuple of (input_type, file_path) or ("text", "") if no media
        """
        video_path = None
        image_path = None
        audio_path = None

        for vtype, vpath in visuals:
            if vtype == "video":
                video_path = vpath
            elif vtype == "image":
                image_path = vpath
            elif vtype == "audio":
                audio_path = vpath

        # Priority: video > image > audio
        if video_path and os.path.exists(video_path):
            return "video", video_path
        if image_path and os.path.exists(image_path):
            return "image", image_path
        if audio_path and os.path.exists(audio_path):
            return "audio", audio_path

        return "text", ""

    # ------------------------------------------------------------------
    # Build multipart request
    # ------------------------------------------------------------------
    def _build_request(
        self, instruction: str, visuals: List[Tuple[str, str]]
    ) -> Tuple[dict, dict, str]:
        """
        Build multipart form data for the Baichuan /v1/chat/multimodal endpoint.

        Args:
            instruction: The text prompt (system + user combined)
            visuals: List of (type, path) tuples

        Returns:
            Tuple of (data dict, files dict, selected_input_type)
        """
        input_type, file_path = self._select_primary_media(visuals)

        data = {
            "text_query": instruction,
            "generate_audio": "false",
        }
        files = {}

        if file_path:
            mime = self._get_mime_type(file_path)
            # Map input_type to the correct form field name
            file_field_map = {
                "video": "video_file",
                "image": "image_file",
                "audio": "audio_file",
            }
            field_name = file_field_map.get(input_type, "video_file")
            files[field_name] = (os.path.basename(file_path), open(file_path, "rb"), mime)

        return data, files, input_type

    def _move_to_device(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, list):
            return [self._move_to_device(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_to_device(v) for v in value)
        return value

    @staticmethod
    def _install_torchaudio_info_compat() -> None:
        try:
            import soundfile as sf
            import torchaudio
        except Exception:
            return

        def _load(uri, normalize=True, **kwargs):
            data, sample_rate = sf.read(uri, always_2d=True, dtype="float32")
            waveform = torch.from_numpy(data.T.copy())
            return waveform, sample_rate

        torchaudio.load = _load
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        if hasattr(torchaudio, "info"):
            return

        def _info(uri):
            sound_info = sf.info(uri)

            class AudioMetaData:
                pass

            metadata = AudioMetaData()
            metadata.sample_rate = int(sound_info.samplerate)
            metadata.num_frames = int(sound_info.frames)
            metadata.num_channels = int(sound_info.channels)
            metadata.bits_per_sample = int(getattr(sound_info, "subtype_info", "0 bit").split()[0]) if str(getattr(sound_info, "subtype_info", "")).split()[:1] and str(getattr(sound_info, "subtype_info", "")).split()[0].isdigit() else 0
            metadata.encoding = str(sound_info.subtype)
            return metadata

        torchaudio.info = _info

    def _ensure_chat_token_ids(self) -> None:
        """Baichuan's local generation config may omit chat role token ids."""
        role_tokens = {
            "user_token_id": "<C_Q>",
            "assistant_token_id": "<C_A>",
        }
        for attr, token in role_tokens.items():
            if hasattr(self._model.generation_config, attr):
                continue
            token_id = self._tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == self._tokenizer.unk_token_id:
                raise ValueError(f"Cannot infer Baichuan chat token id for {token}")
            setattr(self._model.generation_config, attr, token_id)
            eval_logger.info(f"Set Baichuan generation_config.{attr}={token_id} from {token}")

    def _apply_local_resource_limits(self) -> None:
        """Apply explicit, fixed local inference budgets for reproducible runs."""
        config = getattr(self._processor, "config", None)
        if config is None:
            return
        if self.max_video_frames is not None and hasattr(config, "video_config"):
            config.video_config.max_frame_num = self.max_video_frames
            eval_logger.info(f"Set Baichuan video_config.max_frame_num={self.max_video_frames}")
        if self.image_max_pixels is not None and hasattr(config, "visual_config"):
            config.visual_config.max_pixels = self.image_max_pixels
            if getattr(self._processor, "visual_processor", None) is not None:
                self._processor.visual_processor.max_pixels = self.image_max_pixels
            eval_logger.info(f"Set Baichuan visual_config.max_pixels={self.image_max_pixels}")
        if self.video_max_pixels is not None and hasattr(config, "video_config"):
            config.video_config.max_pixels = self.video_max_pixels
            if getattr(self._processor, "video_processor", None) is not None:
                self._processor.video_processor.max_pixels = self.video_max_pixels
            eval_logger.info(f"Set Baichuan video_config.max_pixels={self.video_max_pixels}")

    def _patch_visual_encoder_position_embeddings(self) -> None:
        """Adapt Baichuan's old Qwen2-VL visual forward to newer Transformers."""
        visual_model = getattr(getattr(self._model, "model", None), "visual_model", None)
        if visual_model is None or not hasattr(visual_model, "blocks"):
            return

        def _forward(vm, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
            hidden_states = pixel_values.to(vm.get_dtype())
            grid_thw = grid_thw.to(pixel_values.device)
            hidden_states = vm.patch_embed(hidden_states)
            rotary_pos_emb = vm.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            for blk in vm.blocks:
                if vm.gradient_checkpointing and vm.training:
                    hidden_states = vm._gradient_checkpointing_func(
                        blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                    )
                else:
                    hidden_states = blk(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings,
                    )
            return hidden_states

        visual_model.forward = types.MethodType(_forward, visual_model)

    def _local_prompt(self, system_prompt: str, user_text: str, visuals: List[Tuple[str, str]]) -> str:
        media_chunks = []
        input_type, file_path = self._select_primary_media(visuals)
        selected_visuals = [(input_type, file_path)] if file_path else []
        for vtype, vpath in selected_visuals:
            if not os.path.exists(vpath):
                eval_logger.warning(f"File not found: {vpath}")
                continue
            if vtype == "image" and self._processor.image_start_tag is not None:
                media_chunks.append(
                    f"{self._processor.image_start_tag}{json.dumps({'path': vpath}, ensure_ascii=False)}{self._processor.image_end_tag}"
                )
            elif vtype == "video" and self._processor.video_start_tag is not None:
                media_chunks.append(
                    f"{self._processor.video_start_tag}{json.dumps({'local': vpath}, ensure_ascii=False)}{self._processor.video_end_tag}"
                )
            elif vtype == "audio" and self._processor.audio_start_tag is not None:
                media_chunks.append(
                    f"{self._processor.audio_start_tag}{json.dumps({'path': vpath}, ensure_ascii=False)}{self._processor.audio_end_tag}"
                )
        body = "\n".join([*media_chunks, user_text]).strip()
        if system_prompt:
            body = f"{system_prompt}\n{body}"
        user_token = self._tokenizer.convert_ids_to_tokens(self._model.generation_config.user_token_id)
        assistant_token = self._tokenizer.convert_ids_to_tokens(self._model.generation_config.assistant_token_id)
        return f"{user_token}{body}{assistant_token}"

    def _local_generate(self, system_prompt: str, user_text: str, visuals: List[Tuple[str, str]], gen_kwargs: dict) -> str:
        prompt = self._local_prompt(system_prompt, user_text, visuals)
        inputs = self._processor([prompt])
        unused_generate_keys = {
            "raw_text",
            "index",
            "images_size",
            "videos_size",
            "crop_size",
            "videos_crop_size",
        }
        model_inputs = {
            key: self._move_to_device(value)
            for key, value in inputs.items()
            if value is not None and key not in unused_generate_keys
        }
        input_len = int(model_inputs["input_ids"].shape[-1])
        max_new_tokens = gen_kwargs.get("max_new_tokens", 4096)
        if self.local_max_new_tokens is not None:
            max_new_tokens = min(int(max_new_tokens), self.local_max_new_tokens)
        temperature = gen_kwargs.get("temperature", 0)
        with torch.inference_mode():
            outputs = self._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                use_cache=False,
            )
        new_tokens = outputs[0][input_len:]
        return self._clean_response(self._tokenizer.decode(new_tokens, skip_special_tokens=True))

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    def generate_until(self, requests) -> List[str]:
        """Generate responses for a list of requests."""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Baichuan-Omni Responding",
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

            # Combine system prompt and user text as the instruction
            instruction_parts = []
            if system_prompt:
                instruction_parts.append(
                    f"[System Instructions]\n{system_prompt}\n[End System Instructions]"
                )
            if user_text:
                instruction_parts.append(user_text)
            instruction = "\n\n".join(instruction_parts)

            if self.backend == "local":
                try:
                    response_text = self._local_generate(system_prompt, user_text, visuals, gen_kwargs)
                except Exception as e:
                    eval_logger.exception(f"Baichuan-Omni local generation failed: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if self.fail_fast:
                        raise
                    response_text = ""
            else:
                data, files, input_type = self._build_request(instruction, visuals)
                response_text = ""
                for attempt in range(self.max_retries):
                    try:
                        if attempt > 0:
                            for key in files:
                                try:
                                    files[key][1].close()
                                except Exception:
                                    pass
                            data, files, input_type = self._build_request(instruction, visuals)

                        resp = http_requests.post(
                            self.api_url,
                            data=data,
                            files=files if files else None,
                            timeout=self.timeout,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        response_text = result.get("text", "") or result.get("response", "")
                        response_text = self._clean_response(response_text)
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
                    finally:
                        for key in list(files.keys()):
                            try:
                                files[key][1].close()
                            except Exception:
                                pass

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
        raise NotImplementedError("Multi-round generation not supported for Baichuan-Omni")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Baichuan-Omni API does not support loglikelihood")
