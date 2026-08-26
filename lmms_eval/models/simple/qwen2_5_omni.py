import base64
import gc
import json
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import audioread
import audioread.ffdec
import av
import decord
import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import split_audio
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; Please install it via `pip install qwen-omni-utils[decord]`")


@register_model("qwen2_5_omni")
class Qwen2_5_Omni(lmms):
    """
    Qwen2.5-Omni-7B
    "https://huggingface.co/Qwen/Qwen2.5-Omni-7B"

    For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on your running tasks.
    https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
    """

    is_simple: bool = False  # Use ConfigurableMessagesTask so OmniGUI prompts are preserved.

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[bool] = "eager",
        max_num_frames: int = 768,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        image_max_pixels: int = 1003520,  # Cap per-image pixels to bound memory (tall screenshots otherwise ~2.6M px)
        video_max_frames: int = 4,  # Match qwen3_omni (max_frames_num=4) for prompt/input consistency
        video_max_pixels: int = 200704,  # Cap per-frame pixels to bound memory
        use_audio_in_video: bool = False,  # qwen3_omni sends audio as a separate element, not embedded in video
        response_cache_path: Optional[str] = None,
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.force_use_audio_in_video = bool(use_audio_in_video)

        self.image_max_pixels = int(image_max_pixels)
        self.video_max_frames = int(video_max_frames)
        self.video_max_pixels = int(video_max_pixels)
        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        Qwen2_5OmniForConditionalGeneration._tp_plan = [] if Qwen2_5OmniForConditionalGeneration._tp_plan is None else Qwen2_5OmniForConditionalGeneration._tp_plan
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map, attn_implementation=attn_implementation).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(pretrained)
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.response_cache_path = response_cache_path
        self.response_cache = self._load_response_cache(response_cache_path)
        self._model.disable_talker()
        self.system_prompt = system_prompt
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        audioread.ffdec.COMMANDS = (ffmpeg_exe, "ffmpeg", "avconv")

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_Omni")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def resample_audio(self, audio: np.ndarray, current_sample_rate: int):
        """
        Resample the audio to the target sample rate.
        """
        if current_sample_rate != 16000:  # The sample rate for Qwen2.5-Omni is 16kHz
            if isinstance(audio, np.ndarray):
                audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000).astype(np.float32)
        return audio

    def _check_if_video_has_audio(self, video_path):
        clip = VideoFileClip(video_path)
        return clip.audio is not None

    def _split_media_paths(self, media):
        if isinstance(media, str) and "," in media:
            return [self._resolve_media_path(part.strip()) for part in media.split(",") if part.strip()]
        return [self._resolve_media_path(media)]

    def _resolve_media_path(self, media):
        if not isinstance(media, str) or not media:
            return media

        if os.path.exists(media) or os.path.isabs(media):
            return media

        # Some records store comma-separated media paths. After task-level
        # resolution, the first path may include data/benchmark while later
        # paths remain relative.
        benchmark_path = os.path.join("data", "benchmark", media)
        if os.path.exists(benchmark_path):
            return benchmark_path

        return media

    def _response_cache_key(self, task: str, split: str, doc_id: int) -> str:
        return f"{task}:{split}:{doc_id}"

    def _load_response_cache(self, cache_path: Optional[str]) -> Dict[str, str]:
        if not cache_path or not os.path.exists(cache_path):
            return {}

        cached = {}
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = self._response_cache_key(record["task"], record["split"], int(record["doc_id"]))
                    cached[key] = record.get("response", "")
                except Exception as e:
                    eval_logger.warning(f"Ignoring malformed response cache line: {e}")

        eval_logger.info(f"Loaded {len(cached)} cached Qwen2.5-Omni responses from {cache_path}")
        return cached

    def _append_response_cache(self, task: str, split: str, doc_id: int, response: str) -> None:
        if not self.response_cache_path:
            return

        os.makedirs(os.path.dirname(self.response_cache_path) or ".", exist_ok=True)
        key = self._response_cache_key(task, split, doc_id)
        self.response_cache[key] = response
        with open(self.response_cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"task": task, "split": split, "doc_id": doc_id, "response": response}, ensure_ascii=False) + "\n")

    def _convert_messages_for_qwen(self, messages: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Convert lmms-eval message content ({type, url/text}) to Qwen2.5-Omni
        processor content ({type, image/video/audio/text}).
        """
        converted_messages = []
        use_audio_in_video = False

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", [])

            if isinstance(content, str):
                converted_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue

            if not isinstance(content, list):
                continue

            converted_content = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    converted_content.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image":
                    image = item.get("image", item.get("url"))
                    if image:
                        for image_path in self._split_media_paths(image):
                            img_entry = {"type": "image", "image": image_path}
                            if self.image_max_pixels:
                                img_entry["max_pixels"] = self.image_max_pixels
                            converted_content.append(img_entry)
                elif item_type == "video":
                    video = item.get("video", item.get("url"))
                    if video:
                        # Match qwen3_omni: do not embed audio in video by default.
                        # Only enable when explicitly forced via use_audio_in_video.
                        if self.force_use_audio_in_video and isinstance(video, str):
                            use_audio_in_video = use_audio_in_video or self._check_if_video_has_audio(video)
                        vid_entry = {"type": "video", "video": video}
                        if self.video_max_frames:
                            vid_entry["max_frames"] = self.video_max_frames
                        if self.video_max_pixels:
                            vid_entry["max_pixels"] = self.video_max_pixels
                        converted_content.append(vid_entry)
                elif item_type == "audio":
                    audio = item.get("audio", item.get("url"))
                    if isinstance(audio, dict):
                        audio = audio.get("path", audio)
                    if audio:
                        converted_content.append({"type": "audio", "audio": audio})

            if converted_content:
                converted_messages.append({"role": role, "content": converted_content})

        return converted_messages, use_audio_in_video

    def _generate_single_message(self, message: List[Dict], gen_kwargs: Dict, cache_context: str) -> str:
        use_audio_in_video = gen_kwargs.pop("_use_audio_in_video", False)
        try:
            text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(message, use_audio_in_video=use_audio_in_video)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)

            if self.device_map == "auto":
                inputs = inputs.to("cuda").to(self.model.dtype)
            else:
                inputs = inputs.to(self.model.device).to(self.model.dtype)
        except FileNotFoundError as e:
            eval_logger.warning(f"Skipping sample with missing media: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            self.cache_hook.add_partial("generate_until", (cache_context, gen_kwargs), "")
            return ""
        except Exception as e:
            eval_logger.error(f"Error preprocessing multimodal inputs, skipping sample: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            self.cache_hook.add_partial("generate_until", (cache_context, gen_kwargs), "")
            return ""

        gen_kwargs.setdefault("max_new_tokens", 4096)
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)

        pad_token_id = self.tokenizer.pad_token_id
        try:
            cont = self.model.generate(
                **inputs,
                return_audio=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                use_audio_in_video=use_audio_in_video,
                thinker_do_sample=False,
            )
        except torch.cuda.OutOfMemoryError as e:
            eval_logger.warning(f"CUDA OOM, retrying after clearing cache: {e}")
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
            try:
                audios, images, videos = process_mm_info(message, use_audio_in_video=use_audio_in_video)
                inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
                inputs = inputs.to("cuda").to(self.model.dtype) if self.device_map == "auto" else inputs.to(self.model.device).to(self.model.dtype)
                cont = self.model.generate(
                    **inputs,
                    return_audio=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    use_audio_in_video=use_audio_in_video,
                    thinker_do_sample=False,
                )
            except Exception as e2:
                eval_logger.error(f"Error {e2} in generating (after OOM retry)")
                gc.collect()
                torch.cuda.empty_cache()
                self.cache_hook.add_partial("generate_until", (cache_context, gen_kwargs), "")
                return ""
        except Exception as e:
            eval_logger.error(f"Error {e} in generating")
            self.cache_hook.add_partial("generate_until", (cache_context, gen_kwargs), "")
            return ""

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answer = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self.cache_hook.add_partial("generate_until", (cache_context, gen_kwargs), answer)
        del inputs, cont, generated_ids_trimmed
        gc.collect()
        torch.cuda.empty_cache()
        return answer

    def _generate_until_messages(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Qwen2.5-Omni Responding")

        for request in requests:
            ctx, doc_to_messages_func, gen_kwargs, doc_id, task, split = request.args
            cache_key = self._response_cache_key(task, split, int(doc_id))
            if cache_key in self.response_cache:
                res.append(self.response_cache[cache_key])
                pbar.update(1)
                continue

            doc = self.task_dict[task][split][doc_id]
            raw_messages = doc_to_messages_func(doc)
            message, use_audio_in_video = self._convert_messages_for_qwen(raw_messages)
            gen_kwargs = dict(gen_kwargs)
            gen_kwargs["_use_audio_in_video"] = use_audio_in_video
            answer = self._generate_single_message(message, gen_kwargs, ctx)
            self._append_response_cache(task, split, int(doc_id), answer)
            res.append(answer)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        if requests and len(requests[0].args) >= 6 and callable(requests[0].args[1]):
            return self._generate_until_messages(requests)

        res = []
        current_use_audio = False  # Flag to check whether we are using video or not

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on tasks.
            # https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
            message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        current_use_audio = self._check_if_video_has_audio(visual)
                        if self.use_custom_video_loader:
                            visual = read_video_pyav_base64(visual, num_frm=self.max_num_frames, fps=self.fps, img_format="JPEG", max_image_size=self.max_image_size)
                            image_contents = list(map(lambda x: f"data:image/jpeg;base64,{x}", visual))
                            message.append({"role": "user", "content": [{"type": "video", "video": image_contents}, {"type": "text", "text": context}]})
                        else:  # Model video loader
                            message.append({"role": "user", "content": [{"type": "video", "video": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        single_message = {"role": "user", "content": []}
                        for v in visual:
                            single_message["content"].append({"type": "image", "image": v})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    # Fixed code for audio messages
                    elif isinstance(visual, dict):  # Single audio
                        current_use_audio = True
                        audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
                        single_message = {"role": "user", "content": []}
                        for i in range(len(audio_splits)):
                            single_message["content"].append({"type": "audio", "audio": audio_splits[i]})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, dict) for v in visual):  # Multiple audios
                        current_use_audio = True
                        for i, v in enumerate(visual):
                            audio = self.resample_audio(v["array"], v["sampling_rate"])
                            audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
                            single_message = {"role": "user", "content": []}
                            for j in range(len(audio_splits)):
                                single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
                            single_message["content"].append({"type": "text", "text": context})
                            message.append(single_message)

                    else:
                        raise ValueError(f"Unknown visual type: {type(visual)}")

            text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(message, use_audio_in_video=current_use_audio)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=current_use_audio)

            if self.device_map == "auto":
                inputs = inputs.to("cuda").to(self.model.dtype)
            else:
                inputs = inputs.to(self.model.device).to(self.model.dtype)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            try:
                cont = self.model.generate(
                    **inputs,
                    return_audio=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    use_audio_in_video=current_use_audio,
                    thinker_do_sample=False,
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                answer = ""
                res.append(answer)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                continue

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans
            content = []
            for ans, context in zip(answers, contexts):
                res.append(ans)
                content.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
