import copy
import os
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.env import get_env_value
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    import transformers.utils.hub as _hf_hub

    if not hasattr(_hf_hub, "get_file_from_repo"):
        from transformers.utils.hub import cached_file

        def _get_file_from_repo(path_or_repo, filename, **kwargs):
            return cached_file(path_or_repo, filename, **kwargs)

        _hf_hub.get_file_from_repo = _get_file_from_repo
except Exception:
    pass

try:
    from vita.constants import (
        DEFAULT_AUDIO_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_VIDEO_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
        MAX_IMAGE_LENGTH,
    )
    from vita.conversation import SeparatorStyle, conv_templates
    from vita.model.builder import load_pretrained_model
    from vita.util.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        tokenizer_image_audio_token,
        tokenizer_image_token,
    )
    from vita.util.utils import disable_torch_init
except Exception as e:
    eval_logger.error(f"Error {e} in loading VITA")
    eval_logger.debug("You can set PYTHONPATH to include vita to make the import successful if it is not relative to deps")


class VITA(lmms):
    def __init__(
        self,
        pretrained: str = "VITA-MLLM/VITA-1.5",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_base=None,
        model_type="qwen2p5_instruct",
        frameCat=False,
        device_map="cuda:0",
        conv_template="qwen2p5_instruct",
        use_cache=True,
        max_frames: int = 32,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        pretrained = get_env_value(["VITA_PRETRAINED", "VITA_MODEL_PATH"], default=pretrained)
        model_path = os.path.expanduser(pretrained)
        model_name = get_model_name_from_path(model_path)
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(model_path, model_base, model_name, model_type, device_map=self.device_map)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model_type = model_type

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        self._image_processor = vision_tower.image_processor

        audio_encoder = self.model.get_audio_encoder()
        audio_encoder.to(dtype=torch.float16)
        self._audio_processor = audio_encoder.audio_processor
        self._install_torchaudio_load_compat()
        self._disable_audio_dynamic_chunks(audio_encoder)
        self._config = self._model.config
        self.model.eval()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        self.max_frames = max_frames
        self.frameCat = frameCat
        if self.frameCat:
            from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess

            self.dynamic_preprocess = dynamic_preprocess
        else:
            from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

            self.dynamic_preprocess = dynamic_preprocess
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        raise NotImplementedError("Loglikelihood is not implemented for VITA model")

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    @staticmethod
    def _is_image_path(path):
        return isinstance(path, str) and path.lower().rsplit(".", 1)[-1] in {"jpg", "jpeg", "png", "bmp", "webp"}

    @staticmethod
    def _is_video_path(path):
        return isinstance(path, str) and path.lower().rsplit(".", 1)[-1] in {"mp4", "avi", "mov", "mkv", "webm"}

    @staticmethod
    def _is_audio_path(path):
        return isinstance(path, str) and path.lower().rsplit(".", 1)[-1] in {"wav", "mp3", "m4a", "flac", "ogg"}

    @staticmethod
    def _install_torchaudio_load_compat():
        try:
            import soundfile as sf
            import torchaudio
        except Exception:
            return

        def _load(uri, *args, **kwargs):
            data, sample_rate = sf.read(uri, always_2d=True, dtype="float32")
            return torch.from_numpy(data.T.copy()), sample_rate

        torchaudio.load = _load

    @staticmethod
    def _disable_audio_dynamic_chunks(audio_encoder):
        """Disable training-time random chunking that can fail on very short audio."""
        patched = 0
        for module in audio_encoder.modules():
            if hasattr(module, "transformer_dynamic_chunks"):
                module.transformer_dynamic_chunks = False
                patched += 1
        if patched:
            eval_logger.info(f"Disabled VITA audio dynamic chunks on {patched} module(s)")

    def _prepare_image_tensor(self, image):
        if self.frameCat:
            image, p_num = self.dynamic_preprocess(
                image,
                min_num=2,
                max_num=12,
                image_size=448,
                use_thumbnail=True,
                img_mean=self._image_processor.image_mean,
            )
        else:
            image, p_num = self.dynamic_preprocess(
                image,
                min_num=1,
                max_num=12,
                image_size=448,
                use_thumbnail=True,
            )
        assert len(p_num) == 1
        image_tensor = self.model.process_images(image, self.model.config).to(dtype=self.model.dtype, device="cuda")
        return image_tensor, p_num[0]

    def _prepare_audio(self, audio_path):
        audio_path = os.path.abspath(audio_path)
        audio, audio_for_llm_lens = self._audio_processor.process(audio_path)
        min_audio_frames = 8
        if audio.shape[0] < min_audio_frames:
            pad_shape = (min_audio_frames - audio.shape[0], audio.shape[1])
            if torch.is_tensor(audio):
                pad = torch.zeros(pad_shape, dtype=audio.dtype, device=audio.device)
                audio = torch.cat([audio, pad], dim=0)
            else:
                pad = np.zeros(pad_shape, dtype=audio.dtype)
                audio = np.concatenate([audio, pad], axis=0)
            audio_for_llm_lens = max(int(audio_for_llm_lens), 1)
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        return {
            "audios": audio.half().cuda(),
            "lengths": audio_length.half().cuda(),
            "lengths_for_llm": audio_for_llm_lens.cuda(),
        }

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

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

            prompts_input = contexts[0] if isinstance(contexts, list) or isinstance(contexts, tuple) else contexts

            audios = None
            image_tensors = []
            modality = "lang"
            for visual in visuals:
                if self._is_video_path(visual):
                    visual = os.path.abspath(visual)
                    video_frames, slice_len = self._get_rawvideo_dec(
                        visual,
                        self._image_processor,
                        max_frames=self.max_frames,
                        video_framerate=1,
                        image_aspect_ratio=getattr(self._model.config, "image_aspect_ratio", None),
                    )
                    image_tensors.append(video_frames.half().cuda())
                    # Right now in lmms eval, hasn't got input along with audio, so I keep a dummy case here
                    prompts_input = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + prompts_input
                    modality = "video"
                elif self._is_image_path(visual):
                    image = Image.open(os.path.abspath(visual)).convert("RGB")
                    image_tensor_piece, p_num = self._prepare_image_tensor(image)
                    image_tensors.append(image_tensor_piece)
                    prompts_input = DEFAULT_IMAGE_TOKEN * p_num + "\n" + prompts_input
                    if modality != "video":
                        modality = "image"
                elif self._is_audio_path(visual):
                    audios = self._prepare_audio(visual)
                    prompts_input = prompts_input + DEFAULT_AUDIO_TOKEN
                elif isinstance(visual, Image.Image):
                    image_tensor_piece, p_num = self._prepare_image_tensor(visual.convert("RGB"))
                    image_tensors.append(image_tensor_piece)
                    prompts_input = DEFAULT_IMAGE_TOKEN * p_num + "\n" + prompts_input
                    if modality != "video":
                        modality = "image"
                elif isinstance(visual, dict) and "array" in visual:
                    temp_file_name = f"temp_{self._rank}.wav"
                    sf.write(temp_file_name, visual["array"], visual["sampling_rate"])
                    audios = self._prepare_audio(temp_file_name)
                    prompts_input = prompts_input + DEFAULT_AUDIO_TOKEN
                    os.remove(temp_file_name)
                elif isinstance(visual, dict) and "path" in visual and self._is_audio_path(visual["path"]):
                    audios = self._prepare_audio(visual["path"])
                    prompts_input = prompts_input + DEFAULT_AUDIO_TOKEN

            if image_tensors:
                image_tensor = torch.cat(image_tensors, dim=0)
            else:
                image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=self.model.dtype, device="cuda")

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt(modality)

            if audios:
                input_ids = tokenizer_image_audio_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            else:
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            if not audios:
                audio = torch.zeros(400, 80)
                audio_length = audio.shape[0]
                audio_for_llm_lens = 60
                audio = torch.unsqueeze(audio, dim=0)
                audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
                audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
                audios = dict()
                audios["audios"] = audio.half().cuda()
                audios["lengths"] = audio_length.half().cuda()
                audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    audios=audios,
                    do_sample=False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    shared_v_pid_stride=None,  # 2#16#8#4#1#None,
                )
            output_ids = output_ids.sequences
            input_token_len = input_ids.shape[1]
            if self.model_type == "mixtral-8x7b":
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
                    output_ids = output_ids[:, input_token_len:]
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

            outputs = outputs.strip()
            # Sometimes it contains a ☜, I remove it here
            if outputs.startswith(self.tokenizer.decode(145789)):
                outputs = outputs[len(self.tokenizer.decode(145789)) :]
            if stop_str == "<|im_start|>":
                actual_stop_str = "<|im_end|>"
            else:
                actual_stop_str = stop_str
            if outputs.endswith(actual_stop_str):
                outputs = outputs[: -len(actual_stop_str)]
            outputs = outputs.strip()
            res.append(outputs)
            self.cache_hook.add_partial("generate_until", (prompt, gen_kwargs), outputs)
            pbar.update(1)

        res = re_ords.get_original(res)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")

    def _get_rawvideo_dec(
        self,
        video_path,
        image_processor,
        max_frames=MAX_IMAGE_LENGTH,
        min_frames=4,
        image_resolution=384,
        video_framerate=1,
        s=None,
        e=None,
        image_aspect_ratio="pad",
    ):
        # speed up video decode via decord.

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0.0 else 0.0
            end_time = end_time if end_time >= 0.0 else 0.0
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1

        if os.path.exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # T x 3 x H x W
            sample_fps = int(video_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
            elif len(all_pos) < min_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

            if image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                patch_images = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in patch_images]
                patch_images = [image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0] for i in patch_images]
            else:
                patch_images = [image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0] for i in patch_images]

            patch_images = torch.stack(patch_images)
            slice_len = patch_images.shape[0]

            return patch_images, slice_len
        else:
            print("video path: {} error.".format(video_path))
