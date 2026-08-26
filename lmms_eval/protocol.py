import base64
from io import BytesIO
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from pydantic import BaseModel
from qwen_vl_utils import fetch_video


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any


class ChatVideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: Any


class ChatAudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    url: Any


ChatContent = Union[ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[ChatContent]


class ChatMessages(BaseModel):
    messages: List[ChatMessage]

    def extract_media(self):
        images = []
        videos = []
        audios = []

        for message in self.messages:
            for content in message.content:
                if content.type == "image":
                    images.append(content.url)
                elif content.type == "video":
                    videos.append(content.url)
                elif content.type == "audio":
                    audios.append(content.url)

        return images, videos, audios

    def to_hf_messages(self, video_kwargs: Dict[str, str] = None):
        if video_kwargs is None:
            video_kwargs = {}
        enforce_images = video_kwargs.pop("enforce_images", False)
        num_frames = video_kwargs.get("nframes", 32)
        hf_messages = []
        for message in self.messages:
            hf_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    hf_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    hf_message["content"].append({"type": "image", "image": content.url})
                elif content.type == "video":
                    # Note this is a hacky way if you want to do video in multi-images way
                    if enforce_images:
                        for f in range(num_frames):
                            hf_message["content"].append({"type": "image"})
                    else:
                        hf_message["content"].append({"type": "video", "video": content.url, **video_kwargs})
                elif content.type == "audio":
                    hf_message["content"].append({"type": "audio", "audio": content.url})
            hf_messages.append(hf_message)
        return hf_messages

    def to_openai_messages(self, video_kwargs: Dict[str, str] = None, skip_audio: bool = False):
        """
        Convert messages to OpenAI API format.
        
        Args:
            video_kwargs: Video processing options, including:
                - nframes: Number of frames to extract from video (default: 8)
            skip_audio: If True, skip audio content (for models that don't support audio)
        
        Returns:
            List of messages in OpenAI API format
        """
        if video_kwargs is None:
            video_kwargs = {"nframes": 8}  # Default frames for GPT-4o compatibility
        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(content.url)}"}})
                elif content.type == "video":
                    video_input = fetch_video({"type": "video", "video": content.url, **video_kwargs})
                    for frame in video_input:
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(image)}"}})
                elif content.type == "audio":
                    # Skip audio if not supported by the model (e.g., GPT-4o)
                    if not skip_audio:
                        openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
                    # Audio interface preserved for future models that support it
            openai_messages.append(openai_message)
        return openai_messages

    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str
