import io
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor
from qwen_omni_utils import process_audio_info
from transformers.processing_utils import AudioKwargs, ProcessingKwargs
from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.feature_extraction_utils import BatchFeature

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class ChromaAudioKwargs(AudioKwargs, total=False):
    target_sample_rate: Optional[int]  # 目标采样率


class ChromaProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: ChromaAudioKwargs
    prompt_text: Optional[str]
    prompt_audio: Optional[Union[str, torch.Tensor]]
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "target_sample_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ChromaProcessor(Qwen2_5OmniProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, 'chat_template') or self.chat_template is None:
            self.chat_template = "{% set audio_count = namespace(value=0) %}{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_bos|><|IMAGE|><|vision_eos|>{% elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if add_audio_id %}Audio {{ audio_count.value }}: {% endif %}<|audio_bos|><|AUDIO|><|audio_eos|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_bos|><|VIDEO|><|vision_eos|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    def __call__(self, *args, **kwargs: Unpack[ChromaProcessorKwargs]) -> BatchFeature:
        # extract ChromaProcessor params to apply_chat_template
        prompt_audio = kwargs.pop("prompt_audio", None)
        prompt_text = kwargs.pop("prompt_text", None)
        target_sample_rate = kwargs.pop("target_sample_rate", 24000)

        # thinker processor
        text, audios = self.apply_chat_template(*args, **kwargs)
        thinker_inputs = super().__call__(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        inputs = {f"thinker_{k}": v for k, v in thinker_inputs.items()}

        assert prompt_audio is not None, "prompt_audio can not be empty"
        assert prompt_text is not None, "prompt_text can not be empty"

        prompt_ids = super().__call__(text=prompt_text, return_tensors="pt")

        if isinstance(prompt_audio, str):
            prompt_audio_tensor = self.load_audio(prompt_audio, target_sample_rate)
        elif isinstance(prompt_audio, torch.Tensor):
            prompt_audio_tensor = prompt_audio
        elif isinstance(prompt_audio, np.ndarray):
            prompt_audio_tensor = torch.from_numpy(prompt_audio)
            if prompt_audio_tensor.dim() > 1:
                prompt_audio_tensor = prompt_audio_tensor.squeeze()
        else:
            raise ValueError(
                f"prompt audio must be str, tensor or numpy, but got  {type(prompt_audio)}"
            )

        return BatchFeature(
            data={**inputs, **prompt_ids, "input_values": prompt_audio_tensor},
            tensor_type="pt",
        )

    def load_audio(
        self, audio_path: str, target_sample_rate: int = 24000
    ) -> torch.Tensor:
        """
        process audio file
        Args:
            audio_path:
            target_sample_rate:

        Returns:

        """
        try:
            audio_tensor, sample_rate = torchaudio.load(audio_path)

            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0),
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )

            return audio_tensor
        except Exception as e:
            print(f"load audio : {audio_path}, error{e}")
            raise

    def apply_chat_template(
        self, conversations, chat_template=None, **kwargs
    ) -> Tuple[str, list]:
        """
        apply chat template to conversations
        Args:
            conversations:
            chat_template:
            **kwargs:

        Returns:

        """
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        audios = process_audio_info(conversations, use_audio_in_video=False)
        return (
            super().apply_chat_template(conversations, chat_template, **kwargs),
            audios,
        )


AutoProcessor.register("ChromaProcessor", ChromaProcessor)
