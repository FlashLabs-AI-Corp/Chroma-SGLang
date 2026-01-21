from typing import Any, Dict, List, Tuple, Union, Optional
from dataclasses import fields, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import MODEL_MAPPING, CONFIG_MAPPING, GenerationMixin, PreTrainedModel
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import (
    ModelOutput,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.mimi.modeling_mimi import MimiModel

from .qwen2_5_omni_modeling import Qwen2_5OmniThinkerForConditionalGeneration
from .generation_chroma import ChromaGenerationMixin, sample_topk
from .configuration_chroma import (
    ChromaConfig,
    ChromaDecoderConfig,
    ChromaBackboneConfig,
)

logger = logging.get_logger(__name__)

D_SPECIAL_TOKEN_2_IDS = {
    "<|text_start|>": 151665,
    "<|text_end|>": 151666,
    "<|im_end|>": 151645,
}

PASSTHROUGH_KEYS = [
    "thinker_input_ids",
    "thinker_attention_mask",
    "thinker_cache_position",
    "thinker_past_key_values",
    "thinker_input_features",
    "thinker_feature_attention_mask",
    "thinker_hidden_states",
    "thinker_logits",
    "thinker_flag",
]

ONE_TIME_KEYS = [
    "input_values",  # ref audio waveform
    "thinker_input_features",
    "thinker_feature_attention_mask",
]


@dataclass
class ChromaOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits: torch.FloatTensor | None = None
    past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    cache_position: Optional[int] = None

    # all thinker inputs should be carried through the forward function to the next step
    thinker_loss: Optional[torch.FloatTensor] = None
    thinker_logits: torch.FloatTensor = None
    thinker_past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    thinker_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    thinker_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    thinker_input_ids: Optional[torch.FloatTensor] = None
    thinker_attention_mask: Optional[torch.FloatTensor] = None
    thinker_input_features: Optional[torch.FloatTensor] = None
    thinker_feature_attention_mask: Optional[torch.FloatTensor] = None
    thinker_cache_position: Optional[torch.FloatTensor] = None
    thinker_flag: Optional[bool] = None

    backbone_loss: Optional[torch.FloatTensor] = None
    backbone_logits: torch.FloatTensor = None
    backbone_past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    backbone_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    decoder_loss: Optional[torch.FloatTensor] = None
    decoder_logits: torch.FloatTensor = None
    decoder_past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ChromaLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Identity()


class ChromaPreTrainedModel(PreTrainedModel):
    config_class = ChromaConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else 0.02
        )
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ChromaCodebookHead):
            module.weight.data.normal_(mean=0.0, std=std)


class ChromaAudioEmbedding(nn.Module):
    def __init__(self, audio_num_codebooks, audio_vocab_size, hidden_size):
        super().__init__()
        self.embed_audio_tokens = nn.Embedding(
            num_embeddings=audio_num_codebooks * audio_vocab_size,
            embedding_dim=hidden_size,
        )
        self.audio_vocab_size = audio_vocab_size

    def forward(self, input_ids: torch.Tensor):
        """

        Args:
            input_ids: [B, num_codebooks]

        Returns: [B, num_codebooks, hidden_size]

        """
        num_codebooks = input_ids.shape[-1]
        audio_frames = input_ids + (
            self.audio_vocab_size * torch.arange(num_codebooks, device=input_ids.device)
        )
        embeddings = self.embed_audio_tokens(audio_frames.contiguous().view(-1)).reshape(
            audio_frames.shape + (2048,)
        )
        return embeddings


class ChromaBackboneForCausalLM(ChromaPreTrainedModel):
    config_class = ChromaBackboneConfig
    _supports_flash_attn_2 = True

    def __init__(self, config: ChromaBackboneConfig):
        super().__init__(config)
        self.model = ChromaLlamaModel(LlamaConfig(**config.to_dict()))
        self.codebook0_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

        # During inference, pass parameters of this embedding to decoder
        self.audio_embedding = ChromaAudioEmbedding(
            config.audio_num_codebooks,
            config.vocab_size,
            config.hidden_size,
        )

        self.post_init()

    def emb_audio_frames(
        self, audio_frames: torch.Tensor, add_frame: bool = True
    ) -> torch.Tensor:
        assert (
            audio_frames.dim() > 1
        ), "audio_frames must be a tensor with shape [..., codebook_num]"
        audio_frames = audio_frames.contiguous()
        codebook_num = audio_frames.shape[-1]
        audio_frames = audio_frames.masked_fill(audio_frames == -100, 0)
        audio_embeddings = self.audio_embedding(audio_frames)

        if add_frame:
            # if codebook_num != self.config.audio_num_codebooks:
            #     raise ValueError(f"codebook_num must be equal to audio_num_codebooks {codebook_num} != {self.config.audio_num_codebooks}")
            audio_embeddings = audio_embeddings.sum(dim=-2)
        return audio_embeddings

    def loss_fn(self, logits, labels, ignore_index=-100):
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()

        # Shift so that tokens < n predict n
        labels = F.pad(labels, (0, 1), value=ignore_index)

        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)

        logits = logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.to(logits.device)

        loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index)

        return loss

    def forward(
        self,
        input_embeddings: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        args:
            input_embeddings: [B, seq_len, 2048]  every [2048] is a hidden from qwen
            labels: [B, seq_len]  every element is codebook0 id
        return:
            output: BaseModelOutputWithPast
                loss: [B, seq_len]
                logits: [B, seq_len, 2051]
                hidden_states: [B, seq_len, 2048]
        """
        if input_embeddings is None:
            raise ValueError("input_embeddings is required")

        assert (
            input_embeddings.shape[-1] == self.config.hidden_size
        ), f"input_embeddings must have {self.config.hidden_size} dimensions"

        # Forward
        output: BaseModelOutputWithPast = self.model(
            inputs_embeds=input_embeddings,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = self.codebook0_head(output.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.clone().detach())

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class ChromaCodebookHead(nn.Module):
    def __init__(
        self,
        audio_num_codebooks,
        audio_vocab_size,
        hidden_size,
    ):
        super().__init__()
        self.num_codebooks = audio_num_codebooks
        self.vocab_size = audio_vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.empty(self.num_codebooks, self.hidden_size, self.vocab_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: [B, num_codebook, input_dim]
        return:
            output: [B, num_codebook, output_dim]
        """
        codebook_num = x.shape[1]
        output = torch.bmm(
            x.transpose(0, 1),  # [num_codebook, B, input_dim]
            self.weight[:codebook_num, :, :],  # [num_codebook, input_dim, output_dim]
        )
        return output.transpose(0, 1)  # [B, num_codebook, output_dim]

    def get_logits(self, x: torch.Tensor, codebook_id: int):
        """
        args:
            x: [B, input_dim]
            codebook_id: int
        return:
            logits: [B, ]
        """
        # codebook 0 is in backbone, so the weight is from 1 to num_codebooks
        if codebook_id == 0 or codebook_id > self.num_codebooks:
            raise ValueError(
                f"codebook_id must be between 1 and {self.num_codebooks}, but got {codebook_id}"
            )
        return torch.mm(x, self.weight[codebook_id - 1, :, :])


class ChromaDecoderForCausalLM(ChromaPreTrainedModel, GenerationMixin):
    config_class = ChromaDecoderConfig
    _supports_flash_attn_2 = True

    def __init__(self, config: ChromaDecoderConfig):
        super().__init__(config)

        self.projection = nn.Linear(
            self.config.audio_embedding_dim, self.config.hidden_size, bias=False
        )

        self.model = ChromaLlamaModel(LlamaConfig(**config.to_dict()))

        self.codebook_head = ChromaCodebookHead(
            audio_num_codebooks=self.config.audio_num_codebooks - 1,
            audio_vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
        )

        self.audio_embedding = ChromaAudioEmbedding(
            config.audio_num_codebooks,
            config.vocab_size,
            config.audio_embedding_dim,
        )

        self.post_init()

    def loss_fn(self, logits, labels, ignore_index=-100):
        """
        logits: [B, num_codebooks-1, 2051]
        labels: [B, num_codebooks-1]
        """

        # flatten logits and labels
        vocab_size = logits.size(-1)
        logits_flat = logits.contiguous().view(-1, vocab_size)
        labels_flat = labels.contiguous().view(-1)

        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)

        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
            # During training, just pass inputs_embeds which are embedded by backbone.embedding, so the internal embedding is not used.
            # During inference, pass input_ids and backbone_last_hidden_state, which is aligned with the standard HuggingFace model.
        args:
            input_ids: [B, codebook_num]  every sequence is an audio frame
            backbone_last_hidden_state: [B, 2048]  every [2048] is a hidden from qwen
            inputs_embeds: [B, seq_len, codebook_num, 2048]  every [2048] is a hidden from qwen
            labels: [B, seq_len, num_codebook]  every [n] element is [-100, codebook0-num_codebook-1 id]
        return:
            output: Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]
                loss: [B, seq_len]
                logits: [B, seq_len, codebook_num, 2051]
        """

        if inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds or input_ids is required")

        if inputs_embeds is not None and input_ids is not None:
            raise ValueError(
                "inputs_embeds and input_ids cannot be used at the same time"
            )

        loss = None

        if inputs_embeds is None:
            if past_key_values is not None:
                seq_length = int(past_key_values.get_seq_length())
                past_codebook_num = max(0, seq_length - 1)
            else:
                past_codebook_num = 0

            if past_codebook_num > self.config.audio_num_codebooks - 1:
                raise ValueError(
                    f"past_codebook_num is greater than audio_num_codebooks - 1, "
                    f"{past_codebook_num} > {self.config.audio_num_codebooks - 1}"
                )
            
            codebook_positions = torch.arange(
                input_ids.shape[-1], device=input_ids.device
            ) + past_codebook_num
            if (codebook_positions >= self.config.audio_num_codebooks).any() or (
                codebook_positions < 0
            ).any():
                raise ValueError(
                    f"Invalid codebook_positions={codebook_positions.tolist()} "
                    f"for audio_num_codebooks={self.config.audio_num_codebooks}, "
                    f"past_codebook_num={past_codebook_num}, input_ids_shape={tuple(input_ids.shape)}"
                )
            offset = codebook_positions * self.config.vocab_size
            max_embedding_index = self.audio_embedding.embed_audio_tokens.num_embeddings
            indices = input_ids + offset
            
            if (indices >= max_embedding_index).any() or (indices < 0).any():
                print(f"Error: indices out of bounds!")
            audio_ids_embed = self.audio_embedding.embed_audio_tokens(indices)
            inputs_embeds = (
                torch.cat(
                    [backbone_last_hidden_state.unsqueeze(1), audio_ids_embed], dim=1
                )
                if backbone_last_hidden_state is not None
                else audio_ids_embed
            )

        orig_shape = inputs_embeds.shape

        # if input_embeddings is 4D, it means that the input_embeddings is a batch of sequences
        if inputs_embeds.dim() == 4:
            # [B, seq_len, codebook_num, 2048] -> [B*seq_len, codebook_num, 2048]
            inputs_embeds = inputs_embeds.reshape(
                -1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]
            )
            # [B, seq_len, codebook_num] -> [B*seq_len, codebook_num]
            labels = labels.reshape(-1, labels.shape[-1])

        # cut off the eos any way (delete -1)
        has_eos = inputs_embeds.shape[1] == self.config.audio_num_codebooks + 1
        inputs_embeds = inputs_embeds[:, : self.config.audio_num_codebooks, :]

        # Forward
        inputs_embeds = self.projection(inputs_embeds)
        output: BaseModelOutputWithPast = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        if past_key_values is not None:
            logits = self.codebook_head.get_logits(
                output.last_hidden_state[:, -1, :],
                past_codebook_num + 1,
            ).unsqueeze(1)
        else:
            logits = self.codebook_head(output.last_hidden_state[:, 1:, :])  # (delet 0)

        if labels is not None:
            # the sequence must be full, calculate loss at codebook 1-31
            assert (
                labels.shape[1] == self.config.audio_num_codebooks - 1
            ), f"labels must have {self.config.audio_num_codebooks-1} tokens, but got {labels.shape[1]}"
            assert (
                logits.shape[1] == self.config.audio_num_codebooks - 1
            ), f"logits must have {self.config.audio_num_codebooks-1} tokens, but got {logits.shape[1]}"
            loss = self.loss_fn(logits, labels.clone().detach())

        # Ensure that the output logits sequence length matches the input sequence length
        pad_left = (
            1
            if backbone_last_hidden_state is not None or has_eos or input_ids is None
            else 0
        )
        pad_right = 1 if has_eos else 0
        logits = F.pad(
            logits, (0, 0, pad_left, pad_right), value=0
        )  # if see 0 in the first position, it's just a padding
        logits = logits.reshape(*orig_shape[:-1], logits.shape[-1])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        for decoder generate
        Args:
            input_ids:
            past_key_values:
            attention_mask:
            inputs_embeds:
            cache_position:
            **kwargs:

        Returns:

        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            **kwargs,
        )

        seq_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        is_first_generation_step = (past_key_values is None) or (seq_len == 0)
        if not is_first_generation_step:
            model_inputs.pop("backbone_last_hidden_state", None)
        return model_inputs




class ChromaForConditionalGeneration(ChromaPreTrainedModel, ChromaGenerationMixin):
    base_model_prefix = "chroma"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    _tied_weights_keys = [
        "backbone.audio_embedding.embed_audio_tokens.weight",
        "decoder.audio_embedding.embed_audio_tokens.weight",
    ]

    def __init__(self, config: ChromaConfig):
        super().__init__(config)
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration._from_config(
            config.thinker_config
        )
        self.backbone = ChromaBackboneForCausalLM._from_config(config.backbone_config)
        self.decoder = ChromaDecoderForCausalLM._from_config(config.decoder_config)
        self.codec_model = MimiModel._from_config(config.codec_config)

        assert (
            self.backbone.config.audio_num_codebooks == config.audio_num_codebooks
        ), f"backbone.config.audio_num_codebooks {self.backbone.config.audio_num_codebooks} != config.audio_num_codebooks {config.audio_num_codebooks}"
        assert (
            self.decoder.config.audio_num_codebooks == config.audio_num_codebooks
        ), f"decoder.config.audio_num_codebooks {self.decoder.config.audio_num_codebooks} != config.audio_num_codebooks {config.audio_num_codebooks}"

        self.post_init()

    def _tie_weights(self):
        self._tie_or_clone_weights(
            self.backbone.audio_embedding.embed_audio_tokens,
            self.decoder.audio_embedding.embed_audio_tokens,
        )

    def _embed_text_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "thinker"):
            return self.thinker.model.embed_tokens(ids.to(self.device))
        else:
            return self.embed_tokens(ids.to(self.device))

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        thinker_input_ids: Optional[torch.LongTensor] = None,
        thinker_attention_mask: Optional[torch.LongTensor] = None,
        thinker_cache_position: Optional[torch.LongTensor] = None,
        thinker_past_key_values: Optional[Cache] = None,
        thinker_hidden_states: Optional[torch.FloatTensor] = None,
        thinker_input_features: Optional[torch.FloatTensor] = None,
        thinker_feature_attention_mask: Optional[torch.LongTensor] = None,
        thinker_logits: Optional[torch.FloatTensor] = None,
        prompt_audio: Optional[torch.FloatTensor] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        thinker_flag: bool = True,  # generate at the first step
        **kwargs,
    ):
        """
        args:
            input_ids: [B, seq_len]
            input_values: [B, channels, audio_seq_len]
            past_key_values: [B, num_layers, num_heads, seq_len, hidden_size]
            attention_mask: [B, seq_len]
            inputs_embeds: [B, seq_len, hidden_size]
            cache_position: [B, seq_len]
            thinker_input_ids: [B, seq_len]  None means thinker had generate eos
            thinker_attention_mask: [B, seq_len]
            thinker_cache_position: [B, seq_len]
            thinker_past_key_values:
            thinker_hidden_states: [B, seq_len, hidden_size]
            thinker_input_features: [B, seq_len, hidden_size]
            thinker_feature_attention_mask: [B, seq_len]
            thinker_logits: [B, seq_len, hidden_size]
            prompt_audio: [B, channels, audio_seq_len]
            prompt_ids: [B, seq_len]
            thinker_flag: bool whether thinker need to generate next token and inject into inputs_embeds
        Returns:
            inputs_embeds: [B, seq_len, hidden_size]

        use input_ids to build prompt_embeds, if it is the step that need thinker to generate next token, then inject its hidden states and next token embedding into inputs_embeds
        """

        if input_values is not None:
            # input_values only exists in the first step
            inputs_embeds, attention_mask = self._build_prompt_embeds(
                input_ids, attention_mask, input_values
            )  # type: ignore
        else:
            inputs_embeds = self.backbone.emb_audio_frames(
                input_ids.squeeze(0).to(self.device)  # type: ignore
            ).unsqueeze(0)

        if thinker_input_ids is not None and thinker_flag:
            (
                thinker_input_ids,
                thinker_attention_mask,
                thinker_cache_position,
                thinker_past_key_values,
            ) = self._update_thinker_model_kwargs(
                thinker_input_ids,
                thinker_attention_mask,
                thinker_cache_position,
                thinker_past_key_values,
            )  # type: ignore

            with torch.inference_mode():
                thinker_outputs = self.thinker.forward_transformers_style(
                    input_ids=thinker_input_ids,
                    input_features=thinker_input_features,
                    attention_mask=thinker_attention_mask,
                    feature_attention_mask=thinker_feature_attention_mask,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                    past_key_values=thinker_past_key_values,
                    cache_position=thinker_cache_position,
                    use_audio_in_video=False,
                )

            thinker_hidden_states = thinker_outputs.hidden_states if thinker_outputs.hidden_states is not None else thinker_outputs.logits
            thinker_past_key_values = thinker_outputs.past_key_values
            thinker_logits = thinker_outputs.logits

            thinker_next_ids = thinker_logits[:, -1:, :].argmax(dim=-1)

            next_token_emb = self._embed_text_tokens(thinker_next_ids)

            stop_eos = D_SPECIAL_TOKEN_2_IDS.get("<|im_end|>")
            is_eos = (thinker_next_ids.squeeze(-1) == stop_eos).any()
            thinker_input_ids = thinker_next_ids if not is_eos else None

            thinker_input_embeddings = torch.cat(
                [thinker_hidden_states[:, -1:, :], next_token_emb], dim=1
            )

            if inputs_embeds is not None:
                inputs_embeds = torch.cat(
                    [inputs_embeds, thinker_input_embeddings], dim=1
                )
            else:
                inputs_embeds = thinker_input_embeddings

            if inputs_embeds is not None:
                attention_mask = attention_mask.resize_(
                    (
                        attention_mask.shape[0],
                        attention_mask.shape[1] + thinker_input_embeddings.shape[1],
                    )
                ).fill_(1)

        if input_values is not None:
            attention_mask = attention_mask.resize_(
                (attention_mask.shape[0], inputs_embeds.shape[1])
            ).fill_(1)

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        if thinker_input_ids is None:
            next_thinker_flag = False
        else:
            next_thinker_flag = not thinker_flag

        model_inputs = {
            "input_ids": None,
            "input_embeddings": inputs_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": True,
            "output_hidden_states": True,
            "thinker_past_key_values": thinker_past_key_values,
            "thinker_hidden_states": thinker_hidden_states,
            "thinker_logits": thinker_logits,
            "thinker_input_ids": thinker_input_ids,
            "thinker_attention_mask": thinker_attention_mask,
            "thinker_input_features": thinker_input_features,
            "thinker_feature_attention_mask": thinker_feature_attention_mask,
            "thinker_cache_position": thinker_cache_position,
            "thinker_flag": next_thinker_flag,
        }

        return model_inputs

    def _build_prompt_embeds(self, input_ids, attention_mask=None, input_values=None):
        """
        Build QSM input embeddings according to the specified layout for generation

        Args:
            input_ids: prompt text ids
            attention_mask:
            input_values: prompt audio waveform

        Returns:
        """

        audio_codes = self.codec_model.encode(input_values.unsqueeze(0).unsqueeze(0)).audio_codes
        audio_codes = audio_codes[
            :, : self.config.backbone_config.audio_num_codebooks, :
        ]
        prompt_audio_emb = self.backbone.emb_audio_frames(
            audio_codes.permute(0, 2, 1).to(self.device)
        )

        prompt_text_emb = self._embed_text_tokens(input_ids.to(self.device))

        # TODO: should support batch input
        inputs_emb = []

        # add ref text and audio
        # add text start
        text_start_ids = torch.tensor(
            [D_SPECIAL_TOKEN_2_IDS.get("<|text_start|>")], dtype=torch.long
        )
        text_start_emb = self._embed_text_tokens(text_start_ids).unsqueeze(0)
        inputs_emb.append(text_start_emb)

        inputs_emb.append(prompt_text_emb)

        # add text end
        text_end_ids = torch.tensor(
            [D_SPECIAL_TOKEN_2_IDS.get("<|text_end|>")], dtype=torch.long
        )
        text_end_emb = self._embed_text_tokens(text_end_ids).unsqueeze(0)
        inputs_emb.append(text_end_emb)

        inputs_emb.append(prompt_audio_emb)

        # add eos of ref audio
        eos_token_audio = torch.full(
            (1, prompt_audio_emb.shape[-1]),
            0,
            dtype=prompt_audio_emb.dtype,
            device=prompt_audio_emb.device,
        ).unsqueeze(0)
        inputs_emb.append(eos_token_audio)

        # concat all parts
        input_embeddings = torch.cat(inputs_emb, dim=1)
        # must modify attention_mask inplace to affect model_kwargs
        if attention_mask is not None:
            attention_mask = attention_mask.resize_(
                (
                    attention_mask.shape[0],
                    attention_mask.shape[1] + input_embeddings.shape[1],
                )
            ).fill_(1)
        else:
            attention_mask = torch.ones(
                (1, input_embeddings.shape[1]), device=input_embeddings.device
            )

        return input_embeddings, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # [seq_len, num_codebooks]
        loss_stride: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        input_embeddings: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ChromaOutputWithPast:
        """
        main model forward
        Args:
            input_ids: never used
            position_ids:
            attention_mask:
            feature_attention_mask:
            past_key_values:
            inputs_embeds:
            labels:
            loss_stride:
            use_cache:
            output_attentions:
            output_hidden_states:
            input_embeddings: backbone input embeddings
            cache_position: backbone position
            **kwargs:

        Returns:

        """
        if labels is not None:
            return self.train_forward(
                input_embeddings=input_embeddings,
                labels=labels,
                loss_stride=loss_stride,
                use_cache=use_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                **kwargs,
            )

        # For generation
        backbone_outputs: CausalLMOutputWithPast = self.backbone(
            input_embeddings=input_embeddings,
            labels=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return self._build_outputs(
            loss=backbone_outputs.loss,
            logits=backbone_outputs.logits,
            hidden_states=backbone_outputs.hidden_states,
            past_key_values=backbone_outputs.past_key_values,
            **kwargs,
        )

    def _build_outputs(self, **kwargs) -> ChromaOutputWithPast:
        fields_names = [f.name for f in fields(ChromaOutputWithPast)]
        outputs = ChromaOutputWithPast(
            **{k: v for k, v in kwargs.items() if k in fields_names}
        )
        return outputs

    def train_forward(
        self,
        input_embeddings: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_stride=False,
        **kwargs,
    ):
        """
        Just for calculate loss, nothing to do with thinker, so disable thinker at first to save memory
        args:
            input_embeddings: [B, seq_len, 2048]  every [2048] is a hidden from qwen
            labels: [B, seq_len, num_codebooks]  every [num_codebooks] is num_codebooks audio codebook ids
        return:
            c0_output: Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            decoder_output: Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]
                logits: [B, seq_len, 2051]
        """

        assert labels is not None, "labels is required"
        assert input_embeddings is not None, "input_embeddings is required"
        labels = labels[:, :, : self.config.backbone_config.audio_num_codebooks]

        bs, seq_len, dim = input_embeddings.shape

        # c0 forward
        # with torch.no_grad():
        c0_output = self.backbone(
            input_embeddings=input_embeddings,
            labels=labels[:, :, 0],
        )

        # prepare decoder input: pad labels at seq_len dimension to get shift labels
        c0_hidden_states = c0_output.hidden_states[-1].detach().clone()
        decoder_frames = F.pad(labels.contiguous(), (0, 0, 0, 1), value=-100)[:, 1:, :]

        if loss_stride:
            mask = decoder_frames[..., 0] != -100
            c0_hidden_states = c0_hidden_states[mask][:: self.config.loss_stride]
            decoder_frames = decoder_frames[mask][:: self.config.loss_stride]

        # decoder input
        decoder_input_embeddings = torch.cat(
            [
                c0_hidden_states.unsqueeze(-2),
                self.backbone.emb_audio_frames(decoder_frames, add_frame=False),
            ],
            dim=-2,
        )
        decoder_output = self.decoder(
            inputs_embeds=decoder_input_embeddings, labels=decoder_frames[..., 1:]
        )  # drop the first codebook

        loss = (
            c0_output["loss"] * (1 - self.config.decoder_loss_weight)
            + decoder_output["loss"] * self.config.decoder_loss_weight
        )
        # loss = decoder_output["loss"]

        # merge logits
        if loss_stride:
            # if loss_stride, we just forward to get loss, so logits is not needed
            logits = None
        else:
            # in order to get the valid codebook logits, use shift labels to index the logits
            logits = torch.cat(
                [c0_output.logits.unsqueeze(-2), decoder_output.logits[:, :, 1:-1, :]],
                dim=-2,
            )

        return ChromaOutputWithPast(
            loss=loss,
            logits=logits,
            backbone_hidden_states=c0_output.hidden_states,
            backbone_loss=c0_output.loss,
            backbone_logits=c0_output.logits,
            decoder_loss=decoder_output.loss,
            decoder_logits=decoder_output.logits,
        )

    def get_pred_codebooks(self, output: ChromaOutputWithPast, labels: torch.Tensor):
        """
        get the predicted codebooks from the logits
        args:
            output: ChromaOutputWithPast
            labels: [B, seq_len, num_codebooks]
        return:
            valid_pred: B * [seq_len, num_codebooks]
            shift_labels: B * [seq_len, num_codebooks]
        """
        labels = labels[:, :, : self.config.backbone_config.audio_num_codebooks]
        shift_labels = F.pad(labels.contiguous(), (0, 0, 0, 1), value=-100)[:, 1:, :]
        shift_mask = shift_labels[:, :, 0] != -100

        pred = output.logits.argmax(dim=-1)
        valid_pred = [
            pred[i][shift_mask[i]][:-1, :] for i in range(pred.shape[0])
        ]  # remove the eos
        valid_shift_labels = [
            shift_labels[i][shift_mask[i]][:-1, :] for i in range(shift_labels.shape[0])
        ]  # remove the eos

        return valid_pred, valid_shift_labels

    def gen_pred_codebooks(self, output: ChromaOutputWithPast, labels: torch.Tensor):

        labels = labels[:, :, : self.config.backbone_config.audio_num_codebooks]
        shift_labels = F.pad(labels.contiguous(), (0, 0, 0, 1), value=-100)[:, 1:, :]
        shift_mask = shift_labels[:, :, 0] != -100

        pred = output.logits.argmax(dim=-1)
        valid_pred = [
            pred[i][shift_mask[i]][:-1, :] for i in range(pred.shape[0])
        ]  # remove the eos
        valid_shift_labels = [
            shift_labels[i][shift_mask[i]][:-1, :] for i in range(shift_labels.shape[0])
        ]  # remove the eos

        valid_gen = []
        for i in range(labels.shape[0]):
            valid_gen.append(
                self.decoder.generate(
                    input_ids=valid_pred[i][:, 0:1],
                    backbone_last_hidden_state=output.backbone_hidden_states[-1][i][
                        shift_mask[i]
                    ][:-1, :],
                    max_new_tokens=self.config.audio_num_codebooks - 1,
                    min_new_tokens=self.config.audio_num_codebooks - 1,
                )
            )  # remove the eos

        return valid_gen, valid_shift_labels

    def get_precisions(
        self, codebooks: List[torch.Tensor], valid_shift_labels: List[torch.Tensor]
    ):
        precisions = torch.zeros(self.config.audio_num_codebooks, len(codebooks))
        for i in range(len(codebooks)):
            for j in range(self.config.audio_num_codebooks):
                con = codebooks[i][:, j] == valid_shift_labels[i][:, j]
                p = con.sum() / con.numel()
                precisions[j, i] = p
        return precisions

    def _update_model_kwargs_for_generation(
        self,
        outputs: ChromaOutputWithPast,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:

        for key in PASSTHROUGH_KEYS:
            model_kwargs[key] = getattr(outputs, key, None)

        for key in ONE_TIME_KEYS:
            model_kwargs[key] = None

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        return model_kwargs

    def _update_thinker_model_kwargs(
        self,
        thinker_input_ids: torch.Tensor,
        thinker_attention_mask: Optional[torch.Tensor] = None,
        thinker_cache_position: Optional[torch.Tensor] = None,
        thinker_past_key_values: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Any]]:

        past_seen_tokens = 0
        if thinker_past_key_values is not None:
            if hasattr(thinker_past_key_values, "get_seq_length"):
                past_seen_tokens = int(thinker_past_key_values.get_seq_length())
            elif isinstance(thinker_past_key_values, list) and len(thinker_past_key_values) > 0:
                # (k, v): k shape [B, num_heads, seq_len, head_dim]
                try:
                    past_seen_tokens = int(thinker_past_key_values[0][0].shape[2])
                except Exception:
                    past_seen_tokens = 0

        num_new_tokens = thinker_input_ids.shape[1]

        if thinker_cache_position is None:
            thinker_cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + thinker_input_ids.shape[1],
                device=thinker_input_ids.device,
            )
        else:
            thinker_cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + num_new_tokens,
                device=thinker_input_ids.device,
            )

        if thinker_attention_mask is None:
            thinker_attention_mask = torch.ones(
                (thinker_input_ids.shape[0], num_new_tokens),
                device=thinker_input_ids.device,
            )
        else:
            if thinker_past_key_values is not None:
                thinker_attention_mask = torch.cat(
                    [
                        thinker_attention_mask,
                        thinker_attention_mask.new_ones(
                            (thinker_attention_mask.shape[0], num_new_tokens)
                        ),
                    ],
                    dim=-1,
                )

        return (
            thinker_input_ids,
            thinker_attention_mask,
            thinker_cache_position,
            thinker_past_key_values,
        )


CONFIG_MAPPING.register("chroma", ChromaConfig)
MODEL_MAPPING.register(ChromaConfig, ChromaForConditionalGeneration)
