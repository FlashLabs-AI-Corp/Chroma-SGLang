import math
from typing import Set, List, Tuple, Union, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLMLP,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

try:
    from sglang.srt.layers.quantization import QuantizationConfig
except ImportError as e:
    import warnings

    warnings.warn(f"Can not import QuantizationConfig: {e}, use default config")


    class QuantizationConfig:
        pass

from sglang.utils import logger
from sglang.srt.utils import add_prefix
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.qwen2 import Qwen2MLP, Qwen2Attention
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.managers.schedule_batch import MultimodalInputs, MultimodalDataItem
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding


class Qwen2_5OmniAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = False
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(1, 2)
        ) / math.sqrt(self.head_dim)

        attention_mask = torch.full(
            [1, seq_length, key_states.shape[1]],
            torch.finfo(query_states.dtype).min,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
            ...,
            cu_seqlens[i - 1]: cu_seqlens[i],
            cu_seqlens[i - 1]: cu_seqlens[i],
            ] = 0

        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(
            query_states.dtype
        )

        attn_output = (
            torch.matmul(attn_weights, value_states)
            .transpose(0, 1)
            .reshape(seq_length, self.embed_dim)
        )

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.

        attn_output = self.out_proj(attn_output)

        return attn_output


QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniAudioAttention,
    "sdpa": Qwen2_5OmniAudioAttention,
    "flash_attention_2": Qwen2_5OmniAudioAttention,
}


class Qwen2_5OmniAudioEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES[
            config._attn_implementation
        ](config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2)
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniVisionBlock(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            qkv_backend="sdpa",
            softmax_in_single_precision=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, position_embeddings) -> torch.Tensor:
        seq_len, _ = hidden_states.size()

        normed_hs = self.norm1(hidden_states)
        normed_hs = normed_hs.unsqueeze(0)
        attn = self.attn(
            normed_hs, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + attn
        hidden_states = hidden_states.view(seq_len, -1)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5OmniPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5OmniVisionEncoder(nn.Module):
    _no_split_modules = ["Qwen2_5OmniVisionBlock"]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5OmniVisionBlock(config, quant_config=quant_config)
                for _ in range(config.depth)
            ]
        )
        self.merger = Qwen2_5OmniPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Modification here
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.gate_proj.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.gate_proj.weight.device


class Qwen2_5OmniDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        if (
            config.use_sliding_window
            and config._attn_implementation != "flash_attention_2"
        ):
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        self.rope_scaling = config.rope_scaling
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = hidden_size // self.num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        text_config = config
        self.mlp = Qwen2MLP(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        rope_scaling = config.rope_scaling
        self.rope_theta = getattr(config, "rope_theta", 1000000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            rope_theta=self.rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        text_config = config
        self.mlp = Qwen2MLP(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embedding to q and k.

        Shapes:
          - q: (batch, num_heads, seq_len, head_dim)
          - k: (batch, num_heads|num_kv_heads, seq_len, head_dim)
          - cos/sin: (seq_len, head_dim) OR (batch, seq_len, head_dim)
        """
        if cos.dim() == 2:
            # (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            # (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected cos/sin shape: {cos.shape}")

        # Rotate half
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def _compute_rotary_emb(self, seq_len, device, dtype):
        """Compute rotary embeddings for positions [0..seq_len-1]."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin

    def _compute_rotary_emb_from_positions(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings for arbitrary positions.

        Args:
            position_ids: (batch, seq_len) or (seq_len,)
        Returns:
            cos, sin with shape (batch, seq_len, head_dim) if position_ids is 2D,
            otherwise (seq_len, head_dim).
        """
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )

        if position_ids.dim() == 1:
            pos = position_ids.to(dtype=torch.float32)
            freqs = pos[:, None] * inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)

        if position_ids.dim() == 2:
            pos = position_ids.to(dtype=torch.float32)
            freqs = pos[..., None] * inv_freq[None, None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)

        raise ValueError(f"Unexpected position_ids shape: {position_ids.shape}")

    def _build_combined_attn_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        past_len: int,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build additive attention mask = causal + padding.

        Returns:
            (batch, 1, q_len, k_len) additive mask (0 or -inf).
        """
        neg_inf = torch.finfo(dtype).min

        # Causal in absolute positions:
        # query abs positions: [past_len .. past_len+q_len-1]
        q_pos = torch.arange(past_len, past_len + q_len, device=device)[:, None]
        k_pos = torch.arange(0, k_len, device=device)[None, :]
        causal_allowed = k_pos <= q_pos  # (q_len, k_len)
        causal_mask = torch.where(
            causal_allowed,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), neg_inf, device=device, dtype=dtype),
        ).view(1, 1, q_len, k_len)

        if attention_mask is None:
            return causal_mask

        if attention_mask.dim() != 2:
            # Assume caller passed a pre-built mask
            return attention_mask.to(dtype)

        # Key-side padding mask: (batch, full_seq_len) -> align to k_len
        if attention_mask.shape[-1] > k_len:
            attention_mask = attention_mask[:, -k_len:]
        elif attention_mask.shape[-1] < k_len:
            pad = torch.zeros(
                (attention_mask.shape[0], k_len - attention_mask.shape[-1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=-1)

        key_allowed = attention_mask[:, None, None, :].to(dtype=dtype)  # (b,1,1,k_len)
        pad_mask = (1.0 - key_allowed) * neg_inf
        return causal_mask + pad_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        # SGLang 的层期望 2D 输入 (batch*seq_len, hidden_size)
        orig_shape = hidden_states.shape  # (batch, seq_len, hidden_size)
        hidden_states = hidden_states.view(
            -1, orig_shape[-1]
        )  # (batch*seq_len, hidden_size)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # SGLang 的 Qwen2Attention 期望 2D 输入和 1D position_ids
        position_ids_flat = (
            position_ids.reshape(-1) if position_ids.dim() > 1 else position_ids
        )
        hidden_states = self.self_attn(
            positions=position_ids_flat,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states.view(orig_shape)

        return hidden_states

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Transformers style

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            position_ids: (batch, seq_len) or (seq_len,)
            attention_mask: optional attention mask
        """
        batch_size, seq_len, _ = hidden_states.shape
        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.shape[2]

        residual = hidden_states

        # Layer norm - 需要转换为 2D
        hidden_states_2d = hidden_states.view(-1, self.hidden_size)
        hidden_states_2d = self.input_layernorm(hidden_states_2d)
        hidden_states = hidden_states_2d.view(batch_size, seq_len, -1)

        # Self attention using native PyTorch SDPA
        qkv_proj = self.self_attn.qkv_proj
        o_proj = self.self_attn.o_proj

        # calculate QKV
        qkv_output = qkv_proj(hidden_states_2d)  # SGLang 可能返回元组 (output, bias)
        if isinstance(qkv_output, tuple):
            qkv = qkv_output[0]
        else:
            qkv = qkv_output
        qkv = qkv.view(batch_size, seq_len, -1)

        q = qkv[..., : self.q_size]
        k = qkv[..., self.q_size: self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size:]

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # Apply rotary embeddings using provided position_ids (supports incremental decode)
        if position_ids.dim() == 1:
            pos = position_ids
            if pos.numel() != seq_len:
                pos = torch.arange(
                    past_len,
                    past_len + seq_len,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
        else:
            pos = position_ids
            if pos.shape[-1] != seq_len:
                pos = torch.arange(
                    past_len,
                    past_len + seq_len,
                    device=hidden_states.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1)

        cos, sin = self._compute_rotary_emb_from_positions(
            pos, hidden_states.device, hidden_states.dtype
        )
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV heads if needed (GQA)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # KV cache concat (incremental decoding)
        if past_key_value is not None:
            k = torch.cat([past_k.to(dtype=k.dtype, device=k.device), k], dim=2)
            v = torch.cat([past_v.to(dtype=v.dtype, device=v.device), v], dim=2)

        # Compute attention using SDPA
        # IMPORTANT: keep causal behavior even if a padding mask is provided.
        key_len = k.shape[2]
        sdpa_attn_mask = self._build_combined_attn_mask(
            attention_mask=attention_mask,
            past_len=past_len,
            q_len=seq_len,
            k_len=key_len,
            device=q.device,
            dtype=q.dtype,
        )

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_attn_mask,
            is_causal=False,
        )

        # Reshape and project
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size * seq_len, -1)
        )
        o_output = o_proj(attn_output)
        if isinstance(o_output, tuple):
            attn_output = o_output[0]
        else:
            attn_output = o_output
        attn_output = attn_output.view(batch_size, seq_len, -1)

        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states_2d = hidden_states.view(-1, self.hidden_size)
        hidden_states_2d = self.post_attention_layernorm(hidden_states_2d)
        mlp_output = self.mlp(hidden_states_2d)
        if isinstance(mlp_output, tuple):
            hidden_states_2d = mlp_output[0]
        else:
            hidden_states_2d = mlp_output
        hidden_states = hidden_states_2d.view(batch_size, seq_len, -1)

        hidden_states = residual + hidden_states

        if use_cache:
            # Cache post-rotary, post-GQA K/V
            new_past = (k.detach(), v.detach())
            return hidden_states, new_past

        return hidden_states


class Qwen2_5OmniThinkerModel(nn.Module):
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )

        self.layers = nn.ModuleList(
            [
                Qwen2_5OmniDecoderLayer(config, layer_idx, quant_config=quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_output = decoder_layer(
                positions=positions,
                position_ids=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states = layer_output

        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        assert hidden_states.dim() == 2, hidden_states.shape
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen2_5OmniAudioEncoder(PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2_5OmniAudioEncoderLayer`].

    Args:
        config: Qwen2_5OmniAudioEncoderConfig
    """

    main_input_name = "input_features"
    _no_split_modules = ["Qwen2_5OmniAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        # 确保 feature_lens 是 long 类型
        feature_lens_long = (
            feature_lens.long() if feature_lens.dtype != torch.long else feature_lens
        )
        chunk_num = torch.ceil(feature_lens_long / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = (
            feature_lens_long % (self.n_window * 2)
        ).long()
        chunk_lengths = torch.where(
            chunk_lengths == 0, self.n_window * 2, chunk_lengths
        )

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        (
            padded_feature,
            padded_mask,
            padded_mask_after_cnn,
        ) = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
                                      : padded_embed.shape[1], :
                                      ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        for idx, encoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    cu_seqlens,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    cu_seqlens,
                )

            hidden_states = layer_outputs[0]

        aftercnn_lens_int = [int(x) for x in aftercnn_lens.tolist()]
        hidden_states_list = hidden_states.split(aftercnn_lens_int, dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(
                each_audio_states.transpose(0, 1)
            ).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        return BaseModelOutput(last_hidden_state=token_audio)

    def padded_and_mask_function(
        self, tensor_list, tensor_len, padding_value=0, padding_side="right"
    ):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class Qwen2_5OmniThinkerForConditionalGeneration(nn.Module):
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]
    base_model_prefix = "model"

    @staticmethod
    def is_backend_compatible() -> bool:
        """
        compact SGLang
        """
        return True

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        for auto register class
        """
        pass

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        for transformers
        """
        return cls(config, **kwargs)

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()

        # thinker_config
        if hasattr(config, "thinker_config"):
            config = config.thinker_config
        else:
            config = config

        self.config = config
        self.text_config = config.text_config

        self.audio_tower = Qwen2_5OmniAudioEncoder(config.audio_config)

        self.visual = Qwen2_5OmniVisionEncoder(
            config.vision_config, quant_config=quant_config
        )

        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerModel(
            config.text_config, quant_config=quant_config
        )
        text_config = config.text_config
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        self.is_mrope_enabled = "mrope_section" in self.config.text_config.rope_scaling

        self.logits_processor = LogitsProcessor(config.text_config)

    def get_input_embeddings(self):
        """
        get input embeddings
        Returns:

        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        set input embeddings
        Args:
            value:

        Returns:

        """
        self.model.embed_tokens = value

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.concat([item.pixel_values for item in items], dim=0).type(
            self.visual.dtype
        )

        image_grid_thws = torch.concat([item.image_grid_thws for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thws.dim() == 2, image_grid_thws.dim()
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thws)

        return image_embeds

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        input_features = (
            torch.cat([item.audio_features for item in items])
            .type(self.audio_tower.dtype)
            .to(next(self.audio_tower.parameters()).device)
        )
        feature_attention_mask = torch.cat(
            [item.feature_attention_mask for item in items]
        )
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_attention_mask = torch.sum(feature_attention_mask, dim=1)

        (
            audio_feat_lengths,
            audio_output_lengths,
        ) = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        hs = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            image_data_embedding_func=self.get_image_feature,
            audio_data_embedding_func=self.get_audio_feature,
            positions=positions,
        )
        return self.logits_processor(input_ids, hs, self.lm_head, forward_batch)

    def forward_transformers_style(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thws: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        """
        compact transformers forward

        Args:
            input_ids: token IDs
            attention_mask:
            input_features:
            feature_attention_mask:
            position_ids:
            past_key_values: KV cache
            inputs_embeds:
            labels:
            use_cache:
            output_attentions:
            output_hidden_states:
            return_dict:
            cache_position:
            pixel_values:
            image_grid_thws:
            use_audio_in_video:

        Returns:
            CausalLMOutputWithPast: standard transformers
        """
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        use_cache = bool(use_cache) if use_cache is not None else False

        # SGLang list[(k,v)]
        past_len = 0
        if (
            use_cache
            and past_key_values is not None
            and isinstance(past_key_values, list)
            and len(past_key_values) > 0
        ):
            try:
                past_len = past_key_values[0][0].shape[2]
            except Exception:
                past_len = 0

        # create position_ids
        if position_ids is None:
            if cache_position is not None:
                # cache_position: (seq_len,) absolute positions for current tokens
                if cache_position.dim() == 1:
                    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
                else:
                    position_ids = cache_position
            else:
                start = past_len
                position_ids = (
                    torch.arange(
                        start,
                        start + seq_length,
                        dtype=torch.long,
                        device=input_ids.device
                        if input_ids is not None
                        else inputs_embeds.device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

        # process multimodal inputs
        if input_features is not None or pixel_values is not None:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.model.embed_tokens(input_ids)

            # process audio feature
            if input_features is not None:
                if feature_attention_mask is not None:
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    input_features_flat = input_features.permute(0, 2, 1)[
                        feature_attention_mask.bool()
                    ].permute(1, 0)
                else:
                    audio_feature_lengths = torch.tensor(
                        [input_features.shape[2]] * input_features.shape[0],
                        device=input_features.device,
                    )
                    input_features_flat = input_features.reshape(
                        input_features.shape[1], -1
                    )

                (
                    audio_feat_lengths,
                    _,
                ) = self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths
                )

                audio_outputs = self.audio_tower(
                    input_features_flat,
                    feature_lens=audio_feature_lengths,
                    aftercnn_lens=audio_feat_lengths,
                )
                audio_embeds = audio_outputs.last_hidden_state

                # merge audio embed into text embed
                # 找到音频 token 的位置并替换
                if input_ids is not None:
                    audio_token_mask = input_ids == self.config.audio_token_index
                    if audio_token_mask.any():
                        for batch_idx in range(batch_size):
                            audio_positions = torch.where(audio_token_mask[batch_idx])[
                                0
                            ]
                            if len(audio_positions) > 0:
                                audio_len = min(
                                    len(audio_positions), audio_embeds.shape[0]
                                )
                                inputs_embeds[
                                    batch_idx, audio_positions[:audio_len]
                                ] = audio_embeds[:audio_len]

            # process image feature if necessary
            if pixel_values is not None and image_grid_thws is not None:
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thws)

                if input_ids is not None:
                    image_token_mask = input_ids == self.config.image_token_index
                    if image_token_mask.any():
                        for batch_idx in range(batch_size):
                            image_positions = torch.where(image_token_mask[batch_idx])[
                                0
                            ]
                            if len(image_positions) > 0:
                                image_len = min(
                                    len(image_positions), image_embeds.shape[0]
                                )
                                inputs_embeds[
                                    batch_idx, image_positions[:image_len]
                                ] = image_embeds[:image_len]

            hidden_states = inputs_embeds
        else:
            # text ids
            if inputs_embeds is None:
                hidden_states = self.model.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds

        new_past_key_values = [] if use_cache else None
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            layer_past = None
            if (
                use_cache
                and past_key_values is not None
                and isinstance(past_key_values, list)
                and layer_idx < len(past_key_values)
            ):
                layer_past = past_key_values[layer_idx]

            layer_out = decoder_layer.forward_native(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                hidden_states, layer_new_past = layer_out
                new_past_key_values.append(layer_new_past)
            else:
                hidden_states = layer_out

        # 归一化 - SGLang 的 RMSNorm 需要 2D 输入
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        hidden_states = self.model.norm(hidden_states)
        hidden_states = hidden_states.view(orig_shape)

        # 计算 logits
        if hasattr(self.lm_head, "weight"):
            logits = F.linear(hidden_states, self.lm_head.weight)
        else:
            # 如果是 ParallelLMHead，需要特殊处理
            logits = self.lm_head(hidden_states)

        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values if use_cache else past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,  # sglang 版本不输出 attentions
        )


############################
#    Start Qwen2.5Omni     #
############################


class Qwen2_5OmniModel(nn.Module):
    _no_split_modules = [
        "Qwen2_5OmniTalkerForConditionalGeneration",
        "Qwen2_5OmniToken2WavModel",
    ]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(
            config.thinker_config,
            quant_config=quant_config,
            prefix=add_prefix("thinker", prefix),
        )
        self.has_talker = config.enable_audio_output
        self.speaker_map = {}

        config.enable_audio_output = False
        logger.info(f"Talker is not yet supported")
        if config.enable_audio_output:
            self.enable_talker()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        media_token_ids = [mm_inputs.im_token_id, mm_inputs.audio_token_id]
        pattern = MultiModalityDataPaddingPatternMultimodalTokens(media_token_ids)
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def load_speakers(self, path):
        for key, value in torch.load(path).items():
            self.speaker_map[key] = value
        logger.info("Speaker {} loaded".format(list(self.speaker_map.keys())))

    @classmethod
    def can_generate(cls) -> bool:
        return True

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # 1. Generate from thinker module
        thinker_result = self.thinker.forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
        )

        return thinker_result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # VisionAttention
            (".qkv_proj.", ".q.", "q"),
            (".qkv_proj.", ".k.", "k"),
            (".qkv_proj.", ".v.", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "audio_tower" in name or "talker" in name:
                    continue

                if "visual" in name:
                    # mlp
                    if "gate_proj" in name or "up_proj" in name:
                        continue
                    ...

                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    print(f"skipping {name}")
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:

                if "talker" in name or "token2wav" in name:
                    continue
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            logger.warn(
                f"Some weights are not initialized from checkpoints: {sorted(unloaded_params)}"
            )


EntryClass = Qwen2_5OmniModel
