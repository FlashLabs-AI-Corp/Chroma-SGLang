from transformers import PretrainedConfig
from transformers.utils import logging

from chroma.qwen2_5_omni_config import Qwen2_5OmniThinkerConfig

logger = logging.get_logger(__name__)


class ChromaBackboneConfig(PretrainedConfig):
    model_type = "chroma_backbone"

    def __init__(
        self,
        audio_num_codebooks=32,
        vocab_size=2051,
        max_position_embeddings=2048,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        rope_theta=500000,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 0.5,
            "low_freq_factor": 0.125,
            "original_max_position_embeddings": 1024,
            "rope_type": "llama3",
        },
        head_dim=64,
        return_dict=True,
        torch_dtype="float32",
        max_length=20,
        num_beams=1,
        num_beam_groups=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        typical_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        architectures=["ChromaForConditionalGeneration"],
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_num_codebooks = audio_num_codebooks
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        self.return_dict = return_dict
        self.torch_dtype = torch_dtype
        self.max_length = max_length
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.num_return_sequences = num_return_sequences


class ChromaDecoderConfig(PretrainedConfig):
    model_type = "chroma_decoder"

    def __init__(
        self,
        audio_num_codebooks=32,
        audio_embedding_dim=2048,
        vocab_size=2051,
        max_position_embeddings=33,
        hidden_size=1024,
        intermediate_size=8192,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        rope_theta=500000,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 0.0078125,
            "low_freq_factor": 0.001953125,
            "original_max_position_embeddings": 16,
            "rope_type": "llama3",
        },
        head_dim=128,
        return_dict=True,
        torch_dtype="float32",
        max_length=20,
        num_beams=1,
        num_beam_groups=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        typical_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_num_codebooks = audio_num_codebooks
        self.audio_embedding_dim = audio_embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim
        self.return_dict = return_dict
        self.torch_dtype = torch_dtype
        self.max_length = max_length
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.num_return_sequences = num_return_sequences


class ChromaConfig(PretrainedConfig):
    model_type = "chroma"

    def __init__(
        self,
        thinker_config=None,
        backbone_config=None,
        decoder_config=None,
        codec_config=None,
        loss_stride=16,
        decoder_loss_weight=0.5,
        codebook_pad_token_id=2050,
        codebook_eos_token_id=0,
        audio_num_codebooks=8,
        **kwargs
    ):
        # print(f'thinker_config: {thinker_config}')
        # print(f'backbone_config: {backbone_config}')
        # print(f'decoder_config: {decoder_config}')
        # print(f'codec_config: {codec_config}')
        # print(f'loss_stride: {loss_stride}')
        # print(f'decoder_loss_weight: {decoder_loss_weight}')
        # print(f'codebook_pad_token_id: {codebook_pad_token_id}')
        # print(f'codebook_eos_token_id: {codebook_eos_token_id}')
        # print(f'audio_num_codebooks: {audio_num_codebooks}')
        if isinstance(thinker_config, dict):
            thinker_config = Qwen2_5OmniThinkerConfig(**thinker_config)

        if isinstance(backbone_config, dict):
            backbone_config = ChromaBackboneConfig(**backbone_config)
        elif backbone_config is None:
            # logger.warning("backbone_config is None, using default backbone config.")
            backbone_config = ChromaBackboneConfig(
                audio_num_codebooks=audio_num_codebooks
            )

        if isinstance(decoder_config, dict):
            decoder_config = ChromaDecoderConfig(**decoder_config)
        elif decoder_config is None:
            # logger.warning("decoder_config is None, using default decoder config.")
            decoder_config = ChromaDecoderConfig(
                audio_num_codebooks=audio_num_codebooks
            )

        if codec_config is None:
            codec_config = {"audio_num_codebooks": audio_num_codebooks}

        self.thinker_config = thinker_config
        self.backbone_config = backbone_config
        self.decoder_config = decoder_config
        self.codec_config = codec_config
        self.audio_num_codebooks = audio_num_codebooks

        self.loss_stride = loss_stride
        self.decoder_loss_weight = decoder_loss_weight
        self.codebook_pad_token_id = codebook_pad_token_id
        self.codebook_eos_token_id = codebook_eos_token_id

        super().__init__(**kwargs)

    @property
    def num_hidden_layers(self):
        if hasattr(self, 'backbone_config') and self.backbone_config:
            return self.backbone_config.num_hidden_layers
        return None