import os
import time
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass

import torch
from transformers import (
    GenerationConfig,
    MaxLengthCriteria,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation import GenerationMode, GenerationMixin
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateDecoderOnlyOutput,
)

os.environ["TOKENIZERS_PARALLELISM"] = "0"


def multinomial_sample_one_no_sync(probs):
    """Does multinomial sampling without a cuda synchronization."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """Sample from logits using top-k sampling with temperature."""
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ChromaGenerateOutput(GenerateDecoderOnlyOutput):
    audio: Optional[List[torch.Tensor]] = None


class ChromaGenerationMixin(GenerationMixin):
    def _get_stopping_criteria(self, *args, **kwargs):
        return super()._get_stopping_criteria(*args, **kwargs)

    def _prepare_generation_config(
        self,
        generation_config: Optional[GenerationConfig],
        use_model_defaults: Optional[bool] = None,
        **kwargs: Dict,
    ) -> Tuple[GenerationConfig, Dict]:
        """
        This method overrides [~generation.utils.GenerationMixin._prepare_generation_config].
        It ensures that the decoder generation config is initialized and that passed args as depth_decoder_* are properly handled.
        """
        # extract depth decoder kwargs and remove them from the main kwargs
        depth_decoder_kwargs = {
            k[len("decoder_") :]: v
            for k, v in kwargs.items()
            if k.startswith("decoder_")
        }

        # remove the decoder keys from the original kwargs
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("decoder_")}

        # initialize the generation config
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self.decoder.generation_config.update(**depth_decoder_kwargs)

        # ensure the depth decoder generation config is valid
        decoder_min_new_tokens = getattr(
            self.decoder.generation_config, "min_new_tokens"
        ) or (self.decoder.config.audio_num_codebooks - 1)
        decoder_max_new_tokens = getattr(
            self.decoder.generation_config, "max_new_tokens"
        ) or (self.decoder.config.audio_num_codebooks - 1)

        if {decoder_min_new_tokens, decoder_max_new_tokens} != {
            self.decoder.config.audio_num_codebooks - 1
        }:
            raise ValueError(
                f"depth_decoder_generation_config's min_new_tokens ({decoder_min_new_tokens}) and max_new_tokens ({decoder_max_new_tokens}) must be equal to self.config.num_codebooks - 1 ({self.decoder.config.audio_num_codebooks - 1})"
            )
        elif self.decoder.generation_config.return_dict_in_generate:
            self.decoder.generation_config.return_dict_in_generate = False

        # Monkey patch the get_generation_mode method to support CSM model
        original_get_generation_mode = generation_config.get_generation_mode

        def patched_get_generation_mode(assistant_model=None):
            generation_mode = original_get_generation_mode(assistant_model)
            if generation_mode not in [
                GenerationMode.GREEDY_SEARCH,
                GenerationMode.SAMPLE,
            ]:
                raise ValueError(
                    f"Generation mode {generation_mode} is not supported for CSM model. Please set generation parameters to use greedy or sampling generation."
                )

            return generation_mode

        generation_config.get_generation_mode = patched_get_generation_mode

        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        Args:
            input_ids:
            generation_config:
            logits_processor:
            stopping_criteria:
            synced_gpus:
            **model_kwargs:

        Returns:

        """
        pad_token_id = self.config.codebook_pad_token_id
        has_eos_stopping_criteria = generation_config._eos_token_tensor is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        top_k = generation_config.top_k
        temperature = generation_config.temperature

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(
            cur_len, input_ids.device, model_kwargs
        )

        if input_ids.ndim == 2 and model_kwargs.get("inputs_embeds") is None:
            # in the case where the passed input_ids correspond to text tokens, i.e. don't have a third dimension for codebook ids,
            # we need to remove the input length to the MaxLengthCriteria stopping criteria has such input are not returned
            for criterion in stopping_criteria:
                if isinstance(criterion, MaxLengthCriteria):
                    criterion.max_length -= cur_len

        generated_frames = []

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(
            model_kwargs, generation_config
        )
        if compile_forward:
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
        ):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            model_inputs.update({"output_attentions": output_attentions})
            model_inputs.update({"output_hidden_states": True})

            if is_prefill:
                backbone_outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                backbone_outputs = model_forward(**model_inputs, return_dict=True)

            next_token_logits = backbone_outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            backbone_last_hidden_state = backbone_outputs.hidden_states[-1][:, -1, :]

            model_kwargs = self._update_model_kwargs_for_generation(
                backbone_outputs,
                model_kwargs,
            )

            if synced_gpus and this_peer_finished:
                continue

            # Acquire backbone scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (backbone_outputs.attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (backbone_outputs.hidden_states,)

            # Do sample
            if do_sample:
                next_tokens = sample_topk(next_token_logits, top_k, temperature)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                next_tokens = next_tokens.unsqueeze(0)

            with torch.inference_mode():
                frame_codes = self.decoder.generate(
                    input_ids=next_tokens,
                    backbone_last_hidden_state=backbone_last_hidden_state.clone(),
                    max_new_tokens=self.config.decoder_config.audio_num_codebooks - 1,
                    min_new_tokens=self.config.decoder_config.audio_num_codebooks - 1,
                    do_sample=True,
                    use_cache=True,
                    temperature=temperature,
                    top_k=top_k,
                )

            if frame_codes.shape[-1] != self.config.decoder_config.audio_num_codebooks:
                raise ValueError(
                    f"Generated codebooks shape {frame_codes.shape[-1]} does not match expected "
                    f"audio_num_codebooks {self.config.decoder_config.audio_num_codebooks}"
                )

            next_tokens = frame_codes

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(
                    -1
                ) + pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))

            if next_tokens.sum() != 0:
                generated_frames.append(next_tokens.unsqueeze(1))

            input_ids = next_tokens[:, None, :]

            unfinished_sequences = unfinished_sequences & ~(
                input_ids[:, -1, :-1] == self.config.codebook_eos_token_id
            ).all(-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                torch.cat(generated_frames, dim=1), scores
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del backbone_outputs

            del frame_codes

        if return_dict_in_generate:
            sequences = (
                torch.cat(generated_frames, dim=1)
                if len(generated_frames) > 0
                else input_ids
            )
            return GenerateDecoderOnlyOutput(
                sequences=sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            sequences = (
                torch.cat(generated_frames, dim=1)
                if len(generated_frames) > 0
                else input_ids
            )
            return sequences

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        output_audio: Optional[bool] = False,
        bos_token_id: Optional[int] = 0,
        **kwargs: dict,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        custom generation sampling step:
            1. Infer the backbone model to sample the first codebook token
            2. Call generate on the decoder with the first codebook token as `input_ids` to sample the next codebook tokens
            3. Use these generated codebook tokens as `input_ids` to sample the next first codebook token using the backbone model
            4. Repeat until stopping criteria is met

        Args:
            input_ids:
            input_values:
            input_values_cutoffs:
            generation_config:
            logits_processor:
            stopping_criteria:
            synced_gpus:
            output_audio:
            bos_token_id:
            **kwargs:

        Returns:

        """
        s_time = time.time()
        generate_output = super().generate(
            input_ids=input_ids,
            input_values=input_values,
            input_values_cutoffs=input_values_cutoffs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            synced_gpus=synced_gpus,
            bos_token_id=bos_token_id,
            **kwargs,
        )
        e_time = time.time()
        print(f"Generation time: {e_time - s_time}")

        generate_returned_dict = not isinstance(generate_output, torch.Tensor)
        audio = None
        if output_audio:
            generated_audio_codes = (
                generate_output.sequences if generate_returned_dict else generate_output
            )

            # infer the codec model
            audio = []
            with torch.no_grad():
                for audio_codes_batch in generated_audio_codes:
                    eos_idxs = (
                        (audio_codes_batch == self.config.codebook_eos_token_id)
                        .all(dim=-1)
                        .nonzero()
                    )
                    if eos_idxs.numel() != 0:
                        cutoff_idx = eos_idxs.min()
                    else:
                        cutoff_idx = audio_codes_batch.shape[0]

                    audio_codes_batch = audio_codes_batch[:cutoff_idx]
                    codec_decode_output = self.codec_model.decode(
                        audio_codes_batch.transpose(0, 1).unsqueeze(0)
                    )
                    audio.append(codec_decode_output)

        if generate_returned_dict:
            return ChromaGenerateOutput(audio=audio, **generate_output)
        elif output_audio:
            return audio
        else:
            return generate_output
