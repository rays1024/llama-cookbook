import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache, Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging, is_torch_flex_attn_available
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from llama_cookbook.utils.token_lookup import build_vec_len_lookup

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)

ALL_VECTORS_PATH = "/p/ruishen/processed_waymo_data/training/waymo_vectorized/all_cluster_centroids_10hz_1024.npy"


def vec_token_mse_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute MSE between the expected 2D action vector implied by VEC logits and the ground-truth 2D vector.
    - Extract logits for all VEC_* tokens, softmax over that subset, and take the probability-weighted
      expected 2D vector using the vec_token_vectors lookup (aligned with vec_token_tensor order).
    - Ground-truth vectors are obtained by mapping VEC labels to their semantic 2D vectors using the same lookup.
    - Mask timesteps using labels to keep only VEC label positions.
    """
    if labels is None:
        raise ValueError("labels are required to compute VEC token MSE loss.")

    device = logits.device
    dtype = logits.dtype

    cache = getattr(vec_token_mse_loss, "_cache", None)
    cache_key = (device, dtype, id(tokenizer))
    if cache is None or cache.get("key") != cache_key:
        vocab = tokenizer.get_vocab()
        vec_token_ids = [tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_")]
        vec_token_tensor = torch.tensor(vec_token_ids, device=device, dtype=torch.long)

        vec_to_angle, _ = build_vec_len_lookup(tokenizer, device=device, dtype=dtype)

        try:
            vec_lookup_np = getattr(vec_token_mse_loss, "_vec_lookup_np")
        except AttributeError:
            try:
                vec_lookup_np = np.load(ALL_VECTORS_PATH, allow_pickle=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Vector lookup file not found at {ALL_VECTORS_PATH}") from e
            vec_token_mse_loss._vec_lookup_np = vec_lookup_np

        vec_lookup = torch.as_tensor(np.asarray(vec_lookup_np, dtype=np.float32)[:, :2], device=device, dtype=dtype)

        vec_angles = vec_to_angle[vec_token_tensor]
        valid_mask = (vec_angles >= 0) & (vec_angles < vec_lookup.size(0))
        vec_token_tensor = vec_token_tensor[valid_mask]
        vec_angles = vec_angles[valid_mask]
        vec_token_vectors = vec_lookup[vec_angles]

        cache = {
            "key": cache_key,
            "vec_token_tensor": vec_token_tensor,
            "vec_token_vectors": vec_token_vectors,
            "vec_lookup": vec_lookup,
            "vec_to_angle": vec_to_angle,
        }
        vec_token_mse_loss._cache = cache

    vec_token_tensor = cache["vec_token_tensor"]
    if vec_token_tensor.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    vec_token_vectors = cache["vec_token_vectors"]
    vec_lookup = cache["vec_lookup"]
    vec_to_angle = cache["vec_to_angle"]

    action_mask = torch.isin(labels, vec_token_tensor)
    if not action_mask.any():
        return torch.tensor(0.0, device=device, dtype=dtype)

    vec_logits = logits[..., vec_token_tensor]
    vec_probs = F.softmax(vec_logits, dim=-1)
    v_pred = vec_probs @ vec_token_vectors

    label_angles = vec_to_angle[labels]
    v_gt = vec_lookup[label_angles[action_mask]]
    v_pred = v_pred[action_mask]

    return F.mse_loss(v_pred, v_gt, reduction=reduction)


class LlamaModelBidirectional(LlamaModel):
    """
    Backbone model subclass that overrides attention-mask construction.
    """

    def __init__(self, config):
        super().__init__(config)
        # optionally add custom config attributes here

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
        **kwargs,
    ):
        mask_type_labels = kwargs.pop("mask_type_labels", None)
        if mask_type_labels is None:
            mask_type_labels = attention_mask
        run_default_logic = kwargs.pop("run_default_logic", True)

        # ********** Start of Default Logic **********
        if run_default_logic and mask_type_labels is None:
            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and (attention_mask == 0.0).any():
                    return attention_mask
                return None
            if self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask)
                return attention_mask

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

            # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
            if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

            dtype = input_tensor.dtype
            sequence_length = input_tensor.shape[1]
            if using_compilable_cache:
                target_length = past_key_values.get_max_cache_shape()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
            )

            if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type in ["cuda", "xpu", "npu"]
                and not output_attentions
            ):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                min_dtype = torch.finfo(dtype).min
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

            return causal_mask
            # ********** End of Default Logic **********

        dtype = input_tensor.dtype
        device = input_tensor.device
        batch_size, sequence_length = input_tensor.shape[:2]
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        mask_seq_length = mask_type_labels.shape[-1] if isinstance(mask_type_labels, torch.Tensor) else 0
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(attention_mask, torch.Tensor):
            target_length = attention_mask.shape[-1]
        elif mask_seq_length:
            target_length = mask_seq_length
        else:
            target_length = past_seen_tokens + sequence_length + 1

        if cache_position.numel() > 0:
            target_length = max(target_length, int(cache_position.max()) + 1)
        target_length = max(target_length, mask_seq_length)

        if not isinstance(mask_type_labels, torch.Tensor):
            raise ValueError("`mask_type_labels` must be provided as a tensor for bidirectional masking.")

        def _pad_or_trim(tensor: torch.Tensor, length: int, pad_value: Union[int, float]) -> torch.Tensor:
            current_length = tensor.shape[-1]
            if current_length == length:
                return tensor
            if current_length > length:
                return tensor[..., :length]
            pad_shape = tensor.shape[:-1] + (length - current_length,)
            pad_tensor = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, pad_tensor], dim=-1)

        orig_attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, target_length), device=device, dtype=torch.bool)
        else:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 2:
                raise ValueError("`attention_mask` must be a 2D tensor when using bidirectional masking.")
            attention_mask = attention_mask.to(device=device)
            attention_mask = _pad_or_trim(attention_mask, target_length, 0)
        attention_mask_bool = attention_mask.to(torch.bool)

        mask_type_labels = mask_type_labels.to(device=device)
        mask_type_labels = _pad_or_trim(mask_type_labels, target_length, 0).to(torch.long)

        non_pad_labels = mask_type_labels[attention_mask_bool]
        has_label_two = (non_pad_labels == 2).any()
        has_label_one = (non_pad_labels == 1).any()
        has_label_zero = (non_pad_labels == 0).any()
        if has_label_two:
            context_label, action_label = 1, 2
        elif has_label_zero and has_label_one:
            context_label, action_label = 0, 1
        else:
            context_label, action_label = 1, 2

        query_positions = cache_position.to(device=device, dtype=torch.long)
        if query_positions.shape[-1] != sequence_length:
            raise ValueError("`cache_position` length must match the query sequence length.")
        expanded_query_positions = query_positions.view(1, -1).expand(batch_size, -1)

        key_is_real = attention_mask_bool
        query_is_real = torch.gather(key_is_real, 1, expanded_query_positions)

        key_is_context = mask_type_labels == context_label
        query_labels = torch.gather(mask_type_labels, 1, expanded_query_positions)
        query_is_context = query_labels == context_label
        query_is_action = query_labels == action_label

        key_positions = torch.arange(target_length, device=device).view(1, 1, -1)
        query_positions_broadcast = expanded_query_positions.unsqueeze(-1)

        context_causal = key_is_context.unsqueeze(1) & (key_positions <= query_positions_broadcast)
        action_full = query_is_action.unsqueeze(-1)

        allowed = (query_is_context.unsqueeze(-1) & context_causal) | action_full
        allowed = allowed & key_is_real.unsqueeze(1) & query_is_real.unsqueeze(-1)

        min_dtype = torch.finfo(dtype).min
        additive_mask = torch.full(
            (batch_size, sequence_length, target_length), fill_value=min_dtype, device=device, dtype=dtype
        )
        additive_mask = additive_mask.masked_fill(allowed, 0).unsqueeze(1)

        real_fully_blocked = query_is_real & (~allowed.any(dim=-1))
        if (
            self.config._attn_implementation == "sdpa"
            and orig_attention_mask is not None
            and additive_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
            and real_fully_blocked.any()
        ):
            additive_mask = AttentionMaskConverter._unmask_unattended(additive_mask, min_dtype)
            # Re-mask padded queries/keys after unmasking safeguard.
            if (~query_is_real).any():
                additive_mask = additive_mask.clone()
                additive_mask[:, :, ~query_is_real, :] = min_dtype
            if (~key_is_real).any():
                additive_mask = additive_mask.clone()
                additive_mask[:, :, :, ~key_is_real] = min_dtype

        return additive_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions, **flash_attn_kwargs 
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForBidirectionAttn(LlamaForCausalLM):
    def __init__(self, config):
        """
        Initialize the wrapper and replace default LlamaModel with 
        your bidirectional subclass.

        Must:
            - Call super().__init__(config) first.
            - Replace self.model with LlamaModelBidirectional(config).
            - Preserve any LM head structure and weight tying.
        """
        super().__init__(config)
        self.model = LlamaModelBidirectional(config)
        # optionally add any additional initialization here

    def compute_vec_token_mse_loss(self, logits, labels, tokenizer, reduction: str = "mean") -> torch.Tensor:
        """
        Convenience wrapper around vec_token_mse_loss for callers that already
        have this model instance and want a VEC-only auxiliary objective.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required to compute vec token MSE loss.")
        return vec_token_mse_loss(logits, labels, tokenizer, reduction=reduction)

    def _input_ids_to_embeds_with_empty_tokens(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Convert input_ids to embeddings, replacing any -1 token with a zeroed embedding
        (keeps position slots alive without injecting token semantics).
        """
        neg_mask = input_ids == -1
        safe_ids = input_ids.clone()
        if neg_mask.any():
            safe_ids = safe_ids.masked_fill(neg_mask, 0)

        embeds = self.model.embed_tokens(safe_ids)
        if neg_mask.any():
            embeds = embeds.masked_fill(neg_mask.unsqueeze(-1), 0.0)
            # # fill in random gaussian noise for empty embeddings instead of zeros
            # noise = torch.randn_like(embeds) * 0.1
            # embeds = embeds.masked_fill(neg_mask.unsqueeze(-1), noise)
        return embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Process input_ids into embeddings (treating -1 as empty embeddings), then call super().forward().
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        inputs_embeds = self._input_ids_to_embeds_with_empty_tokens(input_ids)

        output = super().forward(
            input_ids=None,
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            # inputs_embeds=None,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # tokenizer = kwargs.get("tokenizer", None)

        # logits = output.logits
        # loss = self.compute_vec_token_mse_loss(logits, labels, tokenizer, reduction="mean") if labels is not None else None
        # output.loss = loss * 10
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 0,
        do_sample: bool = False,
        greedy: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_scores: bool = False,
        return_dict_in_generate: Optional[bool] = True,
        **kwargs,
    ):
        """
        Run a single forward pass and sample tokens from the resulting logits (non-autoregressive).
        Only sampling-related args (do_sample, top_p, temperature, top_k, repetition_penalty) are applied;
        other generation args are ignored beyond accepting them for signature compatibility.
        If `greedy` is True, directly picks the highest-probability tokens from the forward pass
        (no sampling or logits processing) and returns them in the same output format.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided for generation.")

        if greedy and do_sample:
            raise ValueError("`greedy=True` cannot be combined with `do_sample=True`.")

        device = input_ids.device
        batch_size, orig_seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, orig_seq_len), dtype=torch.long, device=device)

        logits_processors = LogitsProcessorList()
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        logits_warpers = LogitsProcessorList()
        if do_sample:
            if temperature != 1.0:
                logits_warpers.append(TemperatureLogitsWarper(temperature))
            if top_k is not None and top_k > 0:
                logits_warpers.append(TopKLogitsWarper(top_k))
            if top_p is not None and top_p < 1.0:
                logits_warpers.append(TopPLogitsWarper(top_p))

        forward_kwargs = dict(
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        model_outputs = self.forward(input_ids=input_ids, **forward_kwargs)
        logits = model_outputs.logits

        sequences = input_ids.clone()
        scores = []
        decoder_attentions = None
        decoder_hidden_states = None

        if max_new_tokens < 0:
            raise ValueError("`max_new_tokens` must be non-negative for single-pass generation.")
        if max_new_tokens > logits.size(1):
            raise ValueError(
                f"`max_new_tokens`={max_new_tokens} exceeds available positions in logits (seq_len={logits.size(1)}). "
                "Ensure your input includes slots for tokens to sample."
            )

        start_index = logits.size(1) - max_new_tokens if max_new_tokens > 0 else logits.size(1)

        # Slice attention/hidden states for generated positions once if requested.
        if output_attentions and model_outputs.attentions is not None and max_new_tokens > 0:
            decoder_attentions = []
            for pos in range(start_index, logits.size(1)):
                per_layer = tuple(att[:, :, pos : pos + 1, :] for att in model_outputs.attentions)
                decoder_attentions.append(per_layer)
            decoder_attentions = tuple(decoder_attentions)

        if output_hidden_states and model_outputs.hidden_states is not None and max_new_tokens > 0:
            decoder_hidden_states = []
            for pos in range(start_index, logits.size(1)):
                per_layer = tuple(h[:, pos : pos + 1, :] for h in model_outputs.hidden_states)
                decoder_hidden_states.append(per_layer)
            decoder_hidden_states = tuple(decoder_hidden_states)

        if greedy:
            greedy_tokens = torch.argmax(logits, dim=-1)
            if max_new_tokens > 0:
                sequences[:, start_index:] = greedy_tokens[:, start_index:]
                if output_scores:
                    for pos in range(start_index, logits.size(1)):
                        scores.append(logits[:, pos, :])
            else:
                sequences = greedy_tokens
                if output_scores:
                    scores.append(logits[:, -1, :])

            if not return_dict_in_generate:
                return sequences

            scores_tuple = tuple(scores) if output_scores else None
            return GreedySearchDecoderOnlyOutput(
                sequences=sequences,
                scores=scores_tuple,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )

        for pos in range(start_index, logits.size(1)):
            step_logits = logits[:, pos, :]
            tokens_so_far = sequences[:, :pos]
            if len(logits_processors) > 0:
                step_logits = logits_processors(tokens_so_far, step_logits)
            if len(logits_warpers) > 0:
                step_logits = logits_warpers(tokens_so_far, step_logits)
            if output_scores:
                scores.append(step_logits)
            if do_sample:
                probs = torch.softmax(step_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(step_logits, dim=-1)
            sequences[:, pos] = next_tokens

        if max_new_tokens == 0 and output_scores:
            scores.append(logits[:, -1, :])

        if not return_dict_in_generate:
            return sequences

        scores_tuple = tuple(scores) if output_scores else None
        if do_sample:
            return SampleDecoderOnlyOutput(
                sequences=sequences,
                scores=scores_tuple,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        return GreedySearchDecoderOnlyOutput(
            sequences=sequences,
            scores=scores_tuple,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
        )
