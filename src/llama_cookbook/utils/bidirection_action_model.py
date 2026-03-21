import torch
from typing import Any, Optional

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput

from llama_cookbook.utils.action_model import LlamaForCausalLMWithActions
from llama_cookbook.utils.bidirection_attn_llama import LlamaModelBidirectional


class LlamaForBidirectionAttnWithActions(LlamaForCausalLMWithActions):
    """
    Combines bidirectional attention with the action head. Uses the bidirectional
    backbone while keeping the action-head training/generation interface.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelBidirectional(config)
        self.tie_weights()
        self._align_action_head_dtype()

    def _input_ids_to_embeds_with_empty_tokens(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Convert input_ids to embeddings, replacing any -1 token with a zeroed embedding
        so action placeholders keep their positions without injecting token meaning.
        """
        neg_mask = input_ids == -1
        safe_ids = input_ids.clone()
        if neg_mask.any():
            safe_ids = safe_ids.masked_fill(neg_mask, 0)
        
        embeds = self.model.embed_tokens(safe_ids)
        if neg_mask.any():
            embeds = embeds.masked_fill(neg_mask.unsqueeze(-1), 0.0)
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
        mask_type_labels: Optional[torch.Tensor] = None,
        task: str = "language",
        loss_type: Optional[str] = None,
        loss_horizon: Optional[int] = None,
        pred_seq=None,
        use_ce_loss: bool = True,
        **kwargs,
    ):
        """
        Route inputs through the bidirectional backbone while preserving action-head logic.
        Accepts `mask_type_labels` for constructing bidirectional attention masks.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")

        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self._input_ids_to_embeds_with_empty_tokens(input_ids)
            input_ids = None

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task,
            mask_type_labels=mask_type_labels,
            loss_type=loss_type,
            loss_horizon=loss_horizon,
            pred_seq=pred_seq,
            use_ce_loss=use_ce_loss,
            **kwargs,
        )

    def _decode_action_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode each token embedding independently and return a full action sequence.
        """
        batch_size, seq_len, hidden_dim = embeddings.size()
        required_tokens = getattr(self, "_action_token_count", self.horizon)
        if seq_len != required_tokens:
            raise ValueError(
                f"Sequence length ({seq_len}) must equal action token count ({required_tokens}) "
                "when using parallel action decoding."
            )
        reshaped = embeddings.reshape(batch_size * seq_len, hidden_dim)
        decoded = self.action_decoder(reshaped)
        decoded = decoded.reshape(batch_size, seq_len, self._action_chunk_size, self._action_dim)
        actions = decoded.reshape(batch_size, seq_len * self._action_chunk_size, self._action_dim)
        if actions.size(1) > self.horizon:
            actions = actions[:, : self.horizon, :]
        return actions

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
        Single-pass non-autoregressive generation using bidirectional attention.
        Mirrors the bidirectional generator while keeping the action head attached.
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

    @torch.no_grad()
    def action_head_based_generate_actions(
        self,
        input_ids: Optional[torch.LongTensor],
        tokenizer: Any = None,
        attention_mask: Optional[torch.LongTensor] = None,
        mask_type_labels: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        return_generation_output: bool = False,
        **forward_kwargs,
    ):
        """
        Parallel action decoding: runs one forward pass and decodes each action token per step.
        When MoN is enabled, returns all MoN candidates as an expanded batch:
        (batch * mon_num_samples, horizon * action_dim).
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided to generate actions.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        elif attention_mask.size() != input_ids.size():
            raise ValueError("`attention_mask` must have the same shape as `input_ids`.")

        required_steps = self.horizon if max_new_tokens is None else max_new_tokens
        if required_steps < self.horizon:
            raise ValueError(
                f"`max_new_tokens` must be at least the action horizon ({self.horizon}). Received {required_steps}."
            )

        forward_kwargs = dict(forward_kwargs)
        forward_kwargs.setdefault("output_hidden_states", True)
        forward_kwargs.setdefault("return_dict", True)

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_type_labels=mask_type_labels,
            task="action",
            **forward_kwargs,
        )

        flattened_actions = outputs.action_head_output
        if flattened_actions is None:
            raise RuntimeError("Action head did not produce any output during generation.")
        if self._use_mon:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None or len(hidden_states) == 0:
                raise RuntimeError("MoN generation requires hidden states from forward().")
            final_hidden = hidden_states[-1]
            if not isinstance(final_hidden, torch.Tensor):
                raise RuntimeError("Unexpected hidden-state format for MoN generation.")

            action_embeddings = self._select_action_token_embeddings(
                hidden_states=final_hidden,
                attention_mask=attention_mask,
                labels=None,
            )
            sampled_embeddings = self._sample_mon_action_embeddings(action_embeddings)
            sample_count, batch_size, token_count, hidden_dim = sampled_embeddings.size()
            sampled_embeddings = sampled_embeddings.reshape(
                sample_count * batch_size,
                token_count,
                hidden_dim,
            )
            sampled_actions = self._decode_action_embeddings(sampled_embeddings)
            flattened_actions = sampled_actions.reshape(sample_count, batch_size, -1).permute(1, 0, 2).reshape(
                batch_size * sample_count,
                -1,
            )
            outputs["mon_num_samples"] = sample_count

        action_dim = self._action_dim
        predicted_steps = flattened_actions.size(1) // action_dim if action_dim > 0 else 0
        if required_steps > predicted_steps:
            raise ValueError(
                f"Requested {required_steps} action steps but only {predicted_steps} available from action head."
            )
        if required_steps < predicted_steps:
            flattened_actions = flattened_actions[:, : required_steps * action_dim]

        if return_generation_output:
            return flattened_actions, outputs
        return flattened_actions
