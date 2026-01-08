# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import dataclasses
from typing import Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

all_centroids = np.load('/p/ruishen/processed_waymo_data/training/waymo_vectorized/all_cluster_centroids_5hz_1024.npy', allow_pickle=True)

def top_p_filtering(logits, top_p=0.9, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Apply mask
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[0, indices_to_remove] = filter_value
    return logits


@dataclasses.dataclass
class CausalLMOutputWithPastAndActions(CausalLMOutputWithPast):
    action_head_output: Optional[torch.FloatTensor] = None
    action_loss: Optional[torch.FloatTensor] = None
    language_model_loss: Optional[torch.FloatTensor] = None
    cross_entropy_loss: Optional[torch.FloatTensor] = None
    action_prediction_loss: Optional[torch.FloatTensor] = None
    vec_order_loss: Optional[torch.FloatTensor] = None


class MLPResNetBlock(nn.Module):
    """Residual MLP block with pre-layer normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.ffn(x)
        return x + identity


class MLPResNet(nn.Module):
    """Multi-layer perceptron composed of residual blocks."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        return self.fc2(x)

class LlamaForCausalLMWithActions(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.horizon = int(getattr(config, "action_head_horizon", 1))
        if self.horizon <= 0:
            raise ValueError("`action_head_horizon` must be a positive integer.")

        action_hidden_dim = int(getattr(config, "action_head_hidden_dim", config.hidden_size))
        action_num_layers = int(getattr(config, "action_head_num_layers", 2))
        self._default_loss_type = str(getattr(config, "action_head_loss_type", "mse")).lower()
        self._action_dim = int(getattr(config, "action_head_output_dim", 2))
        if self._action_dim <= 0:
            raise ValueError("`action_head_output_dim` must be a positive integer.")
        self._action_hidden_dim = action_hidden_dim
        self._action_num_layers = action_num_layers
        self._action_input_dim = int(config.hidden_size)
        self._action_head_arch = str(getattr(config, "action_head_arch", "resnet")).lower() # resnet or mlp

        self.action_decoder = self._build_action_decoder(
            input_dim=config.hidden_size,
            hidden_dim=action_hidden_dim,
            num_layers=action_num_layers,
            action_dim=self._action_dim,
        )

        self.reset_action_head_parameters()
        self.action_decoder.to(dtype=self.model.embed_tokens.weight.dtype,
                            device=self.model.embed_tokens.weight.device)

        self.config.action_head_output_dim = self._action_dim
        self.config.action_head_hidden_dim = action_hidden_dim
        self.config.action_head_num_layers = action_num_layers
        self.config.action_head_loss_type = self._default_loss_type
        self.config.action_head_horizon = self.horizon
        self.config.use_action_head = True
        self.config.action_head_arch = self._action_head_arch
        self.token_id_to_centroid = None

    def init_token_id_to_centroid(self, tokenizer):
        global all_centroids
        token_id_to_centroid = {}
        for idx in range(all_centroids.shape[0]):
            token_str = f"VEC_{idx}"
            token_id = tokenizer.encode(token_str, add_special_tokens=False)[0]
            token_id_to_centroid[token_id] = all_centroids[idx][:2]  # only use x, y
        all_vec_token_ids = list(token_id_to_centroid.keys())
        token_id_to_centroid["all_vec_token_ids"] = all_vec_token_ids
        self.token_id_to_centroid = token_id_to_centroid

    def _build_action_decoder(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        action_dim: int,
    ) -> nn.Module:
        problems = []
        if input_dim <= 0:
            problems.append(("input_dim", input_dim))
        if hidden_dim <= 0:
            problems.append(("hidden_dim", hidden_dim))
        if action_dim <= 0:
            problems.append(("action_dim", action_dim))
        if problems:
            details = ", ".join(f"{name}={value}" for name, value in problems)
            raise ValueError(
                f"Action decoder configuration produced non-positive dimensions: {details}"
            )

        num_blocks = max(num_layers, 0)
        arch = getattr(self, "_action_head_arch", "resnet") # resnet or mlp
        if arch == "resnet":
            # return MLPResNet(num_blocks=num_blocks, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim)

            # return nn.Linear(input_dim, action_dim, bias=False)
            
            depth = max(num_layers, 4)

            # define layers explicitly
            fc1 = nn.Linear(input_dim, hidden_dim)
            fc2 = nn.Linear(hidden_dim, hidden_dim)
            fc3 = nn.Linear(hidden_dim, hidden_dim)
            fc4 = nn.Linear(hidden_dim, action_dim)

            act = nn.SiLU()

            # helper function that emulates residuals
            def forward_fn(x):
                x = act(fc1(x))

                # two residual blocks, matching depth
                residual = x
                out = act(fc2(x))
                x = residual + 0.3 * out

                residual = x
                out = act(fc3(x))
                x = residual + 0.3 * out

                x = fc4(x)
                return x

            # wrap into a lightweight nn.Module so Sequential-like use still works
            class _Wrapper(nn.Module):
                def forward(self, x):
                    return forward_fn(x)

            wrapper = _Wrapper()
            # register parameters manually so optimizer can see them
            wrapper.fc1, wrapper.fc2, wrapper.fc3, wrapper.fc4 = fc1, fc2, fc3, fc4

            return wrapper

        if arch == "mlp":
            depth = max(num_layers, 4)
            layers: list[nn.Module] = []
            in_dim = input_dim
            for layer_idx in range(depth - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.SiLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, action_dim))
            return nn.Sequential(*layers)

        raise ValueError(
            f"Unsupported action head architecture '{arch}'. Expected 'resnet' or 'mlp'."
        )

    def reset_action_head_parameters(self):
        self.action_decoder.to_empty(device="cuda" if torch.cuda.is_available() else "cpu")
        self.action_decoder.apply(self._init_weights)
        self._align_action_head_dtype()

    def _align_action_head_dtype(self):
        try:
            base_param = next(self.model.parameters())
            base_dtype = base_param.dtype
        except StopIteration:
            base_dtype = torch.get_default_dtype()

        self.action_decoder.to(dtype=base_dtype)

    def _calculate_vector_order_loss(self, base_outputs: CausalLMOutputWithPast, labels: Optional[torch.Tensor]):
        if (
            base_outputs is None
            or base_outputs.logits is None
            or labels is None
            or self.token_id_to_centroid is None
        ):
            return None

        vec_token_ids = self.token_id_to_centroid.get("all_vec_token_ids", [])
        if not vec_token_ids:
            return None

        logits = base_outputs.logits
        if logits.ndim != 3:
            return None

        batch, seq_len, vocab_size = logits.size()
        if seq_len <= 1:
            return None

        device = logits.device
        labels = labels.to(device=device)

        # Align with the teacher-forcing shift (logits predict the next token).
        if labels.ndim != 2 or labels.size(0) != batch:
            return None
        if labels.size(1) != seq_len:
            min_seq = min(labels.size(1), seq_len)
            logits = logits[:, :min_seq, :]
            labels = labels[:, :min_seq]
            batch, seq_len, vocab_size = logits.size()
            if seq_len <= 1:
                return None

        shifted_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shifted_labels = labels[:, 1:].contiguous().view(-1)

        valid_mask = shifted_labels.ne(-100)
        if not torch.any(valid_mask):
            return None

        filtered_logits = shifted_logits[valid_mask]
        filtered_labels = shifted_labels[valid_mask]

        # Build tensors containing the ids and centroids of vec tokens that fall inside the vocab range.
        filtered_vec_ids = []
        vec_centroids = []
        for token_id in vec_token_ids:
            if (
                isinstance(token_id, int)
                and 0 <= token_id < vocab_size
                and token_id in self.token_id_to_centroid
            ):
                filtered_vec_ids.append(token_id)
                vec_centroids.append(self.token_id_to_centroid[token_id])

        if not filtered_vec_ids:
            return None

        vec_token_tensor = torch.tensor(filtered_vec_ids, dtype=torch.long, device=device)
        centroid_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        centroid_tensor = torch.tensor(vec_centroids, dtype=centroid_dtype, device=device)

        vec_token_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        vec_token_mask[vec_token_tensor] = True

        vec_label_mask = vec_token_mask[filtered_labels]
        if not torch.any(vec_label_mask):
            return None

        vec_logits = filtered_logits[vec_label_mask]
        vec_labels = filtered_labels[vec_label_mask]

        if vec_logits.numel() == 0:
            return None

        gathered_logits = torch.index_select(vec_logits, dim=-1, index=vec_token_tensor)

        id_to_index = {token_id: idx for idx, token_id in enumerate(filtered_vec_ids)}
        label_indices = torch.tensor(
            [id_to_index[int(token_id)] for token_id in vec_labels.tolist()],
            dtype=torch.long,
            device=device,
        )
        gt_centroids = centroid_tensor[label_indices]

        diffs = centroid_tensor.unsqueeze(0) - gt_centroids.unsqueeze(1)
        distances = torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-9)
        target_scores = -distances

        alpha = 20.0
        weights = torch.exp(-alpha * distances)
        distance_weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        work_dtype = logits.dtype
        gathered_logits = gathered_logits.to(dtype=work_dtype)
        target_scores = target_scores.to(dtype=work_dtype)
        distance_weights = distance_weights.to(dtype=work_dtype)

        def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
            mean = values.mean(dim=-1, keepdim=True)
            var = values.var(dim=-1, unbiased=False, keepdim=True)
            std = torch.sqrt(var + 1e-6)
            return (values - mean) / std

        norm_logits = _normalize_rows(gathered_logits)
        norm_targets = _normalize_rows(target_scores)

        squared_error = (norm_logits - norm_targets) ** 2
        loss = (distance_weights * squared_error).sum(dim=-1).mean()

        return loss.to(dtype=logits.dtype)

    def set_language_model_trainable(self, trainable: bool) -> None:
        """Enable or disable gradient updates for the base language model (backbone + lm_head)."""
        for parameter in self.model.parameters():
            parameter.requires_grad = trainable
        if hasattr(self, "lm_head"):
            for parameter in self.lm_head.parameters():
                parameter.requires_grad = trainable

    def set_action_head_trainable(self, trainable: bool) -> None:
        """Enable or disable gradient updates for the action decoder head."""
        for parameter in self.action_decoder.parameters():
            parameter.requires_grad = trainable

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
        task: str = "language",
        loss_type: Optional[str] = None,
        loss_horizon: Optional[int] = None,
        use_multistep_loss: bool = False,
        pred_seq=None,
        use_ce_loss: bool = True,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if task == "language":
            effective_output_hidden_states = output_hidden_states or self.config.output_hidden_states
            base_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=effective_output_hidden_states,
                return_dict=True,
                **kwargs,
            )
            if not return_dict:
                return tuple(value for value in base_outputs.values() if value is not None)
            return base_outputs

        effective_output_hidden_states = True
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        vec_order_loss = self._calculate_vector_order_loss(base_outputs, labels)

        hidden_states = base_outputs.hidden_states[-1]

        action_embeddings = self._select_action_token_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels,
        )

        decoded_actions = self._decode_action_embeddings(action_embeddings)
        flattened_actions = decoded_actions.reshape(decoded_actions.size(0), -1)

        action_loss = None
        multistep_losses = None
        if pred_seq is not None:
            effective_loss_type = (loss_type or self._default_loss_type).lower()
            action_loss, multistep_losses = self._compute_action_loss(
                flattened_actions,
                pred_seq,
                effective_loss_type,
                loss_horizon,
                use_multistep_loss=use_multistep_loss,
            )

        ce_loss = base_outputs.loss if (use_ce_loss and base_outputs.loss is not None) else None
        combined_loss = action_loss
        if ce_loss is not None:
            ce_loss = ce_loss.to(dtype=combined_loss.dtype if combined_loss is not None else ce_loss.dtype)
            combined_loss = ce_loss if combined_loss is None else combined_loss + ce_loss
            if vec_order_loss is not None:
                vec_order_loss = vec_order_loss.to(dtype=combined_loss.dtype) * 50.0
                combined_loss = combined_loss + vec_order_loss

        action_outputs = CausalLMOutputWithPastAndActions(
            loss=combined_loss,
            logits=None if task == "action" else base_outputs.logits,
            past_key_values=base_outputs.past_key_values,
            hidden_states=(base_outputs.hidden_states if effective_output_hidden_states else None),
            attentions=base_outputs.attentions,
            action_head_output=flattened_actions,
        )

        action_outputs["action_loss"] = combined_loss
        action_outputs["language_model_loss"] = base_outputs.loss
        action_outputs["cross_entropy_loss"] = ce_loss
        action_outputs["action_prediction_loss"] = action_loss
        if multistep_losses is not None:
            action_outputs["action_per_step_loss"] = multistep_losses.get("per_step")
            action_outputs["action_multistep_loss_5"] = multistep_losses.get("step_5")
            action_outputs["action_multistep_loss_10"] = multistep_losses.get("step_10")
            action_outputs["action_multistep_loss_20"] = multistep_losses.get("step_20")
        action_outputs["vec_order_loss"] = vec_order_loss

        if not return_dict:
            return tuple(value for value in action_outputs.values() if value is not None)
        return action_outputs

    def _select_action_token_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.size()
        device = hidden_states.device

        if labels is not None:
            if labels.dim() != 2 or labels.size(0) != batch_size or labels.size(1) != seq_len:
                raise ValueError(
                    "`labels` must have shape (batch, seq_len) and align with hidden states when task='action'."
                )
            label_mask = labels.ne(-100)
            valid_counts = label_mask.sum(dim=1)
            if torch.any(valid_counts < self.horizon):
                raise ValueError(
                    "Some sequences provide fewer labeled tokens than the configured action horizon. "
                    f"Minimum labeled count: {int(valid_counts.min().item())}, required: {self.horizon}."
                )
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            masked_positions = positions.masked_fill(~label_mask, seq_len)
            sorted_positions, _ = torch.sort(masked_positions, dim=1)
            select_positions = sorted_positions[:, : self.horizon]
            gather_positions = select_positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
            return torch.gather(hidden_states, dim=1, index=gather_positions)

        if attention_mask is None:
            if seq_len < self.horizon:
                raise ValueError(
                    f"Sequence length ({seq_len}) must be at least the configured horizon ({self.horizon})."
                )
            return hidden_states[:, -self.horizon :, :]

        normalized_mask = self._normalize_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        valid_lengths = normalized_mask.sum(dim=1)
        if torch.any(valid_lengths < self.horizon):
            raise ValueError(
                "Some sequences are shorter than the configured action horizon. "
                f"Minimum length in batch: {int(valid_lengths.min().item())}, "
                f"required: {self.horizon}."
            )

        base_positions = valid_lengths - self.horizon
        offsets = torch.arange(self.horizon, device=device).unsqueeze(0)
        gather_positions = base_positions.unsqueeze(1) + offsets
        gather_positions = gather_positions.clamp(min=0, max=seq_len - 1)
        gather_positions = gather_positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return torch.gather(hidden_states, dim=1, index=gather_positions)

    def _normalize_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        if attention_mask.dim() != 2:
            raise ValueError("`attention_mask` must have shape (batch, seq_len).")
        mask = attention_mask.to(device=device)
        if mask.dtype != torch.long:
            mask = mask.to(dtype=torch.long)
        return mask

    def _decode_action_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, hidden_dim = embeddings.size()
        reshaped = embeddings.reshape(batch_size * horizon, hidden_dim)

        decoded = self.action_decoder(reshaped)

        return decoded.reshape(batch_size, horizon, self._action_dim)

    def _convert_actions_to_token_tensor(
        self,
        actions: torch.Tensor,
        tokenizer: Any,
        device: torch.device,
    ) -> torch.LongTensor:
        global all_centroids

        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided when using the action head for generation.")
        if actions.dim() != 2:
            raise ValueError(f"`actions` must have shape (batch, action_dim); received {tuple(actions.size())}.")

        token_ids = []
        actions_np = actions.detach().to("cpu").numpy()
        for action in actions_np:
            diffs = all_centroids[:, :2] - action
            dists = np.linalg.norm(diffs, axis=1)
            nearest_id = int(np.argmin(dists))
            token_id = tokenizer.encode(f"VEC_{nearest_id}", add_special_tokens=False)
            token_ids.append(token_id[0])
        return torch.tensor(token_ids, dtype=torch.long, device=device)


    def _compute_action_loss(
        self,
        action_output: torch.Tensor,
        target_seq: torch.Tensor,
        loss_type: str,
        loss_horizon: Optional[int],
        use_multistep_loss: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, Optional[torch.Tensor]]]]:
        target_seq = target_seq.to(device=action_output.device)
        horizon, action_dim = self._infer_horizon_and_dim(action_output, target_seq)
        steps = horizon
        if loss_horizon is not None:
            if loss_horizon <= 0:
                raise ValueError("`loss_horizon` must be a positive integer when provided.")
            steps = min(loss_horizon, horizon)

        if loss_type == "mse":
            if not use_multistep_loss:
                preds_bf16, targets_bf16 = self._prepare_mse_tensors(action_output, target_seq, steps, action_dim)
                loss = F.mse_loss(preds_bf16, targets_bf16)
                return loss.to(dtype=action_output.dtype), None
            loss, multistep_losses = self._compute_multistep_loss(
                action_output,
                target_seq,
                steps,
                action_dim,
                loss_type,
            )
            return loss.to(dtype=action_output.dtype), multistep_losses
        elif loss_type == "l1":
            if not use_multistep_loss:
                preds_bf16, targets_bf16 = self._prepare_mse_tensors(action_output, target_seq, steps, action_dim)
                loss = torch.mean(torch.abs(preds_bf16 - targets_bf16))
                return loss.to(dtype=action_output.dtype), None
            loss, multistep_losses = self._compute_multistep_loss(
                action_output,
                target_seq,
                steps,
                action_dim,
                loss_type,
            )
            return loss.to(dtype=action_output.dtype), multistep_losses
        elif loss_type == "ade":
            loss = self._compute_ade_loss(action_output, target_seq, horizon, action_dim, steps)
            return loss.to(dtype=action_output.dtype), None
        else:
            raise ValueError(f"Unsupported action loss type '{loss_type}'. Use 'l1', 'mse', or 'ade'.")

    def _prepare_mse_tensors(
        self,
        action_output: torch.Tensor,
        target_seq: torch.Tensor,
        steps: int,
        action_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_sliced = action_output[:, : steps * action_dim]
        pred_sliced = torch.nan_to_num(pred_sliced, nan=0.0, posinf=1e6, neginf=-1e6)
        if target_seq.dim() == 3:
            target_sliced = target_seq[:, :steps, :].reshape(target_seq.size(0), -1)
        else:
            target_sliced = target_seq[:, : steps * action_dim]
        targets_bf16 = torch.nan_to_num(target_sliced.to(dtype=torch.bfloat16), nan=0.0, posinf=1e6, neginf=-1e6)
        return pred_sliced, targets_bf16

    def _compute_multistep_loss(
        self,
        action_output: torch.Tensor,
        target_seq: torch.Tensor,
        steps: int,
        action_dim: int,
        loss_type: str,
    ) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        batch_size = action_output.size(0)
        pred_sliced = action_output[:, : steps * action_dim]
        pred_sliced = torch.nan_to_num(pred_sliced, nan=0.0, posinf=1e6, neginf=-1e6)
        pred_deltas = pred_sliced.reshape(batch_size, steps, action_dim).to(dtype=torch.float32)
        pred_deltas = torch.nan_to_num(pred_deltas, nan=0.0, posinf=1e6, neginf=-1e6)

        if target_seq.dim() == 3:
            target_deltas = target_seq[:, :steps, :]
        else:
            target_deltas = target_seq[:, : steps * action_dim].reshape(batch_size, steps, action_dim)
        target_deltas = torch.nan_to_num(target_deltas.to(dtype=torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)

        def _window_sum(deltas: torch.Tensor, window: int) -> torch.Tensor:
            if window <= 1:
                return deltas
            cumsum = torch.cumsum(deltas, dim=1)
            padded = torch.cat([torch.zeros_like(cumsum[:, :1, :]), cumsum], dim=1)
            return padded[:, window:, :] - padded[:, :-window, :]

        def _compute_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if loss_type == "mse":
                return F.mse_loss(preds, targets)
            return torch.mean(torch.abs(preds - targets))

        per_step_pred = pred_deltas.reshape(batch_size, -1)
        per_step_target = target_deltas.reshape(batch_size, -1)
        per_step_loss = _compute_loss(per_step_pred, per_step_target)

        losses: dict[str, Optional[torch.Tensor]] = {
            "per_step": per_step_loss,
            "step_5": None,
            "step_10": None,
            "step_20": None,
        }

        pred_chunks = [per_step_pred]
        target_chunks = [per_step_target]
        for window, key in ((5, "step_5"), (10, "step_10"), (20, "step_20")):
            if steps < window:
                continue
            pred_window = _window_sum(pred_deltas, window).reshape(batch_size, -1)
            target_window = _window_sum(target_deltas, window).reshape(batch_size, -1)
            losses[key] = _compute_loss(pred_window, target_window)
            pred_chunks.append(pred_window)
            target_chunks.append(target_window)

        combined_pred = torch.cat(pred_chunks, dim=1)
        combined_target = torch.cat(target_chunks, dim=1)
        combined_loss = _compute_loss(combined_pred, combined_target)

        output_dtype = action_output.dtype
        for key, value in losses.items():
            if value is not None:
                losses[key] = value.to(dtype=output_dtype)

        return combined_loss.to(dtype=output_dtype), losses

    def _compute_ade_loss(
        self,
        action_output: torch.Tensor,
        target_seq: torch.Tensor,
        horizon: int,
        action_dim: int,
        steps: int,
    ) -> torch.Tensor:
        pred_deltas = action_output[:, : steps * action_dim].reshape(action_output.size(0), steps, action_dim)
        pred_deltas = torch.nan_to_num(pred_deltas, nan=0.0, posinf=1e6, neginf=-1e6).float()

        target_deltas = self._reshape_delta_sequence(target_seq, horizon, action_dim)[:, :steps, :]
        target_deltas = torch.nan_to_num(target_deltas.to(dtype=torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)

        # distances = torch.linalg.norm(pred_deltas - target_deltas, dim=2)
        distances = torch.sum((pred_deltas - target_deltas) ** 2, dim=2)

        return distances.mean()

    def _reshape_delta_sequence(self, sequence: torch.Tensor, horizon: int, action_dim: int) -> torch.Tensor:
        if sequence.dim() == 2:
            if sequence.size(1) != horizon * action_dim:
                raise ValueError(
                    f"Expected target with dimension {horizon * action_dim}, "
                    f"received {tuple(sequence.size())}."
                )
            return sequence.reshape(sequence.size(0), horizon, action_dim)
        if sequence.dim() == 3:
            if sequence.size(1) != horizon or sequence.size(2) != action_dim:
                raise ValueError(
                    f"Expected target shape (batch, {horizon}, {action_dim}), "
                    f"received {tuple(sequence.size())}."
                )
            return sequence
        raise ValueError(
            "Expected target deltas to have shape (batch, horizon * action_dim) or "
            f"(batch, horizon, action_dim). Received {tuple(sequence.size())}."
        )

    def _infer_horizon_and_dim(self, action_output: torch.Tensor, target_seq: torch.Tensor) -> tuple[int, int]:
        total_dim = action_output.size(1)
        if target_seq.dim() == 3:
            horizon = target_seq.size(1)
            action_dim = target_seq.size(2)
            if horizon <= 0 or action_dim <= 0:
                raise ValueError(f"Invalid target sequence shape {tuple(target_seq.size())}.")
            if horizon * action_dim != total_dim:
                raise ValueError(
                    f"action_head_output_dim ({total_dim}) must equal horizon * action_dim from targets "
                    f"({horizon} * {action_dim})."
                )
            return horizon, action_dim
        action_dim = self._action_dim
        if action_dim <= 0:
            raise ValueError("`action_head_output_dim` must be a positive integer.")
        if total_dim % action_dim != 0:
            raise ValueError(
                f"Flattened action dimension ({total_dim}) must be divisible by per-step action dimension "
                f"({action_dim}) to compute ADE."
            )
        horizon = total_dim // action_dim
        return horizon, action_dim

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        return_generation_output: bool = False,
        pad_token_id: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **generate_kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("`input_ids` or `inputs_embeds` must be provided to generate actions.")
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")

        required_steps = self.horizon if max_new_tokens is None else max_new_tokens
        if required_steps < self.horizon:
            raise ValueError(
                f"`max_new_tokens` must be at least the action horizon ({self.horizon}). Received {required_steps}."
            )

        if input_ids is None and attention_mask is not None:
            raise ValueError("`attention_mask` requires accompanying `input_ids` when `inputs_embeds` is not used.")

        generation_kwargs = dict(generate_kwargs)
        generation_kwargs.setdefault("return_dict_in_generate", True)
        generation_kwargs.setdefault("use_cache", True)
        generation_kwargs.setdefault("output_hidden_states", True)
        generation_kwargs["max_new_tokens"] = max(
            generation_kwargs.get("max_new_tokens", required_steps),
            required_steps,
        )

        generation_output = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            min_new_tokens=required_steps,
            pad_token_id=pad_token_id,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            **generation_kwargs,
        )

        sequences = getattr(generation_output, "sequences", None)
        if sequences is None:
            raise RuntimeError("Generation did not return sequences; ensure `return_dict_in_generate=True`.")

        # Build an attention mask for the generated sequences so that forward(task=\"action\") can
        # gather the final `horizon` tokens in the same manner as teacher forcing.
        if pad_token_id is None:
            pad_token_id = (
                generation_output.get("pad_token_id", None)
                if isinstance(generation_output, dict)
                else getattr(self.generation_config, "pad_token_id", None)
            )
            if pad_token_id is None:
                pad_token_id = getattr(self.config, "pad_token_id", None)

        if pad_token_id is not None:
            attention_mask_generated = sequences.ne(pad_token_id).long()
        else:
            attention_mask_generated = torch.ones_like(sequences, dtype=torch.long)

        hidden_states = None
        generated_hidden_states = getattr(generation_output, "hidden_states", None)
        if generated_hidden_states:
            last_hidden_state = generated_hidden_states[-1]
            if isinstance(last_hidden_state, (tuple, list)):
                last_hidden_state = last_hidden_state[-1]
            if isinstance(last_hidden_state, torch.Tensor):
                if last_hidden_state.dim() == 2:
                    last_hidden_state = last_hidden_state.unsqueeze(1)
                hidden_states = last_hidden_state

        if hidden_states is not None and hidden_states.size(1) >= self.horizon:
            action_embeddings = self._select_action_token_embeddings(
                hidden_states=hidden_states,
                attention_mask=attention_mask_generated,
                labels=None,
            )
            decoded_actions = self._decode_action_embeddings(action_embeddings)
            flattened_actions = decoded_actions.reshape(decoded_actions.size(0), -1)
        else:
            action_outputs = self.forward(
                input_ids=sequences,
                attention_mask=attention_mask_generated,
                output_hidden_states=True,
                return_dict=True,
                task="action",
            )
            flattened_actions = action_outputs.action_head_output

        if return_generation_output:
            return flattened_actions, generation_output
        return flattened_actions

    @torch.no_grad()
    def action_head_based_generate_actions(
        self,
        input_ids: Optional[torch.LongTensor],
        tokenizer: Any,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        return_generation_output: bool = False,
        **forward_kwargs,
    ):
        """Autoregressively generate actions by decoding the action head one token at a time."""
        if input_ids is None:
            raise ValueError("`input_ids` must be provided to generate actions.")
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided so actions can be mapped to token ids.")

        required_steps = self.horizon if max_new_tokens is None else max_new_tokens
        if required_steps < self.horizon:
            raise ValueError(
                f"`max_new_tokens` must be at least the action horizon ({self.horizon}). Received {required_steps}."
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        elif attention_mask.size() != input_ids.size():
            raise ValueError("`attention_mask` must have the same shape as `input_ids`.")

        device = input_ids.device
        generated_sequences = input_ids.clone()
        generated_attention_mask = attention_mask.clone()
        collected_actions = []

        past_key_values = None
        next_input_ids = generated_sequences

        for _ in range(required_steps):
            model_outputs = super().forward(
                input_ids=next_input_ids,
                attention_mask=generated_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                **forward_kwargs,
            )
            past_key_values = model_outputs.past_key_values

            hidden_state = model_outputs.hidden_states[-1]
            if hidden_state.dim() == 2:
                hidden_state = hidden_state.unsqueeze(1)
            last_hidden = hidden_state[:, -1:, :]

            decoded_action = self._decode_action_embeddings(last_hidden).squeeze(1)

            collected_actions.append(decoded_action)
            next_token_tensor = self._convert_actions_to_token_tensor(decoded_action, tokenizer, device=device)

            generated_sequences = torch.cat([generated_sequences, next_token_tensor.unsqueeze(-1)], dim=1)

            padding = torch.ones(
                (generated_attention_mask.size(0), 1),
                dtype=generated_attention_mask.dtype,
                device=generated_attention_mask.device,
            )
            generated_attention_mask = torch.cat([generated_attention_mask, padding], dim=1)

            logits = model_outputs.logits[:, -1, :]
            logits = logits / forward_kwargs["temperature"]
            filtered_logits = top_p_filtering(logits, top_p=forward_kwargs["top_p"])
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_input_ids = torch.multinomial(probabilities, num_samples=1)

            # next_input_ids = next_token_tensor.unsqueeze(-1)

        action_tensor = torch.stack(collected_actions, dim=1)
        flattened_actions = action_tensor.reshape(action_tensor.size(0), -1)

        if return_generation_output:
            return flattened_actions, generated_sequences
        return flattened_actions
