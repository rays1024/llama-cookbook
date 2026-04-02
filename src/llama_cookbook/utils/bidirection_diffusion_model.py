import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_cookbook.utils.action_model import CausalLMOutputWithPastAndActions
from llama_cookbook.utils.bidirection_action_model import LlamaForBidirectionAttnWithActions


class LlamaForBidirectionAttnWithDiffusionActions(LlamaForBidirectionAttnWithActions):
    """
    Diffusion action head conditioned by LLM hidden states.

    Design:
    - Consume standard LLM inputs: `input_ids`, `attention_mask`, `labels`.
    - Convert label token ids (`labels != -100`) into embedding vectors (clean targets x0).
    - Add diffusion noise to clean embeddings to obtain x_t.
    - Concatenate [LM hidden states, x_t, timestep features] per token.
    - Denoise and train against the clean label embeddings (or epsilon when configured).
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_diffusion_steps = int(getattr(config, "diffusion_num_steps", 50))
        if self.num_diffusion_steps <= 0:
            raise ValueError("`diffusion_num_steps` must be a positive integer.")

        self.diffusion_beta_schedule = str(
            getattr(config, "diffusion_beta_schedule", "squaredcos_cap_v2")
        ).lower()

        prediction_type = str(getattr(config, "diffusion_prediction_type", "action")).lower()
        if prediction_type in {"noise", "eps", "epsilon"}:
            prediction_type = "epsilon"
        elif prediction_type in {"action", "x0", "sample", "embedding", "embeddings"}:
            prediction_type = "action"
        else:
            raise ValueError(
                f"Unsupported diffusion prediction type '{prediction_type}'. "
                "Use one of: action/x0/sample/embedding or epsilon/noise/eps."
            )
        self.diffusion_prediction_type = prediction_type

        self.diffusion_embed_dim = int(self.config.hidden_size)
        self.diffusion_time_embed_dim = int(
            getattr(config, "diffusion_time_embed_dim", self.diffusion_embed_dim)
        )
        if self.diffusion_time_embed_dim <= 0:
            raise ValueError("`diffusion_time_embed_dim` must be positive.")

        self.diffusion_head_hidden_dim = int(
            getattr(config, "diffusion_head_hidden_dim", self._action_hidden_dim)
        )
        if self.diffusion_head_hidden_dim <= 0:
            raise ValueError("`diffusion_head_hidden_dim` must be positive.")

        self.diffusion_condition_norm = nn.LayerNorm(self.diffusion_embed_dim)
        self.diffusion_noisy_norm = nn.LayerNorm(self.diffusion_embed_dim)
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(self.diffusion_time_embed_dim, self.diffusion_embed_dim),
            nn.SiLU(),
            nn.Linear(self.diffusion_embed_dim, self.diffusion_embed_dim),
        )

        self.diffusion_fused_dim = self.diffusion_embed_dim * 3
        self.diffusion_fusion_proj = nn.Sequential(
            nn.LayerNorm(self.diffusion_fused_dim),
            nn.Linear(self.diffusion_fused_dim, self.diffusion_embed_dim),
            nn.SiLU(),
        )

        self.diffusion_denoiser = self._build_action_decoder(
            input_dim=self.diffusion_embed_dim,
            hidden_dim=self.diffusion_head_hidden_dim,
            num_layers=self._action_num_layers,
            action_dim=self.diffusion_embed_dim,
        )
        self.diffusion_denoiser.apply(self._init_weights)

        self._register_diffusion_schedule(
            num_steps=self.num_diffusion_steps,
            schedule_name=self.diffusion_beta_schedule,
        )
        self._align_diffusion_modules_dtype()

        self.config.diffusion_num_steps = self.num_diffusion_steps
        self.config.diffusion_beta_schedule = self.diffusion_beta_schedule
        self.config.diffusion_prediction_type = self.diffusion_prediction_type
        self.config.diffusion_time_embed_dim = self.diffusion_time_embed_dim
        self.config.diffusion_head_hidden_dim = self.diffusion_head_hidden_dim
        self.config.diffusion_target_space = "label_embeddings"

    def _align_diffusion_modules_dtype(self) -> None:
        base_weight = self.model.embed_tokens.weight
        modules = [
            self.diffusion_condition_norm,
            self.diffusion_noisy_norm,
            self.diffusion_time_mlp,
            self.diffusion_fusion_proj,
            self.diffusion_denoiser,
        ]
        for module in modules:
            module.to(device=base_weight.device, dtype=base_weight.dtype)

    def reset_action_head_parameters(self):
        """
        Reinitialize diffusion modules while preserving the parent interface.
        """
        super().reset_action_head_parameters()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        diffusion_modules = [
            self.diffusion_condition_norm,
            self.diffusion_noisy_norm,
            self.diffusion_time_mlp,
            self.diffusion_fusion_proj,
            self.diffusion_denoiser,
        ]
        for module in diffusion_modules:
            module.to_empty(device=device)
            module.apply(self._init_weights)
        self._align_diffusion_modules_dtype()

    @staticmethod
    def _alpha_bar_cosine(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    def _betas_for_alpha_bar(self, num_steps: int, max_beta: float = 0.999) -> torch.Tensor:
        betas = []
        for i in range(num_steps):
            t1 = i / num_steps
            t2 = (i + 1) / num_steps
            beta = min(1.0 - self._alpha_bar_cosine(t2) / self._alpha_bar_cosine(t1), max_beta)
            betas.append(beta)
        return torch.tensor(betas, dtype=torch.float32)

    def _build_beta_schedule(self, num_steps: int, schedule_name: str) -> torch.Tensor:
        if schedule_name == "squaredcos_cap_v2":
            return self._betas_for_alpha_bar(num_steps)
        if schedule_name == "linear":
            return torch.linspace(1e-4, 2e-2, num_steps, dtype=torch.float32)
        if schedule_name == "scaled_linear":
            return torch.linspace(1e-4 ** 0.5, 2e-2 ** 0.5, num_steps, dtype=torch.float32) ** 2
        raise ValueError(
            f"Unsupported diffusion beta schedule '{schedule_name}'. "
            "Use one of: squaredcos_cap_v2, linear, scaled_linear."
        )

    def _compute_diffusion_schedule_tensors(
        self, num_steps: int, schedule_name: str
    ) -> dict[str, torch.Tensor]:
        betas = self._build_beta_schedule(num_steps=num_steps, schedule_name=schedule_name).to(
            dtype=torch.float32
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1e-12))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(
            1.0 - alphas_cumprod, min=1e-12
        )
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        return {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
            "posterior_variance": posterior_variance,
        }

    def _schedule_needs_refresh(self) -> bool:
        names = (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        )
        for name in names:
            tensor = getattr(self, name, None)
            if not isinstance(tensor, torch.Tensor):
                return True
            if tensor.dtype != torch.float32:
                return True
            if tensor.numel() != self.num_diffusion_steps:
                return True
            if not torch.isfinite(tensor).all():
                return True

        one_minus_alpha0 = float((1.0 - self.alphas_cumprod[0].detach().float()).item())
        return one_minus_alpha0 <= 1e-8

    def _ensure_diffusion_schedule_precision(
        self, device: Optional[torch.device] = None
    ) -> None:
        if device is None:
            base = getattr(self, "betas", None)
            if isinstance(base, torch.Tensor):
                device = base.device
            else:
                device = self.model.embed_tokens.weight.device

        if not self._schedule_needs_refresh():
            names = (
                "betas",
                "alphas",
                "alphas_cumprod",
                "alphas_cumprod_prev",
                "sqrt_alphas_cumprod",
                "sqrt_one_minus_alphas_cumprod",
                "posterior_variance",
            )
            for name in names:
                tensor = getattr(self, name)
                if tensor.device != device:
                    setattr(self, name, tensor.to(device=device, dtype=torch.float32))
            return

        schedule_tensors = self._compute_diffusion_schedule_tensors(
            num_steps=self.num_diffusion_steps,
            schedule_name=self.diffusion_beta_schedule,
        )
        for name, tensor in schedule_tensors.items():
            setattr(self, name, tensor.to(device=device, dtype=torch.float32))

    def _register_diffusion_schedule(self, num_steps: int, schedule_name: str) -> None:
        schedule_tensors = self._compute_diffusion_schedule_tensors(
            num_steps=num_steps, schedule_name=schedule_name
        )
        self.register_buffer("betas", schedule_tensors["betas"], persistent=True)
        self.register_buffer("alphas", schedule_tensors["alphas"], persistent=True)
        self.register_buffer("alphas_cumprod", schedule_tensors["alphas_cumprod"], persistent=True)
        self.register_buffer(
            "alphas_cumprod_prev", schedule_tensors["alphas_cumprod_prev"], persistent=True
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", schedule_tensors["sqrt_alphas_cumprod"], persistent=True
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            schedule_tensors["sqrt_one_minus_alphas_cumprod"],
            persistent=True,
        )
        self.register_buffer(
            "posterior_variance", schedule_tensors["posterior_variance"], persistent=True
        )

    @staticmethod
    def _sinusoidal_timestep_embedding(
        timesteps: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        device = timesteps.device
        if half == 0:
            return timesteps.float().unsqueeze(-1)
        exponents = torch.arange(half, device=device, dtype=torch.float32)
        exponents = -math.log(max_period) * exponents / max(half - 1, 1)
        freqs = torch.exp(exponents)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def _normalize_timesteps(
        self, timesteps: Optional[torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.num_diffusion_steps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
        elif not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long)

        if timesteps.dim() == 0:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.dim() == 1 and timesteps.size(0) == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.dim() != 1 or timesteps.size(0) != batch_size:
            raise ValueError(
                f"`timesteps` must be shape (batch,), scalar, or (1,). Received {tuple(timesteps.size())}."
            )

        if (timesteps < 0).any() or (timesteps >= self.num_diffusion_steps).any():
            raise ValueError(
                f"`timesteps` must lie in [0, {self.num_diffusion_steps - 1}] for this schedule."
            )
        return timesteps

    def _normalize_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        if attention_mask.dim() != 2 or attention_mask.size() != (batch_size, seq_len):
            raise ValueError(
                "`attention_mask` must have shape (batch, seq_len) matching model inputs. "
                f"Got {tuple(attention_mask.size())}, expected {(batch_size, seq_len)}."
            )
        return attention_mask.to(device=device, dtype=torch.long)

    def _resolve_target_token_mask(
        self,
        *,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask_type_labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Resolve action/target token positions.

        Priority:
        1) `labels != -100` (training target tokens)
        2) `mask_type_labels` action region
        3) `input_ids == -1` empty placeholders (context + empty layout)
        """
        batch_size, seq_len = attention_mask.size()
        device = attention_mask.device
        attn_mask_bool = attention_mask.to(torch.bool)

        if labels is not None:
            if labels.dim() != 2 or labels.size() != (batch_size, seq_len):
                raise ValueError(
                    "`labels` must have shape (batch, seq_len) matching `attention_mask`."
                )
            return labels.to(device=device).ne(-100) & attn_mask_bool

        if mask_type_labels is not None:
            if mask_type_labels.dim() != 2 or mask_type_labels.size() != (batch_size, seq_len):
                raise ValueError(
                    "`mask_type_labels` must have shape (batch, seq_len) matching `attention_mask`."
                )
            mask_type_labels = mask_type_labels.to(device=device, dtype=torch.long)
            non_pad_labels = mask_type_labels[attn_mask_bool]
            has_label_two = (non_pad_labels == 2).any()
            has_label_one = (non_pad_labels == 1).any()
            has_label_zero = (non_pad_labels == 0).any()
            if has_label_two:
                action_label = 2
            elif has_label_zero and has_label_one:
                action_label = 1
            else:
                action_label = 2
            return (mask_type_labels == action_label) & attn_mask_bool

        if input_ids is not None:
            if input_ids.dim() != 2 or input_ids.size() != (batch_size, seq_len):
                raise ValueError(
                    "`input_ids` must have shape (batch, seq_len) matching `attention_mask`."
                )
            empty_mask = input_ids.to(device=device).eq(-1)
            if empty_mask.any():
                return empty_mask & attn_mask_bool

        raise ValueError(
            "Unable to infer target/action token positions. Provide one of: "
            "`labels`, `mask_type_labels`, or `input_ids` with -1 placeholders."
        )

    def _resolve_backbone_mask_type_labels(
        self,
        *,
        attention_mask: torch.Tensor,
        mask_type_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Resolve attention type labels for the bidirectional backbone.
        Uses caller-provided `mask_type_labels` when available, otherwise falls
        back to the normalized `attention_mask` (supports 1/2 encoded schemes).
        """
        if attention_mask.dim() != 2:
            raise ValueError("`attention_mask` must have shape (batch, seq_len).")

        batch_size, seq_len = attention_mask.size()
        device = attention_mask.device
        if mask_type_labels is None:
            return attention_mask.to(device=device, dtype=torch.long)

        if mask_type_labels.dim() != 2 or mask_type_labels.size() != (batch_size, seq_len):
            raise ValueError(
                "`mask_type_labels` must have shape (batch, seq_len) matching `attention_mask`."
            )
        return mask_type_labels.to(device=device, dtype=torch.long)

    def _select_diffusion_positions(self, target_mask: torch.Tensor) -> torch.Tensor:
        if target_mask.dim() != 2:
            raise ValueError("`target_mask` must have shape (batch, seq_len).")

        batch_size = target_mask.size(0)
        positions = []
        for row in range(batch_size):
            pos = torch.nonzero(target_mask[row], as_tuple=False).squeeze(-1)
            if pos.numel() < self.horizon:
                raise ValueError(
                    f"Sample {row} has {int(pos.numel())} target tokens, but horizon={self.horizon}."
                )
            # Keep the last `horizon` target positions.
            positions.append(pos[-self.horizon :])
        return torch.stack(positions, dim=0)

    @staticmethod
    def _gather_sequence_by_positions(
        sequence: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if sequence.dim() != 3:
            raise ValueError(
                f"`sequence` must have shape (batch, seq_len, dim). Got {tuple(sequence.size())}."
            )
        if positions.dim() != 2 or positions.size(0) != sequence.size(0):
            raise ValueError(
                "`positions` must have shape (batch, token_count) aligned with sequence batch size."
            )
        dim = sequence.size(-1)
        gather_index = positions.unsqueeze(-1).expand(-1, -1, dim)
        return torch.gather(sequence, dim=1, index=gather_index)

    def _prepare_optional_diffusion_input(
        self,
        value: Optional[torch.Tensor],
        *,
        name: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        target_positions: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None

        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
        if tensor.dim() != 3 or tensor.size(0) != batch_size or tensor.size(2) != self.diffusion_embed_dim:
            raise ValueError(
                f"`{name}` must have shape (batch, seq_len|horizon, {self.diffusion_embed_dim}). "
                f"Got {tuple(tensor.size())}."
            )

        if tensor.size(1) == self.horizon:
            return tensor
        if tensor.size(1) == seq_len:
            return self._gather_sequence_by_positions(tensor, target_positions)

        raise ValueError(
            f"`{name}` second dimension must be either seq_len ({seq_len}) or horizon "
            f"({self.horizon}). Got {tensor.size(1)}."
        )

    def _labels_to_clean_embeddings(
        self,
        labels: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        gathered_labels = self._gather_target_token_ids(
            labels=labels,
            target_positions=target_positions,
        )
        clean_embeddings = self.model.embed_tokens(gathered_labels)
        return clean_embeddings.to(dtype=torch.float32)

    def _gather_target_token_ids(
        self,
        labels: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError("`labels` must be provided for diffusion action training.")
        if labels.dim() != 2:
            raise ValueError(
                "`labels` must have shape (batch, seq_len). "
                f"Got {tuple(labels.size())}."
            )

        if target_positions.dim() != 2 or target_positions.size(0) != labels.size(0):
            raise ValueError(
                "`target_positions` must have shape (batch, horizon) aligned with labels batch size."
            )
        if target_positions.size(1) != self.horizon:
            raise ValueError(
                f"`target_positions` token count must equal horizon={self.horizon}."
            )

        labels = labels.to(device=target_positions.device, dtype=torch.long)
        gathered_labels = torch.gather(labels, dim=1, index=target_positions)

        vocab_size = self.model.embed_tokens.num_embeddings
        invalid = (gathered_labels < 0) | (gathered_labels >= vocab_size)
        if invalid.any():
            raise ValueError("`labels` contain token ids outside embedding vocabulary range.")

        return gathered_labels

    def _compute_token_ce_from_embeddings(
        self,
        pred_embeddings: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if pred_embeddings.dim() != 3:
            raise ValueError(
                "`pred_embeddings` must have shape (batch, horizon, hidden). "
                f"Got {tuple(pred_embeddings.size())}."
            )
        if target_token_ids.dim() != 2 or target_token_ids.size() != pred_embeddings.size()[:2]:
            raise ValueError(
                "`target_token_ids` must have shape (batch, horizon) aligned with "
                "`pred_embeddings`."
            )

        lm_head = self.get_output_embeddings()
        if lm_head is None:
            lm_head = getattr(self, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("Model does not expose output embeddings (`lm_head`) for CE loss.")

        head_weight = getattr(lm_head, "weight", None)
        logits_input = pred_embeddings
        if head_weight is not None:
            logits_input = logits_input.to(device=head_weight.device, dtype=head_weight.dtype)
            target_token_ids = target_token_ids.to(device=head_weight.device, dtype=torch.long)
        else:
            target_token_ids = target_token_ids.to(device=pred_embeddings.device, dtype=torch.long)

        logits = lm_head(logits_input).float()
        vocab_size = logits.size(-1)
        invalid = (target_token_ids < 0) | (target_token_ids >= vocab_size)
        if invalid.any():
            raise ValueError("`target_token_ids` contain token ids outside vocab range.")

        return F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_token_ids.reshape(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _compute_language_model_ce_from_logits(
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if logits is None or labels is None:
            return None
        if logits.dim() != 3:
            raise ValueError(
                "`logits` must have shape (batch, seq_len, vocab). "
                f"Got {tuple(logits.size())}."
            )
        if labels.dim() != 2:
            raise ValueError(
                "`labels` must have shape (batch, seq_len). "
                f"Got {tuple(labels.size())}."
            )
        if logits.size(0) != labels.size(0):
            raise ValueError(
                "`logits` and `labels` batch dimensions must match. "
                f"Got {logits.size(0)} vs {labels.size(0)}."
            )

        seq_len = min(logits.size(1), labels.size(1))
        if seq_len <= 1:
            return torch.zeros((), device=logits.device, dtype=torch.float32)

        shift_logits = logits[:, : seq_len - 1, :].contiguous().float()
        shift_labels = labels[:, 1:seq_len].contiguous().to(device=shift_logits.device, dtype=torch.long)
        if not shift_labels.ne(-100).any():
            return torch.zeros((), device=shift_logits.device, dtype=torch.float32)

        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

    def _encode_condition_with_backbone(
        self,
        *,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        mask_type_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        lm_outputs = super().forward(
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
            mask_type_labels=mask_type_labels,
            task="language",
            **kwargs,
        )

        if lm_outputs.hidden_states is None or len(lm_outputs.hidden_states) == 0:
            raise RuntimeError("Backbone did not return hidden states for diffusion conditioning.")

        condition_hidden = lm_outputs.hidden_states[-1]
        return condition_hidden, lm_outputs

    def q_sample(
        self, x0: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        if x0.size() != noise.size():
            raise ValueError("`x0` and `noise` must have the same shape for q-sampling.")

        timesteps = timesteps.to(device=x0.device, dtype=torch.long)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1).to(
            dtype=x0.dtype, device=x0.device
        )
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1).to(
            dtype=x0.dtype, device=x0.device
        )
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def _predict_x0_from_noise(
        self, x_t: torch.Tensor, pred_noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        timesteps = timesteps.to(device=x_t.device, dtype=torch.long)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1).to(
            dtype=x_t.dtype, device=x_t.device
        )
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1).to(
            dtype=x_t.dtype, device=x_t.device
        )
        return (x_t - sqrt_one_minus * pred_noise) / torch.clamp(sqrt_alpha, min=1e-12)

    def _predict_noise_from_x0(
        self, x_t: torch.Tensor, pred_x0: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        timesteps = timesteps.to(device=x_t.device, dtype=torch.long)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1).to(
            dtype=x_t.dtype, device=x_t.device
        )
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1).to(
            dtype=x_t.dtype, device=x_t.device
        )
        return (x_t - sqrt_alpha * pred_x0) / torch.clamp(sqrt_one_minus, min=1e-12)

    def _build_diffusion_input(
        self,
        *,
        cond_hidden: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if cond_hidden.dim() != 3 or x_t.dim() != 3:
            raise ValueError(
                "`cond_hidden` and `x_t` must both have shape (batch, seq_len, hidden)."
            )
        if cond_hidden.size() != x_t.size():
            raise ValueError(
                "Conditioning hidden states and noisy embeddings must share the same shape. "
                f"Got cond {tuple(cond_hidden.size())} vs noisy {tuple(x_t.size())}."
            )

        batch_size, seq_len, hidden_dim = cond_hidden.size()
        if hidden_dim != self.diffusion_embed_dim:
            raise ValueError(
                f"Expected hidden dim {self.diffusion_embed_dim}, got {hidden_dim}."
            )

        cond_dtype = next(self.diffusion_condition_norm.parameters()).dtype
        cond_features = self.diffusion_condition_norm(
            cond_hidden.to(device=cond_hidden.device, dtype=cond_dtype)
        )
        noisy_features = self.diffusion_noisy_norm(
            x_t.to(device=cond_hidden.device, dtype=cond_dtype)
        )

        t_embed = self._sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=self.diffusion_time_embed_dim,
        ).to(device=cond_hidden.device, dtype=cond_dtype)
        step_features = self.diffusion_time_mlp(t_embed).unsqueeze(1).expand(batch_size, seq_len, -1)

        fused = torch.cat([cond_features, noisy_features, step_features], dim=-1)
        return self.diffusion_fusion_proj(fused)

    def _predict_diffusion_target_from_xt(
        self,
        *,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond_hidden: torch.Tensor,
    ) -> torch.Tensor:
        diffusion_input = self._build_diffusion_input(
            cond_hidden=cond_hidden,
            x_t=x_t,
            timesteps=timesteps,
        )
        head_dtype = next(self.diffusion_denoiser.parameters()).dtype
        diffusion_input = diffusion_input.to(dtype=head_dtype)

        batch_size, seq_len, input_dim = diffusion_input.size()
        pred_target = self.diffusion_denoiser(diffusion_input.reshape(batch_size * seq_len, input_dim))
        pred_target = pred_target.reshape(batch_size, seq_len, self.diffusion_embed_dim)
        return pred_target.to(dtype=torch.float32)

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
        task: str = "action",
        loss_type: Optional[str] = None,
        loss_horizon: Optional[int] = None,
        pred_seq=None,
        timesteps: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        use_ce_loss: bool = True,
        debug_print_loss: bool = False,
        **kwargs,
    ):
        if task != "action":
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
                mask_type_labels=mask_type_labels,
                task=task,
                loss_type=loss_type,
                loss_horizon=loss_horizon,
                pred_seq=pred_seq,
                use_ce_loss=use_ce_loss,
                **kwargs,
            )

        use_ce_loss = bool(use_ce_loss)
        del loss_type
        del loss_horizon
        del pred_seq

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        effective_output_hidden_states = output_hidden_states or self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")

        source_tensor = input_ids if input_ids is not None else inputs_embeds
        if source_tensor is None:
            raise ValueError("`input_ids` or `inputs_embeds` must be provided for diffusion action mode.")

        batch_size, seq_len = source_tensor.size(0), source_tensor.size(1)
        device = source_tensor.device

        attention_mask = self._normalize_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        backbone_mask_type_labels = self._resolve_backbone_mask_type_labels(
            attention_mask=attention_mask,
            mask_type_labels=mask_type_labels,
        )
        target_mask = self._resolve_target_token_mask(
            attention_mask=attention_mask,
            labels=labels,
            mask_type_labels=mask_type_labels,
            input_ids=input_ids,
        )
        target_positions = self._select_diffusion_positions(target_mask)

        self._ensure_diffusion_schedule_precision(device=device)
        timesteps = self._normalize_timesteps(timesteps=timesteps, batch_size=batch_size, device=device)

        kwargs = dict(kwargs)
        kwargs.pop("tokenizer", None)

        # In diffusion mode, CE is computed from diffusion-predicted embeddings and
        # GT token ids (same target positions), not from backbone next-token loss.
        backbone_labels = None
        cond_hidden, lm_outputs = self._encode_condition_with_backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=backbone_labels,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            mask_type_labels=backbone_mask_type_labels,
            **kwargs,
        )
        cond_hidden = self._gather_sequence_by_positions(cond_hidden, target_positions)

        clean_embeds = None
        if labels is not None:
            clean_embeds = self._labels_to_clean_embeddings(
                labels=labels,
                target_positions=target_positions,
            )

        x_t = self._prepare_optional_diffusion_input(
            noisy_actions,
            name="noisy_actions",
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            target_positions=target_positions,
        )
        target_noise = self._prepare_optional_diffusion_input(
            noise,
            name="noise",
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            target_positions=target_positions,
        )

        if x_t is None:
            if clean_embeds is None:
                raise ValueError(
                    "`labels` are required when `noisy_actions` is not provided because clean "
                    "GT embeddings are needed for q-sampling."
                )
            if target_noise is None:
                target_noise = torch.randn_like(clean_embeds)
            x_t = self.q_sample(clean_embeds, timesteps, target_noise)

        pred_target = self._predict_diffusion_target_from_xt(
            x_t=x_t,
            timesteps=timesteps,
            cond_hidden=cond_hidden,
        )

        if self.diffusion_prediction_type == "epsilon":
            pred_noise = pred_target
            pred_x0 = self._predict_x0_from_noise(x_t, pred_noise, timesteps)
            if target_noise is None and clean_embeds is not None:
                target_noise = self._predict_noise_from_x0(x_t, clean_embeds, timesteps)
            loss_prediction = pred_noise
            loss_target = target_noise
        else:
            pred_x0 = pred_target
            pred_noise = self._predict_noise_from_x0(x_t, pred_x0, timesteps)
            loss_prediction = pred_x0
            loss_target = clean_embeds

        diffusion_loss = None
        if loss_target is not None:
            if loss_prediction.size() != loss_target.size():
                raise ValueError(
                    "Prediction/target shape mismatch for diffusion loss: "
                    f"{tuple(loss_prediction.size())} vs {tuple(loss_target.size())}."
                )
            diffusion_loss = F.mse_loss(
                loss_prediction.float(),
                loss_target.float(),
            ).to(dtype=pred_target.dtype)
            if debug_print_loss:
                timestep_min = int(timesteps.min().item())
                timestep_max = int(timesteps.max().item())
                on_last_step = bool(torch.all(timesteps == 0).item())
                print(
                    "[DEBUG][diffusion_forward] "
                    f"loss={float(diffusion_loss.detach().float().item()):.8f}, "
                    f"timesteps_range=[{timestep_min}, {timestep_max}], "
                    f"all_last_step_t0={on_last_step}"
                )

        diffusion_loss_scale = 10000.0
        scaled_diffusion_loss = None
        if diffusion_loss is not None:
            scaled_diffusion_loss = diffusion_loss.float() * diffusion_loss_scale

        ce_loss = None
        target_token_ids = None
        if use_ce_loss and labels is not None:
            target_token_ids = self._gather_target_token_ids(
                labels=labels,
                target_positions=target_positions,
            )
            ce_loss = self._compute_token_ce_from_embeddings(
                pred_embeddings=pred_x0,
                target_token_ids=target_token_ids,
            )
        if ce_loss is not None:
            ce_loss = ce_loss.float()

        language_model_ce_loss = self._compute_language_model_ce_from_logits(
            logits=lm_outputs.logits,
            labels=labels,
        )
        if language_model_ce_loss is not None:
            language_model_ce_loss = language_model_ce_loss.float()

        combined_loss = scaled_diffusion_loss
        if ce_loss is not None:
            combined_loss = ce_loss if combined_loss is None else combined_loss + ce_loss

        if debug_print_loss:
            ce_loss_val = float(ce_loss.detach().item()) if ce_loss is not None else None
            scaled_val = (
                float(scaled_diffusion_loss.detach().item())
                if scaled_diffusion_loss is not None
                else None
            )
            combined_val = float(combined_loss.detach().item()) if combined_loss is not None else None
            print(
                "[DEBUG][diffusion_forward_loss_terms] "
                f"diffusion_scale={diffusion_loss_scale:.1f}, "
                f"scaled_diffusion_loss={scaled_val}, "
                f"cross_entropy_loss={ce_loss_val}, "
                f"combined_loss={combined_val}"
            )

        flattened_pred_x0 = pred_x0.reshape(pred_x0.size(0), -1)

        outputs = CausalLMOutputWithPastAndActions(
            loss=combined_loss,
            logits=lm_outputs.logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=(lm_outputs.hidden_states if effective_output_hidden_states else None),
            attentions=lm_outputs.attentions,
            action_head_output=flattened_pred_x0,
        )
        outputs["action_loss"] = combined_loss
        outputs["language_model_loss"] = language_model_ce_loss
        outputs["language_model_cross_entropy_loss"] = language_model_ce_loss
        outputs["cross_entropy_loss"] = ce_loss
        outputs["action_prediction_loss"] = scaled_diffusion_loss
        outputs["raw_action_prediction_loss"] = diffusion_loss
        outputs["diffusion_loss_scale"] = diffusion_loss_scale
        outputs["smoothness_loss"] = None
        outputs["vec_order_loss"] = None

        outputs["diffusion_prediction_type"] = self.diffusion_prediction_type
        outputs["predicted_noise"] = pred_noise
        outputs["predicted_x0"] = pred_x0
        outputs["timesteps"] = timesteps
        outputs["noisy_actions"] = x_t
        if clean_embeds is not None:
            outputs["clean_actions"] = clean_embeds
        if target_token_ids is not None:
            outputs["diffusion_target_token_ids"] = target_token_ids
        outputs["diffusion_target_mask"] = target_mask
        outputs["diffusion_target_positions"] = target_positions

        if not return_dict:
            return tuple(value for value in outputs.values() if value is not None)
        return outputs

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
        num_return_sequences: int = 1,
        **kwargs,
    ):
        """
        One-pass LM-head generation for diffusion wrapper.
        This bypasses diffusion sampling and forces language-mode forward pass.
        """
        if num_return_sequences is None:
            num_return_sequences = 1
        if not isinstance(num_return_sequences, int) or num_return_sequences <= 0:
            raise ValueError("`num_return_sequences` must be a positive integer.")

        if input_ids is None:
            raise ValueError("`input_ids` must be provided for generation.")

        generate_kwargs = dict(kwargs)
        # `generate()` should not run diffusion-path logic.
        generate_kwargs["task"] = "language"
        # Drop training/diffusion-only kwargs to reduce memory during inference.
        for key in (
            "labels",
            "pred_seq",
            "timesteps",
            "noisy_actions",
            "noise",
            "loss_type",
            "loss_horizon",
            "use_ce_loss",
            "debug_print_loss",
            "tokenizer",
        ):
            generate_kwargs.pop(key, None)

        if num_return_sequences == 1:
            return super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                greedy=greedy,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **generate_kwargs,
            )

        def _to_cpu_tree(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, tuple):
                return tuple(_to_cpu_tree(v) for v in value)
            if isinstance(value, list):
                return [_to_cpu_tree(v) for v in value]
            return value

        def _concat_tree(items):
            if len(items) == 0:
                return None
            first = items[0]
            if first is None:
                return None
            if isinstance(first, torch.Tensor):
                return torch.cat(items, dim=0)
            if isinstance(first, tuple):
                return tuple(_concat_tree([item[idx] for item in items]) for idx in range(len(first)))
            if isinstance(first, list):
                return [_concat_tree([item[idx] for item in items]) for idx in range(len(first))]
            return first

        sequence_chunks = []
        score_chunks = []
        attention_chunks = []
        hidden_chunks = []
        output_cls = None

        for _ in range(num_return_sequences):
            run_output = super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                greedy=greedy,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **generate_kwargs,
            )

            if not return_dict_in_generate:
                sequence_chunks.append(run_output)
                continue

            output_cls = type(run_output)
            sequence_chunks.append(run_output.sequences)
            if output_scores:
                score_chunks.append(_to_cpu_tree(run_output.scores))
            if output_attentions:
                attention_chunks.append(_to_cpu_tree(run_output.attentions))
            if output_hidden_states:
                hidden_chunks.append(_to_cpu_tree(run_output.hidden_states))

            del run_output

        merged_sequences = torch.cat(sequence_chunks, dim=0)
        if not return_dict_in_generate:
            return merged_sequences

        merged_scores = _concat_tree(score_chunks) if output_scores else None
        merged_attentions = _concat_tree(attention_chunks) if output_attentions else None
        merged_hidden_states = _concat_tree(hidden_chunks) if output_hidden_states else None

        return output_cls(
            sequences=merged_sequences,
            scores=merged_scores,
            attentions=merged_attentions,
            hidden_states=merged_hidden_states,
        )

    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
        timestep: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if timestep == 0:
            return pred_x0.to(device=x_t.device, dtype=x_t.dtype)

        beta_t = self.betas[timestep].to(device=x_t.device, dtype=x_t.dtype)
        alpha_t = self.alphas[timestep].to(device=x_t.device, dtype=x_t.dtype)
        alpha_cumprod_t = self.alphas_cumprod[timestep].to(device=x_t.device, dtype=x_t.dtype)
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep].to(device=x_t.device, dtype=x_t.dtype)

        denom = torch.clamp(1.0 - alpha_cumprod_t, min=1e-12)
        coef_x0 = torch.sqrt(alpha_cumprod_prev) * beta_t / denom
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) / denom
        posterior_mean = coef_x0 * pred_x0 + coef_xt * x_t

        variance = self.posterior_variance[timestep].to(device=x_t.device, dtype=x_t.dtype)
        noise = torch.randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
        return posterior_mean + torch.sqrt(variance) * noise

    @staticmethod
    def _top_k_top_p_filter(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if top_k < 0:
            raise ValueError("`top_k` must be >= 0.")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("`top_p` must be in (0, 1].")

        filtered = logits
        vocab_size = filtered.size(-1)

        if top_k > 0:
            k = min(top_k, vocab_size)
            threshold = torch.topk(filtered, k, dim=-1).values[..., -1, None]
            filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False

            remove_mask = torch.zeros_like(filtered, dtype=torch.bool)
            remove_mask.scatter_(-1, sorted_indices, sorted_remove)
            filtered = filtered.masked_fill(remove_mask, float("-inf"))

        return filtered

    def _decode_embeddings_to_token_ids(
        self,
        sampled_embeddings: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        return_logits: bool = False,
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        if sampled_embeddings.dim() != 3:
            raise ValueError(
                "`sampled_embeddings` must have shape (batch, horizon, hidden). "
                f"Got {tuple(sampled_embeddings.size())}."
            )
        if do_sample and temperature <= 0:
            raise ValueError("`temperature` must be > 0 when `do_sample=True`.")
        if repetition_penalty <= 0:
            raise ValueError("`repetition_penalty` must be > 0.")

        lm_head = self.get_output_embeddings()
        if lm_head is None:
            lm_head = getattr(self, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("Model does not expose output embeddings (`lm_head`) for token decoding.")

        head_weight = getattr(lm_head, "weight", None)
        if head_weight is not None:
            sampled_embeddings = sampled_embeddings.to(
                device=head_weight.device,
                dtype=head_weight.dtype,
            )

        logits = lm_head(sampled_embeddings).float()

        if not do_sample:
            token_ids = torch.argmax(logits, dim=-1)
            if return_logits:
                return token_ids, logits
            return token_ids, None

        batch_size, seq_len, _ = logits.size()
        token_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=logits.device)

        for step in range(seq_len):
            step_logits = logits[:, step, :].clone()
            if repetition_penalty != 1.0 and step > 0:
                prev = token_ids[:, :step]
                for row in range(batch_size):
                    seen_ids = torch.unique(prev[row])
                    if seen_ids.numel() == 0:
                        continue
                    seen_logits = step_logits[row, seen_ids]
                    seen_logits = torch.where(
                        seen_logits < 0,
                        seen_logits * repetition_penalty,
                        seen_logits / repetition_penalty,
                    )
                    step_logits[row, seen_ids] = seen_logits

            step_logits = step_logits / temperature
            step_logits = self._top_k_top_p_filter(step_logits, top_k=top_k, top_p=top_p)
            step_probs = torch.softmax(step_logits, dim=-1)
            sampled_ids = torch.multinomial(step_probs, num_samples=1).squeeze(-1)
            token_ids[:, step] = sampled_ids

        if return_logits:
            return token_ids, logits
        return token_ids, None

    @torch.no_grad()
    def ddpm_sample_actions(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_type_labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        clip_denoised: bool = True,
        clip_range: float = 1.0,
        return_generation_output: bool = False,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("`input_ids` or `inputs_embeds` must be provided for DDPM sampling.")

        source_tensor = input_ids if input_ids is not None else inputs_embeds
        batch_size, seq_len = source_tensor.size(0), source_tensor.size(1)
        device = source_tensor.device

        attention_mask = self._normalize_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        backbone_mask_type_labels = self._resolve_backbone_mask_type_labels(
            attention_mask=attention_mask,
            mask_type_labels=mask_type_labels,
        )
        target_mask = self._resolve_target_token_mask(
            attention_mask=attention_mask,
            labels=labels,
            mask_type_labels=mask_type_labels,
            input_ids=input_ids,
        )
        target_positions = self._select_diffusion_positions(target_mask)

        self._ensure_diffusion_schedule_precision(device=device)

        if num_inference_steps is None:
            num_inference_steps = self.num_diffusion_steps
        if num_inference_steps != self.num_diffusion_steps:
            raise ValueError(
                f"This model was initialized with diffusion_num_steps={self.num_diffusion_steps}; "
                f"received num_inference_steps={num_inference_steps}."
            )

        kwargs = dict(kwargs)
        kwargs.pop("tokenizer", None)

        cond_hidden, _ = self._encode_condition_with_backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            mask_type_labels=backbone_mask_type_labels,
            **kwargs,
        )
        cond_hidden = self._gather_sequence_by_positions(cond_hidden, target_positions)

        x_t = torch.randn(
            (batch_size, self.horizon, self.diffusion_embed_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

        last_pred_noise = None
        last_pred_x0 = None

        for t in reversed(range(num_inference_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_target = self._predict_diffusion_target_from_xt(
                x_t=x_t,
                timesteps=t_batch,
                cond_hidden=cond_hidden,
            ).to(dtype=torch.float32)

            if self.diffusion_prediction_type == "epsilon":
                pred_noise = pred_target
                pred_x0 = self._predict_x0_from_noise(x_t, pred_noise, t_batch)
            else:
                pred_x0 = pred_target
                pred_noise = self._predict_noise_from_x0(x_t, pred_x0, t_batch)

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-clip_range, max=clip_range)
                pred_noise = self._predict_noise_from_x0(x_t, pred_x0, t_batch)

            x_t = self._ddpm_step(x_t=x_t, pred_x0=pred_x0, timestep=t, generator=generator)
            last_pred_noise = pred_noise
            last_pred_x0 = pred_x0

        sampled_embeddings = x_t
        flattened_embeddings = sampled_embeddings.reshape(sampled_embeddings.size(0), -1)

        if return_generation_output:
            generation_output = {
                "sampled_embeddings": sampled_embeddings,
                "predicted_noise": last_pred_noise,
                "predicted_x0": last_pred_x0,
                "target_mask": target_mask,
                "target_positions": target_positions,
            }
            return flattened_embeddings, generation_output
        return flattened_embeddings

    @torch.no_grad()
    def action_head_based_generate_actions(
        self,
        input_ids: Optional[torch.LongTensor],
        tokenizer: Any = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_type_labels: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        decode_tokens: bool = False,
        decode_return_logits: bool = False,
        return_generation_output: bool = False,
        **kwargs,
    ):
        del tokenizer
        del max_new_tokens

        decode_do_sample = False
        decode_temperature = 1.0
        decode_top_p = 1.0
        decode_top_k = 0
        decode_repetition_penalty = 1.0

        if decode_tokens:
            decode_do_sample = bool(kwargs.pop("decode_do_sample", kwargs.pop("do_sample", False)))
            decode_temperature = float(kwargs.pop("decode_temperature", kwargs.pop("temperature", 1.0)))
            decode_top_p = float(kwargs.pop("decode_top_p", kwargs.pop("top_p", 1.0)))
            decode_top_k = int(kwargs.pop("decode_top_k", kwargs.pop("top_k", 0)))
            decode_repetition_penalty = float(
                kwargs.pop("decode_repetition_penalty", kwargs.pop("repetition_penalty", 1.0))
            )

        sampled_output = self.ddpm_sample_actions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mask_type_labels=mask_type_labels,
            return_generation_output=return_generation_output,
            **kwargs,
        )

        if not decode_tokens:
            return sampled_output

        generation_output = None
        if return_generation_output:
            flattened_embeddings, generation_output = sampled_output
            sampled_embeddings = generation_output.get("sampled_embeddings")
        else:
            flattened_embeddings = sampled_output
            sampled_embeddings = None

        if sampled_embeddings is None:
            if not isinstance(flattened_embeddings, torch.Tensor) or flattened_embeddings.dim() != 2:
                raise ValueError(
                    "Expected flattened diffusion output tensor with shape (batch, horizon * hidden)."
                )
            total_dim = flattened_embeddings.size(1)
            if total_dim % self.diffusion_embed_dim != 0:
                raise ValueError(
                    "Cannot reshape flattened diffusion output to embeddings: "
                    f"{total_dim} is not divisible by hidden size {self.diffusion_embed_dim}."
                )
            sampled_embeddings = flattened_embeddings.reshape(
                flattened_embeddings.size(0),
                total_dim // self.diffusion_embed_dim,
                self.diffusion_embed_dim,
            )

        decoded_token_ids, decoded_logits = self._decode_embeddings_to_token_ids(
            sampled_embeddings=sampled_embeddings,
            do_sample=decode_do_sample,
            temperature=decode_temperature,
            top_p=decode_top_p,
            top_k=decode_top_k,
            repetition_penalty=decode_repetition_penalty,
            return_logits=decode_return_logits,
        )

        if return_generation_output:
            if generation_output is None:
                generation_output = {}
            generation_output["decoded_token_ids"] = decoded_token_ids
            if decode_return_logits:
                generation_output["decoded_token_logits"] = decoded_logits
            return decoded_token_ids, generation_output

        return decoded_token_ids
