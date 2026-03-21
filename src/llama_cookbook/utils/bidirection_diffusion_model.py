import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_cookbook.utils.action_model import CausalLMOutputWithPastAndActions
from llama_cookbook.utils.bidirection_action_model import LlamaForBidirectionAttnWithActions


# class LlamaForBidirectionAttnWithDiffusionActions(LlamaForBidirectionAttnWithActions):
#     """
#     Bidirectional action model where the backbone acts as a denoiser:
#     input = [context tokens] + [noisy action tokens x_t] + [timestep token].
#     """

#     def __init__(self, config):
#         super().__init__(config)

#         self.num_diffusion_steps = int(getattr(config, "diffusion_num_steps", 50))
#         if self.num_diffusion_steps <= 0:
#             raise ValueError("`diffusion_num_steps` must be a positive integer.")

#         self.diffusion_beta_schedule = str(
#             getattr(config, "diffusion_beta_schedule", "squaredcos_cap_v2")
#         ).lower()
#         self.diffusion_prediction_type = str(
#             getattr(config, "diffusion_prediction_type", "epsilon")
#         ).lower()
#         if self.diffusion_prediction_type in {"noise", "eps"}:
#             self.diffusion_prediction_type = "epsilon"
#         if self.diffusion_prediction_type != "epsilon":
#             raise ValueError(
#                 f"Unsupported diffusion prediction type '{self.diffusion_prediction_type}'. "
#                 "This class currently supports epsilon/noise prediction only."
#             )

#         # Diffusion path always uses horizon action tokens from the backbone output.
#         self._diffusion_action_token_count = int(self.horizon)
#         self._head_chunk_count = (
#             self._diffusion_action_token_count + self._action_chunk_size - 1
#         ) // self._action_chunk_size
#         self._action_chunk_dim = self._action_dim * self._action_chunk_size
#         hidden_size = int(self.config.hidden_size)
#         self._head_input_dim = hidden_size * self._action_chunk_size

#         self.action_input_proj = nn.Sequential(
#             nn.LayerNorm(self._action_dim),
#             nn.Linear(self._action_dim, hidden_size),
#         )
#         self.time_embed_mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 4),
#             nn.SiLU(),
#             nn.Linear(hidden_size * 4, hidden_size),
#         )

#         # Rebuild decoder so it can consume chunked groups of action-token embeddings.
#         self.action_decoder = self._build_action_decoder(
#             input_dim=self._head_input_dim,
#             hidden_dim=self._action_hidden_dim,
#             num_layers=self._action_num_layers,
#             action_dim=self._action_chunk_dim,
#         )
#         self.action_decoder.apply(self._init_weights)

#         self._register_diffusion_schedule(
#             num_steps=self.num_diffusion_steps,
#             schedule_name=self.diffusion_beta_schedule,
#         )
#         self._align_diffusion_modules_dtype()

#         self.config.diffusion_num_steps = self.num_diffusion_steps
#         self.config.diffusion_beta_schedule = self.diffusion_beta_schedule
#         self.config.diffusion_prediction_type = self.diffusion_prediction_type

#         self.action_mean_std = {
#             "mean": [
#                 0.6479015011446655,
#                 0.03882767111321254
#             ],
#             "std": [
#                 0.535827732824574,
#                 0.4321850248635431
#             ]
#         }

#     def _align_diffusion_modules_dtype(self) -> None:
#         base_weight = self.model.embed_tokens.weight
#         self.action_input_proj.to(device=base_weight.device, dtype=base_weight.dtype)
#         self.time_embed_mlp.to(device=base_weight.device, dtype=base_weight.dtype)

#     @staticmethod
#     def _alpha_bar_cosine(t: float) -> float:
#         return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

#     def _betas_for_alpha_bar(self, num_steps: int, max_beta: float = 0.999) -> torch.Tensor:
#         betas = []
#         for i in range(num_steps):
#             t1 = i / num_steps
#             t2 = (i + 1) / num_steps
#             beta = min(1.0 - self._alpha_bar_cosine(t2) / self._alpha_bar_cosine(t1), max_beta)
#             betas.append(beta)
#         return torch.tensor(betas, dtype=torch.float32)

#     def _build_beta_schedule(self, num_steps: int, schedule_name: str) -> torch.Tensor:
#         if schedule_name == "squaredcos_cap_v2":
#             return self._betas_for_alpha_bar(num_steps)
#         if schedule_name == "linear":
#             return torch.linspace(1e-4, 2e-2, num_steps, dtype=torch.float32)
#         if schedule_name == "scaled_linear":
#             return torch.linspace(1e-4 ** 0.5, 2e-2 ** 0.5, num_steps, dtype=torch.float32) ** 2
#         raise ValueError(
#             f"Unsupported diffusion beta schedule '{schedule_name}'. "
#             "Use one of: squaredcos_cap_v2, linear, scaled_linear."
#         )

#     def _compute_diffusion_schedule_tensors(
#         self, num_steps: int, schedule_name: str
#     ) -> dict[str, torch.Tensor]:
#         betas = self._build_beta_schedule(num_steps=num_steps, schedule_name=schedule_name).to(
#             dtype=torch.float32
#         )
#         alphas = 1.0 - betas
#         alphas_cumprod = torch.cumprod(alphas, dim=0)
#         alphas_cumprod_prev = torch.cat(
#             [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0
#         )

#         sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#         sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1e-12))
#         posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(
#             1.0 - alphas_cumprod, min=1e-12
#         )
#         posterior_variance = torch.clamp(posterior_variance, min=1e-20)

#         return {
#             "betas": betas,
#             "alphas": alphas,
#             "alphas_cumprod": alphas_cumprod,
#             "alphas_cumprod_prev": alphas_cumprod_prev,
#             "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
#             "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
#             "posterior_variance": posterior_variance,
#         }

#     def _schedule_needs_refresh(self) -> bool:
#         names = (
#             "betas",
#             "alphas",
#             "alphas_cumprod",
#             "alphas_cumprod_prev",
#             "sqrt_alphas_cumprod",
#             "sqrt_one_minus_alphas_cumprod",
#             "posterior_variance",
#         )
#         for name in names:
#             tensor = getattr(self, name, None)
#             if not isinstance(tensor, torch.Tensor):
#                 return True
#             if tensor.dtype != torch.float32:
#                 return True
#             if tensor.numel() != self.num_diffusion_steps:
#                 return True
#             if not torch.isfinite(tensor).all():
#                 return True

#         one_minus_alpha0 = float((1.0 - self.alphas_cumprod[0].detach().float()).item())
#         return one_minus_alpha0 <= 1e-8

#     def _ensure_diffusion_schedule_precision(
#         self, device: Optional[torch.device] = None
#     ) -> None:
#         if device is None:
#             base = getattr(self, "betas", None)
#             if isinstance(base, torch.Tensor):
#                 device = base.device
#             else:
#                 device = self.model.embed_tokens.weight.device

#         if not self._schedule_needs_refresh():
#             # Keep schedule tensors colocated with the current execution device.
#             names = (
#                 "betas",
#                 "alphas",
#                 "alphas_cumprod",
#                 "alphas_cumprod_prev",
#                 "sqrt_alphas_cumprod",
#                 "sqrt_one_minus_alphas_cumprod",
#                 "posterior_variance",
#             )
#             for name in names:
#                 tensor = getattr(self, name)
#                 if tensor.device != device:
#                     setattr(self, name, tensor.to(device=device, dtype=torch.float32))
#             return

#         schedule_tensors = self._compute_diffusion_schedule_tensors(
#             num_steps=self.num_diffusion_steps,
#             schedule_name=self.diffusion_beta_schedule,
#         )
#         for name, tensor in schedule_tensors.items():
#             setattr(self, name, tensor.to(device=device, dtype=torch.float32))

#     def _register_diffusion_schedule(self, num_steps: int, schedule_name: str) -> None:
#         schedule_tensors = self._compute_diffusion_schedule_tensors(
#             num_steps=num_steps, schedule_name=schedule_name
#         )
#         self.register_buffer("betas", schedule_tensors["betas"], persistent=True)
#         self.register_buffer("alphas", schedule_tensors["alphas"], persistent=True)
#         self.register_buffer("alphas_cumprod", schedule_tensors["alphas_cumprod"], persistent=True)
#         self.register_buffer(
#             "alphas_cumprod_prev", schedule_tensors["alphas_cumprod_prev"], persistent=True
#         )
#         self.register_buffer(
#             "sqrt_alphas_cumprod", schedule_tensors["sqrt_alphas_cumprod"], persistent=True
#         )
#         self.register_buffer(
#             "sqrt_one_minus_alphas_cumprod",
#             schedule_tensors["sqrt_one_minus_alphas_cumprod"],
#             persistent=True,
#         )
#         self.register_buffer(
#             "posterior_variance", schedule_tensors["posterior_variance"], persistent=True
#         )

#     @staticmethod
#     def _sinusoidal_timestep_embedding(
#         timesteps: torch.Tensor, dim: int, max_period: int = 10000
#     ) -> torch.Tensor:
#         half = dim // 2
#         device = timesteps.device
#         if half == 0:
#             return timesteps.float().unsqueeze(-1)
#         exponents = torch.arange(half, device=device, dtype=torch.float32)
#         exponents = -math.log(max_period) * exponents / max(half - 1, 1)
#         freqs = torch.exp(exponents)
#         args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
#         emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
#         if dim % 2 == 1:
#             emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
#         return emb

#     def _normalize_timesteps(
#         self, timesteps: Optional[torch.Tensor], batch_size: int, device: torch.device
#     ) -> torch.Tensor:
#         if timesteps is None:
#             timesteps = torch.randint(
#                 low=0,
#                 high=self.num_diffusion_steps,
#                 size=(batch_size,),
#                 device=device,
#                 dtype=torch.long,
#             )
#         elif not isinstance(timesteps, torch.Tensor):
#             timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
#         else:
#             timesteps = timesteps.to(device=device, dtype=torch.long)

#         if timesteps.dim() == 0:
#             timesteps = timesteps.expand(batch_size)
#         elif timesteps.dim() == 1 and timesteps.size(0) == 1:
#             timesteps = timesteps.expand(batch_size)
#         elif timesteps.dim() != 1 or timesteps.size(0) != batch_size:
#             raise ValueError(
#                 f"`timesteps` must be shape (batch,), scalar, or (1,). Received {tuple(timesteps.size())}."
#             )

#         if (timesteps < 0).any() or (timesteps >= self.num_diffusion_steps).any():
#             raise ValueError(
#                 f"`timesteps` must lie in [0, {self.num_diffusion_steps - 1}] for this schedule."
#             )
#         return timesteps

#     def _resolve_mask_type_labels(
#         self,
#         attention_mask: torch.Tensor,
#         mask_type_labels: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         input_ids: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if mask_type_labels is not None:
#             if mask_type_labels.dim() != 2:
#                 raise ValueError("`mask_type_labels` must have shape (batch, seq_len).")
#             resolved = mask_type_labels.to(device=attention_mask.device, dtype=torch.long)
#         elif labels is not None:
#             if labels.dim() != 2:
#                 raise ValueError("`labels` must have shape (batch, seq_len) when used to infer masks.")
#             resolved = torch.where(
#                 labels.to(device=attention_mask.device) == -100,
#                 torch.ones_like(labels, dtype=torch.long, device=attention_mask.device),
#                 torch.full_like(labels, 2, dtype=torch.long, device=attention_mask.device),
#             )
#         elif input_ids is not None and (input_ids == -1).any():
#             resolved = torch.where(
#                 input_ids.to(device=attention_mask.device) == -1,
#                 torch.full_like(input_ids, 2, dtype=torch.long, device=attention_mask.device),
#                 torch.ones_like(input_ids, dtype=torch.long, device=attention_mask.device),
#             )
#         else:
#             raise ValueError(
#                 "Unable to infer action token region. Provide `mask_type_labels`, `labels`, "
#                 "or `input_ids` with -1 action placeholders."
#             )

#         if resolved.size() != attention_mask.size():
#             raise ValueError(
#                 "`mask_type_labels`/`labels` shape must match attention mask shape. "
#                 f"Got mask {tuple(resolved.size())} vs attention {tuple(attention_mask.size())}."
#             )
#         return resolved

#     @staticmethod
#     def _infer_context_action_labels(
#         mask_type_labels: torch.Tensor, attention_mask: torch.Tensor
#     ) -> Tuple[int, int]:
#         non_pad = mask_type_labels[attention_mask.to(torch.bool)]
#         has_label_two = (non_pad == 2).any()
#         has_label_one = (non_pad == 1).any()
#         has_label_zero = (non_pad == 0).any()

#         if has_label_two:
#             return 1, 2
#         if has_label_zero and has_label_one:
#             return 0, 1
#         return 1, 2

#     def _select_action_positions(
#         self,
#         mask_type_labels: torch.Tensor,
#         attention_mask: torch.Tensor,
#         required_tokens: int,
#         action_label: int,
#     ) -> torch.Tensor:
#         batch_size = mask_type_labels.size(0)
#         positions = []
#         valid_mask = attention_mask.to(torch.bool)
#         for row in range(batch_size):
#             action_pos = torch.nonzero(
#                 valid_mask[row] & (mask_type_labels[row] == action_label), as_tuple=False
#             ).squeeze(-1)
#             if action_pos.numel() < required_tokens:
#                 raise ValueError(
#                     "Insufficient action tokens for diffusion input. "
#                     f"Sample {row}: found {int(action_pos.numel())}, required {required_tokens}."
#                 )
#             positions.append(action_pos[:required_tokens])
#         return torch.stack(positions, dim=0)

#     @staticmethod
#     def _gather_tokens(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
#         hidden_dim = hidden_states.size(-1)
#         gather_idx = positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
#         return torch.gather(hidden_states, dim=1, index=gather_idx)

#     @staticmethod
#     def _scatter_tokens(
#         base_embeddings: torch.Tensor, positions: torch.Tensor, token_embeddings: torch.Tensor
#     ) -> torch.Tensor:
#         out = base_embeddings.clone()
#         batch_size = out.size(0)
#         batch_idx = torch.arange(batch_size, device=out.device).unsqueeze(1).expand_as(positions)
#         out[batch_idx, positions] = token_embeddings
#         return out

#     def _reshape_pred_seq(self, pred_seq: torch.Tensor) -> torch.Tensor:
#         if pred_seq is None:
#             raise ValueError("`pred_seq` (clean action target) is required for diffusion training.")
#         pred_seq = torch.as_tensor(pred_seq)
#         if pred_seq.dim() == 3:
#             if pred_seq.size(-1) != self._action_dim:
#                 raise ValueError(
#                     f"Expected per-step action dim {self._action_dim}, got {pred_seq.size(-1)}."
#                 )
#             return pred_seq
#         if pred_seq.dim() == 2:
#             if pred_seq.size(1) != self.horizon * self._action_dim:
#                 raise ValueError(
#                     f"Expected flattened action size {self.horizon * self._action_dim}, "
#                     f"got {pred_seq.size(1)}."
#                 )
#             return pred_seq.reshape(pred_seq.size(0), self.horizon, self._action_dim)
#         raise ValueError(
#             "`pred_seq` must have shape (batch, horizon, action_dim) or "
#             f"(batch, horizon*action_dim). Received {tuple(pred_seq.size())}."
#         )

#     def _chunk_action_sequence(self, action_seq: torch.Tensor) -> torch.Tensor:
#         if action_seq.dim() != 3:
#             raise ValueError(
#                 f"`action_seq` must have shape (batch, horizon, action_dim). Got {tuple(action_seq.size())}."
#             )
#         if action_seq.size(-1) != self._action_dim:
#             raise ValueError(
#                 f"`action_seq` last dim must equal {self._action_dim}. Got {action_seq.size(-1)}."
#             )

#         batch_size, horizon, _ = action_seq.size()
#         if horizon != self.horizon:
#             raise ValueError(
#                 f"`action_seq` second dim must equal horizon={self.horizon}. Got {horizon}."
#             )

#         total_steps = self._head_chunk_count * self._action_chunk_size
#         if horizon < total_steps:
#             pad = torch.zeros(
#                 batch_size,
#                 total_steps - horizon,
#                 self._action_dim,
#                 device=action_seq.device,
#                 dtype=action_seq.dtype,
#             )
#             action_seq = torch.cat([action_seq, pad], dim=1)
#         return action_seq.reshape(batch_size, self._head_chunk_count, self._action_chunk_dim)

#     def _unchunk_action_sequence(self, chunk_seq: torch.Tensor) -> torch.Tensor:
#         if chunk_seq.dim() != 3:
#             raise ValueError(
#                 f"`chunk_seq` must have shape (batch, tokens, chunk_dim). Got {tuple(chunk_seq.size())}."
#             )
#         if chunk_seq.size(1) != self._head_chunk_count:
#             raise ValueError(
#                 f"`chunk_seq` token length must be {self._head_chunk_count}. Got {chunk_seq.size(1)}."
#             )
#         if chunk_seq.size(2) != self._action_chunk_dim:
#             raise ValueError(
#                 f"`chunk_seq` last dim must be {self._action_chunk_dim}. Got {chunk_seq.size(2)}."
#             )

#         action = chunk_seq.reshape(
#             chunk_seq.size(0), self._head_chunk_count, self._action_chunk_size, self._action_dim
#         )
#         action = action.reshape(
#             chunk_seq.size(0), self._head_chunk_count * self._action_chunk_size, self._action_dim
#         )
#         if action.size(1) > self.horizon:
#             action = action[:, : self.horizon, :]
#         return action

#     def _prepare_action_sequence_like(self, value: torch.Tensor) -> torch.Tensor:
#         value = torch.as_tensor(value)
#         if (
#             value.dim() == 3
#             and value.size(1) == self.horizon
#             and value.size(2) == self._action_dim
#         ):
#             return value
#         if (
#             value.dim() == 3
#             and value.size(1) == self._head_chunk_count
#             and value.size(2) == self._action_chunk_dim
#         ):
#             return self._unchunk_action_sequence(value)
#         if value.dim() == 2:
#             if value.size(1) == self.horizon * self._action_dim:
#                 return value.reshape(value.size(0), self.horizon, self._action_dim)
#             if value.size(1) == self._head_chunk_count * self._action_chunk_dim:
#                 return self._unchunk_action_sequence(
#                     value.reshape(value.size(0), self._head_chunk_count, self._action_chunk_dim)
#                 )
#         raise ValueError(
#             "Unsupported shape for action/noise tensor. Expected one of:\n"
#             f"- (batch, {self.horizon}, {self._action_dim})\n"
#             f"- (batch, {self._head_chunk_count}, {self._action_chunk_dim})\n"
#             f"- (batch, {self.horizon * self._action_dim})\n"
#             f"- (batch, {self._head_chunk_count * self._action_chunk_dim})"
#         )

#     def _prepare_action_chunks_like(self, value: torch.Tensor) -> torch.Tensor:
#         """
#         Backward-compatible alias for callers using the older helper name.
#         Returns action sequences of shape (batch, horizon, action_dim).
#         """
#         return self._prepare_action_sequence_like(value)

#     def _chunk_hidden_states_for_head(self, action_hidden: torch.Tensor) -> torch.Tensor:
#         if action_hidden.dim() != 3:
#             raise ValueError(
#                 f"`action_hidden` must have shape (batch, horizon, hidden). Got {tuple(action_hidden.size())}."
#             )
#         if action_hidden.size(1) != self._diffusion_action_token_count:
#             raise ValueError(
#                 f"`action_hidden` sequence length must equal horizon={self._diffusion_action_token_count}. "
#                 f"Got {action_hidden.size(1)}."
#             )
#         batch_size, seq_len, hidden_dim = action_hidden.size()
#         total_steps = self._head_chunk_count * self._action_chunk_size
#         if seq_len < total_steps:
#             pad = torch.zeros(
#                 batch_size,
#                 total_steps - seq_len,
#                 hidden_dim,
#                 device=action_hidden.device,
#                 dtype=action_hidden.dtype,
#             )
#             action_hidden = torch.cat([action_hidden, pad], dim=1)
#         return action_hidden.reshape(batch_size, self._head_chunk_count, hidden_dim * self._action_chunk_size)

#     def _decode_noise_chunks(self, embeddings: torch.Tensor) -> torch.Tensor:
#         if embeddings.dim() != 3:
#             raise ValueError(
#                 f"`embeddings` must have shape (batch, horizon, hidden). Got {tuple(embeddings.size())}."
#             )
#         if embeddings.size(1) != self._diffusion_action_token_count:
#             raise ValueError(
#                 f"Expected horizon={self._diffusion_action_token_count} action tokens, "
#                 f"received {embeddings.size(1)}."
#             )

#         chunked_hidden = self._chunk_hidden_states_for_head(embeddings)
#         batch_size, chunk_count, chunk_input_dim = chunked_hidden.size()
#         if chunk_input_dim != self._head_input_dim:
#             raise ValueError(
#                 f"Chunked hidden dim ({chunk_input_dim}) does not match head input dim ({self._head_input_dim})."
#             )

#         decoded = self.action_decoder(chunked_hidden.reshape(batch_size * chunk_count, chunk_input_dim))
#         return decoded.reshape(batch_size, chunk_count, self._action_chunk_dim)

#     def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if noise is None:
#             noise = torch.randn_like(x0)
#         if x0.size() != noise.size():
#             raise ValueError("`x0` and `noise` must have the same shape for q-sampling.")

#         timesteps = timesteps.to(device=x0.device, dtype=torch.long)
#         sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1).to(dtype=x0.dtype, device=x0.device)
#         sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1).to(
#             dtype=x0.dtype, device=x0.device
#         )
#         return sqrt_alpha * x0 + sqrt_one_minus * noise

#     def _predict_x0_from_noise(
#         self, x_t: torch.Tensor, pred_noise: torch.Tensor, timesteps: torch.Tensor
#     ) -> torch.Tensor:
#         timesteps = timesteps.to(device=x_t.device, dtype=torch.long)
#         sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1).to(dtype=x_t.dtype, device=x_t.device)
#         sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1).to(
#             dtype=x_t.dtype, device=x_t.device
#         )
#         return (x_t - sqrt_one_minus * pred_noise) / torch.clamp(sqrt_alpha, min=1e-12)

#     def _build_inputs_with_xt_and_t(
#         self,
#         *,
#         input_ids: Optional[torch.LongTensor],
#         inputs_embeds: Optional[torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         mask_type_labels: Optional[torch.Tensor],
#         labels: Optional[torch.Tensor],
#         position_ids: Optional[torch.LongTensor],
#         x_t_seq: torch.Tensor,
#         timesteps: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")

#         if inputs_embeds is None:
#             if input_ids is None:
#                 raise ValueError("`input_ids` or `inputs_embeds` must be provided.")
#             base_embeds = self._input_ids_to_embeds_with_empty_tokens(input_ids)
#         else:
#             base_embeds = inputs_embeds

#         batch_size, seq_len, _ = base_embeds.size()
#         device = base_embeds.device

#         if attention_mask is None:
#             attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
#         else:
#             attention_mask = attention_mask.to(device=device)
#             if attention_mask.dim() != 2 or attention_mask.size(0) != batch_size or attention_mask.size(1) != seq_len:
#                 raise ValueError(
#                     "`attention_mask` must have shape (batch, seq_len) matching token inputs. "
#                     f"Got {tuple(attention_mask.size())}, expected {(batch_size, seq_len)}."
#                 )

#         resolved_mask_labels = self._resolve_mask_type_labels(
#             attention_mask=attention_mask,
#             mask_type_labels=mask_type_labels,
#             labels=labels,
#             input_ids=input_ids,
#         )
#         _, action_label = self._infer_context_action_labels(
#             mask_type_labels=resolved_mask_labels, attention_mask=attention_mask
#         )
#         action_positions = self._select_action_positions(
#             mask_type_labels=resolved_mask_labels,
#             attention_mask=attention_mask,
#             required_tokens=self._diffusion_action_token_count,
#             action_label=action_label,
#         )

#         if x_t_seq.dim() != 3:
#             raise ValueError(
#                 f"`x_t_seq` must have shape (batch, horizon, action_dim). Got {tuple(x_t_seq.size())}."
#             )
#         if x_t_seq.size(0) != batch_size or x_t_seq.size(1) != self._diffusion_action_token_count:
#             raise ValueError(
#                 f"`x_t_seq` must have shape ({batch_size}, {self._diffusion_action_token_count}, action_dim). "
#                 f"Got {tuple(x_t_seq.size())}."
#             )
#         if x_t_seq.size(2) != self._action_dim:
#             raise ValueError(
#                 f"`x_t_seq` last dimension must equal action_dim={self._action_dim}. "
#                 f"Got {x_t_seq.size(2)}."
#             )

#         proj_dtype = next(self.action_input_proj.parameters()).dtype
#         action_tokens = self.action_input_proj(x_t_seq.to(device=device, dtype=proj_dtype))
#         action_tokens = action_tokens.to(dtype=base_embeds.dtype)
#         with_actions = self._scatter_tokens(base_embeds, action_positions, action_tokens)

#         t_raw = self._sinusoidal_timestep_embedding(timesteps=timesteps, dim=self.config.hidden_size)
#         t_raw = t_raw.to(device=device, dtype=next(self.time_embed_mlp.parameters()).dtype)
#         t_token = self.time_embed_mlp(t_raw).to(dtype=base_embeds.dtype).unsqueeze(1)

#         final_embeds = torch.cat([with_actions, t_token], dim=1)

#         t_attn = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
#         final_attention_mask = torch.cat([attention_mask, t_attn], dim=1)

#         t_labels = torch.full(
#             (batch_size, 1),
#             fill_value=int(action_label),
#             dtype=resolved_mask_labels.dtype,
#             device=device,
#         )
#         final_mask_type_labels = torch.cat([resolved_mask_labels, t_labels], dim=1)

#         final_position_ids = None
#         if position_ids is not None:
#             position_ids = position_ids.to(device=device, dtype=torch.long)
#             if position_ids.dim() != 2 or position_ids.size(0) != batch_size or position_ids.size(1) != seq_len:
#                 raise ValueError(
#                     "`position_ids` must have shape (batch, seq_len) matching token inputs. "
#                     f"Got {tuple(position_ids.size())}, expected {(batch_size, seq_len)}."
#                 )
#             next_pos = position_ids[:, -1:] + 1
#             final_position_ids = torch.cat([position_ids, next_pos], dim=1)

#         return (
#             final_embeds,
#             final_attention_mask,
#             final_mask_type_labels,
#             final_position_ids,
#             action_positions,
#         )

#     def _predict_noise_from_xt(
#         self,
#         *,
#         input_ids: Optional[torch.LongTensor],
#         inputs_embeds: Optional[torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         position_ids: Optional[torch.LongTensor],
#         mask_type_labels: Optional[torch.Tensor],
#         labels: Optional[torch.Tensor],
#         x_t_seq: torch.Tensor,
#         timesteps: torch.Tensor,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         return_dict: bool = True,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Any]:
#         (
#             diffusion_embeds,
#             diffusion_attention_mask,
#             diffusion_mask_labels,
#             diffusion_position_ids,
#             action_positions,
#         ) = self._build_inputs_with_xt_and_t(
#             input_ids=input_ids,
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             mask_type_labels=mask_type_labels,
#             labels=labels,
#             position_ids=position_ids,
#             x_t_seq=x_t_seq,
#             timesteps=timesteps,
#         )

#         lm_outputs = super().forward(
#             input_ids=None,
#             attention_mask=diffusion_attention_mask,
#             position_ids=diffusion_position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=diffusion_embeds,
#             labels=None,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=True,
#             return_dict=True,
#             mask_type_labels=diffusion_mask_labels,
#             task="language",
#             **kwargs,
#         )

#         hidden_states = lm_outputs.hidden_states[-1]
#         action_hidden = self._gather_tokens(hidden_states, action_positions)
#         pred_noise_chunks = self._decode_noise_chunks(action_hidden)

#         if return_dict:
#             return pred_noise_chunks, lm_outputs
#         return pred_noise_chunks, tuple(value for value in lm_outputs.values() if value is not None)

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         mask_type_labels: Optional[torch.Tensor] = None,
#         task: str = "action",
#         loss_type: Optional[str] = None,
#         loss_horizon: Optional[int] = None,
#         pred_seq=None,
#         timesteps: Optional[torch.Tensor] = None,
#         noisy_actions: Optional[torch.Tensor] = None,
#         noise: Optional[torch.Tensor] = None,
#         use_ce_loss: bool = False,
#         **kwargs,
#     ):
#         del use_ce_loss  # diffusion model uses only noise-prediction MSE in action mode.

#         if task != "action":
#             return super().forward(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 inputs_embeds=inputs_embeds,
#                 labels=labels,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 mask_type_labels=mask_type_labels,
#                 task=task,
#                 loss_type=loss_type,
#                 loss_horizon=loss_horizon,
#                 pred_seq=pred_seq,
#                 **kwargs,
#             )

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         effective_output_hidden_states = output_hidden_states or self.config.output_hidden_states

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")

#         source_tensor = input_ids if input_ids is not None else inputs_embeds
#         if source_tensor is None:
#             raise ValueError("`input_ids` or `inputs_embeds` must be provided for diffusion action mode.")
#         batch_size = source_tensor.size(0)
#         device = source_tensor.device
#         self._ensure_diffusion_schedule_precision(device=device)

#         timesteps = self._normalize_timesteps(timesteps=timesteps, batch_size=batch_size, device=device)
#         kwargs = dict(kwargs)
#         kwargs.pop("tokenizer", None)
#         kwargs.pop("loss_type", None)
#         kwargs.pop("loss_horizon", None)

#         target_noise_seq = None
#         clean_seq = None

#         if noisy_actions is not None:
#             x_t_seq = self._prepare_action_sequence_like(noisy_actions).to(
#                 device=device, dtype=torch.float32
#             )
#             if noise is not None:
#                 target_noise_seq = self._prepare_action_sequence_like(noise).to(
#                     device=device, dtype=torch.float32
#                 )
#         else:
#             if pred_seq is None:
#                 raise ValueError(
#                     "Provide `pred_seq` for training (q-sampled internally), or pass `noisy_actions` + `timesteps`."
#                 )
#             clean_seq = self._reshape_pred_seq(pred_seq).to(device=device, dtype=torch.float32)
#             target_noise_seq = (
#                 torch.randn_like(clean_seq)
#                 if noise is None
#                 else self._prepare_action_sequence_like(noise).to(device=device, dtype=torch.float32)
#             )
#             x_t_seq = self.q_sample(clean_seq, timesteps, target_noise_seq)

#         pred_noise_chunks, lm_outputs = self._predict_noise_from_xt(
#             input_ids=input_ids,
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             mask_type_labels=mask_type_labels,
#             labels=labels,
#             x_t_seq=x_t_seq,
#             timesteps=timesteps,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             return_dict=True,
#             **kwargs,
#         )
#         pred_noise_seq = self._unchunk_action_sequence(pred_noise_chunks)

#         diffusion_loss = None
#         if target_noise_seq is not None:
#             diffusion_loss = F.mse_loss(pred_noise_seq.float(), target_noise_seq.float())
#             diffusion_loss = diffusion_loss.to(dtype=pred_noise_seq.dtype)

#         flattened_pred_noise = pred_noise_seq.reshape(pred_noise_seq.size(0), -1)

#         pred_x0_seq = self._predict_x0_from_noise(x_t_seq, pred_noise_seq, timesteps)
#         pred_x0_chunks = self._chunk_action_sequence(pred_x0_seq)
#         pred_x0_actions = self._unchunk_action_sequence(pred_x0_chunks)

#         outputs = CausalLMOutputWithPastAndActions(
#             loss=diffusion_loss,
#             logits=lm_outputs.logits,
#             past_key_values=lm_outputs.past_key_values,
#             hidden_states=(lm_outputs.hidden_states if effective_output_hidden_states else None),
#             attentions=lm_outputs.attentions,
#             action_head_output=flattened_pred_noise,
#         )
#         outputs["action_loss"] = diffusion_loss
#         outputs["language_model_loss"] = None
#         outputs["cross_entropy_loss"] = None
#         outputs["action_prediction_loss"] = diffusion_loss
#         outputs["smoothness_loss"] = None
#         outputs["vec_order_loss"] = None
#         outputs["predicted_noise"] = pred_noise_seq
#         outputs["predicted_noise_chunks"] = pred_noise_chunks
#         outputs["predicted_x0"] = pred_x0_actions
#         outputs["timesteps"] = timesteps
#         outputs["noisy_actions"] = x_t_seq
#         if clean_seq is not None:
#             outputs["clean_actions"] = clean_seq

#         if not return_dict:
#             return tuple(value for value in outputs.values() if value is not None)
#         return outputs

#     def _ddpm_step(
#         self,
#         x_t: torch.Tensor,
#         pred_x0: torch.Tensor,
#         timestep: int,
#         generator: Optional[torch.Generator] = None,
#     ) -> torch.Tensor:
#         if timestep == 0:
#             # Numerically stable terminal step: q(x_{-1} | x_0, x_0) collapses to x_0.
#             return pred_x0.to(device=x_t.device, dtype=x_t.dtype)

#         beta_t = self.betas[timestep].to(device=x_t.device, dtype=x_t.dtype)
#         alpha_t = self.alphas[timestep].to(device=x_t.device, dtype=x_t.dtype)
#         alpha_cumprod_t = self.alphas_cumprod[timestep].to(device=x_t.device, dtype=x_t.dtype)
#         alpha_cumprod_prev = self.alphas_cumprod_prev[timestep].to(device=x_t.device, dtype=x_t.dtype)

#         denom = torch.clamp(1.0 - alpha_cumprod_t, min=1e-12)
#         coef_x0 = torch.sqrt(alpha_cumprod_prev) * beta_t / denom
#         coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) / denom
#         posterior_mean = coef_x0 * pred_x0 + coef_xt * x_t

#         variance = self.posterior_variance[timestep].to(device=x_t.device, dtype=x_t.dtype)
#         noise = torch.randn(
#             x_t.shape,
#             device=x_t.device,
#             dtype=x_t.dtype,
#             generator=generator,
#         )
#         return posterior_mean + torch.sqrt(variance) * noise

#     @torch.no_grad()
#     def _denormalize_actions(self, flattened_actions: torch.Tensor) -> torch.Tensor:
#         stats = getattr(self, "action_mean_std", None)
#         if not isinstance(stats, dict):
#             return flattened_actions

#         mean_values = stats.get("mean")
#         std_values = stats.get("std")
#         if mean_values is None or std_values is None:
#             return flattened_actions

#         mean = torch.as_tensor(
#             mean_values, device=flattened_actions.device, dtype=flattened_actions.dtype
#         )
#         std = torch.as_tensor(
#             std_values, device=flattened_actions.device, dtype=flattened_actions.dtype
#         )

#         if mean.numel() == 0 or std.numel() == 0 or mean.numel() != std.numel():
#             return flattened_actions

#         if flattened_actions.size(-1) % mean.numel() != 0:
#             raise ValueError(
#                 "Cannot denormalize flattened actions: last dimension "
#                 f"({flattened_actions.size(-1)}) is not divisible by action stats size "
#                 f"({mean.numel()})."
#             )

#         action_dim = mean.numel()
#         action_seq = flattened_actions.view(flattened_actions.size(0), -1, action_dim)
#         denorm_seq = action_seq * std.view(1, 1, -1) + mean.view(1, 1, -1)
#         return denorm_seq.reshape(flattened_actions.size(0), -1)

#     @torch.no_grad()
#     def ddpm_sample_actions(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         mask_type_labels: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         num_inference_steps: Optional[int] = None,
#         generator: Optional[torch.Generator] = None,
#         clip_denoised: bool = True,
#         clip_range: float = 1.0,
#         return_generation_output: bool = False,
#         **kwargs,
#     ):
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")
#         if input_ids is None and inputs_embeds is None:
#             raise ValueError("`input_ids` or `inputs_embeds` must be provided for DDPM sampling.")

#         source_tensor = input_ids if input_ids is not None else inputs_embeds
#         batch_size = source_tensor.size(0)
#         device = source_tensor.device
#         self._ensure_diffusion_schedule_precision(device=device)

#         if num_inference_steps is None:
#             num_inference_steps = self.num_diffusion_steps
#         if num_inference_steps != self.num_diffusion_steps:
#             raise ValueError(
#                 f"This model was initialized with diffusion_num_steps={self.num_diffusion_steps}; "
#                 f"received num_inference_steps={num_inference_steps}."
#             )
#         kwargs = dict(kwargs)
#         kwargs.pop("tokenizer", None)
#         kwargs.pop("loss_type", None)
#         kwargs.pop("loss_horizon", None)

#         x_t = torch.randn(
#             (batch_size, self._diffusion_action_token_count, self._action_dim),
#             device=device,
#             dtype=torch.float32,
#             generator=generator,
#         )
#         last_pred_noise = None
#         last_pred_x0 = None

#         for t in reversed(range(num_inference_steps)):
#             t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
#             pred_noise_chunks, _ = self._predict_noise_from_xt(
#                 input_ids=input_ids,
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 mask_type_labels=mask_type_labels,
#                 labels=labels,
#                 x_t_seq=x_t,
#                 timesteps=t_batch,
#                 use_cache=False,
#                 output_attentions=False,
#                 return_dict=True,
#                 **kwargs,
#             )
#             pred_noise_seq = self._unchunk_action_sequence(pred_noise_chunks).to(dtype=torch.float32)
#             pred_x0_chunks = self._chunk_action_sequence(
#                 self._predict_x0_from_noise(x_t, pred_noise_seq, t_batch)
#             )
#             if clip_denoised:
#                 pred_x0_chunks = torch.clamp(pred_x0_chunks, min=-clip_range, max=clip_range)
#             pred_x0_seq = self._unchunk_action_sequence(pred_x0_chunks)
#             x_t_next = self._ddpm_step(x_t=x_t, pred_x0=pred_x0_seq, timestep=t, generator=generator)
#             x_t = x_t_next
#             last_pred_noise = pred_noise_seq
#             last_pred_x0 = pred_x0_chunks

#         final_actions = x_t
#         flattened_actions = final_actions.reshape(final_actions.size(0), -1)

#         flattened_actions = self._denormalize_actions(flattened_actions)

#         if return_generation_output:
#             generation_output = {
#                 "action_sequence": final_actions,
#                 "final_chunk_actions": self._chunk_action_sequence(final_actions),
#                 "predicted_noise": last_pred_noise,
#                 "predicted_x0": self._unchunk_action_sequence(last_pred_x0),
#             }
#             return flattened_actions, generation_output
#         return flattened_actions

#     @torch.no_grad()
#     def action_head_based_generate_actions(
#         self,
#         input_ids: Optional[torch.LongTensor],
#         tokenizer: Any = None,
#         attention_mask: Optional[torch.LongTensor] = None,
#         mask_type_labels: Optional[torch.Tensor] = None,
#         max_new_tokens: Optional[int] = None,
#         return_generation_output: bool = False,
#         **kwargs,
#     ):
#         del tokenizer  # unused for continuous diffusion action sampling
#         if max_new_tokens is not None and max_new_tokens < self.horizon:
#             raise ValueError(
#                 f"`max_new_tokens` must be at least the action horizon ({self.horizon}). "
#                 f"Received {max_new_tokens}."
#             )
#         return self.ddpm_sample_actions(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             mask_type_labels=mask_type_labels,
#             return_generation_output=return_generation_output,
#             **kwargs,
#         )


# ============================================================================
# New diffusion implementation
#
# The legacy `LlamaForBidirectionAttnWithDiffusionActions` above is intentionally
# commented out for reference. The class below keeps the same external interface
# and output contract, but switches to a conditional DDPM head where:
# - LLM hidden states provide conditioning embeddings.
# - The diffusion head can predict either clean action (x0) or noise (epsilon).
# - The prediction target is selected by `diffusion_prediction_type` (default: action).
# ============================================================================


def _valid_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConditionalResBlock1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        groups = _valid_group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.cond_proj(cond).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)

        h = self.norm1(x)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)

        return x + h


class _ConditionalUNet1D(nn.Module):
    """
    Standard DDPM-style 1D UNet adapted for action sequences.
    Input/Output shape: (batch, horizon, action_dim).
    """

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        cond_token_dim: int,
        model_channels: int,
    ):
        super().__init__()

        c1 = model_channels
        c2 = model_channels * 2

        self.in_proj = nn.Conv1d(action_dim, c1, kernel_size=3, padding=1)

        self.down_block = _ConditionalResBlock1D(c1, cond_dim)
        self.downsample = nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1)

        self.mid_block1 = _ConditionalResBlock1D(c2, cond_dim)
        self.mid_block2 = _ConditionalResBlock1D(c2, cond_dim)

        attn_heads = 4
        while c2 % attn_heads != 0 and attn_heads > 1:
            attn_heads -= 1
        self.cond_kv_proj = nn.Linear(cond_token_dim, c2)
        self.cross_attn = nn.MultiheadAttention(c2, num_heads=attn_heads, batch_first=True)

        self.upsample = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up_block = _ConditionalResBlock1D(c1 * 2, cond_dim)

        out_groups = _valid_group_count(c1 * 2)
        self.out_norm = nn.GroupNorm(out_groups, c1 * 2)
        self.out_proj = nn.Conv1d(c1 * 2, action_dim, kernel_size=3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        cond_tokens: torch.Tensor,
        cond_vector: torch.Tensor,
    ) -> torch.Tensor:
        x = x_t.transpose(1, 2)

        h1 = self.in_proj(x)
        h1 = self.down_block(h1, cond_vector)

        h2 = self.downsample(h1)
        h2 = self.mid_block1(h2, cond_vector)
        h2 = self.mid_block2(h2, cond_vector)

        cond_seq = self.cond_kv_proj(cond_tokens)
        h2_seq = h2.transpose(1, 2)
        attn_out, _ = self.cross_attn(h2_seq, cond_seq, cond_seq, need_weights=False)
        h2 = (h2_seq + attn_out).transpose(1, 2)

        up = self.upsample(h2)
        if up.size(-1) != h1.size(-1):
            up = F.interpolate(up, size=h1.size(-1), mode="nearest")

        up = torch.cat([up, h1], dim=1)
        up = self.up_block(up, cond_vector)

        out = self.out_norm(up)
        out = F.silu(out)
        out = self.out_proj(out)

        return out.transpose(1, 2)


class LlamaForBidirectionAttnWithDiffusionActions(LlamaForBidirectionAttnWithActions):
    """
    New diffusion action model (replacement).

    Distinction from the commented legacy class above:
    - This class follows the action-model structure but replaces the direct action
      decoder with a conditional DDPM head.
    - LLM backbone hidden states are used as conditioning embeddings.
    - `action_head_output` is the configured diffusion prediction target:
      clean action (x0, default) or noise (epsilon).
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_diffusion_steps = int(getattr(config, "diffusion_num_steps", 50))
        if self.num_diffusion_steps <= 0:
            raise ValueError("`diffusion_num_steps` must be a positive integer.")

        self.diffusion_beta_schedule = str(
            getattr(config, "diffusion_beta_schedule", "squaredcos_cap_v2")
        ).lower()
        self.diffusion_prediction_type = str(
            getattr(config, "diffusion_prediction_type", "action")
        ).lower()
        if self.diffusion_prediction_type in {"noise", "eps", "epsilon"}:
            self.diffusion_prediction_type = "epsilon"
        elif self.diffusion_prediction_type in {"action", "x0", "sample"}:
            self.diffusion_prediction_type = "action"
        else:
            raise ValueError(
                f"Unsupported diffusion prediction type '{self.diffusion_prediction_type}'. "
                "Use one of: action/x0/sample or epsilon/noise/eps."
            )

        self._diffusion_action_token_count = int(self.horizon)
        if self._action_chunk_size != 1:
            raise ValueError(
                "Diffusion action model expects `action_chunk_size=1` so condition/action tokens "
                "share the same horizon."
            )

        self._head_chunk_count = (
            self._diffusion_action_token_count + self._action_chunk_size - 1
        ) // self._action_chunk_size
        self._action_chunk_dim = self._action_dim * self._action_chunk_size

        hidden_size = int(self.config.hidden_size)
        self.diffusion_input_dim = int(
            getattr(config, "diffusion_input_dim", getattr(config, "diffusion_condition_dim", hidden_size))
        )
        if self.diffusion_input_dim <= 0:
            raise ValueError("`diffusion_input_dim` must be positive.")

        self.diffusion_time_embed_dim = int(
            getattr(config, "diffusion_time_embed_dim", hidden_size)
        )
        if self.diffusion_time_embed_dim <= 0:
            raise ValueError("`diffusion_time_embed_dim` must be positive.")

        self.diffusion_condition_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.diffusion_input_dim),
        )
        self.diffusion_noisy_action_proj = nn.Sequential(
            nn.LayerNorm(self._action_dim),
            nn.Linear(self._action_dim, self.diffusion_input_dim),
        )
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(self.diffusion_time_embed_dim, self.diffusion_input_dim),
            nn.SiLU(),
            nn.Linear(self.diffusion_input_dim, self.diffusion_input_dim),
        )
        self.diffusion_noise_decoder = self._build_action_decoder(
            input_dim=self.diffusion_input_dim,
            hidden_dim=self._action_hidden_dim,
            num_layers=self._action_num_layers,
            action_dim=self._action_dim,
        )
        self.diffusion_noise_decoder.apply(self._init_weights)

        self._register_diffusion_schedule(
            num_steps=self.num_diffusion_steps,
            schedule_name=self.diffusion_beta_schedule,
        )
        self._align_diffusion_modules_dtype()

        self.config.diffusion_num_steps = self.num_diffusion_steps
        self.config.diffusion_beta_schedule = self.diffusion_beta_schedule
        self.config.diffusion_prediction_type = self.diffusion_prediction_type
        self.config.diffusion_input_dim = self.diffusion_input_dim
        self.config.diffusion_condition_dim = self.diffusion_input_dim
        self.config.diffusion_time_embed_dim = self.diffusion_time_embed_dim

        # Temporary hardcoded action stats (legacy-compatible) for inference-time denormalization.
        self.action_mean_std = {
            "mean": [
                0.6479015011446655,
                0.03882767111321254,
            ],
            "std": [
                0.535827732824574,
                0.4321850248635431,
            ],
        }

    def _align_diffusion_modules_dtype(self) -> None:
        base_weight = self.model.embed_tokens.weight
        self.diffusion_condition_proj.to(device=base_weight.device, dtype=base_weight.dtype)
        self.diffusion_noisy_action_proj.to(device=base_weight.device, dtype=base_weight.dtype)
        self.diffusion_time_mlp.to(device=base_weight.device, dtype=base_weight.dtype)
        self.diffusion_noise_decoder.to(device=base_weight.device, dtype=base_weight.dtype)

    def reset_action_head_parameters(self):
        """
        Reinitialize diffusion-head parameters when `untrained_action_head=True`.
        Keeps compatibility with the parent API while resetting diffusion modules.
        """
        # Keep parent behavior for inherited action-decoder members.
        super().reset_action_head_parameters()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        diffusion_modules = [
            self.diffusion_condition_proj,
            self.diffusion_noisy_action_proj,
            self.diffusion_time_mlp,
            self.diffusion_noise_decoder,
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

    def _resolve_mask_type_labels(
        self,
        attention_mask: torch.Tensor,
        mask_type_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask_type_labels is not None:
            if mask_type_labels.dim() != 2:
                raise ValueError("`mask_type_labels` must have shape (batch, seq_len).")
            resolved = mask_type_labels.to(device=attention_mask.device, dtype=torch.long)
        elif labels is not None:
            if labels.dim() != 2:
                raise ValueError("`labels` must have shape (batch, seq_len) when used to infer masks.")
            resolved = torch.where(
                labels.to(device=attention_mask.device) == -100,
                torch.ones_like(labels, dtype=torch.long, device=attention_mask.device),
                torch.full_like(labels, 2, dtype=torch.long, device=attention_mask.device),
            )
        elif input_ids is not None and (input_ids == -1).any():
            resolved = torch.where(
                input_ids.to(device=attention_mask.device) == -1,
                torch.full_like(input_ids, 2, dtype=torch.long, device=attention_mask.device),
                torch.ones_like(input_ids, dtype=torch.long, device=attention_mask.device),
            )
        else:
            raise ValueError(
                "Unable to infer action token region. Provide `mask_type_labels`, `labels`, "
                "or `input_ids` with -1 action placeholders."
            )

        if resolved.size() != attention_mask.size():
            raise ValueError(
                "`mask_type_labels`/`labels` shape must match attention mask shape. "
                f"Got mask {tuple(resolved.size())} vs attention {tuple(attention_mask.size())}."
            )
        return resolved

    @staticmethod
    def _infer_context_action_labels(
        mask_type_labels: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[int, int]:
        non_pad = mask_type_labels[attention_mask.to(torch.bool)]
        has_label_two = (non_pad == 2).any()
        has_label_one = (non_pad == 1).any()
        has_label_zero = (non_pad == 0).any()

        if has_label_two:
            return 1, 2
        if has_label_zero and has_label_one:
            return 0, 1
        return 1, 2

    def _select_action_positions(
        self,
        mask_type_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        required_tokens: int,
        action_label: int,
    ) -> torch.Tensor:
        batch_size = mask_type_labels.size(0)
        positions = []
        valid_mask = attention_mask.to(torch.bool)
        for row in range(batch_size):
            action_pos = torch.nonzero(
                valid_mask[row] & (mask_type_labels[row] == action_label), as_tuple=False
            ).squeeze(-1)
            if action_pos.numel() < required_tokens:
                raise ValueError(
                    "Insufficient action tokens for diffusion conditioning. "
                    f"Sample {row}: found {int(action_pos.numel())}, required {required_tokens}."
                )
            positions.append(action_pos[:required_tokens])
        return torch.stack(positions, dim=0)

    @staticmethod
    def _gather_tokens(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_dim = hidden_states.size(-1)
        gather_idx = positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return torch.gather(hidden_states, dim=1, index=gather_idx)

    def _reshape_pred_seq(self, pred_seq: torch.Tensor) -> torch.Tensor:
        if pred_seq is None:
            raise ValueError("`pred_seq` (clean action target) is required for diffusion training.")
        pred_seq = torch.as_tensor(pred_seq)
        if pred_seq.dim() == 3:
            if pred_seq.size(-1) != self._action_dim:
                raise ValueError(
                    f"Expected per-step action dim {self._action_dim}, got {pred_seq.size(-1)}."
                )
            if pred_seq.size(1) != self.horizon:
                raise ValueError(
                    f"Expected horizon {self.horizon}, got {pred_seq.size(1)}."
                )
            return pred_seq
        if pred_seq.dim() == 2:
            if pred_seq.size(1) != self.horizon * self._action_dim:
                raise ValueError(
                    f"Expected flattened action size {self.horizon * self._action_dim}, "
                    f"got {pred_seq.size(1)}."
                )
            return pred_seq.reshape(pred_seq.size(0), self.horizon, self._action_dim)
        raise ValueError(
            "`pred_seq` must have shape (batch, horizon, action_dim) or "
            f"(batch, horizon*action_dim). Received {tuple(pred_seq.size())}."
        )

    def _chunk_action_sequence(self, action_seq: torch.Tensor) -> torch.Tensor:
        if action_seq.dim() != 3:
            raise ValueError(
                f"`action_seq` must have shape (batch, horizon, action_dim). Got {tuple(action_seq.size())}."
            )
        if action_seq.size(1) != self.horizon or action_seq.size(2) != self._action_dim:
            raise ValueError(
                f"Expected action sequence shape (batch, {self.horizon}, {self._action_dim}); "
                f"got {tuple(action_seq.size())}."
            )

        batch_size = action_seq.size(0)
        total_steps = self._head_chunk_count * self._action_chunk_size
        if self.horizon < total_steps:
            pad = torch.zeros(
                batch_size,
                total_steps - self.horizon,
                self._action_dim,
                device=action_seq.device,
                dtype=action_seq.dtype,
            )
            action_seq = torch.cat([action_seq, pad], dim=1)
        return action_seq.reshape(batch_size, self._head_chunk_count, self._action_chunk_dim)

    def _unchunk_action_sequence(self, chunk_seq: torch.Tensor) -> torch.Tensor:
        if chunk_seq.dim() != 3:
            raise ValueError(
                f"`chunk_seq` must have shape (batch, tokens, chunk_dim). Got {tuple(chunk_seq.size())}."
            )
        if chunk_seq.size(1) != self._head_chunk_count or chunk_seq.size(2) != self._action_chunk_dim:
            raise ValueError(
                f"Expected chunk sequence shape (batch, {self._head_chunk_count}, {self._action_chunk_dim}); "
                f"got {tuple(chunk_seq.size())}."
            )

        action = chunk_seq.reshape(
            chunk_seq.size(0), self._head_chunk_count, self._action_chunk_size, self._action_dim
        )
        action = action.reshape(
            chunk_seq.size(0), self._head_chunk_count * self._action_chunk_size, self._action_dim
        )
        return action[:, : self.horizon, :]

    def _prepare_action_sequence_like(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(value)
        if (
            value.dim() == 3
            and value.size(1) == self.horizon
            and value.size(2) == self._action_dim
        ):
            return value
        if (
            value.dim() == 3
            and value.size(1) == self._head_chunk_count
            and value.size(2) == self._action_chunk_dim
        ):
            return self._unchunk_action_sequence(value)
        if value.dim() == 2:
            if value.size(1) == self.horizon * self._action_dim:
                return value.reshape(value.size(0), self.horizon, self._action_dim)
            if value.size(1) == self._head_chunk_count * self._action_chunk_dim:
                return self._unchunk_action_sequence(
                    value.reshape(value.size(0), self._head_chunk_count, self._action_chunk_dim)
                )
        raise ValueError(
            "Unsupported shape for action/noise tensor. Expected one of:\n"
            f"- (batch, {self.horizon}, {self._action_dim})\n"
            f"- (batch, {self._head_chunk_count}, {self._action_chunk_dim})\n"
            f"- (batch, {self.horizon * self._action_dim})\n"
            f"- (batch, {self._head_chunk_count * self._action_chunk_dim})"
        )

    def _prepare_action_chunks_like(self, value: torch.Tensor) -> torch.Tensor:
        return self._prepare_action_sequence_like(value)

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

    def _encode_condition_from_backbone(
        self,
        *,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        mask_type_labels: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("`input_ids` or `inputs_embeds` must be provided.")

        source = input_ids if input_ids is not None else inputs_embeds
        batch_size = source.size(0)
        seq_len = source.size(1)
        device = source.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device)
            if attention_mask.dim() != 2 or attention_mask.size() != (batch_size, seq_len):
                raise ValueError(
                    "`attention_mask` must have shape (batch, seq_len) matching token inputs. "
                    f"Got {tuple(attention_mask.size())}, expected {(batch_size, seq_len)}."
                )

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

        hidden_states = lm_outputs.hidden_states[-1]
        cond_tokens = self._select_action_token_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels,
        )
        if cond_tokens.size(1) != self._diffusion_action_token_count:
            raise ValueError(
                f"Condition token count ({cond_tokens.size(1)}) must equal horizon "
                f"({self._diffusion_action_token_count})."
            )
        return cond_tokens, lm_outputs

    def _build_diffusion_input(
        self,
        *,
        cond_tokens: torch.Tensor,
        x_t_seq: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if cond_tokens.dim() != 3:
            raise ValueError(
                f"`cond_tokens` must have shape (batch, horizon, hidden). Got {tuple(cond_tokens.size())}."
            )
        if x_t_seq.dim() != 3:
            raise ValueError(
                f"`x_t_seq` must have shape (batch, horizon, action_dim). Got {tuple(x_t_seq.size())}."
            )
        if cond_tokens.size(0) != x_t_seq.size(0) or cond_tokens.size(1) != x_t_seq.size(1):
            raise ValueError(
                "Condition and noisy action sequences must share batch/horizon dimensions. "
                f"Got cond {tuple(cond_tokens.size())} vs noisy {tuple(x_t_seq.size())}."
            )
        if x_t_seq.size(1) != self._diffusion_action_token_count:
            raise ValueError(
                f"Noisy action horizon must be {self._diffusion_action_token_count}; "
                f"got {x_t_seq.size(1)}."
            )
        if x_t_seq.size(2) != self._action_dim:
            raise ValueError(
                f"Noisy action inner dim must be {self._action_dim}; got {x_t_seq.size(2)}."
            )

        cond_dtype = next(self.diffusion_condition_proj.parameters()).dtype
        cond_features = self.diffusion_condition_proj(
            cond_tokens.to(device=cond_tokens.device, dtype=cond_dtype)
        )
        noisy_features = self.diffusion_noisy_action_proj(
            x_t_seq.to(device=cond_tokens.device, dtype=cond_dtype)
        )

        t_embed = self._sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=self.diffusion_time_embed_dim,
        ).to(device=cond_tokens.device, dtype=cond_dtype)
        step_features = self.diffusion_time_mlp(t_embed).unsqueeze(1)

        return cond_features + noisy_features + step_features

    def _predict_diffusion_target_from_xt(
        self,
        *,
        x_t_seq: torch.Tensor,
        timesteps: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        diffusion_input = self._build_diffusion_input(
            cond_tokens=cond_tokens,
            x_t_seq=x_t_seq,
            timesteps=timesteps,
        )
        head_dtype = next(self.diffusion_noise_decoder.parameters()).dtype
        diffusion_input = diffusion_input.to(dtype=head_dtype)
        batch_size, horizon, input_dim = diffusion_input.size()
        pred_target = self.diffusion_noise_decoder(
            diffusion_input.reshape(batch_size * horizon, input_dim)
        )
        pred_target = pred_target.reshape(batch_size, horizon, self._action_dim)
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
        use_ce_loss: bool = False,
        **kwargs,
    ):
        del use_ce_loss

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
                **kwargs,
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        effective_output_hidden_states = output_hidden_states or self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of `input_ids` or `inputs_embeds`.")

        source_tensor = input_ids if input_ids is not None else inputs_embeds
        if source_tensor is None:
            raise ValueError("`input_ids` or `inputs_embeds` must be provided for diffusion action mode.")

        batch_size = source_tensor.size(0)
        device = source_tensor.device
        self._ensure_diffusion_schedule_precision(device=device)

        timesteps = self._normalize_timesteps(timesteps=timesteps, batch_size=batch_size, device=device)
        kwargs = dict(kwargs)
        kwargs.pop("tokenizer", None)
        kwargs.pop("loss_type", None)
        kwargs.pop("loss_horizon", None)

        cond_tokens, lm_outputs = self._encode_condition_from_backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mask_type_labels=mask_type_labels,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        target_noise_seq = None
        target_pred_seq = None
        clean_seq = None

        if noisy_actions is not None:
            x_t_seq = self._prepare_action_sequence_like(noisy_actions).to(
                device=device, dtype=torch.float32
            )
            if noise is not None:
                target_noise_seq = self._prepare_action_sequence_like(noise).to(
                    device=device, dtype=torch.float32
                )
        else:
            if pred_seq is None:
                raise ValueError(
                    "Provide `pred_seq` for training (q-sampled internally), or pass `noisy_actions` + `timesteps`."
                )
            clean_seq = self._reshape_pred_seq(pred_seq).to(device=device, dtype=torch.float32)
            target_noise_seq = (
                torch.randn_like(clean_seq)
                if noise is None
                else self._prepare_action_sequence_like(noise).to(device=device, dtype=torch.float32)
            )
            x_t_seq = self.q_sample(clean_seq, timesteps, target_noise_seq)

        pred_target_seq = self._predict_diffusion_target_from_xt(
            x_t_seq=x_t_seq,
            timesteps=timesteps,
            cond_tokens=cond_tokens,
        )

        if self.diffusion_prediction_type == "epsilon":
            pred_noise_seq = pred_target_seq
            pred_x0_seq = self._predict_x0_from_noise(x_t_seq, pred_noise_seq, timesteps)
        else:
            pred_x0_seq = pred_target_seq
            pred_noise_seq = self._predict_noise_from_x0(x_t_seq, pred_x0_seq, timesteps)

        if self.diffusion_prediction_type == "epsilon":
            target_pred_seq = target_noise_seq
        else:
            if clean_seq is not None:
                target_pred_seq = clean_seq
            elif target_noise_seq is not None:
                target_pred_seq = self._predict_x0_from_noise(x_t_seq, target_noise_seq, timesteps)

        diffusion_loss = None
        if target_pred_seq is not None:
            diffusion_loss = F.mse_loss(pred_target_seq.float(), target_pred_seq.float())
            diffusion_loss = diffusion_loss.to(dtype=pred_target_seq.dtype)

        flattened_pred_target = pred_target_seq.reshape(pred_target_seq.size(0), -1)

        outputs = CausalLMOutputWithPastAndActions(
            loss=diffusion_loss,
            logits=lm_outputs.logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=(lm_outputs.hidden_states if effective_output_hidden_states else None),
            attentions=lm_outputs.attentions,
            action_head_output=flattened_pred_target,
        )
        outputs["action_loss"] = diffusion_loss
        outputs["language_model_loss"] = None
        outputs["cross_entropy_loss"] = None
        outputs["action_prediction_loss"] = diffusion_loss
        outputs["smoothness_loss"] = None
        outputs["vec_order_loss"] = None
        outputs["diffusion_prediction_type"] = self.diffusion_prediction_type
        outputs["predicted_noise"] = pred_noise_seq
        outputs["predicted_noise_chunks"] = self._chunk_action_sequence(pred_noise_seq)
        outputs["predicted_x0"] = pred_x0_seq
        outputs["timesteps"] = timesteps
        outputs["noisy_actions"] = x_t_seq
        if clean_seq is not None:
            outputs["clean_actions"] = clean_seq

        if not return_dict:
            return tuple(value for value in outputs.values() if value is not None)
        return outputs

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

    @torch.no_grad()
    def _denormalize_actions(self, flattened_actions: torch.Tensor) -> torch.Tensor:
        stats = getattr(self, "action_mean_std", None)
        if not isinstance(stats, dict):
            return flattened_actions

        mean_values = stats.get("mean")
        std_values = stats.get("std")
        if mean_values is None or std_values is None:
            return flattened_actions

        mean = torch.as_tensor(
            mean_values, device=flattened_actions.device, dtype=flattened_actions.dtype
        )
        std = torch.as_tensor(
            std_values, device=flattened_actions.device, dtype=flattened_actions.dtype
        )

        if mean.numel() == 0 or std.numel() == 0 or mean.numel() != std.numel():
            return flattened_actions

        if flattened_actions.size(-1) % mean.numel() != 0:
            raise ValueError(
                "Cannot denormalize flattened actions: last dimension "
                f"({flattened_actions.size(-1)}) is not divisible by action stats size "
                f"({mean.numel()})."
            )

        action_dim = mean.numel()
        action_seq = flattened_actions.view(flattened_actions.size(0), -1, action_dim)
        denorm_seq = action_seq * std.view(1, 1, -1) + mean.view(1, 1, -1)
        return denorm_seq.reshape(flattened_actions.size(0), -1)

    @torch.no_grad()
    def _denormalize_action_sequence(self, action_seq: torch.Tensor) -> torch.Tensor:
        flat = action_seq.reshape(action_seq.size(0), -1)
        denorm_flat = self._denormalize_actions(flat)
        return denorm_flat.view(action_seq.size(0), action_seq.size(1), action_seq.size(2))

    @torch.no_grad()
    def ddpm_sample_actions(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_type_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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
        batch_size = source_tensor.size(0)
        device = source_tensor.device
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
        kwargs.pop("loss_type", None)
        kwargs.pop("loss_horizon", None)

        cond_tokens, _ = self._encode_condition_from_backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mask_type_labels=mask_type_labels,
            labels=labels,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            **kwargs,
        )

        x_t = torch.randn(
            (batch_size, self._diffusion_action_token_count, self._action_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        last_pred_noise = None
        last_pred_x0 = None

        for t in reversed(range(num_inference_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_target_seq = self._predict_diffusion_target_from_xt(
                x_t_seq=x_t,
                timesteps=t_batch,
                cond_tokens=cond_tokens,
            ).to(dtype=torch.float32)

            if self.diffusion_prediction_type == "epsilon":
                pred_noise_seq = pred_target_seq
                pred_x0_seq = self._predict_x0_from_noise(x_t, pred_noise_seq, t_batch)
            else:
                pred_x0_seq = pred_target_seq
                pred_noise_seq = self._predict_noise_from_x0(x_t, pred_x0_seq, t_batch)

            if clip_denoised:
                pred_x0_seq = torch.clamp(pred_x0_seq, min=-clip_range, max=clip_range)
                pred_noise_seq = self._predict_noise_from_x0(x_t, pred_x0_seq, t_batch)

            x_t_next = self._ddpm_step(x_t=x_t, pred_x0=pred_x0_seq, timestep=t, generator=generator)
            x_t = x_t_next
            last_pred_noise = pred_noise_seq
            last_pred_x0 = pred_x0_seq

        final_actions = x_t
        flattened_actions = final_actions.reshape(final_actions.size(0), -1)

        if return_generation_output:
            generation_output = {
                "action_sequence": final_actions,
                "final_chunk_actions": self._chunk_action_sequence(final_actions),
                "predicted_noise": last_pred_noise,
                "predicted_x0": last_pred_x0,
            }
            return flattened_actions, generation_output
        return flattened_actions

    @torch.no_grad()
    def action_head_based_generate_actions(
        self,
        input_ids: Optional[torch.LongTensor],
        tokenizer: Any = None,
        attention_mask: Optional[torch.LongTensor] = None,
        mask_type_labels: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        return_generation_output: bool = False,
        **kwargs,
    ):
        del tokenizer
        if max_new_tokens is not None and max_new_tokens < self.horizon:
            raise ValueError(
                f"`max_new_tokens` must be at least the action horizon ({self.horizon}). "
                f"Received {max_new_tokens}."
            )
        sampled = self.ddpm_sample_actions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_type_labels=mask_type_labels,
            return_generation_output=return_generation_output,
            **kwargs,
        )
        if not return_generation_output:
            return self._denormalize_actions(sampled)

        flattened_actions, generation_output = sampled
        denorm_flattened_actions = self._denormalize_actions(flattened_actions)
        denorm_action_sequence = self._denormalize_action_sequence(generation_output["action_sequence"])
        generation_output["action_sequence"] = denorm_action_sequence
        generation_output["final_chunk_actions"] = self._chunk_action_sequence(denorm_action_sequence)
        if generation_output.get("predicted_x0") is not None:
            generation_output["predicted_x0"] = self._denormalize_action_sequence(
                generation_output["predicted_x0"]
            )
        return denorm_flattened_actions, generation_output
