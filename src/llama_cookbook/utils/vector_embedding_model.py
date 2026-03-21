from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_cookbook.utils.bidirection_action_model import LlamaForBidirectionAttnWithActions


ROAD_TYPE_ORDER = [
    "LaneCenter-Freeway",
    "LaneCenter-SurfaceStreet",
    "LaneCenter-BikeLane",
    "RoadLine-BrokenSingleWhite",
    "RoadLine-SolidSingleWhite",
    "RoadLine-SolidDoubleWhite",
    "RoadLine-BrokenSingleYellow",
    "RoadLine-BrokenDoubleYellow",
    "Roadline-SolidSingleYellow",
    "Roadline-SolidDoubleYellow",
    "RoadLine-PassingDoubleYellow",
    "RoadEdgeBoundary",
    "RoadEdgeMedian",
    "StopSign",
    "Crosswalk",
    "SpeedBump",
]

AGENT_TYPE_ORDER = ["Unset", "Vehicle", "Pedestrian", "Cyclist", "Other"]


class VectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        depth = max(int(num_layers), 1)
        layers: List[nn.Module] = [nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolylineVectorEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.hidden_size)
        type_embed_dim = int(getattr(config, "vector_type_embed_dim", 32))
        mlp_hidden_dim = int(getattr(config, "vector_mlp_hidden_dim", hidden_size))
        mlp_layers = int(getattr(config, "vector_mlp_layers", 2))
        self.num_steps = int(getattr(config, "vector_encoder_steps", 8))
        self.normalize_positions = bool(getattr(config, "vector_encoder_normalize", True))
        self.output_scale = float(getattr(config, "vector_output_scale", 0.1))
        self.initializer_range = float(getattr(config, "initializer_range", 0.02))
        self.use_abs_embed = bool(getattr(config, "vector_abs_embed", True))

        self.road_type_to_id = {name: idx for idx, name in enumerate(ROAD_TYPE_ORDER)}
        self.agent_type_to_id = {name: idx for idx, name in enumerate(AGENT_TYPE_ORDER)}
        self.road_unknown_id = len(ROAD_TYPE_ORDER)
        self.agent_unknown_id = len(AGENT_TYPE_ORDER)

        self.road_type_embedding = nn.Embedding(self.road_unknown_id + 1, type_embed_dim)
        self.agent_type_embedding = nn.Embedding(self.agent_unknown_id + 1, type_embed_dim)

        self.hidden_size = hidden_size
        self.vector_mlps = nn.ModuleList()
        self.post_mlp_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_steps)])
        self.context_norm = nn.LayerNorm(hidden_size)
        abs_feat_dim = 5  # center_x, center_y, scale, start_x, start_y
        if self.use_abs_embed:
            self.abs_mlp = nn.Sequential(
                nn.Linear(abs_feat_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.abs_mlp = None
        vector_input_dim = 4 + type_embed_dim
        for step in range(self.num_steps):
            step_input_dim = vector_input_dim if step == 0 else hidden_size * 2
            self.vector_mlps.append(
                VectorMLP(step_input_dim, mlp_hidden_dim, hidden_size, mlp_layers)
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_range = self.initializer_range

        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=init_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=init_range)
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

        self.apply(_init_weights)

    def encode_scene(self, map_payload, trajectory_payload) -> torch.Tensor:
        device = self.road_type_embedding.weight.device
        dtype = self.road_type_embedding.weight.dtype
        polyline_embeddings: List[torch.Tensor] = []

        for segment in map_payload:
            positions = segment.get("positions")
            if positions is None:
                continue
            road_type = segment.get("type", "")
            emb = self._encode_polyline(positions, road_type, True, device, dtype)
            if emb is not None:
                polyline_embeddings.append(emb)

        for agent_data in trajectory_payload.values():
            positions = agent_data.get("positions")
            if positions is None:
                continue
            agent_type = agent_data.get("type", "Unset")
            emb = self._encode_polyline(positions, agent_type, False, device, dtype)
            if emb is not None:
                polyline_embeddings.append(emb)

        if not polyline_embeddings:
            polyline_embeddings.append(torch.zeros(self.hidden_size, device=device, dtype=dtype))

        return torch.stack(polyline_embeddings, dim=0)

    def _encode_polyline(
        self,
        positions,
        type_name: str,
        is_road: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        positions_tensor = torch.as_tensor(positions, dtype=dtype, device=device)
        if positions_tensor.ndim != 2 or positions_tensor.size(0) < 2:
            return None
        if positions_tensor.size(1) > 2:
            positions_tensor = positions_tensor[:, :2]
        elif positions_tensor.size(1) < 2:
            return None
        raw_positions = positions_tensor
        center = positions_tensor.mean(dim=0, keepdim=True)
        centered = positions_tensor - center
        scale = centered.abs().amax()
        scale = torch.clamp(scale, min=1.0)
        if self.normalize_positions:
            positions_tensor = centered / scale
        abs_features = None
        if self.use_abs_embed:
            start = raw_positions[0]
            abs_features = torch.cat(
                [center.squeeze(0), scale.unsqueeze(0), start],
                dim=0,
            )
        starts = positions_tensor[:-1, :]
        ends = positions_tensor[1:, :]

        if is_road:
            type_id = self.road_type_to_id.get(type_name, self.road_unknown_id)
            type_embed = self.road_type_embedding(torch.tensor(type_id, device=device, dtype=torch.long))
        else:
            type_id = self.agent_type_to_id.get(type_name, self.agent_unknown_id)
            type_embed = self.agent_type_embedding(torch.tensor(type_id, device=device, dtype=torch.long))

        type_embed = type_embed.to(dtype=dtype)
        type_expand = type_embed.unsqueeze(0).expand(starts.size(0), -1)
        vector_features = torch.cat([starts, ends, type_expand], dim=-1)
        return self._encode_vectors(vector_features, abs_features=abs_features)

    def _encode_vectors(
        self,
        vector_features: torch.Tensor,
        abs_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = vector_features
        context = None
        for step, mlp in enumerate(self.vector_mlps):
            x = mlp(x)
            x = self.post_mlp_norms[step](x)
            context = torch.max(x, dim=0).values
            context = self.context_norm(context)
            if step < len(self.vector_mlps) - 1:
                context_expand = context.unsqueeze(0).expand(x.size(0), -1)
                x = torch.cat([x, context_expand], dim=-1)
        if self.output_scale != 1.0:
            context = context * self.output_scale
        if abs_features is not None and self.abs_mlp is not None:
            abs_embed = self.abs_mlp(abs_features.to(dtype=context.dtype))
            context = context + abs_embed
        return context

class LlamaForBidirectionAttnWithVectorEmbeddings(LlamaForBidirectionAttnWithActions):
    def __init__(self, config):
        super().__init__(config)
        self.vector_encoder = PolylineVectorEncoder(config)
        self.vector_encoder.reset_parameters()
        # self._rebuild_parallel_action_head()

    def _rebuild_parallel_action_head(self) -> None:
        """
        Rebuild the action decoder to decode one embedding at a time (per step),
        while keeping the flattened output shape downstream.
        """
        self.action_decoder = self._build_action_decoder(
            input_dim=self._action_input_dim,
            hidden_dim=self._action_hidden_dim * 4, # increase hidden dim for better capacity
            # hidden_dim=self._action_hidden_dim,
            num_layers=self._action_num_layers,
            action_dim=self._action_dim * self._action_chunk_size,
        )
        self.reset_action_head_parameters()
        self.config.action_head_output_dim = self._action_dim
        self.config.action_chunk_size = self._action_chunk_size

    def forward(
        self,
        map_payloads=None,
        trajectory_payloads=None,
        pred_seq=None,
        loss_type: Optional[str] = "mse",
        loss_horizon: Optional[int] = None,
        **kwargs,
    ):
        if map_payloads is None or trajectory_payloads is None:
            raise ValueError("`map_payloads` and `trajectory_payloads` must be provided.")

        if not isinstance(map_payloads, (list, tuple)):
            map_payloads = [map_payloads]
            trajectory_payloads = [trajectory_payloads]

        if len(map_payloads) != len(trajectory_payloads):
            raise ValueError("`map_payloads` and `trajectory_payloads` must have the same length.")

        inputs_embeds, attention_mask = self._build_inputs(map_payloads, trajectory_payloads)

        pred_seq = torch.as_tensor(pred_seq, device=inputs_embeds.device) if pred_seq is not None else None
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            task="action",
            pred_seq=pred_seq,
            loss_type=loss_type,
            loss_horizon=loss_horizon,
            use_ce_loss=False,
            **kwargs,
        )

    @torch.no_grad()
    def inference(
        self,
        map_payloads=None,
        trajectory_payloads=None,
        pred_seq=None,
        loss_type: Optional[str] = None,
        loss_horizon: Optional[int] = None,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        kwargs.setdefault("return_dict", True)
        return self.forward(
            map_payloads=map_payloads,
            trajectory_payloads=trajectory_payloads,
            pred_seq=pred_seq,
            loss_type=loss_type,
            loss_horizon=loss_horizon,
            **kwargs,
        )

    def _build_inputs(
        self,
        map_payloads: Sequence,
        trajectory_payloads: Sequence,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.model.embed_tokens.weight.device
        dtype = self.model.embed_tokens.weight.dtype
        horizon = self.horizon

        sequences: List[torch.Tensor] = []
        lengths: List[int] = []

        for map_payload, trajectory_payload in zip(map_payloads, trajectory_payloads):
            scene_embeds = self.vector_encoder.encode_scene(map_payload, trajectory_payload)
            scene_embeds = scene_embeds.to(device=device, dtype=dtype)
            action_embeds = torch.zeros(horizon, self.config.hidden_size, device=device, dtype=dtype)
            sequence = torch.cat([scene_embeds, action_embeds], dim=0)
            sequences.append(sequence)
            lengths.append(sequence.size(0))

        max_len = max(lengths)
        batch_size = len(sequences)
        hidden_size = self.config.hidden_size

        padded = torch.zeros(batch_size, max_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)

        for idx, sequence in enumerate(sequences):
            seq_len = sequence.size(0)
            padded[idx, :seq_len] = sequence
            attention_mask[idx, :seq_len] = 1
            attention_mask[idx, seq_len-horizon:seq_len] = 2

        return padded, attention_mask
