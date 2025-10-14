import json
import h5py
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from llama_cookbook.utils.token_lookup import build_vec_len_lookup

train_h5_path = "/p/ruishen/processed_waymo_data/training/original_ego_traj/merged_variable_trajectories_10hz.h5"
val_h5_path = "/p/ruishen/processed_waymo_data/validation/original_ego_traj/merged_variable_trajectories_10hz.h5"
vec_to_angle = None
len_to_length = None


def load_h5_into_memory(h5_path):
    data_store = {}
    with h5py.File(h5_path, 'r') as f:
        def recurse(name, obj):
            if isinstance(obj, h5py.Group):
                data_store[name] = {
                    "_attrs": dict(obj.attrs),
                    "_datasets": {}
                }
            elif isinstance(obj, h5py.Dataset):
                group_name = "/".join(name.split("/")[:-1])
                if group_name not in data_store:
                    data_store[group_name] = {"_attrs": {}, "_datasets": {}}
                data_store[group_name]["_datasets"][name.split("/")[-1]] = obj[()]
        f.visititems(recurse)
    return data_store

def load_index(h5_path):
    idx_path = h5_path + "_index.json"
    with open(idx_path) as f:
        return json.load(f)

train_index = load_index(train_h5_path)
val_index   = load_index(val_h5_path)

train_raw_traj = load_h5_into_memory(train_h5_path)
val_raw_traj = load_h5_into_memory(val_h5_path)

angle_bins = np.load('/p/ruishen/processed_waymo_data/validation/waymo_vectorized/combined_angle_bins_10hz_512.npy', allow_pickle=True)
all_vectors = np.load('/p/ruishen/processed_waymo_data/training/waymo_vectorized/all_cluster_centroids_10hz_1024.npy', allow_pickle=True)

val_start_heading_path = "/p/ruishen/processed_waymo_data/validation/waymo_vectorized/combined_original_ego_start_heading.json"
train_start_heading_path = "/p/ruishen/processed_waymo_data/training/waymo_vectorized/combined_original_ego_start_heading.json"
val_start_heading = json.load(open(val_start_heading_path, 'r'))
train_start_heading = json.load(open(train_start_heading_path, 'r'))

combined_start_heading = {**train_start_heading, **val_start_heading}

def ce_loss_by_type(logits, labels, tokenizer, ignore_index=-100, reduction="mean"):
    """
    Cross-entropy loss with separate handling for trajectory tokens (VEC_, LEN_, POS_)
    """
    B, T, V = logits.size()
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    logits_flat = shifted_logits.view(-1, V)
    labels_flat = shifted_labels.view(-1)

    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction="none"
    )  # [N]

    # Cache token sets on first call
    if not hasattr(ce_loss_by_type, 'traj_token_ids'):
        vocab = tokenizer.get_vocab()
        ce_loss_by_type.vec_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_"))
        ce_loss_by_type.len_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("LEN_"))
        ce_loss_by_type.pos_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("POS_"))

    # Masks for each type
    vec_mask = torch.tensor([lbl.item() in ce_loss_by_type.vec_token_ids for lbl in labels_flat], device=labels.device)
    len_mask = torch.tensor([lbl.item() in ce_loss_by_type.len_token_ids for lbl in labels_flat], device=labels.device)
    pos_mask = torch.tensor([lbl.item() in ce_loss_by_type.pos_token_ids for lbl in labels_flat], device=labels.device)

    # Compute mean per type (skip empty case)
    def masked_mean(loss, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=loss.device)
        if reduction == "mean":
            return loss[mask].mean()
        elif reduction == "sum":
            return loss[mask].sum()
        else:
            return loss[mask]

    vec_loss = masked_mean(ce_loss, vec_mask)
    len_loss = masked_mean(ce_loss, len_mask)
    pos_loss = masked_mean(ce_loss, pos_mask)

    return {
        "vec_loss": vec_loss.item(),
        "len_loss": len_loss.item(),
        "pos_loss": pos_loss.item()
    }

def load_trajectory_by_key_from_memory(sid, agent_id):
    global train_index, val_index, train_raw_traj, val_raw_traj
    split = "train"
    key = f"{sid}__{agent_id}"
    group_name = train_index.get(key)
    if group_name is None:
        split = "val"
        group_name = val_index.get(key)
        if group_name is None:
            raise ValueError(f"Trajectory with sid {sid} and agent_id {agent_id} not found in index.")

    if split == "train":
        g = train_raw_traj[group_name]
    else:
        g = val_raw_traj[group_name]
    traj = g["_datasets"]["positions"]
    meta = {
        "sid": g["_attrs"]["sid"],
        "agent_id": g["_attrs"]["agent_id"],
        "source_path": g["_attrs"]["source_path"]
    }
    return traj, meta

def centroids_to_global(start_xy, initial_heading, rotated_positions):
    # Ensure numpy array
    rotated_positions = np.asarray(rotated_positions, dtype=float)
    
    # Prepare output
    traj = [np.array(start_xy, dtype=float)]
    current_heading = initial_heading
    current_pos = np.array(start_xy, dtype=float)

    for dx_loc, dy_loc, d_heading in rotated_positions:
        if dx_loc == -1 and dy_loc == -1:
            # Skip invalid positions
            continue
        # Rotate local step into global frame
        cos_h = np.cos(current_heading)
        sin_h = np.sin(current_heading)
        delta_global = np.array([
            cos_h * dx_loc - sin_h * dy_loc,
            sin_h * dx_loc + cos_h * dy_loc
        ])

        # Step to new global position
        current_pos = current_pos + delta_global
        traj.append(current_pos.copy())

        # Update heading (and normalize to [-pi,pi])
        current_heading = current_heading + d_heading
        current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi

    return np.vstack(traj)

def log_normalize_with_target(ade, target_good=0.001, good_max=1.0):
    """
    Log-normalization with scaling:
    - target_good is the best value
    - good_max is the top end of 'good' range
    - above good_max continues increasing but compressed
    """
    # Scale ADE relative to target
    rel = ade / target_good
    # Log1p to compress large ADE, normalize so good_max maps near 1
    norm = torch.log1p(rel) / torch.log1p(torch.tensor(good_max / target_good))
    return norm

# def ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
#     global angle_bins, combined_start_heading

#     if not hasattr(ade_loss, 'vec_token_ids'):
#         vocab = tokenizer.get_vocab()
#         ade_loss.vec_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_"))
#         ade_loss.len_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("LEN_"))
#         ade_loss.pos_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("POS_"))
#         ade_loss.traj_token_ids = torch.tensor(list((ade_loss.vec_token_ids | ade_loss.len_token_ids | ade_loss.pos_token_ids)), device=logits.device)
#         ade_loss.vec_token_ids = torch.tensor(list(ade_loss.vec_token_ids), device=logits.device)
#         ade_loss.len_token_ids = torch.tensor(list(ade_loss.len_token_ids), device=logits.device)

#         ade_loss.vec_to_angle, ade_loss.len_to_length = build_vec_len_lookup(tokenizer, device=logits.device, dtype=logits.dtype)
#         ade_loss.angle_bins_tensor = torch.tensor(angle_bins, device=logits.device, dtype=logits.dtype)

#     # Shift logits and labels so they are aligned
#     logits_shifted = logits[..., :-1, :].contiguous()
#     labels_shifted = labels[..., 1:].contiguous()
#     vals, idx = torch.topk(logits_shifted, top_k, dim=-1)
#     probs = torch.softmax(vals, dim=-1)
#     B, T, K = probs.shape

#     # Precompute all trajectory tensors for the batch
#     norm_raw_traj_tensors = []
#     for b in range(B):
#         raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
#         norm_raw_traj = raw_traj - raw_traj[0]
#         norm_raw_traj = norm_raw_traj[9:]
#         norm_raw_traj_tensor = torch.as_tensor(norm_raw_traj, device=logits.device, dtype=logits.dtype)
#         norm_raw_traj_tensors.append(norm_raw_traj_tensor)

#     ade_values = []
#     label_vec_mask = torch.isin(labels_shifted, ade_loss.vec_token_ids)
#     label_len_mask = torch.isin(labels_shifted, ade_loss.len_token_ids)

#     # logits_flat = logits_shifted.view(-1, logits_shifted.size(-1))
#     # labels_flat = labels_shifted.view(-1)

#     # ce_flat = F.cross_entropy(
#     #     logits_flat,
#     #     labels_flat,
#     #     ignore_index=-100,
#     #     reduction="none"
#     # )
#     # ce_loss = ce_flat.view(B, T)

#     # def get_weighted_ce_loss():
#     #     vec_ids = torch.tensor(list(ade_loss.vec_token_ids), device=idx.device)
#     #     len_ids = torch.tensor(list(ade_loss.len_token_ids), device=idx.device)

#     #     is_vec_pred = torch.isin(idx, vec_ids)   # (B, T, K)
#     #     is_len_pred = torch.isin(idx, len_ids)   # (B, T, K)

#     #     vec_counts = is_vec_pred.sum(dim=-1)     # (B, T)
#     #     len_counts = is_len_pred.sum(dim=-1)     # (B, T)

#     #     vec_ratio = vec_counts.float() / idx.size(-1)
#     #     len_ratio = len_counts.float() / idx.size(-1)

#     #     weights = (label_vec_mask * vec_ratio) + (label_len_mask * len_ratio)
#     #     weighted_ce_loss = ce_loss * weights

#     #     return weighted_ce_loss

#     # masked_ce_loss = get_weighted_ce_loss()
#     # ce_penalty = masked_ce_loss.sum() / (masked_ce_loss != 0).sum().clamp(min=1.0)

#     vec_ade_values = []
#     len_ade_values = []
#     for b in range(B):
#         next_token_ade = []
#         vec_token_ade = []
#         vec_step_count = 1
#         len_step_count = 1
#         for t in range(T):
#             if labels_shifted[b, t] == -100:
#                 continue

#             vec_mask = [i for i in range(K) if idx[b, t, i].item() in ade_loss.vec_token_ids]
#             len_mask = [i for i in range(K) if idx[b, t, i].item() in ade_loss.len_token_ids]

#             if not label_vec_mask[b, t] and not label_len_mask[b, t]:
#                 continue

#             if not (
#                 (len(vec_mask) > 0 and label_vec_mask[b, t]) or
#                 (len(len_mask) > 0 and label_len_mask[b, t])
#             ):
#                 continue

#             start_heading = combined_start_heading[f"{sid[b]}__{ego_id[b]}"][1]
#             if label_vec_mask[b, t]:
#                 if vec_step_count == len(norm_raw_traj_tensors[b]):
#                     continue
#                 p_vec = probs[b, t, vec_mask]
#                 ids_vec = idx[b, t, vec_mask]
#                 vec_angles = ade_loss.vec_to_angle[ids_vec]
#                 len_lengths = torch.arange(0, 3.51, 0.01, device=logits.device, dtype=logits.dtype)
#                 try:
#                     ego_headings_vec = start_heading + ade_loss.angle_bins_tensor[vec_angles]
#                 except (IndexError, RuntimeError):
#                     continue
#                 cos_headings = torch.cos(ego_headings_vec)[:, None]
#                 sin_headings = torch.sin(ego_headings_vec)[:, None]
#                 dx = len_lengths[None, :] * cos_headings
#                 dy = len_lengths[None, :] * sin_headings
#                 pos = torch.stack([dx, dy], dim=-1)
#                 target = norm_raw_traj_tensors[b][vec_step_count] - norm_raw_traj_tensors[b][vec_step_count - 1]
#                 diff = pos - target
#                 norm_diff = torch.norm(diff, dim=-1)
#                 min_diff = torch.min(norm_diff, dim=1).values
#                 step_ade = (p_vec * min_diff).sum()
#                 vec_token_ade.append(step_ade)
#                 vec_ade_values.append(step_ade.item())
#                 vec_step_count += 1

#             elif label_len_mask[b, t]:
#                 if len_step_count == len(norm_raw_traj_tensors[b]):
#                     continue
#                 p_len = probs[b, t, len_mask]
#                 ids_len = idx[b, t, len_mask]
#                 len_lengths = ade_loss.len_to_length[ids_len]
#                 len_gt = ade_loss.len_to_length[labels_shifted[b, t].item()]
#                 len_diff = torch.abs(len_lengths - len_gt)
#                 step_ade = (p_len * len_diff).sum()
#                 next_token_ade.append(step_ade)
#                 len_ade_values.append(step_ade.item())
#                 len_step_count += 1

#         if vec_token_ade:
#             weight_start = 1.0
#             weight_end = 3.0
#             if len(vec_token_ade) > 30:
#                 step_start = 30
#                 step_end = min(80, len(vec_token_ade))
#                 for svw in range(step_start, step_end):
#                     w = weight_start + (weight_end - weight_start) * (svw - step_start) / (80 - step_start)
#                     vec_token_ade[svw] *= w
#             next_token_ade.extend(vec_token_ade)

#         if next_token_ade:
#             ade_values.append(sum(next_token_ade) / len(next_token_ade))

#     if ade_values:
#         ade_value = torch.stack(ade_values).sum()
#     else:
#         ade_value = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

#     vec_ade = np.mean(vec_ade_values) if vec_ade_values else 0.0
#     len_ade = np.mean(len_ade_values) if len_ade_values else 0.0
#     # aux_loss = ade_value + ce_penalty
#     aux_loss = ade_value
#     return aux_loss, ade_value, {"vec_ade": vec_ade, "len_ade": len_ade}
    
def ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
    device = logits.device
    dtype = logits.dtype

    cache = getattr(ade_loss, "_cache", None)
    key = (device, dtype, id(tokenizer))
    if cache is None or cache.get("key") != key:
        vocab = tokenizer.get_vocab()
        vec_token_ids = [tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_")]
        vec_token_tensor = torch.tensor(vec_token_ids, device=device, dtype=torch.long)
        vec_to_angle, _ = build_vec_len_lookup(tokenizer, device=device, dtype=dtype)
        vec_lookup = torch.as_tensor(
            np.asarray(all_vectors, dtype=np.float32)[:, :2], device=device, dtype=dtype
        )
        cache = {
            "key": key,
            "vec_token_tensor": vec_token_tensor,
            "vec_to_angle": vec_to_angle,
            "vec_lookup": vec_lookup,
        }
        ade_loss._cache = cache

    vec_token_tensor = cache["vec_token_tensor"]
    if vec_token_tensor.numel() == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, 0.0

    vec_to_angle = cache["vec_to_angle"]
    vec_lookup = cache["vec_lookup"]

    logits_shifted = logits[..., :-1, :]
    labels_shifted = labels[..., 1:]
    vals, idx = torch.topk(logits_shifted, top_k, dim=-1)
    probs = torch.softmax(vals, dim=-1)

    angle_indices = vec_to_angle[idx]
    valid_mask = angle_indices >= 0

    vec_probs = probs * valid_mask.to(dtype)
    vec_probs_sum = vec_probs.sum(dim=-1, keepdim=True)
    vec_probs = vec_probs / vec_probs_sum.clamp(min=torch.finfo(dtype).tiny)

    clamped_angles = angle_indices.clamp_min(0)
    predicted_steps = vec_lookup[clamped_angles]

    label_vec_mask = torch.isin(labels_shifted, vec_token_tensor)

    ade_values = []
    B, T, K = idx.shape
    for b in range(B):
        raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
        traj_tensor = torch.as_tensor(raw_traj, device=device, dtype=dtype)
        traj_tensor = traj_tensor - traj_tensor[0]
        traj_xy = traj_tensor[9:, :2]
        if traj_xy.size(0) < 2:
            continue
        deltas = traj_xy[1:] - traj_xy[:-1]

        vec_positions = torch.nonzero(label_vec_mask[b], as_tuple=False).flatten()
        if vec_positions.numel() == 0:
            continue

        steps = min(vec_positions.size(0), deltas.size(0))
        if steps == 0:
            continue

        vec_positions = vec_positions[:steps]
        target_steps = deltas[:steps]

        probs_b = vec_probs[b, vec_positions]
        raw_prob_sum = vec_probs_sum[b, vec_positions, 0]
        valid_steps = raw_prob_sum > 0
        if not torch.any(valid_steps):
            continue

        probs_b = probs_b[valid_steps]
        target_steps = target_steps[valid_steps]
        pred_steps = predicted_steps[b, vec_positions][valid_steps]

        diff = pred_steps - target_steps[:, None, :]
        step_errors = (probs_b * diff.norm(dim=-1)).sum(dim=-1)
        ade_values.append(step_errors.mean())

    if ade_values:
        ade_value = torch.stack(ade_values).sum()
    else:
        ade_value = torch.tensor(0.0, device=device, dtype=dtype)

    # aux_loss = ade_value * weight
    aux_loss = ade_value

    return aux_loss, ade_value, 0.0


def ade_loss_all_vec(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
    # top_k argument is kept for signature compatibility with ade_loss but unused here
    device = logits.device
    dtype = logits.dtype

    cache = getattr(ade_loss_all_vec, "_cache", None)
    key = (device, dtype, id(tokenizer))
    if cache is None or cache.get("key") != key:
        vocab = tokenizer.get_vocab()
        vec_token_ids = [tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_")]
        vec_token_tensor = torch.tensor(vec_token_ids, device=device, dtype=torch.long)

        vec_to_angle, _ = build_vec_len_lookup(tokenizer, device=device, dtype=dtype)
        vec_lookup = torch.as_tensor(
            np.asarray(all_vectors, dtype=np.float32)[:, :2], device=device, dtype=dtype
        )

        if vec_token_tensor.numel() > 0:
            vec_angles = vec_to_angle[vec_token_tensor]
            valid_mask = (vec_angles >= 0) & (vec_angles < vec_lookup.size(0))
            vec_token_tensor = vec_token_tensor[valid_mask]
            vec_steps = vec_lookup[vec_angles[valid_mask]]
        else:
            vec_steps = vec_lookup.new_empty((0, 2))

        cache = {
            "key": key,
            "vec_token_tensor": vec_token_tensor,
            "vec_steps": vec_steps,
        }
        ade_loss_all_vec._cache = cache

    vec_token_tensor = cache["vec_token_tensor"]
    if vec_token_tensor.numel() == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, 0.0

    vec_steps = cache["vec_steps"]

    logits_shifted = logits[..., :-1, :]
    labels_shifted = labels[..., 1:]

    vec_logits = logits_shifted[..., vec_token_tensor]
    vec_probs = torch.softmax(vec_logits, dim=-1)

    label_vec_mask = torch.isin(labels_shifted, vec_token_tensor)

    ade_values = []
    B = logits_shifted.size(0)
    for b in range(B):
        raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
        traj_tensor = torch.as_tensor(raw_traj, device=device, dtype=dtype)
        traj_tensor = traj_tensor - traj_tensor[0]
        traj_xy = traj_tensor[9:, :2]
        if traj_xy.size(0) < 2:
            continue
        deltas = traj_xy[1:] - traj_xy[:-1]

        vec_positions = torch.nonzero(label_vec_mask[b], as_tuple=False).flatten()
        if vec_positions.numel() == 0:
            continue

        steps = min(vec_positions.size(0), deltas.size(0))
        if steps == 0:
            continue

        vec_positions = vec_positions[:steps]
        target_steps = deltas[:steps]
        probs_b = vec_probs[b, vec_positions]

        diff = vec_steps.unsqueeze(0) - target_steps.unsqueeze(1)
        step_errors = (probs_b * diff.norm(dim=-1)).sum(dim=-1)
        ade_values.append(step_errors.mean())

    if ade_values:
        ade_value = torch.stack(ade_values).mean()
    else:
        ade_value = torch.tensor(0.0, device=device, dtype=dtype)

    aux_loss = ade_value

    return aux_loss, ade_value, 0.0
