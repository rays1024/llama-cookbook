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

def multi_label_bce_loss(
    logits,
    labels,
    multi_label,
    label_weight=None,
    tokenizer=None,
    ignore_index=-100,
    reduction="mean",
    pos_weight=100.0,
):
    B, T, V = logits.size()
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()

    logits_flat = shifted_logits.view(-1, V)
    labels_flat = shifted_labels.view(-1)
    valid_mask = labels_flat != ignore_index
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return torch.tensor(0.0, device=logits.device)

    if tokenizer is not None:
        cached_id = getattr(multi_label_bce_loss, "_vec_tokenizer_id", None)
        if not hasattr(multi_label_bce_loss, "vec_token_ids") or cached_id != id(tokenizer):
            vec_tokens = tokenizer.convert_tokens_to_ids([f"VEC_{i}" for i in range(1024)])
            vec_token_ids = torch.tensor(vec_tokens, dtype=torch.long)
            vec_token_ids = vec_token_ids[vec_token_ids >= 0]
            if vec_token_ids.numel() == 0:
                raise ValueError("No VEC_* tokens found in tokenizer.")
            multi_label_bce_loss.vec_token_ids = vec_token_ids
            multi_label_bce_loss._vec_tokenizer_id = id(tokenizer)
    if not hasattr(multi_label_bce_loss, "vec_token_ids"):
        raise ValueError("tokenizer is required to cache VEC_* token ids.")
    vec_token_ids = multi_label_bce_loss.vec_token_ids.to(device=logits.device)

    if torch.is_tensor(multi_label):
        multi_label = multi_label.to(device=logits.device)
    if label_weight is not None and torch.is_tensor(label_weight):
        label_weight = label_weight.to(device=logits.device, dtype=logits.dtype)

    if multi_label is not None and multi_label.dim() == 3:
        if multi_label.size(0) != B:
            raise ValueError("multi_label batch size does not match logits.")
        if multi_label.size(1) == labels.size(1):
            multi_label = multi_label[:, 1:, :]
            if label_weight is not None and label_weight.dim() == 3 and label_weight.size(1) == labels.size(1):
                label_weight = label_weight[:, 1:, :]
        if multi_label.size(1) == labels.size(1) - 1:
            multi_label_flat = multi_label.reshape(-1, multi_label.size(-1))[valid_mask]
            if label_weight is None:
                label_weight_flat = torch.ones_like(multi_label_flat, dtype=logits.dtype)
            else:
                label_weight_flat = label_weight.reshape(-1, label_weight.size(-1))[valid_mask]
        else:
            valid_per_sample = (shifted_labels != ignore_index).sum(dim=1).tolist()
            valid_full_per_sample = (labels != ignore_index).sum(dim=1).tolist()
            first_label_valid = (labels[:, 0] != ignore_index).tolist()
            multi_label_chunks = []
            weight_chunks = []
            for b in range(B):
                count = valid_per_sample[b]
                if count == 0:
                    continue
                start = 1 if valid_full_per_sample[b] == count + 1 and first_label_valid[b] else 0
                multi_label_chunks.append(multi_label[b, start:start + count, :])
                if label_weight is not None:
                    weight_chunks.append(label_weight[b, start:start + count, :])
            multi_label_flat = torch.cat(multi_label_chunks, dim=0) if multi_label_chunks else multi_label.new_empty((0, multi_label.size(-1)))
            if label_weight is None:
                label_weight_flat = torch.ones_like(multi_label_flat, dtype=logits.dtype)
            else:
                label_weight_flat = torch.cat(weight_chunks, dim=0) if weight_chunks else label_weight.new_empty((0, label_weight.size(-1)))
    elif multi_label is not None and multi_label.dim() == 2:
        if multi_label.size(0) != valid_count:
            raise ValueError("multi_label length does not match the number of valid labels.")
        multi_label_flat = multi_label
        if label_weight is None:
            label_weight_flat = torch.ones_like(multi_label_flat, dtype=logits.dtype)
        else:
            label_weight_flat = label_weight
    else:
        if not isinstance(multi_label, (list, tuple)) or len(multi_label) != B:
            raise ValueError("multi_label must be a tensor or a list with batch length.")
        if label_weight is not None and (not isinstance(label_weight, (list, tuple)) or len(label_weight) != B):
            raise ValueError("label_weight must be a tensor or a list with batch length.")
        valid_per_sample = (shifted_labels != ignore_index).sum(dim=1).tolist()
        valid_full_per_sample = (labels != ignore_index).sum(dim=1).tolist()
        first_label_valid = (labels[:, 0] != ignore_index).tolist()
        multi_label_flat_list = []
        label_weight_flat_list = []
        for b in range(B):
            sample_multi = list(multi_label[b])
            sample_weight = list(label_weight[b]) if label_weight is not None else None
            if len(sample_multi) == valid_full_per_sample[b] and first_label_valid[b]:
                sample_multi = sample_multi[1:]
                if sample_weight is not None:
                    sample_weight = sample_weight[1:]
            if len(sample_multi) < valid_per_sample[b]:
                raise ValueError("multi_label is shorter than the number of valid labels.")
            if len(sample_multi) > valid_per_sample[b]:
                sample_multi = sample_multi[:valid_per_sample[b]]
                if sample_weight is not None:
                    sample_weight = sample_weight[:valid_per_sample[b]]
            multi_label_flat_list.extend(sample_multi)
            if sample_weight is not None:
                label_weight_flat_list.extend(sample_weight)
        multi_label_flat = torch.as_tensor(multi_label_flat_list, device=logits.device, dtype=torch.long)
        if label_weight is None:
            label_weight_flat = torch.ones_like(multi_label_flat, dtype=logits.dtype)
        else:
            label_weight_flat = torch.as_tensor(label_weight_flat_list, device=logits.device, dtype=logits.dtype)

    if multi_label_flat.size(0) != valid_count:
        raise ValueError("multi_label does not align with the number of valid labels after shifting.")

    valid_logits = logits_flat[valid_mask]
    valid_labels = labels_flat[valid_mask]

    vec_vocab_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
    vec_vocab_mask[vec_token_ids] = True
    non_vec_mask = ~vec_vocab_mask
    vec_target_mask = vec_vocab_mask[valid_labels]

    # BCE #1: classify whether the target token is VEC_* or non-VEC.
    if non_vec_mask.any():
        vec_group_logits = torch.logsumexp(valid_logits[:, vec_vocab_mask], dim=1)
        non_vec_group_logits = torch.logsumexp(valid_logits[:, non_vec_mask], dim=1)
        type_logits = vec_group_logits - non_vec_group_logits
        type_targets = vec_vocab_mask[valid_labels].to(dtype=logits.dtype)
        type_loss = F.binary_cross_entropy_with_logits(type_logits, type_targets, reduction="none")
    else:
        type_loss = torch.zeros(valid_count, device=logits.device, dtype=logits.dtype)

    # BCE #2: multi-label BCE over VEC_* tokens only.
    pos_mask = multi_label_flat != ignore_index
    safe_indices = multi_label_flat.clone()
    safe_indices[~pos_mask] = 0
    pos_is_vec = pos_mask & vec_vocab_mask[safe_indices]

    pos_logits = valid_logits.gather(dim=1, index=safe_indices)
    pos_logsig = F.logsigmoid(pos_logits)
    pos_weights = label_weight_flat * pos_is_vec.to(label_weight_flat.dtype)
    pos_loss = -(pos_weights * pos_logsig).sum(dim=1)
    pos_loss = pos_loss * float(pos_weight)

    vec_logits = valid_logits[:, vec_token_ids]
    neg_logsig = -F.logsigmoid(-vec_logits)

    vec_index = torch.full((V,), -1, dtype=torch.long, device=logits.device)
    vec_index[vec_token_ids] = torch.arange(vec_token_ids.numel(), device=logits.device)
    pos_vec_indices = vec_index[safe_indices]
    pos_vec_indices = pos_vec_indices.masked_fill(~pos_is_vec, -1)
    valid_pos = pos_vec_indices >= 0
    if valid_pos.any():
        pos_vec_counts = torch.zeros((valid_count, vec_token_ids.numel()), dtype=torch.int32, device=logits.device)
        pos_vec_counts.scatter_add_(1, pos_vec_indices.clamp_min(0), valid_pos.int())
        pos_vec_mask = pos_vec_counts > 0
    else:
        pos_vec_mask = torch.zeros((valid_count, vec_token_ids.numel()), dtype=torch.bool, device=logits.device)

    neg_mask = ~pos_vec_mask
    neg_loss = (neg_logsig * neg_mask.to(neg_logsig.dtype)).sum(dim=1)

    pos_loss = pos_loss * vec_target_mask.to(pos_loss.dtype)
    neg_loss = neg_loss * vec_target_mask.to(neg_loss.dtype)

    total_loss = type_loss + pos_loss + neg_loss
    if reduction == "sum":
        return total_loss.sum()
    if reduction == "none":
        output = torch.zeros_like(labels_flat, dtype=total_loss.dtype)
        output[valid_mask] = total_loss
        return output.view(B, T - 1)
    return total_loss.sum() / valid_count
    

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
