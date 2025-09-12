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


val_start_heading_path = "/p/ruishen/processed_waymo_data/validation/waymo_vectorized/combined_original_ego_start_heading.json"
train_start_heading_path = "/p/ruishen/processed_waymo_data/training/waymo_vectorized/combined_original_ego_start_heading.json"
val_start_heading = json.load(open(val_start_heading_path, 'r'))
train_start_heading = json.load(open(train_start_heading_path, 'r'))

combined_start_heading = {**train_start_heading, **val_start_heading}



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

def good_ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
    global angle_bins, combined_start_heading
    if not hasattr(ade_loss, 'vec_token_ids'):
        vocab = tokenizer.get_vocab()
        ade_loss.vec_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_"))
        ade_loss.len_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("LEN_"))
        ade_loss.vec_to_angle, ade_loss.len_to_length = build_vec_len_lookup(tokenizer, device=logits.device, dtype=logits.dtype)
        # Precompute angle_bins as torch tensor on device for faster access and vectorization
        ade_loss.angle_bins_tensor = torch.tensor(angle_bins, device=logits.device, dtype=logits.dtype)
    
    vals, idx = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    B, T, K = probs.shape
    
    # Precompute all trajectory tensors for the batch to avoid per-b loading
    norm_raw_traj_tensors = []
    default_ade_tensors = []
    for b in range(B):
        raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
        norm_raw_traj = raw_traj - raw_traj[0]
        norm_raw_traj = norm_raw_traj[9:]
        norm_raw_traj_tensor = torch.as_tensor(norm_raw_traj, device=logits.device, dtype=logits.dtype)
        norm_raw_traj_tensors.append(norm_raw_traj_tensor)
        default_ade_tensor = torch.norm(norm_raw_traj_tensor, dim=-1).mean() * 0.1
        default_ade_tensors.append(default_ade_tensor)
    
    # Convert default_ade to numpy list at end for return
    default_ade_list = [d.item() for d in default_ade_tensors]
    
    ade_values = []
    for b in range(B):
        next_token_ade = []
        norm_raw_traj_tensor = norm_raw_traj_tensors[b]
        default_ade = default_ade_tensors[b]
        traj_step_count = 1
        for t in range(T-1):
            if labels is not None and labels[b, t] == -100:
                continue

            vec_mask = [i for i in range(K) if idx[b, t, i].item() in ade_loss.vec_token_ids]
            len_mask = [i for i in range(K) if idx[b, t+1, i].item() in ade_loss.len_token_ids]

            if len(vec_mask) == 0 or len(len_mask) == 0:
                continue

            p_vec = probs[b, t, vec_mask]
            ids_vec = idx[b, t, vec_mask]
            p_len = probs[b, t+1, len_mask]
            ids_len = idx[b, t+1, len_mask]

            pair_probs = p_vec[:, None] * p_len[None, :]

            # Vectorized computation: get angles and lengths as tensors
            vec_angles = ade_loss.vec_to_angle[ids_vec]
            len_lengths = ade_loss.len_to_length[ids_len]
            start_heading = combined_start_heading[f"{sid[b]}__{ego_id[b]}"][1]
            # Compute ego_headings for all vec (shape: num_vec)
            try:
                ego_headings_vec = start_heading + ade_loss.angle_bins_tensor[vec_angles]
            except (IndexError, RuntimeError):
                continue  # Skip if any vec index is invalid
            # Compute dx, dy for all pairs (shape: num_vec, num_len)
            cos_headings = torch.cos(ego_headings_vec)[:, None]  # (num_vec, 1)
            sin_headings = torch.sin(ego_headings_vec)[:, None]  # (num_vec, 1)
            dx = len_lengths[None, :] * cos_headings  # (num_vec, num_len)
            dy = len_lengths[None, :] * sin_headings  # (num_vec, num_len)
            pos = torch.stack([dx, dy], dim=-1)  # (num_vec, num_len, 2)
            target = norm_raw_traj_tensor[traj_step_count] - norm_raw_traj_tensor[traj_step_count - 1]
            # Compute norms for all pairs (broadcasting handles shapes)
            diff = pos - target  # (num_vec, num_len, 2)
            norm_diff = torch.norm(diff, dim=-1)  # (num_vec, num_len)
            # Weighted ADE sum for this step
            weighted_ade = pair_probs * norm_diff
            step_ade = weighted_ade.sum()
            next_token_ade.append(step_ade)
            traj_step_count += 1
            if len(next_token_ade) >= len(norm_raw_traj_tensor) - 1:
                break

        if next_token_ade:
            if len(next_token_ade) < len(norm_raw_traj_tensor) - 1:
                next_token_ade.append(step_ade * len(norm_raw_traj_tensor) / len(next_token_ade))
            ade_values.append(sum(next_token_ade) / len(next_token_ade))
        else:
            ade_values.append(logits.sum() * 0 + default_ade)  # Maintain tensor type for gradient flow

    if ade_values:
        ade_value = torch.stack(ade_values).sum()
    else:
        ade_value = logits.sum() * 0 + np.mean(default_ade_list)

    aux_loss = ade_value
    default_ade = np.mean(default_ade_list)
    return aux_loss, ade_value, 0



def ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
    global angle_bins, combined_start_heading
    if not hasattr(ade_loss, 'vec_token_ids'):
        vocab = tokenizer.get_vocab()
        ade_loss.vec_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_"))
        ade_loss.len_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("LEN_"))
        ade_loss.vec_to_angle, ade_loss.len_to_length = build_vec_len_lookup(tokenizer, device=logits.device, dtype=logits.dtype)
        # Precompute angle_bins as torch tensor on device for faster access and vectorization
        ade_loss.angle_bins_tensor = torch.tensor(angle_bins, device=logits.device, dtype=logits.dtype)
    
    vals, idx = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    B, T, K = probs.shape
    
    # Precompute all trajectory tensors for the batch to avoid per-b loading
    norm_raw_traj_tensors = []
    for b in range(B):
        raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
        norm_raw_traj = raw_traj - raw_traj[0]
        norm_raw_traj = norm_raw_traj[9:]
        norm_raw_traj_tensor = torch.as_tensor(norm_raw_traj, device=logits.device, dtype=logits.dtype)
        norm_raw_traj_tensors.append(norm_raw_traj_tensor)
        
    ade_values = []
    for b in range(B):
        next_token_ade = []
        norm_raw_traj_tensor = norm_raw_traj_tensors[b]
        traj_step_count = 1
        for t in range(T-1):
            if labels is not None and labels[b, t] == -100:
                continue

            vec_mask = [i for i in range(K) if idx[b, t, i].item() in ade_loss.vec_token_ids]
            len_mask = [i for i in range(K) if idx[b, t+1, i].item() in ade_loss.len_token_ids]

            if len(vec_mask) == 0 or len(len_mask) == 0:
                continue

            p_vec = probs[b, t, vec_mask]
            ids_vec = idx[b, t, vec_mask]
            p_len = probs[b, t+1, len_mask]
            ids_len = idx[b, t+1, len_mask]

            pair_probs = p_vec[:, None] * p_len[None, :]

            # Vectorized computation: get angles and lengths as tensors
            vec_angles = ade_loss.vec_to_angle[ids_vec]
            len_lengths = ade_loss.len_to_length[ids_len]
            start_heading = combined_start_heading[f"{sid[b]}__{ego_id[b]}"][1]
            # Compute ego_headings for all vec (shape: num_vec)
            try:
                ego_headings_vec = start_heading + ade_loss.angle_bins_tensor[vec_angles]
            except (IndexError, RuntimeError):
                continue  # Skip if any vec index is invalid
            # Compute dx, dy for all pairs (shape: num_vec, num_len)
            cos_headings = torch.cos(ego_headings_vec)[:, None]  # (num_vec, 1)
            sin_headings = torch.sin(ego_headings_vec)[:, None]  # (num_vec, 1)
            dx = len_lengths[None, :] * cos_headings  # (num_vec, num_len)
            dy = len_lengths[None, :] * sin_headings  # (num_vec, num_len)
            pos = torch.stack([dx, dy], dim=-1)  # (num_vec, num_len, 2)
            target = norm_raw_traj_tensor[traj_step_count] - norm_raw_traj_tensor[traj_step_count - 1]
            # Compute norms for all pairs (broadcasting handles shapes)
            diff = pos - target  # (num_vec, num_len, 2)
            norm_diff = torch.norm(diff, dim=-1)  # (num_vec, num_len)
            # Weighted ADE sum for this step
            weighted_ade = pair_probs * norm_diff
            step_ade = weighted_ade.sum()
            next_token_ade.append(step_ade)
            traj_step_count += 1
            if len(next_token_ade) >= len(norm_raw_traj_tensor) - 1:
                break

        if next_token_ade:
            ade_values.append(sum(next_token_ade) / len(next_token_ade) * len(norm_raw_traj_tensor) / len(next_token_ade))
        else:
            ade_values.append(torch.tensor(0.0, device=logits.device, dtype=logits.dtype))

    if ade_values:
        ade_value = torch.stack(ade_values).sum()
    else:
        ade_value = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    aux_loss = ade_value
    return aux_loss, ade_value, 0
