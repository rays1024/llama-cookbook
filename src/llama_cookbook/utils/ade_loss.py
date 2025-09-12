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

# def ade_loss(generated_texts, sid, ego_id, weight=0.5, context=None):
#     global angle_bins, combined_start_heading
#     pattern = r"VEC_(-?\d+(?:\.\d+)?)LEN_(-?\d+(?:\.\d+)?)"
#     ade_values = []
#     history_pattern = r"EGO_TRAJ_START.*?EGO_TRAJ_END"
#     default_ade_list = []

#     for i, text in enumerate(generated_texts):
#         history = re.search(history_pattern, context[i], re.DOTALL)
#         matches = history.group(0)
#         _, ego_heading = combined_start_heading[f"{sid[i]}__{ego_id[i]}"]
#         history = [[0,0]]
#         matches = re.findall(pattern, matches)
#         for match in matches:
#             v = int(float(match[0]))
#             l = float(match[1])
#             try:
#                 ego_heading += angle_bins[v]
#             except IndexError:
#                 continue
#             dx = l * np.cos(ego_heading)
#             dy = l * np.sin(ego_heading)
#             history.append([history[-1][0] + dx, history[-1][1] + dy])

#         matches = re.findall(pattern, text)
#         raw_traj, _ = load_trajectory_by_key_from_memory(sid[i], ego_id[i])
#         norm_raw_traj = raw_traj - raw_traj[0]
#         default_ade = np.linalg.norm(norm_raw_traj, axis=1).mean() * 2
#         default_ade_list.append(default_ade)
#         if not matches:
#             ade_values.append(default_ade)
#             continue
#         llm_points = history
#         for match in matches:
#             v = int(float(match[0]))
#             l = float(match[1])
#             try:
#                 ego_heading += angle_bins[v]
#             except IndexError:
#                 continue
#             ego_heading = (ego_heading + np.pi) % (2 * np.pi) - np.pi
#             dx = l * np.cos(ego_heading)
#             dy = l * np.sin(ego_heading)
#             llm_points.append([llm_points[-1][0] + dx, llm_points[-1][1] + dy])
#             if len(llm_points) > len(norm_raw_traj):
#                 break
#         llm_points = np.array(llm_points)

#         if len(llm_points) < len(norm_raw_traj):
#             llm_points = np.concatenate([llm_points, np.zeros((len(norm_raw_traj) - len(llm_points), 2))])
#         elif len(llm_points) > len(norm_raw_traj):
#             llm_points = llm_points[:len(norm_raw_traj)]
#         ade = np.linalg.norm(llm_points - norm_raw_traj, axis=1).mean()
#         ade_values.append(ade)
    
#     ade_value = np.mean(ade_values)
#     aux_loss = log_normalize_with_target(ade_value)
#     aux_loss = aux_loss * weight
#     default_ade = np.mean(default_ade_list)

#     return ade_value, aux_loss, default_ade

# def ade_loss(generated_texts, sid, ego_id, weight=1.0, context=None, **kwargs):
#     global angle_bins, combined_start_heading
#     pattern = r"VEC_(-?\d+(?:\.\d+)?)LEN_(-?\d+(?:\.\d+)?)"
#     ade_values = []
#     default_ade_list = []

#     for i, text in enumerate(generated_texts):
#         matches = re.findall(pattern, text)
#         raw_traj, _ = load_trajectory_by_key_from_memory(sid[i], ego_id[i])
#         norm_raw_traj = raw_traj - raw_traj[0]
#         default_ade = np.linalg.norm(norm_raw_traj, axis=1).mean() * 2
#         default_ade_list.append(default_ade)
#         if not matches:
#             ade_values.append(default_ade)
#             continue
#         next_token_ade = []
#         delta_raw_traj = np.diff(norm_raw_traj, axis=0, prepend=np.array([[0,0]]))[9:]
#         heading_raw_traj = np.arctan2(delta_raw_traj[:,1], delta_raw_traj[:,0])
#         for t_i, match in enumerate(matches):
#             v = int(float(match[0]))
#             l = float(match[1])
#             try:
#                 ego_heading = heading_raw_traj[t_i] + angle_bins[v]
#             except IndexError:
#                 continue
#             ego_heading = (ego_heading + np.pi) % (2 * np.pi) - np.pi
#             dx = l * np.cos(ego_heading)
#             dy = l * np.sin(ego_heading)
#             next_token_ade.append(np.linalg.norm(np.array([dx, dy]) - delta_raw_traj[t_i]))
#             if t_i > len(norm_raw_traj):
#                 break
#         ade_values.append(np.mean(next_token_ade))
    
#     ade_value = np.mean(ade_values)
#     aux_loss = log_normalize_with_target(ade_value)
#     aux_loss = aux_loss * weight
#     default_ade = np.mean(default_ade_list)

#     return ade_value, aux_loss, default_ade








# def ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
#     global angle_bins, combined_start_heading
#     if not hasattr(ade_loss, 'vec_token_ids'):
#         vocab = tokenizer.get_vocab()
#         ade_loss.vec_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_"))
#         ade_loss.len_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("LEN_"))
#         ade_loss.vec_to_angle, ade_loss.len_to_length = build_vec_len_lookup(tokenizer, device=logits.device, dtype=logits.dtype)
#         ade_loss.traj_token_ids = ade_loss.vec_token_ids.union(ade_loss.len_token_ids)
#         ade_loss.traj_token_ids = torch.tensor(list(ade_loss.traj_token_ids), device=logits.device)
#         ade_loss.angle_bins_tensor = torch.tensor(angle_bins, device=logits.device, dtype=logits.dtype)
    
#     vals, idx = torch.topk(logits, top_k, dim=-1)
#     probs = torch.softmax(vals, dim=-1)
#     B, T, K = probs.shape
    
#     # Precompute all trajectory tensors for the batch to avoid per-b loading
#     norm_raw_traj_tensors = []
#     for b in range(B):
#         raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
#         norm_raw_traj = raw_traj - raw_traj[0]
#         norm_raw_traj = norm_raw_traj[11:]
#         norm_raw_traj_tensor = torch.as_tensor(norm_raw_traj, device=logits.device, dtype=logits.dtype)
#         norm_raw_traj_tensors.append(norm_raw_traj_tensor)
        
#     ade_values = []
#     for b in range(B):
#         next_token_ade = []
#         norm_raw_traj_tensor = norm_raw_traj_tensors[b]
#         traj_step_count = 1
#         for t in range(T-1):
#             if labels is not None and labels[b, t] == -100:
#                 continue

#             vec_mask = [i for i in range(K) if idx[b, t, i].item() in ade_loss.vec_token_ids]
#             len_mask_t = [i for i in range(K) if idx[b, t, i].item() in ade_loss.len_token_ids]
#             len_mask = [i for i in range(K) if idx[b, t+1, i].item() in ade_loss.len_token_ids]

#             if len(vec_mask) == 0 and len(len_mask_t) == 0:
#                 ce_loss = F.cross_entropy(logits[b, t, :].unsqueeze(0), labels[b, t].unsqueeze(0))
#                 next_token_ade.append(ce_loss)
#                 continue

#             if len(vec_mask) == 0 or len(len_mask) == 0:
#                 ce_loss = F.cross_entropy(logits[b, t, :].unsqueeze(0), labels[b, t].unsqueeze(0))
#                 next_token_ade.append(ce_loss)
#                 continue

#             p_vec = probs[b, t, vec_mask]
#             ids_vec = idx[b, t, vec_mask]
#             p_len = probs[b, t+1, len_mask]
#             ids_len = idx[b, t+1, len_mask]

#             pair_probs = p_vec[:, None] * p_len[None, :]

#             # Vectorized computation: get angles and lengths as tensors
#             vec_angles = ade_loss.vec_to_angle[ids_vec]
#             len_lengths = ade_loss.len_to_length[ids_len]
#             start_heading = combined_start_heading[f"{sid[b]}__{ego_id[b]}"][1]
#             # Compute ego_headings for all vec (shape: num_vec)
#             try:
#                 ego_headings_vec = start_heading + ade_loss.angle_bins_tensor[vec_angles]
#             except (IndexError, RuntimeError):
#                 continue  # Skip if any vec index is invalid
#             # Compute dx, dy for all pairs (shape: num_vec, num_len)
#             cos_headings = torch.cos(ego_headings_vec)[:, None]  # (num_vec, 1)
#             sin_headings = torch.sin(ego_headings_vec)[:, None]  # (num_vec, 1)
#             dx = len_lengths[None, :] * cos_headings  # (num_vec, num_len)
#             dy = len_lengths[None, :] * sin_headings  # (num_vec, num_len)
#             pos = torch.stack([dx, dy], dim=-1)  # (num_vec, num_len, 2)
#             target = norm_raw_traj_tensor[traj_step_count] - norm_raw_traj_tensor[traj_step_count - 1]
#             # Compute norms for all pairs (broadcasting handles shapes)
#             diff = pos - target  # (num_vec, num_len, 2)
#             norm_diff = torch.norm(diff, dim=-1)  # (num_vec, num_len)
#             # Weighted ADE sum for this step
#             weighted_ade = pair_probs * norm_diff
#             step_ade = weighted_ade.sum()
#             next_token_ade.append(step_ade)
#             traj_step_count += 1
#             if traj_step_count == len(norm_raw_traj_tensor):
#                 break
#         ade_values.append(sum(next_token_ade) / len(next_token_ade))

#     ade_value = torch.stack(ade_values).sum()

#     # aux_loss = log_normalize_with_target(ade_value)
#     # aux_loss = aux_loss * weight
#     aux_loss = ade_value
#     return ade_value, aux_loss, 0









# def ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
#     """
#     ADE/CE hybrid loss (updated).

#     - ADE is computed from top-k VEC@t × LEN@(t+1) pairs (as before).
#     - When do_ade is True at t:
#         * Do NOT apply token-level CE at t.
#         * Apply ADE expectation term (top-k).
#         * Apply group-CE loss that encourages mass on VEC@t and LEN@t+1,
#           punishing non-traj tokens implicitly.
#     - When do_ade is False at t: use the original CE (with paired-LEN masking).
#     """
#     global angle_bins, combined_start_heading

#     device, dtype = logits.device, logits.dtype
#     B, T, V = logits.shape

#     # ---- lazy init / cache tied to device+dtype ----
#     need_init = (
#         not hasattr(ade_loss, "_cache")
#         or ade_loss._cache.get("device") != device
#         or ade_loss._cache.get("dtype")  != dtype
#     )
#     if need_init:
#         vocab = tokenizer.get_vocab()
#         max_id = max(vocab.values())
#         lut_size = max_id + 1

#         vec_token_ids = [tid for tok, tid in vocab.items() if tok.startswith("VEC_")]
#         len_token_ids = [tid for tok, tid in vocab.items() if tok.startswith("LEN_")]

#         vec_lut = torch.zeros(lut_size, dtype=torch.bool, device=device)
#         len_lut = torch.zeros(lut_size, dtype=torch.bool, device=device)
#         if len(vec_token_ids) > 0:
#             vec_lut[torch.tensor(vec_token_ids, device=device)] = True
#         if len(len_token_ids) > 0:
#             len_lut[torch.tensor(len_token_ids, device=device)] = True

#         vec_to_angle, len_to_length = build_vec_len_lookup(
#             tokenizer, device=device, dtype=dtype
#         )

#         ade_loss._cache = dict(
#             device=device,
#             dtype=dtype,
#             vec_lut=vec_lut,
#             len_lut=len_lut,
#             vec_to_angle=vec_to_angle,
#             len_to_length=len_to_length,
#         )

#     vec_lut      = ade_loss._cache["vec_lut"]
#     len_lut      = ade_loss._cache["len_lut"]
#     vec_to_angle = ade_loss._cache["vec_to_angle"]
#     len_to_length= ade_loss._cache["len_to_length"]
#     angle_bins_tensor = torch.as_tensor(angle_bins, device=device, dtype=dtype)

#     vec_mask_all = vec_lut[:V]
#     len_mask_all = len_lut[:V]

#     # ---- top-k once ----
#     topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)  # (B,T,K)
#     topk_probs = torch.softmax(topk_vals, dim=-1)            # (B,T,K)
#     vec_flags  = vec_lut[topk_idx]
#     len_flags  = len_lut[topk_idx]

#     has_vec      = vec_flags.any(dim=-1)         # (B,T)
#     has_len_next = len_flags[:, 1:, :].any(-1)   # (B,T-1)

#     # --- mask CE for paired LEN tokens ---
#     safe_labels = labels.clone()
#     safe_labels[safe_labels < 0] = 0
#     is_len_token = len_lut[safe_labels]
#     is_vec_token = vec_lut[safe_labels]
#     is_paired_len = torch.zeros_like(labels, dtype=torch.bool)
#     is_paired_len[:, 1:] = is_vec_token[:, :-1] & is_len_token[:, 1:]

#     labels_short   = labels[:, :T-1]
#     labels_valid   = (labels_short != -100)
#     ce_mask = labels_valid.clone()
#     is_paired_len_short = is_paired_len[:, :T-1]
#     ce_mask[:, 1:] &= ~is_paired_len_short[:, 1:]

#     ce_targets = labels_short.clone()
#     ce_targets[~ce_mask] = 0
#     ce_loss_all = F.cross_entropy(
#         logits[:, :T-1, :].reshape(-1, V),
#         ce_targets.reshape(-1),
#         reduction="none"
#     ).reshape(B, T-1)

#     ade_allowed = labels_valid & has_vec[:, :T-1] & has_len_next

#     # ---- cache all probs for speed ----
#     all_probs = torch.softmax(logits, dim=-1)  # (B,T,V)

#     # preload trajectories
#     start_headings = torch.empty((B,), device=device, dtype=dtype)
#     traj_tensors   = []
#     for b in range(B):
#         raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
#         norm_traj = raw_traj - raw_traj[0]
#         norm_traj = norm_traj[11:]
#         traj_tensors.append(torch.as_tensor(norm_traj, device=device, dtype=dtype))
#         start_headings[b] = torch.as_tensor(
#             combined_start_heading[f"{sid[b]}__{ego_id[b]}"][1],
#             device=device, dtype=dtype
#         )

#     batch_losses, ade_values, ce_values = [], [], []
#     eps = torch.finfo(dtype).eps

#     for b in range(B):
#         traj = traj_tensors[b]
#         L = traj.size(0)
#         step_ptr = 0
#         step_losses = []

#         for t in range(T - 1):
#             do_ade = bool(ade_allowed[b, t])

#             if do_ade:
#                 if step_ptr + 1 >= L:
#                     break

#                 vec_mask_t    = vec_flags[b, t]
#                 len_mask_next = len_flags[b, t+1]
#                 p_vec = topk_probs[b, t][vec_mask_t]
#                 ids_v = topk_idx[b, t][vec_mask_t]
#                 p_len = topk_probs[b, t+1][len_mask_next]
#                 ids_l = topk_idx[b, t+1][len_mask_next]

#                 if p_vec.numel() == 0 or p_len.numel() == 0:
#                     if bool(ce_mask[b, t]):
#                         step_losses.append(ce_loss_all[b, t])
#                         ce_values.append(ce_loss_all[b, t].item())
#                     continue

#                 vec_angles = vec_to_angle[ids_v.long()].long()
#                 len_lengths = len_to_length[ids_l.long()]
#                 try:
#                     headings = start_headings[b] + angle_bins_tensor[vec_angles]
#                 except Exception:
#                     continue
#                 cos_h = torch.cos(headings)[:, None]
#                 sin_h = torch.sin(headings)[:, None]
#                 dx = len_lengths[None, :] * cos_h
#                 dy = len_lengths[None, :] * sin_h
#                 pos = torch.stack([dx, dy], -1)

#                 target = traj[step_ptr+1] - traj[step_ptr]
#                 norm_diff = torch.norm(pos - target, dim=-1)
#                 pair_probs = p_vec[:, None] * p_len[None, :]
#                 step_ade = (pair_probs * norm_diff).sum()
#                 step_losses.append(step_ade)
#                 ade_values.append(step_ade.item())

#                 # group CE for VEC at t and LEN at t+1
#                 probs_t = all_probs[b, t]
#                 probs_t1 = all_probs[b, t+1]
#                 logp_vec = torch.log(probs_t[vec_mask_all] + eps)
#                 logp_len = torch.log(probs_t1[len_mask_all] + eps)
#                 group_ce_vec = -logp_vec.mean()
#                 group_ce_len = -logp_len.mean()
#                 nontraj_ce_loss = group_ce_vec + group_ce_len
#                 step_losses.append(nontraj_ce_loss)
#                 ce_values.append(float(nontraj_ce_loss))

#                 step_ptr += 1
#                 if step_ptr == L - 1:
#                     break

#             else:
#                 if bool(ce_mask[b, t]):
#                     step_losses.append(ce_loss_all[b, t])
#                     ce_values.append(ce_loss_all[b, t].item())

#         batch_losses.append(
#             torch.stack(step_losses).mean() if step_losses else torch.zeros((), device=device)
#         )
#     aux_loss = torch.stack(batch_losses).sum()
#     ade = np.mean(ade_values) if ade_values else 0.0
#     ce  = np.mean(ce_values)  if ce_values else 0.0
#     return aux_loss, ade, ce





# def ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
#     """
#     ADE-guided auxiliary loss:
#       • ADE computed from top-k VEC@t x LEN@(t+1) geometry.
#       • Geometry mapped to soft targets over all VEC/LEN tokens.
#       • No extra CE-style penalties (to avoid double punishment).
#       • Combine externally with CE loss for full supervision:
#             total_loss = ce_loss_outside + lambda_ade * aux_loss
#     """

#     global angle_bins, combined_start_heading

#     device, dtype = logits.device, logits.dtype
#     B, T, V = logits.shape
#     eps = torch.finfo(dtype).eps

#     # ---------- cache LUTs ----------
#     need_init = (
#         not hasattr(ade_loss, "_cache")
#         or ade_loss._cache.get("device") != device
#         or ade_loss._cache.get("dtype")  != dtype
#     )
#     if need_init:
#         vocab = tokenizer.get_vocab()
#         max_id = max(vocab.values())
#         lut_size = max_id + 1

#         vec_token_ids = [tid for tok, tid in vocab.items() if tok.startswith("VEC_")]
#         len_token_ids = [tid for tok, tid in vocab.items() if tok.startswith("LEN_")]

#         vec_lut = torch.zeros(lut_size, dtype=torch.bool, device=device)
#         len_lut = torch.zeros(lut_size, dtype=torch.bool, device=device)
#         if len(vec_token_ids) > 0:
#             vec_lut[torch.tensor(vec_token_ids, device=device, dtype=torch.long)] = True
#         if len(len_token_ids) > 0:
#             len_lut[torch.tensor(len_token_ids, device=device, dtype=torch.long)] = True

#         vec_to_angle, len_to_length = build_vec_len_lookup(
#             tokenizer, device=device, dtype=dtype
#         )

#         ade_loss._cache = dict(
#             device=device, dtype=dtype,
#             vec_lut=vec_lut, len_lut=len_lut,
#             vec_to_angle=vec_to_angle, len_to_length=len_to_length,
#         )

#     vec_lut        = ade_loss._cache["vec_lut"]
#     len_lut        = ade_loss._cache["len_lut"]
#     vec_to_angle   = ade_loss._cache["vec_to_angle"]
#     len_to_length  = ade_loss._cache["len_to_length"]
#     angle_bins_tensor = torch.as_tensor(angle_bins, device=device, dtype=dtype)

#     # Limit LUTs to current vocab
#     vec_mask_all = vec_lut[:V]
#     len_mask_all = len_lut[:V]
#     vec_ids_all  = torch.nonzero(vec_mask_all, as_tuple=False).squeeze(-1)
#     len_ids_all  = torch.nonzero(len_mask_all, as_tuple=False).squeeze(-1)

#     # Precompute top-k once
#     topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)   # (B,T,K)
#     topk_probs = torch.softmax(topk_vals, dim=-1)              # (B,T,K)
#     vec_flags = vec_mask_all[topk_idx]
#     len_flags = len_mask_all[topk_idx]

#     # Label typing
#     safe_labels = labels.clone()
#     safe_labels[safe_labels < 0] = 0
#     is_vec_lbl = vec_lut[safe_labels]
#     labels_valid = (labels != -100)

#     # Only compute ADE at t if label[t] is VEC (paired with LEN at t+1)
#     elig_t = (labels_valid[:, :-1]) & (is_vec_lbl[:, :-1])

#     # Load trajectories and headings
#     start_headings = torch.empty((B,), device=device, dtype=dtype)
#     traj_tensors   = []
#     for b in range(B):
#         raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
#         norm_traj = raw_traj - raw_traj[0]
#         norm_traj = norm_traj[11:]
#         traj_tensors.append(torch.as_tensor(norm_traj, device=device, dtype=dtype))
#         start_headings[b] = torch.as_tensor(
#             combined_start_heading[f"{sid[b]}__{ego_id[b]}"][1],
#             device=device, dtype=dtype
#         )

#     # Hyperparameters
#     gamma = 5.0   # softness of ADE→target mapping
#     alpha = 1.0   # weight for KL/CE within groups

#     batch_losses = []
#     ade_values = []

#     for b in range(B):
#         traj = traj_tensors[b]
#         L = traj.size(0)
#         step_ptr = 0
#         step_losses = []

#         for t in range(T - 1):
#             if not bool(elig_t[b, t]):
#                 continue
#             if not bool(labels_valid[b, t+1]):
#                 continue

#             # Full softmax at t and t+1
#             probs_t  = torch.softmax(logits[b, t],   dim=-1)
#             probs_t1 = torch.softmax(logits[b, t+1], dim=-1)

#             # Group probabilities
#             p_vec_all = probs_t[vec_ids_all]
#             p_len_all = probs_t1[len_ids_all]

#             # Top-k presence check
#             vec_mask_t    = vec_flags[b, t]
#             len_mask_next = len_flags[b, t+1]
#             if not (vec_mask_t.any() and len_mask_next.any() and (step_ptr + 1 < L)):
#                 continue

#             # Gather top-k vec/len
#             ids_v  = topk_idx[b, t][vec_mask_t]
#             p_vec  = topk_probs[b, t][vec_mask_t]
#             ids_l  = topk_idx[b, t+1][len_mask_next]
#             p_len  = topk_probs[b, t+1][len_mask_next]

#             # Normalize within groups
#             s_vec = p_vec.sum().clamp_min(eps).detach()
#             s_len = p_len.sum().clamp_min(eps).detach()
#             p_vec_c = p_vec / s_vec
#             p_len_c = p_len / s_len

#             # Map tokens to geometry
#             vec_angles  = vec_to_angle[ids_v.long()].long()
#             len_lengths = len_to_length[ids_l.long()]

#             try:
#                 headings = start_headings[b] + angle_bins_tensor[vec_angles]
#             except Exception:
#                 continue

#             cos_h = torch.cos(headings)[:, None]
#             sin_h = torch.sin(headings)[:, None]
#             dx = len_lengths[None, :] * cos_h
#             dy = len_lengths[None, :] * sin_h
#             pos = torch.stack([dx, dy], dim=-1)

#             target = traj[step_ptr + 1] - traj[step_ptr]
#             norm_diff = torch.norm(pos - target, dim=-1)

#             # Expected ADE (metric only)
#             step_ade = (p_vec_c[:, None] * p_len_c[None, :] * norm_diff).sum()
#             ade_values.append(step_ade.item())

#             # Marginal errors
#             err_vec_topk = (norm_diff * p_len_c[None, :]).sum(dim=1)
#             err_len_topk = (norm_diff * p_vec_c[:, None]).sum(dim=0)

#             # Full-size score vectors
#             n_vec_all = vec_ids_all.numel()
#             n_len_all = len_ids_all.numel()
#             scores_vec = torch.full((n_vec_all,), -gamma * (err_vec_topk.max().detach() + 1.0),
#                                     device=device, dtype=dtype)
#             scores_len = torch.full((n_len_all,), -gamma * (err_len_topk.max().detach() + 1.0),
#                                     device=device, dtype=dtype)

#             # Map ids → positions
#             id_to_vecpos = torch.full((V,), -1, device=device, dtype=torch.long)
#             id_to_lenpos = torch.full((V,), -1, device=device, dtype=torch.long)
#             id_to_vecpos[vec_ids_all] = torch.arange(n_vec_all, device=device, dtype=torch.long)
#             id_to_lenpos[len_ids_all] = torch.arange(n_len_all, device=device, dtype=torch.long)

#             vec_pos = id_to_vecpos[ids_v.long()]
#             len_pos = id_to_lenpos[ids_l.long()]
#             scores_vec[vec_pos] = -gamma * err_vec_topk
#             scores_len[len_pos] = -gamma * err_len_topk

#             # Soft targets
#             q_vec = torch.softmax(scores_vec, dim=-1)
#             q_len = torch.softmax(scores_len, dim=-1)

#             # Model’s distributions
#             p_vec_group = p_vec_all / p_vec_all.sum().clamp_min(eps).detach()
#             p_len_group = p_len_all / p_len_all.sum().clamp_min(eps).detach()

#             # KL/CE shaping
#             L_vec = -(q_vec * torch.log(p_vec_group.clamp_min(eps))).sum()
#             L_len = -(q_len * torch.log(p_len_group.clamp_min(eps))).sum()

#             group_loss = alpha * (L_vec + L_len)
#             step_losses.append(group_loss)

#             step_ptr += 1
#             if step_ptr == L - 1:
#                 break

#         batch_losses.append(
#             torch.stack(step_losses).mean() if step_losses else torch.zeros((), device=device, dtype=dtype)
#         )

#     aux_loss = weight * torch.stack(batch_losses).sum()
#     ade = (sum(ade_values) / len(ade_values)) if ade_values else 0.0
#     ce  = 0.0  # irrelevant here
#     return aux_loss, ade, ce








# def ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
#     global angle_bins, combined_start_heading
#     if not hasattr(ade_loss, 'vec_token_ids'):
#         vocab = tokenizer.get_vocab()
#         ade_loss.vec_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_"))
#         ade_loss.len_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("LEN_"))
#         ade_loss.vec_to_angle, ade_loss.len_to_length = build_vec_len_lookup(tokenizer, device=logits.device, dtype=logits.dtype)
#         ade_loss.traj_token_ids = ade_loss.vec_token_ids.union(ade_loss.len_token_ids)
#         ade_loss.traj_token_ids = torch.tensor(list(ade_loss.traj_token_ids), device=logits.device)
#         ade_loss.angle_bins_tensor = torch.tensor(angle_bins, device=logits.device, dtype=logits.dtype)
    
#     vals, idx = torch.topk(logits, top_k, dim=-1)
#     probs = torch.softmax(vals, dim=-1)
#     B, T, K = probs.shape
    
#     # Precompute all trajectory tensors for the batch to avoid per-b loading
#     norm_raw_traj_tensors = []
#     for b in range(B):
#         raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
#         norm_raw_traj = raw_traj - raw_traj[0]
#         norm_raw_traj = norm_raw_traj[11:]
#         norm_raw_traj_tensor = torch.as_tensor(norm_raw_traj, device=logits.device, dtype=logits.dtype)
#         norm_raw_traj_tensors.append(norm_raw_traj_tensor)
    
#     ade_values = []
#     for b in range(B):
#         next_token_ade = []
#         norm_raw_traj_tensor = norm_raw_traj_tensors[b]
#         traj_step_count = 1
#         for t in range(T-1):
#             if labels is not None and labels[b, t] == -100:
#                 continue

#             vec_mask = [i for i in range(K) if idx[b, t, i].item() in ade_loss.vec_token_ids]
#             len_mask = [i for i in range(K) if idx[b, t+1, i].item() in ade_loss.len_token_ids]

#             if len(vec_mask) == 0 or len(len_mask) == 0:
#                 continue

#             p_vec = probs[b, t, vec_mask]
#             ids_vec = idx[b, t, vec_mask]
#             p_len = probs[b, t+1, len_mask]
#             ids_len = idx[b, t+1, len_mask]

#             pair_probs = p_vec[:, None] * p_len[None, :]

#             # Vectorized computation: get angles and lengths as tensors
#             vec_angles = ade_loss.vec_to_angle[ids_vec]
#             len_lengths = ade_loss.len_to_length[ids_len]
#             start_heading = combined_start_heading[f"{sid[b]}__{ego_id[b]}"][1]
#             # Compute ego_headings for all vec (shape: num_vec)
#             try:
#                 ego_headings_vec = start_heading + ade_loss.angle_bins_tensor[vec_angles]
#             except (IndexError, RuntimeError):
#                 continue  # Skip if any vec index is invalid
#             # Compute dx, dy for all pairs (shape: num_vec, num_len)
#             cos_headings = torch.cos(ego_headings_vec)[:, None]  # (num_vec, 1)
#             sin_headings = torch.sin(ego_headings_vec)[:, None]  # (num_vec, 1)
#             dx = len_lengths[None, :] * cos_headings  # (num_vec, num_len)
#             dy = len_lengths[None, :] * sin_headings  # (num_vec, num_len)
#             pos = torch.stack([dx, dy], dim=-1)  # (num_vec, num_len, 2)
#             target = norm_raw_traj_tensor[traj_step_count] - norm_raw_traj_tensor[traj_step_count - 1]
#             # Compute norms for all pairs (broadcasting handles shapes)
#             diff = pos - target  # (num_vec, num_len, 2)
#             norm_diff = torch.norm(diff, dim=-1)  # (num_vec, num_len)
#             # Weighted ADE sum for this step
#             weighted_ade = pair_probs * norm_diff
#             step_ade = weighted_ade.sum()
#             next_token_ade.append(step_ade)
#             traj_step_count += 1
#             if traj_step_count == len(norm_raw_traj_tensor):
#                 break
#         if len(next_token_ade) == 0:
#             ade_values.append(0.0)
#         else:
#             ade_values.append(sum(next_token_ade) / len(next_token_ade))

#     ade_value = torch.stack(ade_values).sum()

#     aux_loss = ade_value
#     return aux_loss, ade_value.item(), 0






# def ade_loss(logits, top_k, sid, ego_id, weight=1.0, tokenizer=None, labels=None):
#     global angle_bins, combined_start_heading
#     if not hasattr(ade_loss, 'vec_token_ids'):
#         vocab = tokenizer.get_vocab()
#         ade_loss.vec_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("VEC_"))
#         ade_loss.len_token_ids = set(tok_id for tok, tok_id in vocab.items() if tok.startswith("LEN_"))
#         ade_loss.vec_to_angle, ade_loss.len_to_length = build_vec_len_lookup(tokenizer, device=logits.device, dtype=logits.dtype)
#         # Precompute angle_bins as torch tensor on device for faster access and vectorization
#         ade_loss.angle_bins_tensor = torch.tensor(angle_bins, device=logits.device, dtype=logits.dtype)
    
#     vals, idx = torch.topk(logits, top_k, dim=-1)
#     probs = torch.softmax(vals, dim=-1)
#     B, T, K = probs.shape
    
#     # Precompute all trajectory tensors for the batch to avoid per-b loading
#     norm_raw_traj_tensors = []
#     for b in range(B):
#         raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
#         norm_raw_traj = raw_traj - raw_traj[0]
#         norm_raw_traj = norm_raw_traj[9:]
#         norm_raw_traj_tensor = torch.as_tensor(norm_raw_traj, device=logits.device, dtype=logits.dtype)
#         norm_raw_traj_tensors.append(norm_raw_traj_tensor)
        
#     ade_values = []
#     for b in range(B):
#         next_token_ade = []
#         norm_raw_traj_tensor = norm_raw_traj_tensors[b]
#         traj_step_count = 1
#         for t in range(T-1):
#             if labels is not None and labels[b, t] == -100:
#                 continue

#             vec_mask = [i for i in range(K) if idx[b, t, i].item() in ade_loss.vec_token_ids]
#             len_mask = [i for i in range(K) if idx[b, t+1, i].item() in ade_loss.len_token_ids]

#             if len(vec_mask) == 0 or len(len_mask) == 0:
#                 continue

#             p_vec = probs[b, t, vec_mask]
#             ids_vec = idx[b, t, vec_mask]
#             p_len = probs[b, t+1, len_mask]
#             ids_len = idx[b, t+1, len_mask]

#             pair_probs = p_vec[:, None] * p_len[None, :]

#             # Vectorized computation: get angles and lengths as tensors
#             vec_angles = ade_loss.vec_to_angle[ids_vec]
#             len_lengths = ade_loss.len_to_length[ids_len]
#             start_heading = combined_start_heading[f"{sid[b]}__{ego_id[b]}"][1]
#             # Compute ego_headings for all vec (shape: num_vec)
#             try:
#                 ego_headings_vec = start_heading + ade_loss.angle_bins_tensor[vec_angles]
#             except (IndexError, RuntimeError):
#                 continue  # Skip if any vec index is invalid
#             # Compute dx, dy for all pairs (shape: num_vec, num_len)
#             cos_headings = torch.cos(ego_headings_vec)[:, None]  # (num_vec, 1)
#             sin_headings = torch.sin(ego_headings_vec)[:, None]  # (num_vec, 1)
#             dx = len_lengths[None, :] * cos_headings  # (num_vec, num_len)
#             dy = len_lengths[None, :] * sin_headings  # (num_vec, num_len)
#             pos = torch.stack([dx, dy], dim=-1)  # (num_vec, num_len, 2)
#             target = norm_raw_traj_tensor[traj_step_count] - norm_raw_traj_tensor[traj_step_count - 1]
#             # Compute norms for all pairs (broadcasting handles shapes)
#             diff = pos - target  # (num_vec, num_len, 2)
#             norm_diff = torch.norm(diff, dim=-1)  # (num_vec, num_len)
#             # Weighted ADE sum for this step
#             weighted_ade = pair_probs * norm_diff
#             step_ade = weighted_ade.sum()
#             next_token_ade.append(step_ade)
#             traj_step_count += 1
#             if len(next_token_ade) >= len(norm_raw_traj_tensor) - 1:
#                 break

#         if next_token_ade:
#             ade_values.append(sum(next_token_ade) / len(next_token_ade))
#         else:
#             ade_values.append(torch.tensor(0.0, device=logits.device, dtype=logits.dtype))

#     if ade_values:
#         ade_value = torch.stack(ade_values).sum()
#     else:
#         ade_value = 0.0
#     aux_loss = ade_value
#     aux_loss = aux_loss * weight
#     return aux_loss, ade_value.item(), 0.0



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
    default_ade_tensors = []
    for b in range(B):
        raw_traj, _ = load_trajectory_by_key_from_memory(sid[b], ego_id[b])
        norm_raw_traj = raw_traj - raw_traj[0]
        norm_raw_traj = norm_raw_traj[9:]
        norm_raw_traj_tensor = torch.as_tensor(norm_raw_traj, device=logits.device, dtype=logits.dtype)
        norm_raw_traj_tensors.append(norm_raw_traj_tensor)
        default_ade_tensor = torch.norm(norm_raw_traj_tensor, dim=-1).mean()
        default_ade_tensors.append(default_ade_tensor)
    
    # Convert default_ade to numpy list at end for return
    default_ade_list = [d.item() for d in default_ade_tensors]
    
    ade_values = []
    for b in range(B):
        next_token_ade = []
        none_traj_token_ade = []
        norm_raw_traj_tensor = norm_raw_traj_tensors[b]
        default_ade = default_ade_tensors[b]
        traj_step_count = 1
        for t in range(T-1):
            if labels is not None and labels[b, t] == -100:
                continue

            vec_mask = [i for i in range(K) if idx[b, t, i].item() in ade_loss.vec_token_ids]
            len_mask = [i for i in range(K) if idx[b, t+1, i].item() in ade_loss.len_token_ids]

            if len(vec_mask) == 0 or len(len_mask) == 0:
                # TODO: fill in how to handle cases where neither vec nor len tokens are found at token t
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
            if len(next_token_ade) < len(norm_raw_traj_tensor) - 1:
                next_token_ade.append(step_ade * len(norm_raw_traj_tensor) / len(next_token_ade))
        if next_token_ade:
            ade_values.append(sum(next_token_ade) / len(next_token_ade))
        else:
            ade_values.append(logits.sum() * 0 + default_ade)  # Maintain tensor type for gradient flow

    if ade_values:
        ade_value = torch.stack(ade_values).sum()
    else:
        ade_value = logits.sum() * 0 + np.mean(default_ade_list)
    aux_loss = log_normalize_with_target(ade_value)
    aux_loss = aux_loss * weight
    default_ade = np.mean(default_ade_list)
    return aux_loss, ade_value.item(), 0