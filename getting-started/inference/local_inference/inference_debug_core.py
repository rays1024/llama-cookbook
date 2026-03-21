#!/usr/bin/env python3
import argparse
import json
import os
import re
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from accelerate.utils import is_xpu_available
from transformers import AutoTokenizer, logging

from llama_cookbook.inference.model_utils import load_model, load_peft_model
from llama_cookbook.utils.aux_loss import multi_label_bce_loss

logging.set_verbosity_error()

all_centroids = np.load(
    '/p/ruishen/processed_waymo_data/training/waymo_vectorized/all_cluster_centroids_10hz_1024.npy',
    allow_pickle=True,
)

ROAD_TYPE_TOKEN = [
    "LaneCenter-Freeway",
    "LaneCenter-SurfaceStreet",
    "RoadEdgeBoundary",
    "RoadEdgeMedian",
    "StopSign",
    "Crosswalk",
    "SpeedBump",
]

VEC_RE = re.compile(r"^VEC_(\d+)$")


def build_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    custom_tokens = [f"VEC_{i}" for i in range(1024)]
    custom_tokens.extend(ROAD_TYPE_TOKEN)
    custom_tokens.extend(
        [
            "AGENT_ID_",
            "AGENT_TYPE_Vehicle",
            "AGENT_TYPE_Pedestrian",
            "AGENT_TYPE_Cyclist",
            "AGENT_TYPE_Other",
            "AGENT_TYPE_Unset",
            "TRAJ_NONE",
            "CTRL_NONE",
            "POS_",
            "POS_NONE",
            "EGO_TRAJ_START",
            "EGO_TRAJ_END",
            "AGENT_TRAJ_START",
            "AGENT_TRAJ_END",
            "MAP_START",
            "MAP_END",
            "ROAD_START",
            "ROAD_END",
        ]
    )
    tokenizer.add_tokens(custom_tokens)
    return tokenizer


def load_row_by_index(data_path, row_index, batch_size=1024):
    pf = pq.ParquetFile(data_path)
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        batch_dict = batch.to_pydict()
        batch_len = len(next(iter(batch_dict.values())))
        if row_index < seen + batch_len:
            i = row_index - seen
            return {k: batch_dict[k][i] for k in batch_dict}
        seen += batch_len
    raise IndexError(f"Row index {row_index} out of range for {data_path}")


def iter_rows_by_range(data_path, start_index, num_rows, batch_size=1024):
    if num_rows <= 0:
        return
    pf = pq.ParquetFile(data_path)
    seen = 0
    yielded = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        batch_dict = batch.to_pydict()
        batch_len = len(next(iter(batch_dict.values())))
        for i in range(batch_len):
            global_index = seen + i
            if global_index < start_index:
                continue
            if yielded >= num_rows:
                return
            row = {k: batch_dict[k][i] for k in batch_dict}
            yield global_index, row
            yielded += 1
        seen += batch_len
        if yielded >= num_rows:
            return
    if yielded < num_rows:
        raise IndexError(
            f"Requested {num_rows} rows starting at {start_index}, but only {yielded} rows available."
        )


def vec_token_to_xy(token):
    token_clean = token.lstrip("\u2581")
    match = VEC_RE.match(token_clean)
    if not match:
        return None
    idx = int(match.group(1))
    if idx < 0 or idx >= len(all_centroids):
        return None
    vec = all_centroids[idx]
    return [float(vec[0]), float(vec[1])]


def to_jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _is_numeric_scalar(value):
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def init_scalar_debug_averager():
    return {
        "sum_by_key": {},
        "count_by_key": {},
    }


def update_scalar_debug_averager(averager, payload):
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if not _is_numeric_scalar(value):
            continue
        value_float = float(value)
        if not np.isfinite(value_float):
            continue
        averager["sum_by_key"][key] = averager["sum_by_key"].get(key, 0.0) + value_float
        averager["count_by_key"][key] = averager["count_by_key"].get(key, 0) + 1


def finalize_scalar_debug_averager(averager):
    avg_payload = {}
    for key, value_sum in averager["sum_by_key"].items():
        count = averager["count_by_key"].get(key, 0)
        if count > 0:
            avg_payload[key] = value_sum / float(count)
    return avg_payload


def init_gt_metrics():
    return {
        "steps_with_gt": 0,
        "gt_in_topk": 0,
        "gt_rank_sum": 0.0,
        "gt_prob_sum": 0.0,
        "vec_steps_with_gt": 0,
        "vec_gt_in_topk": 0,
        "vec_gt_rank_sum": 0.0,
        "vec_gt_prob_sum": 0.0,
        "vec_top5_l2_sum": 0.0,
        "vec_top5_l2_steps": 0,
    }


def finalize_gt_metrics(acc, topk_report):
    def safe_div(num, denom):
        return num / denom if denom else None

    return {
        "topk_report": topk_report,
        "steps_with_gt": acc["steps_with_gt"],
        "gt_in_topk": acc["gt_in_topk"],
        "gt_in_topk_rate": safe_div(acc["gt_in_topk"], acc["steps_with_gt"]),
        "avg_gt_rank_in_topk": safe_div(acc["gt_rank_sum"], acc["gt_in_topk"]),
        "avg_gt_prob_in_topk": safe_div(acc["gt_prob_sum"], acc["gt_in_topk"]),
        "vec_steps_with_gt": acc["vec_steps_with_gt"],
        "vec_gt_in_topk": acc["vec_gt_in_topk"],
        "vec_gt_in_topk_rate": safe_div(acc["vec_gt_in_topk"], acc["vec_steps_with_gt"]),
        "avg_vec_gt_rank_in_topk": safe_div(acc["vec_gt_rank_sum"], acc["vec_gt_in_topk"]),
        "avg_vec_gt_prob_in_topk": safe_div(acc["vec_gt_prob_sum"], acc["vec_gt_in_topk"]),
        "vec_top5_l2_steps": acc["vec_top5_l2_steps"],
        "avg_vec_top5_l2": safe_div(acc["vec_top5_l2_sum"], acc["vec_top5_l2_steps"]),
    }

def prepare_context(row, tokenizer, context_mode="labels"):
    input_ids = list(row["input_ids"])
    attention_mask = list(row["attention_mask"])
    labels = list(row.get("labels", []))

    # Trim padding to keep alignment stable.
    if tokenizer.pad_token_id is not None:
        input_ids = [x for x in input_ids if x != tokenizer.pad_token_id]
        attention_mask = [x for x in attention_mask if x != 0]
    if labels:
        labels = labels[: len(input_ids)]

    if context_mode == "full" or not labels:
        context_ids = input_ids
        context_mask = attention_mask
        target_ids = [x for x in labels if x != -100]
        return context_ids, context_mask, target_ids, input_ids

    context_ids = [input_ids[i] for i, label in enumerate(labels) if label == -100]
    context_mask = [attention_mask[i] for i, label in enumerate(labels) if label == -100]
    target_ids = [label for label in labels if label != -100]
    return context_ids, context_mask, target_ids, input_ids


def prepare_inputs_and_labels(row, tokenizer):
    input_ids = list(row["input_ids"])
    attention_mask = list(row["attention_mask"])
    labels = list(row.get("labels", []))

    # Trim padding to keep alignment stable.
    if tokenizer.pad_token_id is not None:
        input_ids = [x for x in input_ids if x != tokenizer.pad_token_id]
        attention_mask = [x for x in attention_mask if x != 0]
    if labels:
        labels = labels[: len(input_ids)]
    return input_ids, attention_mask, labels


def compute_gt_info(
    step_logits,
    gt_id,
    tokenizer,
    topk_ids,
    vec_token_ids,
    vec_id_to_index,
    row_metrics,
    overall_metrics,
):
    gt_token = tokenizer.convert_ids_to_tokens([gt_id])[0]
    log_denom = torch.logsumexp(step_logits, dim=-1)
    gt_prob = torch.exp(step_logits[0, gt_id] - log_denom[0]).item()
    gt_in_topk = gt_id in topk_ids
    gt_rank = topk_ids.index(gt_id) + 1 if gt_in_topk else None
    gt_vec_xy = vec_token_to_xy(gt_token)
    gt_vec_top5_l2 = None

    row_metrics["steps_with_gt"] += 1
    overall_metrics["steps_with_gt"] += 1
    if gt_in_topk:
        row_metrics["gt_in_topk"] += 1
        row_metrics["gt_rank_sum"] += gt_rank
        row_metrics["gt_prob_sum"] += gt_prob
        overall_metrics["gt_in_topk"] += 1
        overall_metrics["gt_rank_sum"] += gt_rank
        overall_metrics["gt_prob_sum"] += gt_prob

    if gt_vec_xy is not None:
        row_metrics["vec_steps_with_gt"] += 1
        overall_metrics["vec_steps_with_gt"] += 1
        if vec_token_ids.numel() > 0:
            vec_logits = step_logits.index_select(dim=-1, index=vec_token_ids)
            k = min(5, vec_logits.size(-1))
            _, vec_top_idx = torch.topk(vec_logits, k, dim=-1)
            vec_top_ids = vec_token_ids[vec_top_idx.squeeze(0)].detach().cpu().tolist()
            vec_indices = [vec_id_to_index.get(tok_id) for tok_id in vec_top_ids]
            vec_indices = [i for i in vec_indices if i is not None]
            if vec_indices:
                top_vecs = np.asarray([all_centroids[i][:2] for i in vec_indices], dtype=np.float32)
                gt_vec = np.asarray(gt_vec_xy, dtype=np.float32)
                gt_vec_top5_l2 = float(np.linalg.norm(top_vecs - gt_vec, axis=1).mean())
                row_metrics["vec_top5_l2_sum"] += gt_vec_top5_l2
                row_metrics["vec_top5_l2_steps"] += 1
                overall_metrics["vec_top5_l2_sum"] += gt_vec_top5_l2
                overall_metrics["vec_top5_l2_steps"] += 1
        if gt_in_topk:
            row_metrics["vec_gt_in_topk"] += 1
            row_metrics["vec_gt_rank_sum"] += gt_rank
            row_metrics["vec_gt_prob_sum"] += gt_prob
            overall_metrics["vec_gt_in_topk"] += 1
            overall_metrics["vec_gt_rank_sum"] += gt_rank
            overall_metrics["vec_gt_prob_sum"] += gt_prob

    return {
        "token": gt_token,
        "token_id": gt_id,
        "in_topk": gt_in_topk,
        "rank_in_topk": gt_rank,
        "prob": gt_prob,
        "vec_xy": gt_vec_xy,
        "vec_top5_l2": gt_vec_top5_l2,
    }


def rank_for_token(step_logits, token_id):
    vocab_size = step_logits.size(-1)
    if token_id is None or token_id < 0 or token_id >= vocab_size:
        return None
    token_logit = step_logits[0, token_id]
    return int((step_logits > token_logit).sum().item()) + 1


def apply_top_k_top_p(logits, top_k=0, top_p=1.0):
    filtered = logits.clone()
    vocab_size = filtered.size(-1)

    if top_k is not None and top_k > 0 and top_k < vocab_size:
        kth_vals = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = torch.where(filtered < kth_vals, torch.tensor(-float("inf"), device=filtered.device), filtered)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        # Keep at least one token.
        sorted_mask[..., 0] = False
        filtered_sorted = sorted_logits.masked_fill(sorted_mask, -float("inf"))
        filtered = torch.full_like(filtered, -float("inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)

    return filtered


def topk_from_logits(logits, k, tokenizer):
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k, dim=-1)
    top_probs = top_probs.squeeze(0).tolist()
    top_ids = top_ids.squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(top_ids)
    return list(zip(tokens, top_ids, top_probs))


def topk_vec_tokens(logits, vec_token_ids, k, tokenizer):
    vec_logits = logits.index_select(dim=-1, index=vec_token_ids)
    vec_probs = torch.softmax(vec_logits, dim=-1)
    top_probs, top_idx = torch.topk(vec_probs, k, dim=-1)
    top_probs = top_probs.squeeze(0).tolist()
    top_idx = top_idx.squeeze(0).tolist()
    top_ids = vec_token_ids[top_idx].tolist()
    tokens = tokenizer.convert_ids_to_tokens(top_ids)
    return list(zip(tokens, top_ids, top_probs))


def topk_logits_from_logits(logits, k, tokenizer):
    k = min(k, logits.size(-1))
    top_logits, top_ids = torch.topk(logits, k, dim=-1)
    top_logits = top_logits.squeeze(0).tolist()
    top_ids = top_ids.squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(top_ids)
    return list(zip(tokens, top_ids, top_logits))


def format_topk(name, topk_items, limit=10):
    entries = []
    for tok, tok_id, prob in topk_items[:limit]:
        entries.append(f"{tok}:{tok_id}={prob:.4f}")
    return f"{name}: " + ", ".join(entries)


def format_topk_logits(name, topk_items, limit=10):
    entries = []
    for tok, tok_id, logit in topk_items[:limit]:
        entries.append(f"{tok}:{tok_id}={logit:.4f}")
    return f"{name}: " + ", ".join(entries)


def _to_device_tensor(value, device, dtype=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    if isinstance(value, np.ndarray):
        tensor = torch.as_tensor(value, device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return None


def _align_multi_label_for_shifted_labels(logits, labels, multi_label, label_weight, ignore_index=-100):
    B, T, V = logits.size()
    shifted_labels = labels[..., 1:].contiguous()
    labels_flat = shifted_labels.reshape(-1)
    valid_mask = labels_flat != ignore_index
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return shifted_labels, labels_flat, valid_mask, valid_count, None, None

    device = logits.device
    dtype = logits.dtype
    multi_label = _to_device_tensor(multi_label, device=device)
    label_weight = _to_device_tensor(label_weight, device=device, dtype=dtype)

    if torch.is_tensor(multi_label) and multi_label.dim() == 3:
        if multi_label.size(0) != B:
            raise ValueError("multi_label batch size does not match logits.")
        if multi_label.size(1) == labels.size(1):
            multi_label = multi_label[:, 1:, :]
            if label_weight is not None and label_weight.dim() == 3 and label_weight.size(1) == labels.size(1):
                label_weight = label_weight[:, 1:, :]

        if multi_label.size(1) == labels.size(1) - 1:
            multi_label_flat = multi_label.reshape(-1, multi_label.size(-1))[valid_mask]
            if label_weight is None:
                label_weight_flat = torch.ones_like(multi_label_flat, dtype=dtype)
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
            if multi_label_chunks:
                multi_label_flat = torch.cat(multi_label_chunks, dim=0)
            else:
                multi_label_flat = multi_label.new_empty((0, multi_label.size(-1)))
            if label_weight is None:
                label_weight_flat = torch.ones_like(multi_label_flat, dtype=dtype)
            else:
                if weight_chunks:
                    label_weight_flat = torch.cat(weight_chunks, dim=0)
                else:
                    label_weight_flat = label_weight.new_empty((0, label_weight.size(-1)))

    elif torch.is_tensor(multi_label) and multi_label.dim() == 2:
        if multi_label.size(0) != valid_count:
            raise ValueError("multi_label length does not match the number of valid labels.")
        multi_label_flat = multi_label
        if label_weight is None:
            label_weight_flat = torch.ones_like(multi_label_flat, dtype=dtype)
        else:
            label_weight_flat = label_weight

    else:
        if not isinstance(multi_label, (list, tuple)) or len(multi_label) != B:
            raise ValueError("multi_label must be a tensor or a list with batch length.")
        if label_weight is not None and (
            not isinstance(label_weight, (list, tuple)) or len(label_weight) != B
        ):
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
        multi_label_flat = torch.as_tensor(multi_label_flat_list, device=device, dtype=torch.long)
        if label_weight is None:
            label_weight_flat = torch.ones_like(multi_label_flat, dtype=dtype)
        else:
            label_weight_flat = torch.as_tensor(label_weight_flat_list, device=device, dtype=dtype)

    if multi_label_flat.size(0) != valid_count:
        raise ValueError("multi_label does not align with valid labels after shifting.")

    multi_label_flat = multi_label_flat.to(device=device, dtype=torch.long)
    label_weight_flat = label_weight_flat.to(device=device, dtype=dtype)
    return shifted_labels, labels_flat, valid_mask, valid_count, multi_label_flat, label_weight_flat


def _safe_grad(loss_scalar, wrt, retain_graph):
    if not torch.is_tensor(loss_scalar) or not loss_scalar.requires_grad:
        return torch.zeros_like(wrt)
    grad = torch.autograd.grad(
        loss_scalar,
        wrt,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(wrt)
    return grad


def _vec_l2_to_gt(token_id, gt_token_id, vec_id_to_index):
    pred_idx = vec_id_to_index.get(int(token_id))
    gt_idx = vec_id_to_index.get(int(gt_token_id))
    if pred_idx is None or gt_idx is None:
        return None
    pred_xy = np.asarray(all_centroids[pred_idx][:2], dtype=np.float32)
    gt_xy = np.asarray(all_centroids[gt_idx][:2], dtype=np.float32)
    return float(np.linalg.norm(pred_xy - gt_xy))


def _get_centroids_xy():
    cached = getattr(_get_centroids_xy, "_cache", None)
    if cached is not None:
        return cached
    try:
        centroids_xy = np.asarray(all_centroids, dtype=np.float32)
        if centroids_xy.ndim != 2 or centroids_xy.shape[1] < 2:
            centroids_xy = None
        else:
            centroids_xy = centroids_xy[:, :2]
    except Exception:
        centroids_xy = None
    _get_centroids_xy._cache = centroids_xy
    return centroids_xy


def build_distance_rank_plot_path(output_json_path, row_index):
    base, _ = os.path.splitext(output_json_path)
    return f"{base}_row{row_index}_distance_rank_trend.png"


def build_logit_distance_plot_path(output_json_path, row_index):
    base, _ = os.path.splitext(output_json_path)
    return f"{base}_row{row_index}_logit_distance_trend.png"


def build_distance_group_distance_plot_path(output_json_path, row_index):
    base, _ = os.path.splitext(output_json_path)
    return f"{base}_row{row_index}_distance_group_distance_trend.png"


def build_negative_bce_distribution_plot_path(output_json_path, row_index):
    base, _ = os.path.splitext(output_json_path)
    return f"{base}_row{row_index}_negative_bce_distribution.png"


def _save_four_line_plot(step_index, series, plot_path, ylabel, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib_unavailable: {exc}"

    if not step_index:
        return "no_step_index_for_plot"

    plt.figure(figsize=(10, 6))
    plotted = 0
    if isinstance(series, dict):
        series_items = series.items()
    else:
        return "invalid_series_format"

    for (key, label), vals in series_items:
        xs = []
        ys = []
        for x, y in zip(step_index, vals):
            if y is None:
                continue
            xs.append(x)
            ys.append(y)
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=label)
            plotted += 1

    if plotted == 0:
        plt.close()
        return "no_valid_points_to_plot"

    plt.xlabel("Step index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return None


def _save_four_line_plot_distance_x(step_index, series, plot_path, xlabel, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib_unavailable: {exc}"

    if not step_index:
        return "no_step_index_for_plot"

    if isinstance(series, dict):
        series_items = series.items()
    else:
        return "invalid_series_format"

    plt.figure(figsize=(10, 6))
    plotted = 0
    for (key, label), vals in series_items:
        xs = []
        ys = []
        for step, distance in zip(step_index, vals):
            if distance is None:
                continue
            xs.append(distance)
            ys.append(step)
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=label)
            plotted += 1

    if plotted == 0:
        plt.close()
        return "no_valid_points_to_plot"

    plt.xlabel(xlabel)
    plt.ylabel("Step index")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return None


def save_distance_rank_trend_plot(distance_rank_trend, plot_path):
    step_index = distance_rank_trend.get("step_index", [])
    series = {
        ("avg_logit_rank_top1_by_step", "Top-1 by distance"): distance_rank_trend.get("avg_logit_rank_top1_by_step", []),
        ("avg_logit_rank_top5_by_step", "Top-5 by distance"): distance_rank_trend.get("avg_logit_rank_top5_by_step", []),
        ("avg_logit_rank_top10_by_step", "Top-10 by distance"): distance_rank_trend.get("avg_logit_rank_top10_by_step", []),
        ("avg_logit_rank_top20_by_step", "Top-20 by distance"): distance_rank_trend.get("avg_logit_rank_top20_by_step", []),
    }
    return _save_four_line_plot(
        step_index=step_index,
        series=series,
        plot_path=plot_path,
        ylabel="Average logit rank (lower is better)",
        title="Distance-group token rank trend",
    )


def save_logit_distance_trend_plot(logit_distance_trend, plot_path):
    step_index = logit_distance_trend.get("step_index", [])
    series = {
        ("avg_distance_top1_by_step", "Top-1 by logit"): logit_distance_trend.get("avg_distance_top1_by_step", []),
        ("avg_distance_top5_by_step", "Top-5 by logit"): logit_distance_trend.get("avg_distance_top5_by_step", []),
        ("avg_distance_top10_by_step", "Top-10 by logit"): logit_distance_trend.get("avg_distance_top10_by_step", []),
        ("avg_distance_top20_by_step", "Top-20 by logit"): logit_distance_trend.get("avg_distance_top20_by_step", []),
    }
    return _save_four_line_plot(
        step_index=step_index,
        series=series,
        plot_path=plot_path,
        ylabel="Average L2 distance to GT VEC",
        title="Logit-group token distance trend",
    )


def save_distance_group_distance_trend_plot(distance_group_distance_trend, plot_path):
    step_index = distance_group_distance_trend.get("step_index", [])
    series = {
        ("avg_distance_top5_by_step", "Top-5 by distance"): distance_group_distance_trend.get("avg_distance_top5_by_step", []),
        ("avg_distance_top10_by_step", "Top-10 by distance"): distance_group_distance_trend.get("avg_distance_top10_by_step", []),
        ("avg_distance_top20_by_step", "Top-20 by distance"): distance_group_distance_trend.get("avg_distance_top20_by_step", []),
        ("avg_distance_top50_by_step", "Top-50 by distance"): distance_group_distance_trend.get("avg_distance_top50_by_step", []),
    }
    return _save_four_line_plot(
        step_index=step_index,
        series=series,
        plot_path=plot_path,
        ylabel="Average L2 distance to GT VEC",
        title="Distance-group token distance trend",
    )


def save_negative_bce_distribution_plot(multi_label_bce_debug, plot_path, bins=30):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib_unavailable: {exc}"

    steps = multi_label_bce_debug.get("steps", [])
    values = []
    for step in steps:
        neg_terms = step.get("sampled_negative_terms", step.get("sampled_negatives_top_loss", []))
        if not isinstance(neg_terms, list):
            continue
        for term in neg_terms:
            if not isinstance(term, dict):
                continue
            val = term.get("raw_bce_term", None)
            if val is None:
                continue
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue

    if len(values) == 0:
        return "no_negative_bce_terms"

    if bins is None or bins <= 0:
        bins = 30
    bins = max(5, min(int(bins), 200))

    counts, edges = np.histogram(np.asarray(values, dtype=np.float32), bins=bins)
    widths = np.diff(edges)

    plt.figure(figsize=(10, 6))
    plt.bar(edges[:-1], counts, width=widths, align="edge", alpha=0.85, edgecolor="black", linewidth=0.5)
    plt.xlabel("Negative token raw BCE term")
    plt.ylabel("Count")
    plt.title("Negative BCE loss distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return None


def debug_teacher_forcing_multi_label_bce(
    logits,
    labels,
    multi_label,
    label_weight,
    tokenizer,
    vec_token_ids,
    vec_id_to_index,
    topk_report=10,
    focus_step=None,
    max_steps=None,
    neg_token_limit=20,
    ignore_index=-100,
    pos_weight=100.0,
):
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("Expected logits [B, T, V] and labels [B, T].")
    if logits.size(0) != 1:
        raise ValueError("BCE debug currently expects batch size 1.")

    multi_label_input = multi_label
    label_weight_input = label_weight
    if isinstance(multi_label_input, np.ndarray):
        multi_label_input = torch.as_tensor(multi_label_input, device=logits.device, dtype=torch.long)
    if isinstance(label_weight_input, np.ndarray):
        label_weight_input = torch.as_tensor(label_weight_input, device=logits.device, dtype=logits.dtype)
    if isinstance(multi_label_input, (list, tuple)):
        # Row-level parquet samples are often [steps, k] without a batch dimension.
        try:
            multi_label_tensor = torch.as_tensor(multi_label_input, device=logits.device, dtype=torch.long)
            if multi_label_tensor.dim() == 2:
                multi_label_input = multi_label_tensor
                if label_weight_input is not None and not torch.is_tensor(label_weight_input):
                    try:
                        label_weight_tensor = torch.as_tensor(
                            label_weight_input,
                            device=logits.device,
                            dtype=logits.dtype,
                        )
                        if label_weight_tensor.dim() == 2:
                            label_weight_input = label_weight_tensor
                    except Exception:
                        pass
        except Exception:
            if len(multi_label_input) != 1:
                multi_label_input = [multi_label_input]
            if (
                label_weight_input is not None
                and isinstance(label_weight_input, (list, tuple))
                and len(label_weight_input) != 1
            ):
                label_weight_input = [label_weight_input]

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if logits.device.type == "cuda":
        cuda_rng_state = torch.cuda.get_rng_state(logits.device)

    reference_loss = multi_label_bce_loss(
        logits,
        labels,
        multi_label_input,
        label_weight=label_weight_input,
        tokenizer=tokenizer,
        ignore_index=ignore_index,
        reduction="mean",
        pos_weight=pos_weight,
    )

    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, logits.device)

    (
        shifted_labels,
        labels_flat,
        valid_mask,
        valid_count,
        multi_label_flat,
        label_weight_flat,
    ) = _align_multi_label_for_shifted_labels(
        logits,
        labels,
        multi_label_input,
        label_weight_input,
        ignore_index=ignore_index,
    )

    if valid_count == 0:
        return {
            "valid_count": 0,
            "reference_full_mean_loss": float(reference_loss.item()),
            "debug_note": "No valid labels after shifting.",
            "message": "No valid labels after shifting.",
            "steps": [],
        }

    B, T, V = logits.size()
    shifted_logits = logits[..., :-1, :].contiguous()
    logits_flat = shifted_logits.reshape(-1, V)
    valid_logits = logits_flat[valid_mask]
    valid_labels = labels_flat[valid_mask]

    vec_vocab_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
    vec_vocab_mask[vec_token_ids] = True
    non_vec_mask = ~vec_vocab_mask
    vec_target_mask = vec_vocab_mask[valid_labels]

    valid_logits_leaf = valid_logits.detach().clone().requires_grad_(True)

    if non_vec_mask.any():
        vec_group_logits = torch.logsumexp(valid_logits_leaf[:, vec_vocab_mask], dim=1)
        non_vec_group_logits = torch.logsumexp(valid_logits_leaf[:, non_vec_mask], dim=1)
        type_logits_dbg = vec_group_logits - non_vec_group_logits
        type_targets_dbg = vec_target_mask.to(dtype=logits.dtype)
        type_loss_dbg = F.binary_cross_entropy_with_logits(
            type_logits_dbg,
            type_targets_dbg,
            reduction="none",
        )
    else:
        type_logits_dbg = torch.zeros(valid_count, device=logits.device, dtype=logits.dtype)
        type_targets_dbg = vec_target_mask.to(dtype=logits.dtype)
        type_loss_dbg = torch.zeros(valid_count, device=logits.device, dtype=logits.dtype)

    pos_mask = multi_label_flat != ignore_index
    safe_indices = multi_label_flat.clone()
    safe_indices[~pos_mask] = 0
    pos_is_vec = pos_mask & vec_vocab_mask[safe_indices]

    pos_logits_dbg = valid_logits_leaf.gather(dim=1, index=safe_indices)
    pos_logsig_dbg = F.logsigmoid(pos_logits_dbg)
    pos_weights_dbg = label_weight_flat * pos_is_vec.to(label_weight_flat.dtype)
    pos_denom_dbg = pos_is_vec.sum(dim=1).clamp_min(1).to(pos_logsig_dbg.dtype)
    pos_contrib_slot = -(pos_weights_dbg * pos_logsig_dbg) / pos_denom_dbg[:, None]
    pos_loss_dbg = pos_contrib_slot.sum(dim=1) * float(pos_weight)

    vec_logits_dbg = valid_logits_leaf[:, vec_token_ids]
    neg_logsig_dbg = -F.logsigmoid(-vec_logits_dbg)

    vec_index = torch.full((V,), -1, dtype=torch.long, device=logits.device)
    vec_index[vec_token_ids] = torch.arange(vec_token_ids.numel(), device=logits.device)
    pos_vec_indices = vec_index[safe_indices]
    pos_vec_indices = pos_vec_indices.masked_fill(~pos_is_vec, -1)
    valid_pos = pos_vec_indices >= 0
    if valid_pos.any():
        pos_vec_counts = torch.zeros(
            (valid_count, vec_token_ids.numel()),
            dtype=torch.int32,
            device=logits.device,
        )
        pos_vec_counts.scatter_add_(1, pos_vec_indices.clamp_min(0), valid_pos.int())
        pos_vec_mask = pos_vec_counts > 0
    else:
        pos_vec_mask = torch.zeros(
            (valid_count, vec_token_ids.numel()),
            dtype=torch.bool,
            device=logits.device,
        )

    neg_candidate_mask_dbg = ~pos_vec_mask
    if neg_candidate_mask_dbg.any():
        neg_keep = torch.rand_like(neg_logsig_dbg) < 0.5
        neg_mask_dbg = neg_candidate_mask_dbg & neg_keep
    else:
        neg_mask_dbg = neg_candidate_mask_dbg
    neg_denom_dbg = neg_mask_dbg.sum(dim=1).clamp_min(1).to(neg_logsig_dbg.dtype)
    neg_contrib_slot = (neg_logsig_dbg * neg_mask_dbg.to(neg_logsig_dbg.dtype)) / neg_denom_dbg[:, None]
    neg_loss_dbg = neg_contrib_slot.sum(dim=1)

    pos_loss_dbg = pos_loss_dbg * vec_target_mask.to(pos_loss_dbg.dtype)
    neg_loss_dbg = neg_loss_dbg * vec_target_mask.to(neg_loss_dbg.dtype)
    total_loss_dbg = type_loss_dbg + pos_loss_dbg + neg_loss_dbg

    denom = float(valid_count)
    type_mean = type_loss_dbg.sum() / denom
    pos_mean = pos_loss_dbg.sum() / denom
    neg_mean = neg_loss_dbg.sum() / denom
    full_debug_mean = total_loss_dbg.sum() / denom
    pos_neg_mean = (pos_loss_dbg.sum() + neg_loss_dbg.sum()) / denom

    # Focus gradients on multi-label BCE terms (positive + negative), excluding type loss.
    grad_total = _safe_grad(pos_neg_mean, valid_logits_leaf, retain_graph=True)
    grad_pos = _safe_grad(pos_mean, valid_logits_leaf, retain_graph=True)
    grad_neg = _safe_grad(neg_mean, valid_logits_leaf, retain_graph=False)

    valid_positions = [
        i + 1 for i, lbl in enumerate(shifted_labels[0].tolist()) if int(lbl) != ignore_index
    ]
    if len(valid_positions) != valid_count:
        raise ValueError("Unexpected valid position count mismatch.")

    step_indices = list(range(valid_count))
    if focus_step is not None:
        if focus_step < 0 or focus_step >= valid_count:
            raise ValueError(f"--debug_focus_step={focus_step} is outside [0, {valid_count - 1}]")
        step_indices = [focus_step]
    elif max_steps is not None and max_steps > 0:
        step_indices = step_indices[:max_steps]

    centroids_xy = _get_centroids_xy()
    vec_index_to_id = {int(vec_idx): int(tok_id) for tok_id, vec_idx in vec_id_to_index.items()}
    distance_group_rank_sizes = [1, 5, 10, 20]
    distance_group_distance_sizes = [5, 10, 20, 50]
    trend_top_by_step = {k: [] for k in distance_group_rank_sizes}
    trend_dist_by_logit_top_by_step = {k: [] for k in distance_group_rank_sizes}
    trend_dist_by_distance_top_by_step = {k: [] for k in distance_group_distance_sizes}
    trend_step_index = []
    trend_label_position = []
    trend_gt_is_vec = []
    trend_vec_steps = 0
    trend_vec_steps_logit = 0

    steps_payload = []
    topk_report = max(1, int(topk_report))
    neg_token_limit = int(neg_token_limit)
    for step_idx in step_indices:
        step_logits = valid_logits[step_idx]
        gt_id = int(valid_labels[step_idx].item())
        gt_token = tokenizer.convert_ids_to_tokens([gt_id])[0]
        gt_is_vec = bool(vec_vocab_mask[gt_id].item())
        gt_rank = rank_for_token(step_logits.unsqueeze(0), gt_id)
        gt_vec_xy = vec_token_to_xy(gt_token)
        distance_group_avg_rank = {f"top{k}": None for k in distance_group_rank_sizes}
        logit_group_avg_distance = {f"top{k}": None for k in distance_group_rank_sizes}
        distance_group_avg_distance = {f"top{k}": None for k in distance_group_distance_sizes}

        if gt_is_vec and centroids_xy is not None:
            gt_vec_idx = vec_id_to_index.get(gt_id)
            if gt_vec_idx is not None and 0 <= gt_vec_idx < centroids_xy.shape[0]:
                trend_vec_steps += 1
                dist_to_gt = np.linalg.norm(centroids_xy - centroids_xy[gt_vec_idx], axis=1)
                sorted_vec_idx = np.argsort(dist_to_gt)
                logits_2d = step_logits.unsqueeze(0)
                for k in distance_group_rank_sizes:
                    selected = sorted_vec_idx[: min(k, sorted_vec_idx.shape[0])]
                    rank_vals = []
                    for vec_idx in selected:
                        tok_id = vec_index_to_id.get(int(vec_idx))
                        if tok_id is None:
                            continue
                        rank = rank_for_token(logits_2d, tok_id)
                        if rank is not None:
                            rank_vals.append(float(rank))
                    if rank_vals:
                        distance_group_avg_rank[f"top{k}"] = float(sum(rank_vals) / len(rank_vals))
                for k in distance_group_distance_sizes:
                    selected = sorted_vec_idx[: min(k, sorted_vec_idx.shape[0])]
                    if selected.size > 0:
                        distance_group_avg_distance[f"top{k}"] = float(np.mean(dist_to_gt[selected]))

                if vec_token_ids.numel() > 0:
                    trend_vec_steps_logit += 1
                    vec_step_logits = step_logits.index_select(dim=-1, index=vec_token_ids)
                    sorted_vec_local_idx = torch.argsort(vec_step_logits, descending=True)
                    gt_vec_xy_arr = centroids_xy[gt_vec_idx]
                    for k in distance_group_rank_sizes:
                        top_local = sorted_vec_local_idx[: min(k, sorted_vec_local_idx.numel())]
                        dist_vals = []
                        for local_idx in top_local.tolist():
                            tok_id = int(vec_token_ids[local_idx].item())
                            vec_idx = vec_id_to_index.get(tok_id)
                            if vec_idx is None or vec_idx < 0 or vec_idx >= centroids_xy.shape[0]:
                                continue
                            dist_vals.append(float(np.linalg.norm(centroids_xy[vec_idx] - gt_vec_xy_arr)))
                        if dist_vals:
                            logit_group_avg_distance[f"top{k}"] = float(sum(dist_vals) / len(dist_vals))

        trend_step_index.append(step_idx)
        trend_label_position.append(valid_positions[step_idx])
        trend_gt_is_vec.append(gt_is_vec)
        for k in distance_group_rank_sizes:
            trend_top_by_step[k].append(distance_group_avg_rank[f"top{k}"])
            trend_dist_by_logit_top_by_step[k].append(logit_group_avg_distance[f"top{k}"])
        for k in distance_group_distance_sizes:
            trend_dist_by_distance_top_by_step[k].append(distance_group_avg_distance[f"top{k}"])

        k = min(topk_report, step_logits.numel())
        top_vals, top_ids = torch.topk(step_logits, k=k, dim=-1)
        top_vals = top_vals.detach().cpu().tolist()
        top_ids = top_ids.detach().cpu().tolist()
        top_tokens = tokenizer.convert_ids_to_tokens(top_ids)

        topk_payload = []
        for rank_idx, (tok_id, tok, logit_val) in enumerate(zip(top_ids, top_tokens, top_vals), start=1):
            topk_payload.append(
                {
                    "rank": rank_idx,
                    "token": tok,
                    "token_id": int(tok_id),
                    "logit": float(logit_val),
                    "is_vec": bool(vec_id_to_index.get(int(tok_id), None) is not None),
                    "vec_l2_to_gt": _vec_l2_to_gt(tok_id, gt_id, vec_id_to_index) if gt_is_vec else None,
                }
            )

        pos_slots = torch.nonzero(pos_is_vec[step_idx], as_tuple=False).flatten().tolist()
        step_gate = float(vec_target_mask[step_idx].item())
        positive_payload = []
        for slot in pos_slots:
            tok_id = int(safe_indices[step_idx, slot].item())
            tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
            token_logit = float(pos_logits_dbg[step_idx, slot].detach().item())
            token_logsig = float(pos_logsig_dbg[step_idx, slot].detach().item())
            token_weight = float(label_weight_flat[step_idx, slot].item())
            token_raw_bce = float((-pos_logsig_dbg[step_idx, slot]).detach().item())
            token_weighted = token_raw_bce * token_weight
            token_after_denom = token_weighted / float(pos_denom_dbg[step_idx].detach().item())
            token_after_pos_weight = token_after_denom * float(pos_weight)
            token_final = token_after_pos_weight * step_gate
            positive_payload.append(
                {
                    "slot": int(slot),
                    "token": tok,
                    "token_id": tok_id,
                    "rank": rank_for_token(step_logits.unsqueeze(0), tok_id),
                    "weight": token_weight,
                    "logit": token_logit,
                    "sigmoid_logit": float(torch.sigmoid(torch.tensor(token_logit)).item()),
                    "logsigmoid_logit": token_logsig,
                    "raw_bce_term": token_raw_bce,
                    "weighted_raw_bce_term": token_weighted,
                    "after_pos_denom": token_after_denom,
                    "after_pos_weight": token_after_pos_weight,
                    "final_loss_contribution": token_final,
                    "included_in_final_loss": bool(step_gate > 0.0),
                    "grad_total": float(grad_total[step_idx, tok_id].item()),
                    "grad_pos": float(grad_pos[step_idx, tok_id].item()),
                    "grad_neg": float(grad_neg[step_idx, tok_id].item()),
                    "vec_l2_to_gt": _vec_l2_to_gt(tok_id, gt_id, vec_id_to_index) if gt_is_vec else None,
                }
            )

        step_neg_idx = torch.nonzero(neg_mask_dbg[step_idx], as_tuple=False).flatten()
        step_neg_ids = vec_token_ids[step_neg_idx]
        step_neg_contrib = neg_contrib_slot[step_idx, step_neg_idx]
        step_neg_final_contrib = step_neg_contrib * step_gate
        keep_k = int(step_neg_ids.numel()) if neg_token_limit <= 0 else min(neg_token_limit, int(step_neg_ids.numel()))
        negative_payload = []
        if keep_k > 0:
            top_neg_rel = torch.topk(step_neg_final_contrib, k=keep_k, dim=-1).indices
            for rel_i in top_neg_rel.tolist():
                tok_id = int(step_neg_ids[rel_i].item())
                tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
                vec_local_idx = int(step_neg_idx[rel_i].item())
                token_logit = float(step_logits[tok_id].item())
                token_logsig_neg = float(F.logsigmoid(torch.tensor(-token_logit)).item())
                token_raw_bce = float((-token_logsig_neg))
                token_after_denom = token_raw_bce / float(neg_denom_dbg[step_idx].detach().item())
                token_final = token_after_denom * step_gate
                negative_payload.append(
                    {
                        "token": tok,
                        "token_id": tok_id,
                        "rank": rank_for_token(step_logits.unsqueeze(0), tok_id),
                        "logit": token_logit,
                        "sigmoid_logit": float(torch.sigmoid(torch.tensor(token_logit)).item()),
                        "logsigmoid_neg_logit": token_logsig_neg,
                        "raw_bce_term": token_raw_bce,
                        "after_neg_denom": token_after_denom,
                        "final_loss_contribution": token_final,
                        "included_in_final_loss": bool(step_gate > 0.0),
                        "grad_total": float(grad_total[step_idx, tok_id].item()),
                        "grad_neg": float(grad_neg[step_idx, tok_id].item()),
                        "grad_pos": float(grad_pos[step_idx, tok_id].item()),
                        "vec_l2_to_gt": _vec_l2_to_gt(tok_id, gt_id, vec_id_to_index) if gt_is_vec else None,
                        "vec_local_index": vec_local_idx,
                    }
                )

        grad_abs = grad_total[step_idx].abs()
        grad_topk = min(topk_report, grad_abs.numel())
        grad_vals, grad_ids = torch.topk(grad_abs, k=grad_topk, dim=-1)
        grad_tokens = tokenizer.convert_ids_to_tokens(grad_ids.tolist())
        grad_payload = []
        for tok_id, tok, abs_grad in zip(grad_ids.tolist(), grad_tokens, grad_vals.tolist()):
            grad_payload.append(
                {
                    "token": tok,
                    "token_id": int(tok_id),
                    "abs_grad_total": float(abs_grad),
                    "grad_total": float(grad_total[step_idx, tok_id].item()),
                    "grad_pos": float(grad_pos[step_idx, tok_id].item()),
                    "grad_neg": float(grad_neg[step_idx, tok_id].item()),
                    "logit": float(step_logits[tok_id].item()),
                    "vec_l2_to_gt": _vec_l2_to_gt(tok_id, gt_id, vec_id_to_index) if gt_is_vec else None,
                }
            )

        steps_payload.append(
            {
                "step_index": step_idx,
                "label_position": valid_positions[step_idx],
                "logit_position": valid_positions[step_idx] - 1,
                "gt_token": gt_token,
                "gt_token_id": gt_id,
                "gt_rank_by_logit": gt_rank,
                "gt_is_vec": gt_is_vec,
                "gt_vec_xy": gt_vec_xy,
                "is_vec_target_step": bool(vec_target_mask[step_idx].item()),
                "pos_loss": float(pos_loss_dbg[step_idx].detach().item()),
                "neg_loss": float(neg_loss_dbg[step_idx].detach().item()),
                "pos_neg_step_loss": float((pos_loss_dbg[step_idx] + neg_loss_dbg[step_idx]).detach().item()),
                "pos_denom": float(pos_denom_dbg[step_idx].detach().item()),
                "neg_denom": float(neg_denom_dbg[step_idx].detach().item()),
                "neg_candidates": int(neg_candidate_mask_dbg[step_idx].sum().item()),
                "neg_sampled": int(neg_mask_dbg[step_idx].sum().item()),
                "topk_by_logit": topk_payload,
                "positive_terms": positive_payload,
                "sampled_negative_terms": negative_payload,
                "positive_targets": positive_payload,
                "sampled_negatives_top_loss": negative_payload,
                "top_tokens_by_abs_grad": grad_payload,
                "avg_logit_rank_of_distance_groups": distance_group_avg_rank,
                "avg_distance_of_logit_groups": logit_group_avg_distance,
                "avg_distance_of_distance_groups": {
                    "top5": distance_group_avg_distance["top5"],
                    "top10": distance_group_avg_distance["top10"],
                    "top20": distance_group_avg_distance["top20"],
                    "top50": distance_group_avg_distance["top50"],
                },
            }
        )

    distance_rank_trend = {
        "step_index": trend_step_index,
        "label_position": trend_label_position,
        "gt_is_vec": trend_gt_is_vec,
        "avg_logit_rank_top1_by_step": trend_top_by_step[1],
        "avg_logit_rank_top5_by_step": trend_top_by_step[5],
        "avg_logit_rank_top10_by_step": trend_top_by_step[10],
        "avg_logit_rank_top20_by_step": trend_top_by_step[20],
        "vec_steps_with_distance": trend_vec_steps,
    }
    logit_distance_trend = {
        "step_index": trend_step_index,
        "label_position": trend_label_position,
        "gt_is_vec": trend_gt_is_vec,
        "avg_distance_top1_by_step": trend_dist_by_logit_top_by_step[1],
        "avg_distance_top5_by_step": trend_dist_by_logit_top_by_step[5],
        "avg_distance_top10_by_step": trend_dist_by_logit_top_by_step[10],
        "avg_distance_top20_by_step": trend_dist_by_logit_top_by_step[20],
        "vec_steps_with_distance": trend_vec_steps_logit,
    }
    distance_group_distance_trend = {
        "step_index": trend_step_index,
        "label_position": trend_label_position,
        "gt_is_vec": trend_gt_is_vec,
        "avg_distance_top5_by_step": trend_dist_by_distance_top_by_step[5],
        "avg_distance_top10_by_step": trend_dist_by_distance_top_by_step[10],
        "avg_distance_top20_by_step": trend_dist_by_distance_top_by_step[20],
        "avg_distance_top50_by_step": trend_dist_by_distance_top_by_step[50],
        "vec_steps_with_distance": trend_vec_steps,
    }

    return {
        "valid_count": valid_count,
        "reference_full_mean_loss": float(reference_loss.item()),
        "debug_full_mean_loss": float(full_debug_mean.detach().item()),
        "full_loss_abs_diff": float((full_debug_mean.detach() - reference_loss.detach()).abs().item()),
        "pos_mean": float(pos_mean.detach().item()),
        "neg_mean": float(neg_mean.detach().item()),
        "pos_neg_mean": float(pos_neg_mean.detach().item()),
        "type_mean_ignored": float(type_mean.detach().item()),
        "gradient_focus": "pos_neg_terms_only",
        "debug_note": "Token-level terms and gradients focus on positive and negative multi-label BCE; type loss is excluded from token diagnostics.",
        "step_count_reported": len(steps_payload),
        "distance_rank_trend": distance_rank_trend,
        "logit_distance_trend": logit_distance_trend,
        "distance_group_distance_trend": distance_group_distance_trend,
        "steps": steps_payload,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal inference debug for logit/sampling behavior.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--peft_model", default=None)
    parser.add_argument("--quantization", default=None, choices=[None, "4bit", "8bit"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--row_index", type=int, default=0)
    parser.add_argument("--num_rows", type=int, default=1)
    parser.add_argument("--context_mode", choices=["labels", "full"], default="labels")
    parser.add_argument("--run_mode", choices=["generate", "teacher_forcing"], default="generate")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_chars", type=int, default=400)
    parser.add_argument("--topk_report", type=int, default=10)
    parser.add_argument("--debug_multi_label_bce", action="store_true")
    parser.add_argument("--debug_focus_step", type=int, default=None)
    parser.add_argument(
        "--debug_neg_tokens",
        type=int,
        default=20,
        help="Max sampled negative VEC tokens reported per step. Use <=0 to dump all sampled negatives.",
    )
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    if is_xpu_available():
        torch.xpu.manual_seed(args.seed)

    tokenizer = build_tokenizer(args.model_name)
    model = load_model(args.model_name, args.quantization, use_fast_kernels=False)
    if args.peft_model:
        model = load_peft_model(model, args.peft_model)
    model.eval()

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    device = model.device

    vec_tokens = tokenizer.convert_tokens_to_ids([f"VEC_{i}" for i in range(1024)])
    vec_token_ids = torch.tensor([x for x in vec_tokens if x >= 0], device=device, dtype=torch.long)
    vec_id_to_index = {tok_id: i for i, tok_id in enumerate(vec_tokens) if tok_id >= 0}

    if args.num_rows <= 0:
        raise ValueError("--num_rows must be >= 1")

    output_json = args.output_json
    if output_json is None:
        output_json = f"inference_debug_{time.strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_json = output_json.strip()
        if not output_json:
            output_json = f"inference_debug_{time.strftime('%Y%m%d_%H%M%S')}.json"
        else:
            if "/" in output_json:
                dir_part, base = output_json.rsplit("/", 1)
                if not base.startswith("inference_debug_"):
                    base = f"inference_debug_{base}"
                output_json = f"{dir_part}/{base}"
            else:
                if not output_json.startswith("inference_debug_"):
                    output_json = f"inference_debug_{output_json}"

    debug_payload = {
        "meta": {
            "model_name": args.model_name,
            "peft_model": args.peft_model,
            "quantization": args.quantization,
            "row_index_start": args.row_index,
            "num_rows_requested": args.num_rows,
            "context_mode": args.context_mode,
            "run_mode": args.run_mode,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "topk_report": args.topk_report,
            "debug_multi_label_bce": args.debug_multi_label_bce,
            "debug_focus_step": args.debug_focus_step,
            "debug_neg_tokens": args.debug_neg_tokens,
        },
        "rows": [],
    }

    aggregate_bce_debug_only = (
        args.run_mode == "teacher_forcing"
        and args.debug_multi_label_bce
        and args.num_rows > 1
    )
    debug_payload["meta"]["aggregate_debug_only"] = aggregate_bce_debug_only

    overall_metrics = init_gt_metrics()
    rows_processed = 0
    gt_rank_total = 0.0
    gt_rank_count = 0
    multi_rank_total = 0.0
    multi_rank_count = 0
    bce_debug_scalar_averager = init_scalar_debug_averager()
    bce_debug_rows_total = 0
    bce_debug_rows_succeeded = 0
    bce_debug_rows_failed = 0

    for row_index, row in iter_rows_by_range(args.data_path, args.row_index, args.num_rows):
        rows_processed += 1
        full_input_ids, full_attention_mask, full_labels = prepare_inputs_and_labels(row, tokenizer)
        context_ids, context_mask, target_ids, _ = prepare_context(
            row, tokenizer, context_mode=args.context_mode
        )

        sid = to_jsonable(row.get("sid", None))
        ego_id = to_jsonable(row.get("ego_id", None))

        context_text = tokenizer.decode(context_ids, skip_special_tokens=True)
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True) if target_ids else ""

        # print(f"Row {row_index} | sid={sid} ego_id={ego_id}")
        # print(f"context_mode={args.context_mode} context_len={len(context_ids)} target_len={len(target_ids)}")
        # print(f"context_text: {context_text[: args.print_chars]}")
        # if target_text:
        #     print(f"target_text: {target_text[: args.print_chars]}")

        if args.run_mode == "teacher_forcing":
            input_ids = torch.tensor([full_input_ids], device=device)
            attention_mask = torch.tensor([full_attention_mask], device=device)
            labels_tensor = (
                torch.tensor([full_labels], device=device)
                if full_labels
                else None
            )
            with torch.no_grad():
                if labels_tensor is None:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels_tensor,
                        use_cache=False,
                    )

            logits = outputs.logits
            target_positions = [i for i, label in enumerate(full_labels) if label != -100]
            if args.max_new_tokens and args.max_new_tokens > 0:
                target_positions = target_positions[: args.max_new_tokens]
            multi_labels = row.get("multi_label", None)
            label_weight = row.get("label_weight", None)

            for step, label_pos in enumerate(target_positions):
                logit_pos = label_pos - 1
                if logit_pos < 0 or logit_pos >= logits.size(1):
                    continue

                step_logits = logits[:, logit_pos, :]
                gt_id = int(full_labels[label_pos])
                gt_rank = rank_for_token(step_logits, gt_id)
                if gt_rank is not None:
                    gt_rank_total += gt_rank
                    gt_rank_count += 1

                step_multi = None
                if isinstance(multi_labels, (list, tuple, np.ndarray)) and step < len(multi_labels):
                    step_multi = multi_labels[step]
                elif torch.is_tensor(multi_labels) and multi_labels.dim() >= 2 and step < multi_labels.size(0):
                    step_multi = multi_labels[step]
                if step_multi is not None:
                    if torch.is_tensor(step_multi):
                        step_multi = step_multi.detach().cpu().tolist()
                    if isinstance(step_multi, np.ndarray):
                        step_multi = step_multi.tolist()
                    if isinstance(step_multi, (list, tuple)) and step_multi:
                        ranks = []
                        for token_id in step_multi:
                            try:
                                token_id_int = int(token_id)
                            except (TypeError, ValueError):
                                continue
                            rank = rank_for_token(step_logits, token_id_int)
                            if rank is not None:
                                ranks.append(rank)
                        if ranks:
                            avg_rank = float(sum(ranks)) / float(len(ranks))
                            multi_rank_total += avg_rank
                            multi_rank_count += 1

            if args.debug_multi_label_bce:
                row_payload = None
                if not aggregate_bce_debug_only:
                    row_payload = {
                        "row_index": row_index,
                        "sid": sid,
                        "ego_id": ego_id,
                        "context_len": len(context_ids),
                        "target_len": len(target_ids),
                        "context_text_preview": context_text[: args.print_chars],
                        "target_text_preview": target_text[: args.print_chars] if target_text else "",
                    }

                bce_debug = None
                bce_debug_error = None
                if labels_tensor is None:
                    bce_debug_error = "No labels in row; cannot run teacher-forcing BCE debug."
                elif multi_labels is None:
                    bce_debug_error = "Row has no multi_label column."
                else:
                    try:
                        bce_debug = debug_teacher_forcing_multi_label_bce(
                            logits=logits.detach(),
                            labels=labels_tensor,
                            multi_label=multi_labels,
                            label_weight=label_weight,
                            tokenizer=tokenizer,
                            vec_token_ids=vec_token_ids,
                            vec_id_to_index=vec_id_to_index,
                            topk_report=args.topk_report,
                            focus_step=args.debug_focus_step,
                            max_steps=args.max_new_tokens,
                            neg_token_limit=args.debug_neg_tokens,
                            ignore_index=-100,
                            pos_weight=100.0,
                        )
                    except Exception as exc:
                        bce_debug_error = str(exc)

                if aggregate_bce_debug_only:
                    bce_debug_rows_total += 1
                    if bce_debug is not None and bce_debug_error is None:
                        update_scalar_debug_averager(bce_debug_scalar_averager, bce_debug)
                        bce_debug_rows_succeeded += 1
                    else:
                        bce_debug_rows_failed += 1
                else:
                    if bce_debug_error is not None:
                        row_payload["multi_label_bce_debug_error"] = bce_debug_error
                    elif bce_debug is not None:
                        row_payload["multi_label_bce_debug"] = bce_debug
                        trend_payload = bce_debug.get("distance_rank_trend", {})
                        plot_path = build_distance_rank_plot_path(output_json, row_index)
                        plot_error = save_distance_rank_trend_plot(trend_payload, plot_path)
                        if plot_error is None:
                            row_payload["distance_rank_trend_plot"] = plot_path
                        else:
                            row_payload["distance_rank_trend_plot_error"] = plot_error

                        reverse_trend_payload = bce_debug.get("logit_distance_trend", {})
                        reverse_plot_path = build_logit_distance_plot_path(output_json, row_index)
                        reverse_plot_error = save_logit_distance_trend_plot(
                            reverse_trend_payload,
                            reverse_plot_path,
                        )
                        if reverse_plot_error is None:
                            row_payload["logit_distance_trend_plot"] = reverse_plot_path
                        else:
                            row_payload["logit_distance_trend_plot_error"] = reverse_plot_error

                        distance_group_distance_payload = bce_debug.get("distance_group_distance_trend", {})
                        distance_group_distance_plot_path = build_distance_group_distance_plot_path(
                            output_json,
                            row_index,
                        )
                        distance_group_distance_plot_error = save_distance_group_distance_trend_plot(
                            distance_group_distance_payload,
                            distance_group_distance_plot_path,
                        )
                        if distance_group_distance_plot_error is None:
                            row_payload["distance_group_distance_trend_plot"] = (
                                distance_group_distance_plot_path
                            )
                        else:
                            row_payload["distance_group_distance_trend_plot_error"] = (
                                distance_group_distance_plot_error
                            )

                        neg_dist_plot_path = build_negative_bce_distribution_plot_path(
                            output_json,
                            row_index,
                        )
                        neg_dist_plot_error = save_negative_bce_distribution_plot(
                            bce_debug,
                            neg_dist_plot_path,
                            bins=30,
                        )
                        if neg_dist_plot_error is None:
                            row_payload["negative_bce_distribution_plot"] = neg_dist_plot_path
                        else:
                            row_payload["negative_bce_distribution_plot_error"] = neg_dist_plot_error
                    debug_payload["rows"].append(row_payload)

            continue

        input_ids = torch.tensor([context_ids], device=device)
        attention_mask = torch.tensor([context_mask], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        temp = args.temperature if args.temperature and args.temperature > 0 else 1.0
        scaled_logits = logits / temp
        topk_raw = topk_from_logits(scaled_logits, args.topk_report, tokenizer)
        # print(format_topk("raw_topk", topk_raw, limit=args.topk_report))

        filtered_logits = apply_top_k_top_p(scaled_logits, top_k=args.top_k, top_p=args.top_p)
        topk_filtered = topk_from_logits(filtered_logits, args.topk_report, tokenizer)
        # print(format_topk("filtered_topk", topk_filtered, limit=args.topk_report))

        if vec_token_ids.numel() > 0:
            vec_topk = topk_vec_tokens(
                scaled_logits, vec_token_ids, min(args.topk_report, vec_token_ids.numel()), tokenizer
            )
            # print(format_topk("vec_topk", vec_topk, limit=args.topk_report))
        else:
            print("vec_topk: no VEC tokens found in tokenizer.")

        if args.do_sample:
            probs = torch.softmax(filtered_logits, dim=-1)
            sampled_id = torch.multinomial(probs, num_samples=1).item()
            # print(f"sampled_token: {tokenizer.convert_ids_to_tokens([sampled_id])[0]} ({sampled_id})")

        row_payload = {
            "row_index": row_index,
            "sid": sid,
            "ego_id": ego_id,
            "context_len": len(context_ids),
            "target_len": len(target_ids),
            "context_text_preview": context_text[: args.print_chars],
            "target_text_preview": target_text[: args.print_chars] if target_text else "",
            "steps": [],
        }
        row_metrics = init_gt_metrics()

        generated_ids = []
        current_attention_mask = attention_mask
        next_input_ids = input_ids

        for step in range(args.max_new_tokens):
            if step == 0:
                step_logits = logits
            else:
                with torch.no_grad():
                    outputs = model(
                        input_ids=next_input_ids,
                        attention_mask=current_attention_mask,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                step_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

            step_topk = topk_logits_from_logits(step_logits, args.topk_report, tokenizer)
            step_topk_str = format_topk_logits("logits_topk", step_topk, limit=args.topk_report)
            topk_ids = [tok_id for _, tok_id, _ in step_topk]

            gt_info = None
            if step < len(target_ids):
                gt_id = int(target_ids[step])
                gt_info = compute_gt_info(
                    step_logits,
                    gt_id,
                    tokenizer,
                    topk_ids,
                    vec_token_ids,
                    vec_id_to_index,
                    row_metrics,
                    overall_metrics,
                )

            if args.do_sample:
                temp = args.temperature if args.temperature and args.temperature > 0 else 1.0
                step_logits = step_logits / temp
                step_logits = apply_top_k_top_p(step_logits, top_k=args.top_k, top_p=args.top_p)
                probs = torch.softmax(step_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token_id = torch.argmax(step_logits, dim=-1).item()

            decoded_token = tokenizer.decode([next_token_id], skip_special_tokens=False)
            # print(f"step={step} token={decoded_token!r} id={next_token_id} {step_topk_str}")
            row_payload["steps"].append(
                {
                    "step": step,
                    "token": decoded_token,
                    "token_id": next_token_id,
                    "logits_topk": [
                        {"token": tok, "token_id": tok_id, "logit": logit}
                        for tok, tok_id, logit in step_topk
                    ],
                    "gt": gt_info,
                }
            )

            generated_ids.append(next_token_id)
            next_input_ids = torch.tensor([[next_token_id]], device=device)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.size(0), 1),
                        device=device,
                        dtype=current_attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                break

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        row_payload["generated_text"] = generated_text[: args.print_chars]
        row_payload["metrics"] = finalize_gt_metrics(row_metrics, args.topk_report)
        # debug_payload["rows"].append(row_payload)
        # print(f"generated_text: {generated_text[: args.print_chars]}")

        # Optional: quick view of ground-truth VEC sequence for comparison
        if target_ids:
            gt_vecs = re.findall(r"VEC_(\\d+)", tokenizer.decode(target_ids, skip_special_tokens=True))
            # if gt_vecs:
            #     print(f"ground_truth_vec_ids (first 10): {gt_vecs[:10]}")

    if rows_processed == 0:
        raise IndexError(f"No rows found starting at index {args.row_index}.")

    if args.run_mode == "teacher_forcing":
        avg_gt_rank = (gt_rank_total / gt_rank_count) if gt_rank_count else None
        avg_multi_label_rank = (multi_rank_total / multi_rank_count) if multi_rank_count else None
        if args.debug_multi_label_bce:
            debug_payload["meta"]["num_rows_processed"] = rows_processed
            metrics_payload = {
                "avg_gt_rank": avg_gt_rank,
                "avg_multi_label_rank": avg_multi_label_rank,
            }
            if aggregate_bce_debug_only:
                metrics_payload["avg_multi_label_bce_debug"] = finalize_scalar_debug_averager(
                    bce_debug_scalar_averager
                )
                metrics_payload["multi_label_bce_debug_rows_total"] = bce_debug_rows_total
                metrics_payload["multi_label_bce_debug_rows_succeeded"] = bce_debug_rows_succeeded
                metrics_payload["multi_label_bce_debug_rows_failed"] = bce_debug_rows_failed
                debug_payload.pop("rows", None)
            debug_payload["metrics"] = metrics_payload
        else:
            debug_payload = {
                "avg_gt_rank": avg_gt_rank,
                "avg_multi_label_rank": avg_multi_label_rank,
            }
    else:
        debug_payload["meta"]["num_rows_processed"] = rows_processed
        debug_payload["metrics"] = finalize_gt_metrics(overall_metrics, args.topk_report)
    with open(output_json, "w") as f:
        json.dump(debug_payload, f, indent=2)
    print(f"saved_debug_json: {output_json}")


if __name__ == "__main__":
    main()
