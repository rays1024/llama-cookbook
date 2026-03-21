#!/usr/bin/env python3
import argparse
import json
import random
import re
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from accelerate.utils import is_xpu_available
from datasets import Dataset
from transformers import AutoTokenizer, logging

from llama_cookbook.inference.model_utils import load_model, load_peft_model

logging.set_verbosity_error()


ROAD_TYPE_TOKEN = [
    "LaneCenter-Freeway",
    "LaneCenter-SurfaceStreet",
    "RoadEdgeBoundary",
    "RoadEdgeMedian",
    "StopSign",
    "Crosswalk",
    "SpeedBump",
]

VEC_TOKEN_RE = re.compile(r"^VEC_\d+$")
STRAIGHT_TOKEN_RE = re.compile(r"^STRAIGHT_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-forcing CE eval for two token types only: STRAIGHT_* and VEC_*, "
            "restricted to tokens between ROAD_START and ROAD_END."
        )
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--peft_model", type=str, default=None)
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit"])
    parser.add_argument("--use_fast_kernels", action="store_true")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/p/ruishen/processed_waymo_data/all/validation/waymo_tokenized/trimmed_combined_context_next_token_10hz_all_vec_norm_True_straight_token.parquet",
    )
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--output_summary",
        type=str,
        default="ce_token_type_eval_summary.json",
        help="JSON file with only avg_ce_straight and avg_ce_vec.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


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
    custom_tokens.extend([f"STRAIGHT_{i}" for i in range(30)])
    tokenizer.add_tokens(custom_tokens)
    return tokenizer


def trim_row(
    row: Dict,
    pad_token_id: Optional[int],
) -> Tuple[List[int], List[int]]:
    input_ids = list(row["input_ids"])
    attention_mask = list(row["attention_mask"])

    if pad_token_id is not None:
        input_ids = [x for x in input_ids if x != pad_token_id]
        attention_mask = [x for x in attention_mask if x != 0]

    if len(attention_mask) != len(input_ids):
        attention_mask = [1] * len(input_ids)

    return input_ids, attention_mask


def normalize_token(token: str) -> str:
    return token.lstrip("▁").lstrip("Ġ")


def classify_token_type(token: str) -> Optional[str]:
    if VEC_TOKEN_RE.match(token):
        return "vec"
    if STRAIGHT_TOKEN_RE.match(token):
        return "straight"
    return None


def build_road_token_mask(input_ids: List[int], tokenizer: AutoTokenizer) -> List[bool]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    tokens = [normalize_token(t) for t in tokens]

    in_road = False
    mask = [False] * len(tokens)
    for idx, tok in enumerate(tokens):
        if tok == "ROAD_START":
            in_road = True
            continue
        if tok == "ROAD_END":
            in_road = False
            continue
        if in_road:
            mask[idx] = True
    return mask


def safe_mean(total: float, count: int) -> Optional[float]:
    if count <= 0:
        return None
    return total / float(count)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = build_tokenizer(args.model_name)

    model = load_model(
        args.model_name,
        args.quantization,
        args.use_fast_kernels,
    )
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if args.peft_model:
        model = load_peft_model(model, args.peft_model)
    model.eval()

    device = model.device

    table = pq.read_table(args.data_path)
    num_rows = min(table.num_rows, args.max_rows) if args.max_rows > 0 else table.num_rows

    rows = []
    read_batch_size = 1000
    for i in range(0, num_rows, read_batch_size):
        batch = table.slice(i, min(read_batch_size, num_rows - i))
        batch_dict = batch.to_pydict()
        row_len = len(batch_dict[next(iter(batch_dict))])
        for j in range(row_len):
            rows.append({k: batch_dict[k][j] for k in batch_dict})

    dataset = Dataset.from_list(rows)
    sample_count = len(dataset) if args.num_samples <= 0 else min(args.num_samples, len(dataset))
    sampled = dataset.shuffle(seed=args.seed).select(range(sample_count))

    token_type_acc = {
        "vec": {"sum_ce": 0.0, "count": 0},
        "straight": {"sum_ce": 0.0, "count": 0},
    }
    overall_sum_ce = 0.0
    overall_count = 0

    for i in range(0, sample_count, args.batch_size):
        batch_rows = [sampled[j] for j in range(i, min(i + args.batch_size, sample_count))]

        input_tensors = []
        attention_tensors = []
        road_masks = []

        for row in batch_rows:
            input_ids, attention_mask = trim_row(row, tokenizer.pad_token_id)
            if len(input_ids) <= 1:
                continue

            input_tensors.append(torch.tensor(input_ids, dtype=torch.long))
            attention_tensors.append(torch.tensor(attention_mask, dtype=torch.long))
            road_masks.append(build_road_token_mask(input_ids, tokenizer))

        if not input_tensors:
            continue

        input_ids_batch = torch.nn.utils.rnn.pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device)
        attention_mask_batch = torch.nn.utils.rnn.pad_sequence(
            attention_tensors,
            batch_first=True,
            padding_value=0,
        ).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            logits = outputs.logits

        # Teacher-forcing NTP over the full input_ids sequence.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids_batch[:, 1:].contiguous()

        per_token_ce = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
        ).view_as(shift_targets)

        valid_mask = attention_mask_batch[:, 1:].contiguous().bool()

        for b_idx in range(shift_targets.size(0)):
            row_valid_positions = torch.nonzero(valid_mask[b_idx], as_tuple=False).squeeze(-1)
            if row_valid_positions.numel() == 0:
                continue

            road_mask = road_masks[b_idx]
            for shifted_pos in row_valid_positions.tolist():
                ce_loss = float(per_token_ce[b_idx, shifted_pos].item())
                overall_sum_ce += ce_loss
                overall_count += 1

                sequence_pos = shifted_pos + 1
                if sequence_pos < 0 or sequence_pos >= len(road_mask):
                    continue
                if not road_mask[sequence_pos]:
                    continue

                token_id = int(shift_targets[b_idx, shifted_pos].item())
                token = normalize_token(tokenizer.convert_ids_to_tokens([token_id])[0])
                token_type = classify_token_type(token)
                if token_type is None:
                    continue

                token_type_acc[token_type]["sum_ce"] += ce_loss
                token_type_acc[token_type]["count"] += 1

    avg_ce_straight = safe_mean(token_type_acc["straight"]["sum_ce"], token_type_acc["straight"]["count"])
    avg_ce_vec = safe_mean(token_type_acc["vec"]["sum_ce"], token_type_acc["vec"]["count"])
    avg_ce_overall = safe_mean(overall_sum_ce, overall_count)

    summary = {
        "avg_ce_straight": avg_ce_straight,
        "avg_ce_vec": avg_ce_vec,
        "count_straight_tokens": token_type_acc["straight"]["count"],
        "count_vec_tokens": token_type_acc["vec"]["count"],
        "avg_ce_overall": avg_ce_overall,
    }

    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"avg_ce_straight: {avg_ce_straight}")
    print(f"avg_ce_vec: {avg_ce_vec}")
    print(f"count_straight_tokens: {token_type_acc['straight']['count']}")
    print(f"count_vec_tokens: {token_type_acc['vec']['count']}")
    print(f"avg_ce_overall: {avg_ce_overall}")


if __name__ == "__main__":
    main()
