#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
import pyarrow.parquet as pq
from datasets import Dataset

from llama_cookbook.utils.action_model import LlamaForCausalLMWithActions
from llama_cookbook.utils.bidirection_action_model import LlamaForBidirectionAttnWithActions
from llama_cookbook.utils.bidirection_diffusion_model import LlamaForBidirectionAttnWithDiffusionActions

from tqdm import tqdm  # Add tqdm import


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run action-head inference from a local checkpoint.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Hugging Face-format checkpoint directory containing config.json and weights.",
    )
    parser.add_argument(
        "--input_parquet",
        type=str,
        required=True,
        help="Parquet file with columns: input_ids, attention_mask, pred_seq.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="inference_results.json",
        help="Filename for the output JSON (saved relative to the current working directory).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=6,
        help="Number of inference passes to run per row.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=0,
        help="Optional number of tokens to generate for language output; 0 disables text generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for text generation when max_new_tokens > 0.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p for text generation when max_new_tokens > 0.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process from the input Parquet file.",
    )
    return parser.parse_args()


def _to_tensor(value, dtype=None, device="cpu"):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor.to(device)

def trim_input_ids(input_ids, labels, attention_mask, keep_first_n=0):
    label_indexes = [i for i, id in enumerate(labels) if id != -100]
    if keep_first_n > 0:
        label_indexes = [index for i, index in enumerate(label_indexes) if i >= keep_first_n]
    input_ids = input_ids[: label_indexes[0]]
    attention_mask = attention_mask[: label_indexes[0]]
    return input_ids, attention_mask

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_path)
    config.use_action_head = True
    # model = LlamaForCausalLMWithActions.from_pretrained(args.model_path, config=config)
    config.bidirectional_attention = True
    model = LlamaForBidirectionAttnWithActions.from_pretrained(args.model_path, config=config)
    # model = LlamaForBidirectionAttnWithDiffusionActions.from_pretrained(args.model_path, config=config)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    ROAD_TYPE_TOKEN = [
        "LaneCenter-Freeway",
        "LaneCenter-SurfaceStreet",
        "RoadEdgeBoundary",
        "RoadEdgeMedian",
        "StopSign",
        "Crosswalk",
        "SpeedBump"
    ]

    # ROAD_TYPE_TOKEN = [
    #     "LaneCenter-Freeway",
    #     "LaneCenter-SurfaceStreet",
    #     "RoadLine-BrokenSingleWhite",
    #     "RoadLine-SolidSingleWhite",
    #     "RoadLine-SolidDoubleWhite",
    #     "RoadLine-BrokenSingleYellow",
    #     "RoadLine-BrokenDoubleYellow",
    #     "Roadline-SolidSingleYellow",
    #     "Roadline-SolidDoubleYellow",
    #     "RoadLine-PassingDoubleYellow",
    #     "StopSign",
    #     "Crosswalk",
    #     "SpeedBump"
    # ]

    # ROAD_TYPE_TOKEN = [
    #     "LaneCenter-Freeway",
    #     "LaneCenter-SurfaceStreet",
    #     "LaneCenter-BikeLane",
    #     "RoadLine-BrokenSingleWhite",
    #     "RoadLine-SolidSingleWhite",
    #     "RoadLine-SolidDoubleWhite",
    #     "RoadLine-BrokenSingleYellow",
    #     "RoadLine-BrokenDoubleYellow",
    #     "Roadline-SolidSingleYellow",
    #     "Roadline-SolidDoubleYellow",
    #     "RoadLine-PassingDoubleYellow",
    #     "RoadEdgeBoundary",
    #     "RoadEdgeMedian",
    #     "StopSign",
    #     "Crosswalk",
    #     "SpeedBump"
    # ]

    # # vel_type = [f'VEL_{round(i/10, 2)}' for i in list(range(0, 41))]
    # acc_type = [f'ACC_{round(i, 3)}' for i in [x * 0.005 for x in range(-20, 21)]]
    # len_type = [f'LEN_{round(i/10, 2)}' for i in list(range(0, 51, 5))]
    # dir_type = [f'VEC_{i}' for i in range(360)]

    # veh_vec = [f'VEH_VEC_{i}' for i in range(512)]
    # ped_vec = [f'PED_VEC_{i}' for i in range(512)]
    # cyc_vec = [f'CYCL_VEC_{i}' for i in range(512)]

    # custom_tokens = []

    # # custom_tokens.extend(vel_type)
    # custom_tokens.extend(acc_type)
    # custom_tokens.extend(len_type)
    # custom_tokens.extend(dir_type)

    # custom_tokens.extend(veh_vec)
    # custom_tokens.extend(ped_vec)
    # custom_tokens.extend(cyc_vec)

    # for l in len_type:
    #     for d in dir_type:
    #         custom_tokens.append(f'{d}{l}')

    # for v in vel_type:
    #     for d in dir_type:
    #         custom_tokens.append(f'{d}{v}')

    # for a in acc_type:
    #     for d in dir_type:
    #         custom_tokens.append(f'{d}{a}')


    # angle_bins = np.load('/p/ruishen/processed_waymo_data/validation/waymo_vectorized/combined_angle_bins_10hz_512.npy', allow_pickle=True)
    # len_vals = np.arange(0, 3.51, 0.01)
    # len_type = [f"LEN_{val:.2f}" for val in len_vals]
    # len_type.append("LEN_10.00")
    # dir_type = [f"VEC_{i}" for i in range(len(angle_bins))]
    # acc_type = [f'ACC_{round(i, 3)}' for i in [x * 0.005 for x in range(-20, 21)]]

    # custom_tokens = []
    # custom_tokens.extend(acc_type)
    # custom_tokens.extend(len_type)
    # custom_tokens.extend(dir_type)

    custom_tokens = [f"VEC_{i}" for i in range(1024)]

    custom_tokens.extend(ROAD_TYPE_TOKEN)

    # custom_tokens.append('<ROAD_START>')
    # custom_tokens.append('<ROAD_END>')
    # custom_tokens.append('<ROAD_VECTOR_START>')
    # custom_tokens.append('<ROAD_VECTOR_END>')
    # custom_tokens.append('AGENT_TRAJ_START')
    # custom_tokens.append('AGENT_TRAJ_END')
    # custom_tokens.append('START_')
    # custom_tokens.append('AGENT_ID_')
    # custom_tokens.append('AGENT_TYPE_Vehicle')
    # custom_tokens.append('AGENT_TYPE_Pedestrian')
    # custom_tokens.append('AGENT_TYPE_Cyclist')
    # custom_tokens.append('AGENT_TYPE_Other')
    # custom_tokens.append('AGENT_TYPE_Unset')
    # custom_tokens.append('TRAJ_NONE')
    # custom_tokens.append('CTRL_NONE')
    # custom_tokens.append('EGO_TRAJ_START')
    # custom_tokens.append('EGO_TRAJ_END')
    # custom_tokens.append('AGENT_TRAJ_START')
    # custom_tokens.append('AGENT_TRAJ_END')
    # custom_tokens.append('MAP_START')
    # custom_tokens.append('MAP_END')
    # custom_tokens.append('INITIAL_HEADING_')

    # custom_tokens.append('START_')
    # custom_tokens.append('AGENT_ID_')
    # custom_tokens.append('AGENT_TYPE_Vehicle')
    # custom_tokens.append('AGENT_TYPE_Pedestrian')
    # custom_tokens.append('AGENT_TYPE_Cyclist')
    # custom_tokens.append('AGENT_TYPE_Other')
    # custom_tokens.append('AGENT_TYPE_Unset')
    # custom_tokens.append('TRAJ_NONE')
    # custom_tokens.append('CTRL_NONE')
    # custom_tokens.append('POS_')
    # custom_tokens.append('POS_NONE')
    # custom_tokens.append('EGO_TRAJ_START')
    # custom_tokens.append('EGO_TRAJ_END')
    # custom_tokens.append('AGENT_TRAJ_START')
    # custom_tokens.append('AGENT_TRAJ_END')
    # custom_tokens.append('MAP_START')
    # custom_tokens.append('MAP_END')
    # custom_tokens.append('INITIAL_HEADING_')

    custom_tokens.append('AGENT_ID_')
    custom_tokens.append('AGENT_TYPE_Vehicle')
    custom_tokens.append('AGENT_TYPE_Pedestrian')
    custom_tokens.append('AGENT_TYPE_Cyclist')
    custom_tokens.append('AGENT_TYPE_Other')
    custom_tokens.append('AGENT_TYPE_Unset')
    custom_tokens.append('TRAJ_NONE')
    custom_tokens.append('CTRL_NONE')
    custom_tokens.append('POS_')
    custom_tokens.append('POS_NONE')
    custom_tokens.append('EGO_TRAJ_START')
    custom_tokens.append('EGO_TRAJ_END')
    custom_tokens.append('AGENT_TRAJ_START')
    custom_tokens.append('AGENT_TRAJ_END')
    custom_tokens.append('MAP_START')
    custom_tokens.append('MAP_END')
    custom_tokens.append('ROAD_START')
    custom_tokens.append('ROAD_END')

    tokenizer.add_tokens(custom_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    # df = pd.read_parquet(args.input_parquet)
    # if args.max_samples is not None:
    #     df = df.head(args.max_samples)

    table = pq.read_table(args.input_parquet)

    num_rows = table.num_rows
    num_rows = 1000

    # Step 2: Use tqdm to visualize loading progress
    batch_size = 1000
    rows = []
    for i in tqdm(range(0, num_rows, batch_size), desc="Building dataset"):
        batch = table.slice(i, batch_size)
        batch_dict = batch.to_pydict()
        # Reorganize row-wise
        for j in range(len(batch_dict[next(iter(batch_dict))])):
            rows.append({k: batch_dict[k][j] for k in batch_dict})

    # Step 3: Wrap into HuggingFace Dataset
    custom_dataset = Dataset.from_list(rows)

    # Shuffle the dataset and select a batch of samples
    seed = 42
    num_samples = min(args.max_samples, len(custom_dataset))
    random_rows = custom_dataset.shuffle(seed=seed)[:num_samples]
    df = pd.DataFrame(random_rows)

    output_path = Path(args.output_json)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference Progress"):
            row_input_ids = row["input_ids"]
            row_labels = row["labels"]
            row_attention_mask = row.get("attention_mask", None)
            # row_input_ids, row_attention_mask = trim_input_ids(row_input_ids, row_labels, row_attention_mask, keep_first_n=40)
            row_pred_seq = row.get("pred_seq", None)
            row_pred_seq = np.stack(row_pred_seq)

            row_sid = row.get("sid", None)
            row_ego_id = row.get("ego_id", None)

            # if row_sid != "5fd55a1c669f6e40" or str(row_ego_id) != "1324":
            #     continue

            input_ids = _to_tensor(row_input_ids, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = None
            if row_attention_mask is not None:
                attention_mask = _to_tensor(row_attention_mask, dtype=torch.long, device=device).unsqueeze(0)

            pred_seq = None
            if row_pred_seq is not None:
                pred_seq = _to_tensor(row_pred_seq, device=device).unsqueeze(0)

            with torch.no_grad():
                if getattr(model, "_use_mon", False):
                    mask_type_labels = row_labels.copy()
                    mask_type_labels = _to_tensor(mask_type_labels, dtype=torch.long, device=device).unsqueeze(0)
                    mask_type_labels = torch.where(mask_type_labels == -100, 1, 2)

                    target_mask = (mask_type_labels == 2).long()
                    mask_id = -1
                    mon_input_ids = torch.where(target_mask == 1, mask_id, input_ids)

                    mon_action_output, _ = model.action_head_based_generate_actions(
                        input_ids=mon_input_ids,
                        attention_mask=attention_mask,
                        return_generation_output=True,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                        tokenizer=tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        mask_type_labels=mask_type_labels,
                    )

                    mon_action_output = mon_action_output.detach().cpu()
                    mon_action_batch = mon_action_output.reshape(mon_action_output.size(0), -1, 2).numpy()
                    for mon_candidate_idx, mon_candidate in enumerate(mon_action_batch):
                        json_line = {
                            "ground_truth": row_pred_seq.tolist() if row_pred_seq is not None else None,
                            "llm_answer": mon_candidate.tolist(),
                            "mon_candidate_idx": mon_candidate_idx,
                            "sid": row_sid,
                            "ego_id": row_ego_id,
                            "decoded_text": None,
                        }
                        f.write(json.dumps(json_line) + "\n")
                    continue

                for _ in range(args.num_runs):
                    # train_input_ids = _to_tensor(row["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                    # train_attention_mask = _to_tensor(row.get("attention_mask", None), dtype=torch.long, device=device)
                    # train_labels = _to_tensor(row.get("labels", None), dtype=torch.long, device=device).unsqueeze(0)
                    # action_output = model(input_ids=train_input_ids, attention_mask=train_attention_mask, labels=train_labels, pred_seq=pred_seq, task="action")
                    # llm = action_output['action_head_output'].detach().cpu().numpy().reshape(79, 2)
                    # json_line = {
                    #     "ground_truth": row_pred_seq.tolist() if row_pred_seq is not None else None,
                    #     "llm_answer": llm.tolist(),
                    #     "action_loss": 0,
                    # }
                    # f.write(json.dumps(json_line) + "\n")
                    # continue
                    
                    # action_output, generation_output = model.generate_actions(
                    #     input_ids=input_ids,
                    #     attention_mask=attention_mask,
                    #     return_generation_output=True,
                    #     pad_token_id=tokenizer.pad_token_id,
                    #     temperature=args.temperature,
                    #     top_p=args.top_p,
                    #     do_sample=True,
                    # )

                    mask_type_labels = row_labels.copy()
                    mask_type_labels = _to_tensor(mask_type_labels, dtype=torch.long, device=device).unsqueeze(0)
                    mask_type_labels = torch.where(mask_type_labels == -100, 1, 2)

                    target_mask = (mask_type_labels == 2).long()
                    mask_id = -1
                    input_ids = torch.where(target_mask == 1, mask_id, input_ids)

                    action_output, generation_output = model.action_head_based_generate_actions(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_generation_output=True,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                        tokenizer=tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        mask_type_labels=mask_type_labels,
                    )

                    # generation_output = model.generate(
                    #     input_ids=input_ids,
                    #     attention_mask=attention_mask,
                    #     max_new_tokens=args.max_new_tokens,
                    #     pad_token_id=tokenizer.pad_token_id,
                    #     temperature=args.temperature,
                    #     top_p=args.top_p,
                    #     do_sample=True,
                    # )                    

                    action_output_flat = action_output
                    action_loss = None
                    if pred_seq is not None:
                        action_output_flat = action_output_flat.reshape(-1, 2).detach().cpu().numpy()
                        llm = np.cumsum(action_output_flat, axis=0)
                        gt = pred_seq.squeeze(0).detach().cpu().numpy()
                        gt = np.cumsum(gt, axis=0)
                        ade = np.mean(np.linalg.norm(gt - llm, axis=-1))
                        action_loss = float(ade)

                    decoded_text = None
                    # if tokenizer is not None and args.max_new_tokens > 0:
                    #     generated_ids = generation_output.sequences
                    #     # decoded_text = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)
                    #     decoded_text = tokenizer.decode(generated_ids[0][(mask_type_labels == -100).sum():], skip_special_tokens=True)

                    json_line = {
                        "ground_truth": row_pred_seq.tolist() if row_pred_seq is not None else None,
                        "llm_answer": action_output_flat.tolist(),
                        "action_loss": action_loss,
                        "sid": row_sid,
                        "ego_id": row_ego_id,
                        "decoded_text": decoded_text,
                    }
                    f.write(json.dumps(json_line) + "\n")

    print(f"Saved inference results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
