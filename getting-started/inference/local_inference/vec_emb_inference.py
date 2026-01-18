#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from llama_cookbook.configs.datasets import custom_dataset as CustomDatasetConfig
from llama_cookbook.utils.dataset_utils import get_preprocessed_dataset
from llama_cookbook.utils.vector_embedding_model import LlamaForBidirectionAttnWithVectorEmbeddings

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vector-embedding inference from a local checkpoint.")
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
        help="Parquet file with map/trajectory payloads and optional pred_seq fields.",
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


def _load_dataset(input_parquet: str):
    dataset_config = CustomDatasetConfig()
    dataset_config.train_path = input_parquet
    dataset_config.data_path = input_parquet
    return get_preprocessed_dataset(None, dataset_config, split="train")


def _parse_payload(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _extract_vector_row(row):
    sid_agent_id = row.get("sid_agent_id")
    if sid_agent_id is None:
        sid_agent_id = row.get("sid")
    ego_id = row.get("ego_id")

    sid = None
    if isinstance(sid_agent_id, str) and "__" in sid_agent_id:
        sid, ego = sid_agent_id.split("__", 1)
        if ego_id is None:
            try:
                ego_id = int(ego)
            except ValueError:
                ego_id = ego
    elif sid_agent_id is not None:
        sid = sid_agent_id

    map_payload = row.get("map_payloads")
    if map_payload is None:
        map_payload = row.get("map")
    map_payload = _parse_payload(map_payload, [])

    trajectory_payload = row.get("trajectory_payloads")
    if trajectory_payload is None:
        trajectory_payload = row.get("trajectories")
    trajectory_payload = _parse_payload(trajectory_payload, {})

    pred_seq = row.get("pred_seq") or []
    pred_seq = _parse_payload(pred_seq, [])

    return map_payload, trajectory_payload, pred_seq, sid, ego_id


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_path)
    config.use_action_head = True
    config.bidirectional_attention = True
    config.vec_emb_model = True
    model = LlamaForBidirectionAttnWithVectorEmbeddings.from_pretrained(args.model_path, config=config)
    model.to(device)
    model.eval()

    dataset = _load_dataset(args.input_parquet)
    seed = 42
    num_samples = min(args.max_samples, len(dataset))
    if num_samples == 0:
        print("No samples found in dataset.")
        return
    dataset = dataset.shuffle(seed=seed).select(list(range(num_samples)))

    output_path = Path(args.output_json)
    with output_path.open("w", encoding="utf-8") as f:
        for idx in tqdm(range(num_samples), desc="Inference Progress"):
            row = dataset[idx]
            map_payload, trajectory_payload, pred_seq, row_sid, row_ego_id = _extract_vector_row(row)

            pred_seq_array = None
            pred_seq_batch = None
            if pred_seq is not None:
                pred_seq_array = np.asarray(pred_seq)
                if pred_seq_array.size == 0:
                    pred_seq_array = None
                else:
                    if pred_seq_array.ndim == 1:
                        pred_seq_array = pred_seq_array.reshape(1, -1)
                    if pred_seq_array.ndim == 2:
                        pred_seq_batch = pred_seq_array[None, ...]
                    elif pred_seq_array.ndim == 3:
                        pred_seq_batch = pred_seq_array
                    else:
                        pred_seq_array = None
                        pred_seq_batch = None

            ground_truth = None
            if pred_seq_array is not None:
                gt = pred_seq_array
                if gt.ndim == 3:
                    gt = gt[0]
                ground_truth = gt.tolist()

            for _ in range(args.num_runs):
                output = model.inference(
                    map_payloads=[map_payload],
                    trajectory_payloads=[trajectory_payload],
                    pred_seq=pred_seq_batch,
                )

                action_output_flat = None
                action_loss = None
                if output.action_head_output is not None:
                    action_output_flat = (
                        output.action_head_output.reshape(-1, 2).detach().cpu().numpy()
                    )

                    if pred_seq_array is not None:
                        gt = pred_seq_array
                        if gt.ndim == 3:
                            gt = gt[0]
                        steps = min(gt.shape[0], action_output_flat.shape[0])
                        if steps > 0:
                            ade = np.mean(np.linalg.norm(gt[:steps] - action_output_flat[:steps], axis=-1))
                            action_loss = float(ade)

                json_line = {
                    "ground_truth": ground_truth,
                    "llm_answer": action_output_flat.tolist() if action_output_flat is not None else None,
                    "action_loss": action_loss,
                    "sid": row_sid,
                    "ego_id": row_ego_id,
                    "decoded_text": None,
                }
                f.write(json.dumps(json_line) + "\n")

    print(f"Saved inference results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
