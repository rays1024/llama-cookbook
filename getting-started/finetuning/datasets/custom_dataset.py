# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets
import numpy as np
import json
import pyarrow.parquet as pq
import pyarrow as pa
import tqdm
import psutil
import os
import warnings
from datasets import Dataset, concatenate_datasets

process = psutil.Process(os.getpid())

class CustomDataCollator:
    def __init__(self, config=None):
        self.config = config
        self.pred_seq_len = None
        if config is not None:
            self.pred_seq_len = getattr(config, "pred_seq_len", None)
            if self.pred_seq_len is None:
                self.pred_seq_len = getattr(config, "action_head_horizon", None)

    def __call__(self, data):
        if not data:
            return {}

        map_payloads = []
        trajectory_payloads = []
        pred_seqs = []
        sids = []
        ego_ids = []

        for row in data:
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

            sids.append(sid)
            ego_ids.append(ego_id)

            map_payload = row.get("map")
            if map_payload is None:
                map_payload = row.get("map_payloads")
            map_payloads.append(json.loads(map_payload) if isinstance(map_payload, str) else map_payload)

            traj_payload = row.get("trajectories")
            if traj_payload is None:
                traj_payload = row.get("trajectory_payloads")
            if isinstance(traj_payload, str):
                traj_payload = json.loads(traj_payload)
            trajectory_payloads.append(traj_payload)

            pred_seq = row.get("pred_seq") or []
            pred_seqs.append(json.loads(pred_seq) if isinstance(pred_seq, str) else pred_seq)

        pred_seq_batch = pred_seqs

        return {
            "map_payloads": map_payloads,
            "trajectory_payloads": trajectory_payloads,
            "pred_seq": pred_seq_batch,
            "sid": sids,
            "ego_id": ego_ids,
        }


def get_data_collator(dataset_processer, dataset_config):
    if "embedding" in dataset_config.train_path:
        return CustomDataCollator()
    else:
        return None

def mem_gb():
    return process.memory_info().rss / (1024 ** 3)


def _build_dataset_direct(parquet_file, keep_cols, num_rows):
    table = parquet_file.read(columns=keep_cols).slice(0, num_rows)
    return Dataset(table)


def _empty_dataset_from_schema(parquet_file, keep_cols):
    schema = parquet_file.schema_arrow
    empty_arrays = [pa.array([], type=schema.field(col).type) for col in keep_cols]
    return Dataset(pa.Table.from_arrays(empty_arrays, names=keep_cols))


def _build_dataset_batched(parquet_file, keep_cols, num_rows, batch_size=1024):
    if num_rows == 0:
        return _empty_dataset_from_schema(parquet_file, keep_cols)

    datasets_buffer = []
    processed_rows = 0

    pbar = tqdm.tqdm(
        parquet_file.iter_batches(batch_size=batch_size, columns=keep_cols),
        total=(num_rows + batch_size - 1) // batch_size if num_rows else 0,
        desc="Building dataset (batched fallback)",
    )

    for record_batch in pbar:
        if processed_rows >= num_rows:
            break

        remaining = num_rows - processed_rows
        if record_batch.num_rows > remaining:
            record_batch = record_batch.slice(0, remaining)

        table = pa.Table.from_batches([record_batch])
        datasets_buffer.append(Dataset(table))
        processed_rows += record_batch.num_rows

        pbar.set_postfix(rows=processed_rows, mem=f"{mem_gb():.2f} GB")

    if datasets_buffer:
        return datasets_buffer[0] if len(datasets_buffer) == 1 else concatenate_datasets(datasets_buffer)

    return _empty_dataset_from_schema(parquet_file, keep_cols)


def _build_dataset_pandas(data_path, keep_cols, num_rows):
    import pandas as pd

    df = pd.read_parquet(data_path, columns=keep_cols)
    if num_rows is not None:
        df = df.iloc[:num_rows]
    return Dataset.from_pandas(df, preserve_index=False)

def get_custom_dataset(dataset_config, tokenizer, split):
    # Load parquet file into dataset object
    dataset = None
    data_path = dataset_config.data_path

    # if split == "validation":
    #     data_path = data_path.replace("training", "validation")
    #     data_path = data_path.replace("sampling_factor_2", "sampling_factor_5")


    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/hierarchical_reasoning_validation.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/hierarchical_reasoning_training.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_traj_prediction.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_traj_prediction.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_traj_qa_10hz_long.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_traj_qa_10hz_long.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_traj_qa_10hz_long_pos.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_traj_qa_10hz_long_pos.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_traj_qa_10hz_long_grid.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_traj_qa_10hz_long_grid.parquet"

    # data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/hierarchical_reasoning_training_low_to_traj.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/language_condition/validation/waymo_tokenized/trimmed_combined_language_condition_10hz_long.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/language_condition/training/waymo_tokenized/trimmed_combined_language_condition_10hz_long.parquet"
    
    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_long.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_10hz_long.parquet"
    
    # small dataset for overfitting test
    # data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/small_overfitting_10hz_long.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_map_next_token_10hz_long.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_map_next_token_10hz_long.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_map_next_token_10hz_long_grid.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_map_next_token_10hz_long_grid.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_long.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_10hz_long.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_map_next_token_10hz_all_vec.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_map_next_token_10hz_all_vec.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_context_next_token_10hz_all_vec.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_context_next_token_10hz_all_vec.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_traj_next_token_10hz_all_vec.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_traj_next_token_10hz_all_vec.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_qa_10hz_all_vec.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_qa_10hz_all_vec.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/traj_pred_raw_traj.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/traj_pred_raw_traj.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/token_to_centroid.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/token_to_centroid.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/centroid_to_token.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/centroid_to_token.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq_noisy_10percent.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq_noisy_10percent.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_traj_next_token_10hz_all_vec_with_seq.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_traj_next_token_10hz_all_vec_with_seq.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_context_next_token_10hz_all_vec_norm_True.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_context_next_token_10hz_all_vec_norm_True.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_context_next_token_5hz_all_vec_norm_True.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_context_next_token_5hz_all_vec_norm_True.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/all/validation/waymo_tokenized/trimmed_combined_context_next_token_5hz_all_vec_norm_True.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/all/training/waymo_tokenized/trimmed_combined_context_next_token_5hz_all_vec_norm_True.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/all/validation/waymo_tokenized/trimmed_combined_context_next_token_10hz_all_vec_norm_True.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/all/training/waymo_tokenized/trimmed_combined_context_next_token_10hz_all_vec_norm_True.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq_norm_True.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq_norm_True.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_5hz_all_vec_with_seq_norm_True.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_5hz_all_vec_with_seq_norm_True.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq_noisy_10percent_norm_True.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq_noisy_10percent_norm_True.parquet"

    # if split == "validation":
    #     data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq_norm_True_parallel_decode.parquet"
    # else:
    #     data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_traj_prediction_10hz_all_vec_with_seq_norm_True_parallel_decode.parquet"

    # if split == "validation":
    #     data_path = "/p/liverobotics/Rui/gsm8k_tokenized_val.parquet"
    # else:
    #     data_path = "/p/liverobotics/Rui/gsm8k_tokenized_train.parquet"


    if split == "validation":
        data_path = dataset_config.train_path.replace("training", "validation")
    else:
        data_path = dataset_config.train_path

    DROP_COLUMNS = {"higher", "lower", "question", "answer", "raw_traj", "sid", "ego_id"}

    parquet_file = pq.ParquetFile(data_path)

    num_rows = parquet_file.metadata.num_rows
    if split == "validation":
        num_rows = num_rows // 40


    # batch_size = 1000
    # datasets = []

    # processed_rows = 0

    # pbar = tqdm.tqdm(
    #     parquet_file.iter_batches(batch_size=batch_size),
    #     total=(num_rows + batch_size - 1) // batch_size,
    #     desc="Building dataset",
    # )

    # for record_batch in pbar:
    #     if processed_rows >= num_rows:
    #         break

    #     keep_cols = [c for c in record_batch.schema.names if c not in DROP_COLUMNS]
    #     record_batch = record_batch.select(keep_cols)

    #     remaining = num_rows - processed_rows
    #     if record_batch.num_rows > remaining:
    #         record_batch = record_batch.slice(0, remaining)

    #     table = pa.Table.from_batches([record_batch])
    #     ds = Dataset(table)
    #     datasets.append(ds)

    #     processed_rows += record_batch.num_rows

    #     pbar.set_postfix(
    #         rows=processed_rows,
    #         mem=f"{mem_gb():.2f} GB",
    #     )

    # dataset = concatenate_datasets(datasets)    


    keep_cols = [name for name in parquet_file.schema_arrow.names if name not in DROP_COLUMNS]
    if num_rows == 0:
        return _empty_dataset_from_schema(parquet_file, keep_cols)
    try:
        dataset = _build_dataset_direct(parquet_file, keep_cols, num_rows)
    except OSError as direct_err:
        if "List index overflow" not in str(direct_err):
            raise

        warnings.warn(
            f"Direct parquet read failed with '{direct_err}'. "
            "Falling back to batched Arrow loading.",
            RuntimeWarning,
        )

        try:
            dataset = _build_dataset_batched(parquet_file, keep_cols, num_rows)
        except Exception as batched_err:
            warnings.warn(
                f"Batched Arrow loading also failed with '{batched_err}'. "
                "Falling back to pandas.read_parquet.",
                RuntimeWarning,
            )
            dataset = _build_dataset_pandas(data_path, keep_cols, num_rows)

    return dataset
