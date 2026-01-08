# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets
import pyarrow.parquet as pq
import pyarrow as pa
import tqdm
import psutil
import os
from datasets import Dataset, concatenate_datasets

import psutil
import os

process = psutil.Process(os.getpid())

def mem_gb():
    return process.memory_info().rss / (1024 ** 3)

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

    batch_size = 1000
    datasets = []

    parquet_file = pq.ParquetFile(data_path)

    num_rows = parquet_file.metadata.num_rows
    if split == "validation":
        num_rows = num_rows // 40

    processed_rows = 0

    pbar = tqdm.tqdm(
        parquet_file.iter_batches(batch_size=batch_size),
        total=(num_rows + batch_size - 1) // batch_size,
        desc="Building dataset",
    )

    for record_batch in pbar:
        if processed_rows >= num_rows:
            break

        keep_cols = [c for c in record_batch.schema.names if c not in DROP_COLUMNS]
        record_batch = record_batch.select(keep_cols)

        remaining = num_rows - processed_rows
        if record_batch.num_rows > remaining:
            record_batch = record_batch.slice(0, remaining)

        table = pa.Table.from_batches([record_batch])
        ds = Dataset(table)
        datasets.append(ds)

        processed_rows += record_batch.num_rows

        pbar.set_postfix(
            rows=processed_rows,
            mem=f"{mem_gb():.2f} GB",
        )

    dataset = concatenate_datasets(datasets)    

    return dataset
