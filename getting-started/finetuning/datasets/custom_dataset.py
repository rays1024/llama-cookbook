# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets
import pyarrow.parquet as pq
import tqdm
from datasets import Dataset

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

    if split == "validation":
        data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_traj_qa_10hz_long.parquet"
    else:
        data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_traj_qa_10hz_long.parquet"
    
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


    table = pq.read_table(data_path)

    # if split == "validation":
    #     # sample a validation set
    #     table = table.slice(0, int(0.1 * table.num_rows))
    # else:
    #     # sample a training set
    #     table = table.slice(int(0.1 * table.num_rows), int(0.9 * table.num_rows))

    # # for hierarchical reasoning, need to remove the columns that are not used
    table = table.drop([key for key in table.column_names if key in ["higher", "lower", "sid", "ego_id", "question", "answer"]])

    num_rows = table.num_rows
    num_rows = num_rows

    # Step 2: Use tqdm to visualize loading progress
    batch_size = 1000
    rows = []
    for i in tqdm.tqdm(range(0, num_rows, batch_size), desc="Building dataset"):
        batch = table.slice(i, batch_size)
        batch_dict = batch.to_pydict()
        # Reorganize row-wise
        for j in range(len(batch_dict[next(iter(batch_dict))])):
            rows.append({k: batch_dict[k][j] for k in batch_dict})

    # Step 3: Wrap into HuggingFace Dataset
    dataset = Dataset.from_list(rows)

    return dataset
