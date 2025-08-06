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

    if split == "validation":
        data_path = data_path.replace("training", "validation")
        data_path = data_path.replace("sampling_factor_2", "sampling_factor_5")


    # dataset = datasets.Dataset.from_parquet(data_path)
    # dataset = dataset.remove_columns(["answer"])

    # if split == "training":
    #     map_qa = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_ego_centric_map_qa.parquet"
    #     map_qa = datasets.Dataset.from_parquet(map_qa)
    # elif split == "validation":
    #     map_qa = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_ego_centric_map_qa.parquet"
    #     map_qa = datasets.Dataset.from_parquet(map_qa)
    
    # map_qa = map_qa.remove_columns(["answer"])
    # # map_qa = map_qa.shuffle(seed=42).select(range(int(len(map_qa)/3)))
    # dataset = datasets.concatenate_datasets([dataset, map_qa])

    # dataset = dataset.shuffle(seed=42)

    # for testing purposes, only select a subset of the dataset
    # dataset = dataset.select(range(int(len(dataset)/4)))

    table = pq.read_table(data_path)
    num_rows = table.num_rows

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
