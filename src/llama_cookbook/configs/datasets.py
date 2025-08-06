# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_cookbook/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_cookbook/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/p/liverobotics/Rui/llama-cookbook/src/llama_cookbook/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "/p/liverobotics/Rui/llama-cookbook/getting-started/finetuning/datasets/custom_dataset.py"
    train_split: str = "training"
    test_split: str = "validation"
    data_path: str = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/trimmed_combined_sampling_factor_2.parquet"
    
@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"

@dataclass 
class ocrvqa_dataset:
    dataset: str = "ocrvqa_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = "/p/liverobotics/Rui/llama-cookbook/getting-started/finetuning/datasets/ocrvqa_dataset.py"