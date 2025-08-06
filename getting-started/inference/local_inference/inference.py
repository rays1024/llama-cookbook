# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from collections import Counter
import os
import re
import sys
import time

import fire

import torch

from accelerate.utils import is_xpu_available
from llama_cookbook.inference.model_utils import load_model, load_peft_model

from llama_cookbook.inference.safety_utils import AgentType, get_safety_checker
from transformers import AutoTokenizer
from transformers import logging
logging.set_verbosity_error()

import datasets
from tqdm import tqdm

from openai import OpenAI
import json
from typing import List, Tuple

import pyarrow.parquet as pq
from datasets import Dataset

import gc    

# question_batch, higher_batch, lower_batch are optional
def save_to_jsonl(
    ground_truths: List[str],
    context_batch: List[str],
    llm_answers: List[str],
    output_file: str,
    sid_batch: List[str],
    ego_id_batch: List[str],
    question_batch: List[str] = None,
    higher_batch: List[str] = None,
    lower_batch: List[str] = None
):
    with open(output_file, 'a') as f:
        for idx in range(len(ground_truths)):
            entry = {
                'ground_truth': ground_truths[idx],
                'context': context_batch[idx],
                'llm_answer': llm_answers[idx],
                'sid': sid_batch[idx],
                'agent_id': ego_id_batch[idx],
            }
            if question_batch is not None:
                entry['question'] = question_batch[idx]
            if higher_batch is not None:
                entry['higher'] = higher_batch[idx]
            if lower_batch is not None:
                entry['lower'] = lower_batch[idx]
            f.write(json.dumps(entry) + '\n')

def save_attention_to_jsonl(
    top_k_indices: List[int],
    top_k_weights: List[float],
    output_file: str,
    sid_batch: List[str],
    ego_id_batch: List[str],
    input_ids: List[int] = None,
    question_batch: List[str] = None,
    higher_batch: List[str] = None,
    lower_batch: List[str] = None
):
    with open(output_file, 'a') as f:
        for i in range(len(top_k_indices)):
            idx_list = top_k_indices[i]
            weight_list = top_k_weights[i]
            sid = sid_batch[i]
            ego_id = ego_id_batch[i]
            entry = {
                'sid': sid,
                'ego_id': ego_id,
            }
            if question_batch is not None:
                entry['question'] = question_batch[i]
            if higher_batch is not None:
                entry['higher'] = higher_batch[i]
            if lower_batch is not None:
                entry['lower'] = lower_batch[i]
            if input_ids is not None:
                input_ids_list = input_ids[i].tolist()
                entry['input_ids'] = input_ids_list
            idx_weight_dict = {idx: weight_list[j] for j, idx in enumerate(idx_list)}
            entry['idx_weight_dict'] = idx_weight_dict
            f.write(json.dumps(entry) + '\n')

def main(
    model_name,
    peft_model: str = None,
    quantization: str = None, # Options: 4bit, 8bit
    max_new_tokens: int = 100,  # The maximum numbers of tokens to generate
    prompt_file: str = None,
    seed: int = 42,  # seed value for reproducibility
    do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 0.9,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 0.3,  # [optional] The value used to modulate the next token probabilities.
    top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1,  # [optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool = False,
    max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False,  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    share_gradio: bool = False,  # Enable endpoint creation for gradio.live
    **kwargs,
):
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ROAD_TYPE_TOKEN = [
        "LaneCenter-Freeway",
        "LaneCenter-SurfaceStreet",
        "RoadLine-BrokenSingleWhite",
        "RoadLine-SolidSingleWhite",
        "RoadLine-SolidDoubleWhite",
        "RoadLine-BrokenSingleYellow",
        "RoadLine-BrokenDoubleYellow",
        "Roadline-SolidSingleYellow",
        "Roadline-SolidDoubleYellow",
        "RoadLine-PassingDoubleYellow",
        "StopSign",
        "Crosswalk",
        "SpeedBump"
    ]

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

    # vel_type = [f'VEL_{round(i/10, 2)}' for i in list(range(0, 41))]
    acc_type = [f'ACC_{round(i, 3)}' for i in [x * 0.005 for x in range(-20, 21)]]
    len_type = [f'LEN_{round(i/10, 2)}' for i in list(range(0, 51, 5))]
    dir_type = [f'VEC_{i}' for i in range(360)]

    veh_vec = [f'VEH_VEC_{i}' for i in range(512)]
    ped_vec = [f'PED_VEC_{i}' for i in range(512)]
    cyc_vec = [f'CYCL_VEC_{i}' for i in range(512)]

    custom_tokens = []

    # custom_tokens.extend(vel_type)
    custom_tokens.extend(acc_type)
    custom_tokens.extend(len_type)
    custom_tokens.extend(dir_type)

    custom_tokens.extend(veh_vec)
    custom_tokens.extend(ped_vec)
    custom_tokens.extend(cyc_vec)

    # for l in len_type:
    #     for d in dir_type:
    #         custom_tokens.append(f'{d}{l}')

    # for v in vel_type:
    #     for d in dir_type:
    #         custom_tokens.append(f'{d}{v}')

    # for a in acc_type:
    #     for d in dir_type:
    #         custom_tokens.append(f'{d}{a}')

    custom_tokens.extend(ROAD_TYPE_TOKEN)

    # custom_tokens.append('<ROAD_START>')
    # custom_tokens.append('<ROAD_END>')
    # custom_tokens.append('<ROAD_VECTOR_START>')
    # custom_tokens.append('<ROAD_VECTOR_END>')
    # custom_tokens.append('AGENT_TRAJ_START')
    # custom_tokens.append('AGENT_TRAJ_END')
    custom_tokens.append('START_')
    custom_tokens.append('AGENT_ID_')
    custom_tokens.append('AGENT_TYPE_Vehicle')
    custom_tokens.append('AGENT_TYPE_Pedestrian')
    custom_tokens.append('AGENT_TYPE_Cyclist')
    custom_tokens.append('AGENT_TYPE_Other')
    custom_tokens.append('AGENT_TYPE_Unset')
    custom_tokens.append('TRAJ_NONE')
    custom_tokens.append('CTRL_NONE')
    custom_tokens.append('EGO_TRAJ_START')
    custom_tokens.append('EGO_TRAJ_END')
    custom_tokens.append('AGENT_TRAJ_START')
    custom_tokens.append('AGENT_TRAJ_END')
    custom_tokens.append('MAP_START')
    custom_tokens.append('MAP_END')
    custom_tokens.append('INITIAL_HEADING_')

    tokenizer.add_tokens(custom_tokens)

    output_file = "inference_result.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)
    attention_file = "attention_result.jsonl"
    if os.path.exists(attention_file):
        os.remove(attention_file)

    model = load_model(model_name, quantization, use_fast_kernels, **kwargs)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))


    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()


    def inference(
        batch,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        **kwargs,
    ):

        sid_batch = kwargs.pop('sid_batch', None)
        ego_id_batch = kwargs.pop('ego_id_batch', None)
        question_batch = kwargs.pop('question_batch', [])
        higher_batch = kwargs.pop('higher_batch', [])
        lower_batch = kwargs.pop('lower_batch', [])
        context_batch = kwargs.pop('context_batch', [])
        ground_truths = kwargs.pop('ground_truths', [])

        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                # output_attentions=True,
                # output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )

        # target_input_ids = tokenizer.encode("Instruction: The vehicle accelerates to a moderate speed, stays in its lane, approaches the intersection, and crosses the intersection.\n")

        sequences = outputs.sequences
        # attentions = outputs.attentions
        # attention_tensors = []
        # for attn in attentions:
        #     if isinstance(attn, tuple):
        #         attention_tensors.extend(attn)
        #     else:
        #         attention_tensors.append(attn)
        
        # # Only get the last layer's attention
        # shapes = [attn.shape for attn in attention_tensors]
        # most_common_shape = Counter(shapes).most_common(1)[0][0]
        # filtered_attentions = [attn for attn in attention_tensors if attn.shape == most_common_shape]
        
        # # Get only the last layer instead of stacking all layers
        # last_layer_attention = filtered_attentions[-1].detach().cpu()
        
        # # Process attention for all tokens in the sequence
        # batch_size, num_heads, seq_len, _ = last_layer_attention.shape
        
        # # Average attention weights across all heads
        # mean_attention = last_layer_attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
        


        # for b in range(batch_size):
        #     # Sum over attention *received* by each token (column-wise)
        #     token_attention_scores = mean_attention[b].sum(dim=0)  # [seq_len]
            
        #     # Get rankings for all tokens
        #     sorted_indices = torch.argsort(token_attention_scores, descending=True)
        #     rankings = torch.empty_like(sorted_indices)
        #     rankings[sorted_indices] = torch.arange(len(sorted_indices))
            
        #     # Find specific tokens in the input sequence and get their rankings
        #     input_ids_list = batch['input_ids'][b].tolist()
        #     token_rankings = []
            
        #     for token_id in target_input_ids:
        #         # Find all positions where this token appears
        #         positions = [i for i, x in enumerate(input_ids_list) if x == token_id]
                
        #         for pos in positions:
        #             if pos < len(token_attention_scores):
        #                 rank = rankings[pos].item() + 1  # +1 for 1-based ranking
        #                 attention_score = token_attention_scores[pos].item()
        #                 token_text = tokenizer.decode([token_id])
                        
        #                 token_rankings.append({
        #                     'token_id': token_id,
        #                     'token_text': token_text,
        #                     'position': pos,
        #                     'rank': rank,
        #                     'attention_score': attention_score,
        #                     'total_tokens': len(token_attention_scores)
        #                 })
            
        #     # Print the rankings
        #     print(f"Batch {b} - Token Rankings:")
        #     for item in token_rankings:
        #         print(f"Token '{item['token_text']}' (ID: {item['token_id']}) at position {item['position']}: "
        #             f"Rank {item['rank']}/{item['total_tokens']} (score: {item['attention_score']:.4f})")
        
        # exit()


        # K = 400
        # all_token_attentions = []  # to store top-K attention values
        # all_token_indices = []     # to store corresponding token indices

        # for b in range(batch_size):
        #     # Sum over attention *received* by each token (i.e., column-wise)
        #     token_attention_scores = mean_attention[b].sum(dim=0)  # [seq_len]
            
        #     # Get top K token indices
        #     topk_values, topk_indices = torch.topk(token_attention_scores, k=min(K, seq_len), largest=True)
        #     # normalize the attention weights
        #     topk_values = topk_values / topk_values.sum()
        #     topk_values = topk_values.tolist()
        #     topk_indices = topk_indices.tolist()

        #     # Append to lists
        #     all_token_attentions.append(topk_values)
        #     all_token_indices.append(topk_indices)
        
        # save_attention_to_jsonl(all_token_indices, all_token_attentions, attention_file, sid_batch, ego_id_batch, batch['input_ids'])

        llm_answer = []
        context_batch = []
        list_of_tokens = []
        for output_token in sequences:
            len_input_ids = len(batch['input_ids'][0].tolist())
            list_of_tokens.append(output_token[len_input_ids:].detach().cpu())
            input_ids = batch['input_ids'][0].tolist()
            extracted_content = tokenizer.decode(output_token[len_input_ids:], skip_special_tokens=True)
            context_tokens = tokenizer.decode(output_token[:len_input_ids], skip_special_tokens=True)
            context_batch.append(context_tokens)
            # Check if the input tokens contain "EGO_TRAJ_START"
            llm_answer.append(extracted_content)

        save_to_jsonl(ground_truths, context_batch, llm_answer, output_file, sid_batch, ego_id_batch)


    # data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_sampling_factor_5_10hz.parquet"
    # data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/hierarchical_reasoning_validation.parquet"
    # data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz.parquet"
    # data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/trimmed_combined_traj_prediction_10hz_long.parquet"
    # data_path = "/p/ruishen/processed_waymo_data/language_condition/validation/waymo_tokenized/combined_language_condition_10hz.parquet"
    # data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/combined_traj_qa_2hz.parquet"
    # data_path = "/p/ruishen/processed_waymo_data/test/waymo_tokenized/combined_traj_prediction_10hz.parquet"
    # data_path = "/p/ruishen/processed_waymo_data/training/waymo_tokenized/combined_traj_qa_10hz_long.parquet"
    # data_path = "/p/ruishen/processed_waymo_data/validation/waymo_tokenized/small_overfitting_10hz_long.parquet"
    data_path = "/p/ruishen/processed_waymo_data/test/waymo_tokenized/combined_traj_prediction_10hz.parquet"

    # custom_dataset = datasets.Dataset.from_parquet(data_path)

    table = pq.read_table(data_path)

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
    import random
    seed = 42
    num_samples = min(100, len(custom_dataset))
    random_rows = custom_dataset.shuffle(seed=seed)[:num_samples]
    batch_size = 1

    # target_sid, target_ego_id = "fe9abb8ae49ba98a", 2056
    target_sid, target_ego_id = random_rows['sid'][20], int(random_rows['ego_id'][20])
    print(f"Target SID: {target_sid}, Target Ego ID: {target_ego_id}")

    for i in tqdm(range(0, num_samples, batch_size), desc="Processing samples"):
        input_ids_batch = []
        attention_mask_batch = []
        ground_truths = []
        sid_batch = []
        ego_id_batch = []
        question_batch = []
        # higher_batch = []
        # lower_batch = []
        # context_batch = []

        for j in range(i, min(i+batch_size, num_samples)):
            # input_ids = random_rows['input_ids_a'][j]
            # attention_mask = random_rows['attention_mask_a'][j]
            # labels = random_rows['labels_a'][j]
            
            sid = random_rows['sid'][j]
            ego_id = random_rows['ego_id'][j]
            # question = random_rows['question'][j]
            # higher = random_rows['higher'][j]
            # lower = random_rows['lower'][j]

            input_ids = random_rows['input_ids'][j]
            attention_mask = random_rows['attention_mask'][j]
            labels = random_rows['labels'][j]


            # to_be_removed = "AGENT_TRAJ_STARTSTART_[725.056, 6424.752]INITIAL_HEADING_-0.036AGENT_ID_2056AGENT_TYPE_VehicleVEH_VEC_27VEH_VEC_72VEH_VEC_107VEH_VEC_295VEH_VEC_107VEH_VEC_65VEH_VEC_295VEH_VEC_107VEH_VEC_65VEH_VEC_4VEH_VEC_237VEH_VEC_107VEH_VEC_237VEH_VEC_107VEH_VEC_65VEH_VEC_65VEH_VEC_65VEH_VEC_65VEH_VEC_4VEH_VEC_4VEH_VEC_179VEH_VEC_4VEH_VEC_113VEH_VEC_65VEH_VEC_4VEH_VEC_22VEH_VEC_65VEH_VEC_22VEH_VEC_22VEH_VEC_66VEH_VEC_113VEH_VEC_196VEH_VEC_113VEH_VEC_89VEH_VEC_89VEH_VEC_66VEH_VEC_106VEH_VEC_238VEH_VEC_141VEH_VEC_141VEH_VEC_238VEH_VEC_141VEH_VEC_141VEH_VEC_71VEH_VEC_95VEH_VEC_136VEH_VEC_165VEH_VEC_11VEH_VEC_95VEH_VEC_165VEH_VEC_11VEH_VEC_95VEH_VEC_231VEH_VEC_33VEH_VEC_95VEH_VEC_367VEH_VEC_33VEH_VEC_95VEH_VEC_87VEH_VEC_95VEH_VEC_367VEH_VEC_95VEH_VEC_87VEH_VEC_95VEH_VEC_33VEH_VEC_367VEH_VEC_95VEH_VEC_87VEH_VEC_33VEH_VEC_95VEH_VEC_33VEH_VEC_367VEH_VEC_112VEH_VEC_367VEH_VEC_112VEH_VEC_367VEH_VEC_33VEH_VEC_112VEH_VEC_231VEH_VEC_33VEH_VEC_165VEH_VEC_112VEH_VEC_367VEH_VEC_112VEH_VEC_367VEH_VEC_95VEH_VEC_87VEH_VEC_33VEH_VEC_95VEH_VEC_367AGENT_TRAJ_END"
            # to_be_removed = tokenizer.encode(to_be_removed, add_special_tokens=False)
            # def remove_exact_sublist(sublist, mainlist):
            #     n = len(sublist)
            #     for i in range(len(mainlist) - n + 1):
            #         if mainlist[i:i+n] == sublist:
            #             return mainlist[:i] + mainlist[i+n:], i, i + n-1
            #     return mainlist, -1, -1
            # input_ids, start, end = remove_exact_sublist(to_be_removed, input_ids)
            # attention_mask = attention_mask[:start] + attention_mask[end+1:]
            # labels = labels[:start] + labels[end+1:]

            # if sid == target_sid and int(ego_id) == target_ego_id:
            #     new_instruction = "Instruction: The vehicle makes a right turn at the first intersection.\n"
            #     new_instruction = "\nPredict the future 80 steps of the ego vehicle trajectory.\n"
            #     new_instruction_ids = tokenizer.encode(new_instruction, add_special_tokens=False)
            #     old_start_idx = input_ids.index(17077)
            #     old_end_idx = len(input_ids) - 1 - input_ids[::-1].index(130229)
            #     input_ids = input_ids[:old_start_idx] + new_instruction_ids + input_ids[old_end_idx:]
            #     attention_mask = attention_mask[:old_start_idx] + [1] * len(new_instruction_ids) + attention_mask[old_end_idx:]
            #     labels = labels[:old_start_idx] + [-100] * len(new_instruction_ids) + labels[old_end_idx:]

            #     task_type = "<TRAJECTORY_PREDICTION>\n"
            #     # task_type = "<NATURAL_LANGUAGE_ANSWER>\n"
            #     task_type_ids = tokenizer.encode(task_type, add_special_tokens=False)
            #     old_start_idx = input_ids.index(27)
            #     old_end_idx = len(input_ids) - 1 - input_ids[::-1].index(397)
            #     input_ids = input_ids[:old_start_idx] + task_type_ids + input_ids[old_end_idx+1:]
            #     attention_mask = attention_mask[:old_start_idx] + [1] * len(task_type_ids) + attention_mask[old_end_idx+1:]
            #     labels = labels[:old_start_idx] + [-100] * len(task_type_ids) + labels[old_end_idx+1:]



            # remove the trailing padding
            input_ids = [x for x in input_ids if x != tokenizer.pad_token_id]
            attention_mask = [x for x in attention_mask if x != 0]
            labels = labels[:len(input_ids)]

            # Get indices for labels conditions
            indices_neg_100 = [i for i, label in enumerate(labels) if label == -100]
            indices_greater_0 = [i for i, label in enumerate(labels) if label > 0]

            # Extract input_ids and attention_mask for both conditions
            input_ids_neg_100 = [input_ids[i] for i in indices_neg_100]
            attention_mask_neg_100 = [attention_mask[i] for i in indices_neg_100]

            input_ids_greater_0 = [input_ids[i] for i in indices_greater_0]

            decoded_context = tokenizer.decode(input_ids_neg_100, skip_special_tokens=True)
            if "Question:" in decoded_context:
                ego_traj_start_input_id = tokenizer("Answer:", return_tensors="pt").input_ids[0][-1].item()
            else:
                ego_traj_start_input_id = tokenizer("EGO_TRAJ_START", return_tensors="pt").input_ids[0][-1].item()
            input_ids_neg_100.append(ego_traj_start_input_id)
            attention_mask_neg_100.append(1)

            # Store the batch data
            input_ids_batch.append(input_ids_neg_100)
            attention_mask_batch.append(attention_mask_neg_100)

            # Decode the ground truth for labels > 0
            ground_truth = tokenizer.decode(input_ids_greater_0, skip_special_tokens=True)
            ground_truths.append(ground_truth)

            # context_ids = torch.tensor(random_rows['context_ids_b'][j])
            # context_ids = context_ids[context_ids != tokenizer.pad_token_id]

            sid_batch.append(sid)
            ego_id_batch.append(ego_id)
            # question_batch.append(question)
            # higher_batch.append(higher)
            # lower_batch.append(lower)
            # context_batch.append(context_ids)
        
        # if sid != target_sid or int(ego_id) != target_ego_id:
        #     continue
        
        input_ids_batch = torch.tensor(input_ids_batch)
        attention_mask_batch = torch.tensor(attention_mask_batch)

        # # Pad input_ids and attention_mask to the same length
        # input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        #     [torch.tensor(ids) for ids in input_ids_batch], batch_first=True, padding_value=tokenizer.pad_token_id
        # )
        # attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        #     [torch.tensor(mask) for mask in attention_mask_batch], batch_first=True, padding_value=0
        # )

        # Create the batch dictionary
        batch = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch
        }

        kwargs = {
            'sid_batch': sid_batch,
            'ego_id_batch': ego_id_batch,
            'ground_truths': ground_truths,
            # 'question_batch': question_batch
            # 'higher_batch': higher_batch,
            # 'lower_batch': lower_batch,
            # 'context_batch': context_batch
        }

        for i in range(6):
            inference(batch, temperature, top_p, top_k, max_new_tokens, **kwargs)
        # inference(batch, temperature, top_p, top_k, max_new_tokens, **kwargs)

        # if sid == target_sid and int(ego_id) == target_ego_id:
        #     exit()


if __name__ == "__main__":
    fire.Fire(main)
