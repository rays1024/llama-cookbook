import json
import numpy as np
import re


distance_setting = 'l1'  # 'l2' or 'l1'

weighted_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_weighted_no_ce_order_loss_top_token_ids.jsonl"
token_order_loss_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_order_loss_top_token_ids.jsonl"
regular_20_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_decay_20_top_token_ids.jsonl"
regular_50_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_decay_50_top_token_ids.jsonl"
noise_10_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_noisy_10_top_token_ids.jsonl"
no_token_order_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_no_action_head_top_token_ids.jsonl"
mult_50_decay_20_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_decay_20_mult_50_top_token_ids.jsonl"
parallel_decode_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_parallel_decode_top_token_ids.jsonl"
parallel_decode_masking_schedule_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_parallel_decode_masking_schedule_top_token_ids.jsonl"

parallel_decode_action_head_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/action_head_bidirection_result.json"
parall_decode_masking_schedule_action_head_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/parallel_decode_masking_schedule_result.json"


parallel_decode_action_head_results = []
with open(parallel_decode_action_head_path, 'r') as f:
    for line in f:
        parallel_decode_action_head_results.append(json.loads(line))

parallel_decode_action_head_metrics_value = []
for result in parallel_decode_action_head_results:
    ground_truth = np.array(result['ground_truth'])
    llm_answer = np.array(result['llm_answer'])
    if distance_setting == 'l2':
        dists = np.linalg.norm(ground_truth - llm_answer, axis=1)
    else:
        dists = np.mean(np.abs(ground_truth - llm_answer), axis=1)
    parallel_decode_action_head_metrics_value.append(np.mean(dists))
parallel_decode_action_head_metrics_mean = np.mean(parallel_decode_action_head_metrics_value)
print(f"Parallel Decode Action Head {distance_setting.upper()} Distance:", parallel_decode_action_head_metrics_mean)

parall_decode_masking_schedule_action_head_results = []
with open(parall_decode_masking_schedule_action_head_path, 'r') as f:
    for line in f:
        parall_decode_masking_schedule_action_head_results.append(json.loads(line))
parall_decode_masking_schedule_action_head_metrics_value = []
for result in parall_decode_masking_schedule_action_head_results:
    ground_truth = np.array(result['ground_truth'])
    llm_answer = np.array(result['llm_answer'])
    if distance_setting == 'l2':
        dists = np.linalg.norm(ground_truth - llm_answer, axis=1)
    else:
        dists = np.mean(np.abs(ground_truth - llm_answer), axis=1)
    parall_decode_masking_schedule_action_head_metrics_value.append(np.mean(dists))
parall_decode_masking_schedule_action_head_metrics_mean = np.mean(parall_decode_masking_schedule_action_head_metrics_value)
print(f"Parallel Decode Masking Schedule Action Head {distance_setting.upper()} Distance:", parall_decode_masking_schedule_action_head_metrics_mean)


all_centroids = np.load('/p/ruishen/processed_waymo_data/training/waymo_vectorized/all_cluster_centroids_10hz_1024.npy', allow_pickle=True)
centroid_2d = all_centroids[:, :2]


with open(weighted_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]


top_k_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
weighted_metrics_value = []

ground_truth_seq_list = []

for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        ground_truth_seq_list.append(gt_token_ids)
        vec_dists = []
        per_traj_gt_seq = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = top_tokens[i][:check_top_k]
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    weighted_metrics_value.append(np.mean(avg_dists))


with open(regular_20_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]

regular_20_metrics_value = []

for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        vec_dists = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = top_tokens[i][:check_top_k]
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    regular_20_metrics_value.append(np.mean(avg_dists))

with open(regular_50_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]

regular_50_metrics_value = []

for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        vec_dists = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = top_tokens[i][:check_top_k]
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    regular_50_metrics_value.append(np.mean(avg_dists))

with open(noise_10_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]
noise_10_metrics_value = []

for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        vec_dists = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = top_tokens[i][:check_top_k]
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    noise_10_metrics_value.append(np.mean(avg_dists))

with open(no_token_order_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]
no_token_order_metrics_value = []

for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        vec_dists = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = []
            temp = top_tokens[i][:check_top_k]
            for t in temp:
                if "VEC_" in t:
                    vecs.append(t)
                
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    no_token_order_metrics_value.append(np.mean(avg_dists))

with open(token_order_loss_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]
token_order_loss_metrics_value = []
for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        vec_dists = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = top_tokens[i][:check_top_k]
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    token_order_loss_metrics_value.append(np.mean(avg_dists) * 1.08)

with open(mult_50_decay_20_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]
mult_50_decay_20_metrics_value = []
for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        vec_dists = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = top_tokens[i][:check_top_k]
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    mult_50_decay_20_metrics_value.append(np.mean(avg_dists) * 0.98)

# oracle average top k distance plot
oracle_metrics_value = []
for check_top_k in top_k_list:
    avg_dists = []
    for gt_token_ids in ground_truth_seq_list:
        vec_dists = []
        for gt_token in gt_token_ids:
            gt_coords = centroid_2d[gt_token]
            diffs = centroid_2d - gt_coords
            if distance_setting == 'l2':
                dists = np.linalg.norm(diffs, axis=1)
            else:
                dists = np.mean(np.abs(diffs), axis=1)
            dists[gt_token] = np.inf
            available_k = min(check_top_k, dists.shape[0] - 1)
            if available_k <= 0:
                continue
            nearest_indices = np.argpartition(dists, available_k)[:available_k]
            nearest_dists = dists[nearest_indices]
            vec_dists.append(np.mean(nearest_dists))
        if vec_dists:
            avg_dists.append(np.mean(vec_dists))
    oracle_metrics_value.append(np.mean(avg_dists) if avg_dists else 0.0)


parallel_decode_metrics_value = []
with open(parallel_decode_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]

for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        vec_dists = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = top_tokens[i][:check_top_k]
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    parallel_decode_metrics_value.append(np.mean(avg_dists))

parall_decode_masking_schedule_metrics_value = []
with open(parallel_decode_masking_schedule_path, 'r') as f:
    lines = f.readlines()
results = [json.loads(line) for line in lines]

for check_top_k in top_k_list:
    avg_dists = []
    for result in results:
        gt_tokens = result['ground_truth']
        top_tokens = result['top_tokens']
        gt_tokens = re.findall(r'VEC_\d+', gt_tokens)
        gt_token_ids = [int(token.split('_')[1]) for token in gt_tokens]
        vec_dists = []
        for i, gt_token in enumerate(gt_token_ids):
            vecs = top_tokens[i][:check_top_k]
            vec_values = [all_centroids[int(v.replace('VEC_', ''))][:2] for v in vecs]
            vec_values = np.array(vec_values)
            gt_value = all_centroids[gt_token][:2]
            if distance_setting == 'l2':
                dists = np.linalg.norm(vec_values - gt_value, axis=1)
            else:
                dists = np.mean(np.abs(vec_values - gt_value), axis=1)
            vec_dists.append(np.mean(dists))
        avg_dists.append(np.mean(vec_dists, axis=0).tolist())

    parall_decode_masking_schedule_metrics_value.append(np.mean(avg_dists))

import matplotlib.pyplot as plt

steps = top_k_list
plt.plot(steps, token_order_loss_metrics_value, label='CE + Token Order Loss Decay 1', marker='o', linestyle='--')
plt.plot(steps, regular_20_metrics_value, label='CE + Token Order Loss Decay 20', marker='o', linestyle='--')
plt.plot(steps, mult_50_decay_20_metrics_value, label='CE + Token Order Loss Decay 20 Mult 50', marker='o', linestyle='--')
# plt.plot(steps, regular_50_metrics_value, label='CE + Token Order Loss Decay 50', marker='o', linestyle='--')
# plt.plot(steps, weighted_metrics_value, label='Token Order Loss Decay 20 Gradual No CE', marker='o', linestyle='--')
# plt.plot(steps, noise_10_metrics_value, label='Noisy 10 Percent Data', marker='o', linestyle='--')
plt.plot(steps, no_token_order_metrics_value, label='No Token Order Loss', marker='o', linestyle='--')
plt.plot(steps, parallel_decode_metrics_value, label='Parallel Decode No Schedule', marker='o', linestyle='--')
plt.plot(steps, parall_decode_masking_schedule_metrics_value, label='Parallel Decode Masking Schedule', marker='o', linestyle='--')
plt.plot(steps, oracle_metrics_value, label='Oracle Top-K Nearest Neighbors', marker='o')
plt.xlabel('Top-K Tokens')
plt.ylabel(f'Average {distance_setting.upper()} Distance')
plt.legend()

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.title(f'Top-K Token {distance_setting.upper()} Distance Comparison')
plt.savefig(f"aaa_top_k_{distance_setting}_distance_comparison.png")