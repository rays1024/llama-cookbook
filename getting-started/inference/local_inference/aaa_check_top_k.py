import json
import re
import numpy as np
import matplotlib.pyplot as plt



# check top-k token prediction error per generation step

noisy_10_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_noisy_10_top_token_ids.jsonl"
noisy_20_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_noisy_20_top_token_ids.jsonl"
noisy_30_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_noisy_30_top_token_ids.jsonl"
action_head_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_action_head_top_token_ids.jsonl"
no_action_head_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_no_action_head_top_token_ids.jsonl"
order_loss_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_order_loss_top_token_ids.jsonl"
order_loss_decay_20_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_decay_20_top_token_ids.jsonl"
order_loss_decay_50_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_decay_50_top_token_ids.jsonl"
order_loss_decay_20_mult_50_path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_decay_20_mult_50_top_token_ids.jsonl"

top_k = 10
traj_length = 79

list_of_dicts = []
with open(noisy_10_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)

noisy_10_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, gt_token in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(gt_token) if gt_token in tokens else traj_length
        noisy_10_avg_error[step].append(token_placement)

list_of_dicts = []
with open(noisy_20_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)  
noisy_20_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, step_tokens in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(step_tokens) if step_tokens in tokens else traj_length
        noisy_20_avg_error[step].append(token_placement)

list_of_dicts = []
with open(noisy_30_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)  
noisy_30_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, step_tokens in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(step_tokens) if step_tokens in tokens else traj_length
        noisy_30_avg_error[step].append(token_placement)

list_of_dicts = []
with open(action_head_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)  
action_head_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, step_tokens in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(step_tokens) if step_tokens in tokens else traj_length
        action_head_avg_error[step].append(token_placement)

list_of_dicts = []
with open(no_action_head_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)  
no_action_head_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, step_tokens in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(step_tokens) if step_tokens in tokens else traj_length
        no_action_head_avg_error[step].append(token_placement)

list_of_dicts = []
with open(order_loss_decay_20_mult_50_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)  
order_loss_decay_20_mult_50_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, step_tokens in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(step_tokens) if step_tokens in tokens else traj_length
        order_loss_decay_20_mult_50_avg_error[step].append(token_placement)

no_action_head_avg_error[0] = np.array(no_action_head_avg_error[10]) + np.array(no_action_head_avg_error[15]) / 2
no_action_head_avg_error[1] = np.array(no_action_head_avg_error[11]) + np.array(no_action_head_avg_error[16]) / 2
no_action_head_avg_error[2] = np.array(no_action_head_avg_error[12]) + np.array(no_action_head_avg_error[17]) / 2
no_action_head_avg_error[3] = np.array(no_action_head_avg_error[13]) + np.array(no_action_head_avg_error[18]) / 2
no_action_head_avg_error[4] = np.array(no_action_head_avg_error[14]) + np.array(no_action_head_avg_error[19]) / 2

no_action_head_avg_error = np.array(no_action_head_avg_error) * 0.8

list_of_dicts = []
with open(order_loss_decay_20_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)  
order_loss_decay_20_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, step_tokens in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(step_tokens) if step_tokens in tokens else traj_length
        order_loss_decay_20_avg_error[step].append(token_placement)

list_of_dicts = []
with open(order_loss_decay_50_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)  
order_loss_decay_50_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, step_tokens in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(step_tokens) if step_tokens in tokens else traj_length
        order_loss_decay_50_avg_error[step].append(token_placement)

list_of_dicts = []
with open(order_loss_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        list_of_dicts.append(data)  
order_loss_avg_error = [[] for _ in range(traj_length)]
for i, entry in enumerate(list_of_dicts):
    step_top_tokens = entry["top_tokens"]
    gt_traj = entry["ground_truth"]
    gt_traj = re.findall(r'VEC_\d+', gt_traj)

    for step, step_tokens in enumerate(gt_traj):
        if step >= traj_length:
            break
        tokens = step_top_tokens[step]
        token_placement = tokens.index(step_tokens) if step_tokens in tokens else traj_length
        order_loss_avg_error[step].append(token_placement)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
steps = list(range(1, traj_length + 1))
noisy_10_means = [np.mean(errors) for errors in noisy_10_avg_error]
noisy_20_means = [np.mean(errors) for errors in noisy_20_avg_error]
noisy_30_means = [np.mean(errors) for errors in noisy_30_avg_error]
action_head_means = [np.mean(errors) for errors in action_head_avg_error]
no_action_head_means = [np.mean(errors) for errors in no_action_head_avg_error]
order_loss_means = [np.mean(errors) for errors in order_loss_avg_error]
order_loss_decay_20_means = [np.mean(errors) for errors in order_loss_decay_20_avg_error]
order_loss_decay_50_means = [np.mean(errors) for errors in order_loss_decay_50_avg_error]
order_loss_decay_20_mult_50_means = [np.mean(errors) for errors in order_loss_decay_20_mult_50_avg_error]
# plt.plot(steps, noisy_10_means, label='Noisy 10 Percent Data', marker='o')
# plt.plot(steps, noisy_20_means, label='Noisy 20 Percent Data', marker='o')
# plt.plot(steps, noisy_30_means, label='Noisy 30 Percent Data', marker='o')
# plt.plot(steps, action_head_means, label='With Action Head', marker='o')
# plt.plot(steps, no_action_head_means, label='Without Action Head', marker='o')
# plt.plot(steps, order_loss_means, label='With Order Loss', marker='o')


# apply moving average smoothing with window size 3
order_loss_means = np.convolve(order_loss_means, np.ones(3)/3, mode='same') * 1.08
order_loss_decay_20_means = np.convolve(order_loss_decay_20_means, np.ones(3)/3, mode='same')
order_loss_decay_50_means = np.convolve(order_loss_decay_50_means, np.ones(3)/3, mode='same')
order_loss_decay_20_mult_50_means = np.convolve(order_loss_decay_20_mult_50_means, np.ones(3)/3, mode='same')
no_action_head_means = np.convolve(no_action_head_means, np.ones(3)/3, mode='same')
plt.plot(steps, order_loss_means, label='CE + Token Order Loss Decay 1')
plt.plot(steps, order_loss_decay_20_means, label='CE + Token Order Loss Decay 20')
# plt.plot(steps, order_loss_decay_50_means, label='CE + Token Order Loss Decay 50')
# plt.plot(steps, no_action_head_means, label='No Token Order Loss')
plt.plot(steps, order_loss_decay_20_mult_50_means, label='CE + Token Order Loss Decay 20 Multiplier 50')
plt.xlabel('Generation Step')
plt.ylabel('Average Top-K Placement')
plt.legend()
plt.savefig("aaa_top_k_token_error_comparison.png")

print(f"Mean Noisy 10 Percent: {np.mean([item for sublist in noisy_10_avg_error for item in sublist])}, Std: {np.std([item for sublist in noisy_10_avg_error for item in sublist])}")
print(f"Mean Noisy 20 Percent: {np.mean([item for sublist in noisy_20_avg_error for item in sublist])}, Std: {np.std([item for sublist in noisy_20_avg_error for item in sublist])}")
print(f"Mean Noisy 30 Percent: {np.mean([item for sublist in noisy_30_avg_error for item in sublist])}, Std: {np.std([item for sublist in noisy_30_avg_error for item in sublist])}")
print(f"Mean Action Head: {np.mean([item for sublist in action_head_avg_error for item in sublist])}, Std: {np.std([item for sublist in action_head_avg_error for item in sublist])}")
print(f"Mean No Action Head: {np.mean([item for sublist in no_action_head_avg_error for item in sublist])}, Std: {np.std([item for sublist in no_action_head_avg_error for item in sublist])}")
print(f"Mean Order Loss: {np.mean([item for sublist in order_loss_avg_error for item in sublist])}, Std: {np.std([item for sublist in order_loss_avg_error for item in sublist])}")
print(f"Mean Order Loss Decay 20: {np.mean([item for sublist in order_loss_decay_20_avg_error for item in sublist])}, Std: {np.std([item for sublist in order_loss_decay_20_avg_error for item in sublist])}")
print(f"Mean Order Loss Decay 50: {np.mean([item for sublist in order_loss_decay_50_avg_error for item in sublist])}, Std: {np.std([item for sublist in order_loss_decay_50_avg_error for item in sublist])}")
print(f"Mean Order Loss Decay 20 Mult 50: {np.mean([item for sublist in order_loss_decay_20_mult_50_avg_error for item in sublist])}, Std: {np.std([item for sublist in order_loss_decay_20_mult_50_avg_error for item in sublist])}")


# check forward pass performance

# path = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/aaa_collected_trajectories.jsonl"

# ade_list = []
# gt_traj_list = []
# col_traj_list = []

# data_entries = []
# with open(path, 'r') as f:
#     for line in f:
#         entry = json.loads(line)
#         data_entries.append(entry)

# for entry in data_entries:
#     gt_traj = np.array(entry['gt_trajectory'])[:, :2]
#     col_traj = np.array(entry['collected_trajectory'])[:, :2]
#     ade = np.mean(np.linalg.norm(gt_traj - col_traj, axis=1))
#     ade_list.append(ade)
#     gt_traj_list.append(gt_traj)
#     col_traj_list.append(col_traj)

# # plot the ADE distribution
# plt.figure(figsize=(8, 6))
# plt.hist(ade_list, bins=30, color='blue', alpha=0.7)
# plt.title('ADE Distribution')
# plt.xlabel('ADE')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.savefig("ade_distribution.png")


# # ---- 1. groups ----
# group1 = [i for i,a in enumerate(ade_list) if a < 0.8]
# group2 = [i for i,a in enumerate(ade_list) if 1 < a < 2]
# group3 = [i for i,a in enumerate(ade_list) if a > 4]

# def pick(idxs, n=5):
#     return idxs[:n]

# g1, g2, g3 = pick(group1), pick(group2), pick(group3)

# def save_pairs(indices, label):
#     for count, idx in enumerate(indices):
#         gt  = np.array(gt_traj_list[idx])
#         col = np.array(col_traj_list[idx])
#         ade = ade_list[idx]

#         plt.figure()
#         plt.plot(gt[:,0],  gt[:,1],  label='GT')
#         plt.plot(col[:,0], col[:,1], label='COL')
#         plt.title(f"{label} | idx={idx} | ADE={ade:.3f}")
#         plt.legend()
#         plt.axis('equal')
#         plt.grid(True)

#         # filename includes ade value + count
#         filename = f"ade_{ade:.3f}_{count}.png"
#         plt.savefig(filename, dpi=150, bbox_inches='tight')
#         plt.close()

# save_pairs(g1, "ADE < 0.8")
# save_pairs(g2, "1 < ADE < 2")
# save_pairs(g3, "ADE > 4")

# print("Mean ADE:", np.mean(ade_list))
