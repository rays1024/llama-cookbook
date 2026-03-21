# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib


import torch
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json


from llama_cookbook.model_checkpointing import save_fsdp_model_checkpoint_full, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint, save_model_checkpoint
from llama_cookbook.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_cookbook.utils.memory_utils import MemoryTrace
from llama_cookbook.utils.aux_loss import ade_loss, ce_loss_by_type, ade_loss_all_vec, multi_label_bce_loss
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_cookbook.utils.flop_utils import FlopMeasure
import torch.nn as nn

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predictions

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # if train_config.action_head:
    #     model.set_action_head_trainable(False)
    #     llm_trainable = True
    #     ah_trainable = False
    #     start_cotrain_step = len(train_dataloader) * 3/4
    steps_per_epoch = len(train_dataloader)

    past_val_losses = []
    masking_prob = 0.6
    last_change_step = 0
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        print(f"train_config.max_train_step: {train_config.max_train_step}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    sid = batch.pop("sid", None)
                    ego_id = batch.pop("ego_id", None)
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.vec_emb_model:
                            break # skip moving to device here, will be handled in the model forward
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                if batch[key] is None:
                                    continue
                                if key in ["pred_seq", "multi_label", "label_weight"]:
                                    batch[key] = torch.tensor(batch[key])
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            elif torch.cuda.is_available():
                                if batch[key] is None:
                                    continue
                                if key in ["pred_seq", "multi_label", "label_weight"]:
                                    batch[key] = torch.tensor(batch[key])
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        if 'input_ids_a' in batch:
                            outputs_a = model(input_ids=batch["input_ids_a"], attention_mask=batch["attention_mask_a"], labels=batch["labels_a"])
                            loss_a = outputs_a.loss

                            logits_a = outputs_a.logits.detach().float()
                            output_a_tokens = logits_a.argmax(dim=-1)

                            # for each entry in the batch, concat context_ids_b, output_a_tokens, prompt_ids_b, and gt_ids_b
                            input_ids_b = []
                            attention_mask_b = []
                            labels_b = []
                            for i in range(len(batch["context_ids_b"])):
                                context_ids_b = batch["context_ids_b"][i]
                                context_ids_b = context_ids_b[context_ids_b != tokenizer.pad_token_id]
                                output_a_tokens_i = output_a_tokens[i]
                                output_a_tokens_i = output_a_tokens_i[output_a_tokens_i != tokenizer.pad_token_id]
                                prompt_ids_b = batch["prompt_ids_b"][i]
                                prompt_ids_b = prompt_ids_b[prompt_ids_b != tokenizer.pad_token_id]
                                gt_ids_b = batch["gt_ids_b"][i]
                                gt_ids_b = gt_ids_b[gt_ids_b != tokenizer.pad_token_id]

                                # concat context + output + prompt + gt + eos
                                eos_tensor = torch.tensor([tokenizer.eos_token_id], dtype=torch.long).to(context_ids_b.device)
                                context_token = torch.cat((context_ids_b, output_a_tokens_i, prompt_ids_b), dim=0)
                                gt_ids_b = torch.cat((gt_ids_b, eos_tensor), dim=0)
                                input_id = torch.cat((context_token, gt_ids_b), dim=0)
                                input_ids_b.append(input_id)
                                attention_mask_b.append(torch.ones_like(input_id))
                                labels_b.append(torch.cat((torch.full_like(context_token, -100), gt_ids_b), dim=0))

                            input_ids_b = torch.nn.utils.rnn.pad_sequence(input_ids_b, batch_first=True)
                            attention_mask_b = torch.nn.utils.rnn.pad_sequence(attention_mask_b, batch_first=True)
                            labels_b = torch.nn.utils.rnn.pad_sequence(labels_b, batch_first=True)
                            
                            # move to device
                            if train_config.enable_fsdp:
                                if is_xpu_available():
                                    input_ids_b = input_ids_b.to(torch.device(f"xpu:{local_rank}"))
                                    attention_mask_b = attention_mask_b.to(torch.device(f"xpu:{local_rank}"))
                                    labels_b = labels_b.to(torch.device(f"xpu:{local_rank}"))
                                else:
                                    input_ids_b = input_ids_b.to(local_rank)
                                    attention_mask_b = attention_mask_b.to(local_rank)
                                    labels_b = labels_b.to(local_rank)
                            else:
                                if is_xpu_available():
                                    input_ids_b = input_ids_b.to('xpu:0')
                                    attention_mask_b = attention_mask_b.to('xpu:0')
                                    labels_b = labels_b.to('xpu:0')
                                elif torch.cuda.is_available():
                                    input_ids_b = input_ids_b.to('cuda:0')
                                    attention_mask_b = attention_mask_b.to('cuda:0')
                                    labels_b = labels_b.to('cuda:0')

                            # forward pass and compute
                            loss_b = model(input_ids=input_ids_b, attention_mask=attention_mask_b, labels=labels_b).loss

                            loss = loss_a + loss_b * train_config.loss_weight
                        else:
                            if 'identifier' in batch:
                                loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                                targets = batch["labels"]
                                mask = (targets != -100).float()
                                task_type = batch["identifier"] # [batch_size, seq_len]
                                per_token_loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                                per_sample_loss = per_token_loss.view(logits.size(0), -1)
                                sum_loss_per_sample = (per_sample_loss * mask).sum(dim=1)
                                count_per_sample = mask.sum(dim=1).clamp(min=1)
                                mean_loss_per_sample = sum_loss_per_sample / count_per_sample
                                task_type_sum = task_type.sum(dim=1) # [batch_size]
                                weights = torch.where(task_type_sum > 0, train_config.loss_weight, 1.0) # 0 is qa and 1 is traj
                                weighted_per_sample = mean_loss_per_sample * weights.squeeze(-1)
                                loss = weighted_per_sample.sum() / len(weighted_per_sample)
                                # get individual loss for each task type
                                loss_qa = mean_loss_per_sample[task_type_sum == 0].mean() if (task_type_sum == 0).any() else torch.tensor(0.0, device=mean_loss_per_sample.device)
                                loss_traj = mean_loss_per_sample[task_type_sum != 0].mean() if (task_type_sum != 0).any() else torch.tensor(0.0, device=mean_loss_per_sample.device)
                            elif train_config.action_head and not train_config.bidirectional_attention:
                                output = model(**batch, task='action')
                                loss = output.loss
                                action_loss = output.action_prediction_loss
                                ce_loss = output.cross_entropy_loss
                                vec_order_loss = output.vec_order_loss

                                loss = ce_loss + vec_order_loss

                                # if total_train_steps <= (len(train_dataloader) * 1/2):
                                #     loss = ce_loss + vec_order_loss
                                # elif total_train_steps <= (len(train_dataloader) * 3/4):
                                #     loss = action_loss
                                # elif (total_train_steps - start_cotrain_step) / (len(train_dataloader) * 0.5) <= 1.0:
                                #     progress = (step + epoch * len(train_dataloader) - start_cotrain_step) / (len(train_dataloader) * 0.5)
                                #     progress = max(0.0, min(1.0, progress))
                                #     action_loss_weight_scale = 1e-3 + progress * (1 - 1e-3)
                                #     loss = action_loss * action_loss_weight_scale + ce_loss + vec_order_loss
                            elif train_config.bidirectional_attention and not train_config.action_head and not train_config.vec_emb_model:
                                mask_type_labels = batch["labels"].clone()
                                mask_type_labels = torch.where(mask_type_labels == -100,
                                                               torch.tensor(1, device=mask_type_labels.device, dtype=mask_type_labels.dtype),
                                                                torch.tensor(2, device=mask_type_labels.device, dtype=mask_type_labels.dtype)) 
                                batch["mask_type_labels"] = mask_type_labels
                                loss = model(**batch, tokenizer=tokenizer).loss
                            elif train_config.bidirectional_attention and train_config.action_head and train_config.action_model_type == "default":
                                mask_type_labels = batch["labels"].clone()
                                mask_type_labels = torch.where(mask_type_labels == -100,
                                                               torch.tensor(1, device=mask_type_labels.device, dtype=mask_type_labels.dtype),
                                                                torch.tensor(2, device=mask_type_labels.device, dtype=mask_type_labels.dtype)) 
                                batch["mask_type_labels"] = mask_type_labels

                                if train_config.target_masking_schedule:
                                    if len(past_val_losses) == 0:
                                        masking_prob = 0.6
                                else:
                                    masking_prob = 1.0
                                mask_id = -1
                                target_mask = batch["labels"] != -100
                                random_tensor = torch.rand(batch["labels"].shape, device=batch["labels"].device)
                                mask_positions = (random_tensor < masking_prob) & target_mask
                                batch["input_ids"][mask_positions] = mask_id

                                output = model(**batch, tokenizer=tokenizer, task='action', loss_type='ade')
                                loss = output.loss
                                action_loss = output.action_prediction_loss
                                smoothness_loss = output.smoothness_loss
                                ce_loss = output.cross_entropy_loss
                                vec_order_loss = output.vec_order_loss

                                if train_config.multi_label_bce:
                                    mlbce_loss = multi_label_bce_loss(
                                        output.logits,
                                        batch["labels"],
                                        batch["multi_label"],
                                        batch["label_weight"],
                                        tokenizer=tokenizer,
                                        ignore_index=-100,
                                    )
                                    loss = mlbce_loss + action_loss

                                # loss = ce_loss
                                # loss = ce_loss + action_loss
                                loss = action_loss

                                # loss = action_loss + smoothness_loss * 1000
                            elif train_config.bidirectional_attention and train_config.action_head and train_config.action_model_type == "diffusion":
                                mask_type_labels = batch["labels"].clone()
                                mask_type_labels = torch.where(mask_type_labels == -100,
                                                               torch.tensor(1, device=mask_type_labels.device, dtype=mask_type_labels.dtype),
                                                                torch.tensor(2, device=mask_type_labels.device, dtype=mask_type_labels.dtype)) 
                                batch["mask_type_labels"] = mask_type_labels

                                if train_config.target_masking_schedule:
                                    if len(past_val_losses) == 0:
                                        masking_prob = 0.6
                                else:
                                    masking_prob = 1.0
                                mask_id = -1
                                target_mask = batch["labels"] != -100
                                random_tensor = torch.rand(batch["labels"].shape, device=batch["labels"].device)
                                mask_positions = (random_tensor < masking_prob) & target_mask
                                batch["input_ids"][mask_positions] = mask_id

                                output = model(**batch, tokenizer=tokenizer, task='action')
                                loss = output.action_loss

                            elif train_config.vec_emb_model:
                                output = model(map_payloads=batch['map_payloads'],
                                             trajectory_payloads=batch['trajectory_payloads'],
                                             pred_seq=batch['pred_seq'],
                                             loss_horizon=80)
                                loss = output.action_prediction_loss
                            elif train_config.multi_label_bce:
                                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                                logits = outputs.logits
                                mlbce_loss = multi_label_bce_loss(
                                    logits,
                                    batch["labels"],
                                    batch["multi_label"],
                                    batch["label_weight"],
                                    tokenizer=tokenizer,
                                    ignore_index=-100,
                                )
                                loss = mlbce_loss

                            else:
                                loss = model(**batch).loss

                                # # ce and ade loss only
                                # outputs = model(**batch)
                                # logits = outputs.logits
                                # labels = batch["labels"]
                                # aux_loss, ade, _ = ade_loss_all_vec(logits, top_k=10, sid=sid, ego_id=ego_id, weight=1.0, tokenizer=tokenizer, labels=labels)
                                # ce = outputs.loss
                                # aux_weight = 0.1
                                # # cur_step = step + epoch * len(train_dataloader)
                                # # aux_weight = min(1.0, 0.3 + (cur_step/2400) * 0.1)
                                # loss = (1-aux_weight) * ce + aux_weight * aux_loss

                                # traj_token_loss = ce_loss_by_type(logits, labels, tokenizer, ignore_index=-100, reduction="mean")

                                # if step % 600 == 0:
                                #     folder_name = (
                                #             train_config.dist_checkpoint_root_folder
                                #             + "/"
                                #             + train_config.dist_checkpoint_folder
                                #             + "-"
                                #             + train_config.model_name
                                #         )
                                #     if not os.path.exists(f"{folder_name}/logs"):
                                #         os.makedirs(f"{folder_name}/logs", exist_ok=True)
                                #     try:
                                #         if len(os.listdir(f"{folder_name}/logs")) > 30:
                                #             files = os.listdir(f"{folder_name}/logs")
                                #             files = [os.path.join(f"{folder_name}/logs", f) for f in files]
                                #             files.sort(key=os.path.getmtime)
                                #             for f in files[:3]:
                                #                 os.remove(f)
                                #         torch.save({
                                #             "epoch": epoch,
                                #             "step": step,
                                #             "logits": logits.detach().cpu(),
                                #             "labels": labels.detach().cpu(),
                                #         }, f"{folder_name}/logs/logits_labels_rank{local_rank}_epoch{epoch}_step{step}.pt")
                                #     except Exception as e:
                                #         pass

                    total_loss += loss.detach().float()
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()

                        # # check gradient and update
                        # for name, param in model.module.named_parameters():
                        #     # check action_decoder gradients
                        #     if param.grad is not None:
                        #         grad_mean = param.grad.abs().mean()
                        #         print(f"Step {step}, Param: {name}, Grad Mean: {grad_mean.item():.6f}")
                        #     else:
                        #         print(f"Step {step}, Param: {name}, Grad is None")
                        # breakpoint()

                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float() * gradient_accumulation_steps,
                            })
                        if "identifier" in batch:
                            wandb_run.log({
                                'train/loss_qa': loss_qa.detach().float(),
                                'train/loss_traj': loss_traj.detach().float(),
                            })
                        if "ce" in locals() and "ade" in locals():
                            wandb_run.log({
                                'train/aux_loss': aux_loss,
                                'train/ce': ce,
                                'train/ade': ade,
                            })
                        if "traj_token_loss" in locals():
                            wandb_run.log({
                                'train/vec_ce_loss': traj_token_loss["vec_loss"],
                                'train/len_ce_loss': traj_token_loss["len_loss"],
                                'train/pos_ce_loss': traj_token_loss["pos_loss"],
                                "train/ade_vec": ade_dict["vec_ade"],
                                "train/ade_len": ade_dict["len_ade"],
                            })
                        if "action_loss" in locals() and "ce_loss" in locals():
                            wandb_run.log({
                                'train/action_loss': action_loss.detach().float(),
                                'train/ce_loss': ce_loss.detach().float(),
                                'train/vec_order_loss': vec_order_loss.detach().float(),
                                'train/masking_prob': masking_prob,
                                # 'train/ml_bce_loss': mlbce_loss.detach().float() if 'mlbce_loss' in locals() else 0.0,
                                'train/smoothness_loss': smoothness_loss.detach().float() if smoothness_loss is not None else 0.0,
                            })
                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")


                    # if (step + epoch * len(train_dataloader)) % 300 == 0 and step > 0:
                    #     eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                    #     # Check for >20 subdirs and delete oldest if needed
                    #     output_dir = train_config.output_dir
                    #     if not os.path.exists(output_dir):
                    #         os.makedirs(output_dir, exist_ok=True)
                    #     subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
                    #     if len(subdirs) > 20:
                    #         oldest_dir = min(subdirs, key=lambda d: os.path.getmtime(os.path.join(output_dir, d)))
                    #         oldest_dir_path = os.path.join(output_dir, oldest_dir)
                    #         try:
                    #             import shutil
                    #             shutil.rmtree(oldest_dir_path)
                    #             print(f"Deleted oldest directory: {oldest_dir_path}")
                    #         except Exception as e:
                    #             print(f"Failed to delete {oldest_dir_path}: {e}")
                    #     epoch_output_dir = os.path.join(train_config.output_dir, f"step_{step + epoch * len(train_dataloader)}")
                    #     if not os.path.exists(epoch_output_dir):
                    #         os.makedirs(epoch_output_dir, exist_ok=True)
                    #     save_peft_checkpoint(model, epoch_output_dir)

                    # if train_config.action_head:
                    #     if (step + epoch * len(train_dataloader)) > (len(train_dataloader) * 1/2) and not ah_trainable:
                    #         model.set_action_head_trainable(True)
                    #         model.set_language_model_trainable(False)
                    #         llm_trainable = False
                    #         ah_trainable = True
                    #     if (step + epoch * len(train_dataloader)) > (len(train_dataloader) * 3/4) and not llm_trainable:
                    #         model.set_language_model_trainable(True)
                    #         llm_trainable = True

                    if (step + epoch * len(train_dataloader)) % 2500 == 0 and step > 0:
                        # run evaluation and save model
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                        past_val_losses.append(eval_epoch_loss)


                        if train_config.target_masking_schedule:
                            window_size = 4
                            min_history = window_size * 2
                            margin = 1.2

                            if len(past_val_losses) >= min_history:
                                recent = torch.tensor(past_val_losses[-window_size:])
                                older  = torch.tensor(past_val_losses[-min_history:-window_size])

                                recent_mean = recent.mean()
                                older_mean = older.mean()

                                # estimate noise from recent history
                                noise = torch.std(recent)

                                # estimate trend magnitude
                                trend = (recent_mean - older_mean).abs()

                                if trend < noise * margin:
                                    if len(past_val_losses) - last_change_step >= window_size:
                                        masking_prob = min(1.0, masking_prob + 0.1)
                                        last_change_step = len(past_val_losses)


                        print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                        print("=====================================================")
                        folder_name = (
                            train_config.dist_checkpoint_root_folder
                            + "/"
                            + train_config.dist_checkpoint_folder
                            + "-"
                            + train_config.model_name
                        )
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name, exist_ok=True)
                        subdirs = [d for d in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, d))]
                        if len(subdirs) > 10:
                            oldest_dir = min(subdirs, key=lambda d: os.path.getmtime(os.path.join(folder_name, d)))
                            oldest_dir_path = os.path.join(folder_name, oldest_dir)
                            try:
                                import shutil
                                shutil.rmtree(oldest_dir_path)
                                print(f"Deleted oldest directory: {oldest_dir_path}")
                            except Exception as e:
                                print(f"Failed to delete {oldest_dir_path}: {e}")
                        if not train_config.enable_fsdp:
                            # create epoch output dir
                            epoch_output_dir = os.path.join(train_config.output_dir, f"step_{step + epoch * len(train_dataloader)}")
                            if not os.path.exists(epoch_output_dir):
                                os.makedirs(epoch_output_dir, exist_ok=True)
                            save_model_checkpoint(model, epoch_output_dir)
                        else:
                            if train_config.save_optimizer:
                                save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer, epoch=step+epoch*len(train_dataloader))
                            else:
                                save_model_and_optimizer_sharded(model, rank, train_config, epoch=step+epoch*len(train_dataloader))

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        should_save_model = train_config.save_model
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            past_val_losses.append(eval_epoch_loss)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss
        
        should_save_model = True
        checkpoint_start_time = time.perf_counter()
        if should_save_model:
            if train_config.enable_fsdp:
                dist.barrier()
            if train_config.use_peft:
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"we are about to save the PEFT modules")
                else:
                    print(f"we are about to save the PEFT modules")
                
                epoch_output_dir = os.path.join(train_config.output_dir, f"epoch_{epoch}")
                if not os.path.exists(epoch_output_dir):
                    os.makedirs(epoch_output_dir, exist_ok=True)
                save_peft_checkpoint(model, epoch_output_dir)
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                else:
                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

            else:
                if not train_config.enable_fsdp:
                    # create epoch output dir
                    epoch_output_dir = os.path.join(train_config.output_dir, f"epoch_{epoch}")
                    if not os.path.exists(epoch_output_dir):
                        os.makedirs(epoch_output_dir, exist_ok=True)
                    save_model_checkpoint(model, epoch_output_dir)
                    
                elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
                    print("=====================================================")
                    save_fsdp_model_checkpoint_full(
                        model, optimizer, rank, train_config, epoch=epoch
                    )
                    
                    if train_config.save_optimizer:
                        print(" Saving the FSDP optimizer using FULL_STATE_DICT")
                        print("=====================================================")
                        save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                    
                elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:

                    if train_config.save_optimizer:
                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        print("=====================================================")
                        save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                    else:
                        print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                        print("=====================================================")
                        save_model_and_optimizer_sharded(model, rank, train_config, epoch=epoch)

                    
            if train_config.enable_fsdp:
                dist.barrier()
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    if train_config.action_head:
        eval_ade = 0.0
        eval_ce = 0.0
        eval_aux_loss = 0.0
        action_loss = 0.0
        ce_loss = 0.0
        vec_order_loss = 0.0
        mlbce_loss = 0.0
        smoothness_loss = 0.0
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            sid = batch.pop("sid", None)
            ego_id = batch.pop("ego_id", None)
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
                if train_config.vec_emb_model:
                    break # skip moving to device for vec emb model
                if train_config.enable_fsdp:
                    if batch[key] is None:
                        continue
                    if key in ["pred_seq", "multi_label", "label_weight"]:
                        batch[key] = torch.tensor(batch[key])
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        if batch[key] is None:
                            continue
                        if key in ["pred_seq", "multi_label", "label_weight"]:
                            batch[key] = torch.tensor(batch[key])
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                if 'input_ids_a' in batch:
                    outputs_a = model(input_ids=batch["input_ids_a"], attention_mask=batch["attention_mask_a"], labels=batch["labels_a"])
                    loss_a = outputs_a.loss

                    logits_a = outputs_a.logits.detach().float()
                    output_a_tokens = logits_a.argmax(dim=-1)

                    # for each entry in the batch, concat context_ids_b, output_a_tokens, prompt_ids_b, and gt_ids_b
                    input_ids_b = []
                    attention_mask_b = []
                    labels_b = []
                    for i in range(len(batch["context_ids_b"])):
                        context_ids_b = batch["context_ids_b"][i]
                        context_ids_b = context_ids_b[context_ids_b != tokenizer.pad_token_id]
                        output_a_tokens_i = output_a_tokens[i]
                        output_a_tokens_i = output_a_tokens_i[output_a_tokens_i != tokenizer.pad_token_id]
                        prompt_ids_b = batch["prompt_ids_b"][i]
                        prompt_ids_b = prompt_ids_b[prompt_ids_b != tokenizer.pad_token_id]
                        gt_ids_b = batch["gt_ids_b"][i]
                        gt_ids_b = gt_ids_b[gt_ids_b != tokenizer.pad_token_id]

                        # concat context + output + prompt + eos
                        eos_tensor = torch.tensor([tokenizer.eos_token_id], dtype=torch.long).to(context_ids_b.device)
                        context_token = torch.cat((context_ids_b, output_a_tokens_i, prompt_ids_b), dim=0)
                        gt_ids_b = torch.cat((gt_ids_b, eos_tensor), dim=0)
                        input_id = torch.cat((context_token, gt_ids_b), dim=0)
                        input_ids_b.append(input_id)
                        attention_mask_b.append(torch.ones_like(input_id))
                        labels_b.append(torch.cat((torch.full_like(context_token, -100), gt_ids_b), dim=0))

                    input_ids_b = torch.nn.utils.rnn.pad_sequence(input_ids_b, batch_first=True)
                    attention_mask_b = torch.nn.utils.rnn.pad_sequence(attention_mask_b, batch_first=True)
                    labels_b = torch.nn.utils.rnn.pad_sequence(labels_b, batch_first=True)
                    
                    # move to device
                    if train_config.enable_fsdp:
                        if is_xpu_available():
                            input_ids_b = input_ids_b.to(torch.device(f"xpu:{local_rank}"))
                            attention_mask_b = attention_mask_b.to(torch.device(f"xpu:{local_rank}"))
                            labels_b = labels_b.to(torch.device(f"xpu:{local_rank}"))
                        else:
                            input_ids_b = input_ids_b.to(local_rank)
                            attention_mask_b = attention_mask_b.to(local_rank)
                            labels_b = labels_b.to(local_rank)
                    else:
                        if is_xpu_available():
                            input_ids_b = input_ids_b.to('xpu:0')
                            attention_mask_b = attention_mask_b.to('xpu:0')
                            labels_b = labels_b.to('xpu:0')
                        elif torch.cuda.is_available():
                            input_ids_b = input_ids_b.to('cuda:0')
                            attention_mask_b = attention_mask_b.to('cuda:0')
                            labels_b = labels_b.to('cuda:0')

                    # forward pass and compute
                    loss_b = model(input_ids=input_ids_b, attention_mask=attention_mask_b, labels=labels_b).loss

                    loss = loss_a + loss_b * train_config.loss_weight
                else:
                    if "identifier" in batch:
                        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                        targets = batch["labels"]
                        mask = (targets != -100).float()
                        task_type = batch["identifier"]
                        per_token_loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                        per_sample_loss = per_token_loss.view(logits.size(0), -1)
                        sum_loss_per_sample = (per_sample_loss * mask).sum(dim=1)
                        count_per_sample = mask.sum(dim=1).clamp(min=1)
                        mean_loss_per_sample = sum_loss_per_sample / count_per_sample
                        task_type_sum = task_type.sum(dim=1) # [batch_size]
                        weights = torch.where(task_type_sum > 0, train_config.loss_weight, 1.0) # 0 is qa and 1 is traj
                        weighted_per_sample = mean_loss_per_sample * weights.squeeze(-1)
                        loss = weighted_per_sample.sum() / len(weighted_per_sample)
                        # get individual loss for each task type
                        loss_qa = mean_loss_per_sample[task_type_sum == 0].mean() if (task_type_sum == 0).any() else torch.tensor(0.0, device=mean_loss_per_sample.device)
                        loss_traj = mean_loss_per_sample[task_type_sum != 0].mean() if (task_type_sum != 0).any() else torch.tensor(0.0, device=mean_loss_per_sample.device)
                    elif train_config.action_head and not train_config.bidirectional_attention:
                        output = model(**batch, task='action')
                        loss = output.loss
                        action_loss += output.action_prediction_loss
                        ce_loss += output.cross_entropy_loss
                        vec_order_loss += output.vec_order_loss
                    elif train_config.bidirectional_attention and not train_config.action_head and not train_config.vec_emb_model:
                        mask_type_labels = batch["labels"].clone()
                        mask_type_labels = torch.where(mask_type_labels == -100,
                                                       torch.tensor(1, device=mask_type_labels.device, dtype=mask_type_labels.dtype),
                                                        torch.tensor(2, device=mask_type_labels.device, dtype=mask_type_labels.dtype)) 
                        batch["mask_type_labels"] = mask_type_labels
                        loss = model(**batch, tokenizer=tokenizer).loss
                    elif train_config.bidirectional_attention and train_config.action_head and train_config.action_model_type == "default":
                        mask_type_labels = batch["labels"].clone()
                        mask_type_labels = torch.where(mask_type_labels == -100,
                                                        torch.tensor(1, device=mask_type_labels.device, dtype=mask_type_labels.dtype),
                                                        torch.tensor(2, device=mask_type_labels.device, dtype=mask_type_labels.dtype)) 
                        batch["mask_type_labels"] = mask_type_labels
                        masking_prob = 1.0
                        mask_id = -1
                        target_mask = batch["labels"] != -100
                        random_tensor = torch.rand(batch["labels"].shape, device=batch["labels"].device)
                        mask_positions = (random_tensor < masking_prob) & target_mask
                        batch["input_ids"][mask_positions] = mask_id
                        output = model(**batch, tokenizer=tokenizer, task='action', loss_type='ade')
                        action_loss += output.action_prediction_loss
                        ce_loss += output.cross_entropy_loss
                        vec_order_loss += output.vec_order_loss

                        smoothness_loss += output.smoothness_loss
                        loss = output.action_prediction_loss

                        if train_config.multi_label_bce:
                            logits = output.logits
                            mlbce_loss += multi_label_bce_loss(
                                logits,
                                batch["labels"],
                                batch["multi_label"],
                                batch["label_weight"],
                                tokenizer=tokenizer,
                                ignore_index=-100,
                            )
                    elif train_config.bidirectional_attention and train_config.action_head and train_config.action_model_type == "diffusion":
                        mask_type_labels = batch["labels"].clone()
                        mask_type_labels = torch.where(mask_type_labels == -100,
                                                        torch.tensor(1, device=mask_type_labels.device, dtype=mask_type_labels.dtype),
                                                        torch.tensor(2, device=mask_type_labels.device, dtype=mask_type_labels.dtype)) 
                        batch["mask_type_labels"] = mask_type_labels

                        masking_prob = 1.0
                        mask_id = -1
                        target_mask = batch["labels"] != -100
                        random_tensor = torch.rand(batch["labels"].shape, device=batch["labels"].device)
                        mask_positions = (random_tensor < masking_prob) & target_mask
                        batch["input_ids"][mask_positions] = mask_id

                        output = model(**batch, tokenizer=tokenizer, task='action')
                        loss = output.action_loss

                    elif train_config.vec_emb_model:
                        output = model(map_payloads=batch['map_payloads'],
                                     trajectory_payloads=batch['trajectory_payloads'],
                                     pred_seq=batch['pred_seq'],
                                     loss_horizon=80)
                        loss = output.action_prediction_loss
                    elif train_config.multi_label_bce:
                        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                        logits = output.logits
                        mlbce_loss = multi_label_bce_loss(
                            logits,
                            batch["labels"],
                            batch["multi_label"],
                            batch["label_weight"],
                            tokenizer=tokenizer,
                            ignore_index=-100,
                        )
                        loss = mlbce_loss
                    else:
                        loss = model(**batch).loss
                        # if train_config.get("multi_class_ce", False):
                        #     loss = multi_class_ce(output.logits, batch["labels"])
                        # else:
                        #     loss = output.loss

                        # # ce and ade loss only
                        # outputs = model(**batch)
                        # logits = outputs.logits
                        # labels = batch["labels"]
                        # aux_loss, ade, _ = ade_loss_all_vec(logits, top_k=10, sid=sid, ego_id=ego_id, weight=1.0, tokenizer=tokenizer, labels=labels)
                        # ce = outputs.loss
                        # aux_weight = 0.1
                        # loss = (1-aux_weight) * ce + aux_weight * aux_loss
                        # eval_ade += ade.detach().float()
                        # eval_ce += ce.detach().float()
                        # eval_aux_loss += aux_loss.detach().float()
                        # traj_token_loss = ce_loss_by_type(logits, labels, tokenizer, ignore_index=-100, reduction="mean")
            
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()

            # # Decode predictions and add to evaluation predictions list
            # preds = torch.argmax(outputs.logits, -1)
            # eval_preds.extend(
            #     tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            # )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        if train_config.action_head and isinstance(action_loss, torch.Tensor):
            dist.all_reduce(action_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(ce_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(vec_order_loss, op=dist.ReduceOp.SUM)
            # dist.all_reduce(mlbce_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(smoothness_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.action_head and isinstance(action_loss, torch.Tensor):
        eval_ade = eval_ade / len(eval_dataloader)
        eval_ce = eval_ce / len(eval_dataloader)
        eval_aux_loss = eval_aux_loss / len(eval_dataloader)
        action_loss = action_loss / len(eval_dataloader)
        ce_loss = ce_loss / len(eval_dataloader)
        vec_order_loss = vec_order_loss / len(eval_dataloader)
        # mlbce_loss = mlbce_loss / len(eval_dataloader)
        smoothness_loss = smoothness_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
        if train_config.action_head and isinstance(action_loss, torch.Tensor):
            eval_ade = eval_ade / world_size
            eval_ce = eval_ce / world_size
            eval_aux_loss = eval_aux_loss / world_size
            action_loss = action_loss / world_size
            ce_loss = ce_loss / world_size
            vec_order_loss = vec_order_loss / world_size
            # mlbce_loss = mlbce_loss / world_size
            smoothness_loss = smoothness_loss / world_size

    if train_config.action_head and isinstance(action_loss, torch.Tensor):
        eval_ppl = torch.exp(ce_loss)
    else:
        eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)
        if "identifier" in batch:
            wandb_run.log({
                'eval/loss_qa': loss_qa.detach().float(),
                'eval/loss_traj': loss_traj.detach().float(),
            }, commit=False)
        if "ce" in locals() and "ade" in locals():
            wandb_run.log({
                'eval/aux_loss': eval_aux_loss,
                'eval/ce': eval_ce,
                'eval/ade': eval_ade,
            }, commit=False)
        if "traj_token_loss" in locals():
            wandb_run.log({
                'eval/vec_ce_loss': traj_token_loss["vec_loss"],
                'eval/len_ce_loss': traj_token_loss["len_loss"],
                'eval/pos_ce_loss': traj_token_loss["pos_loss"],
                "eval/ade_vec": ade_dict["vec_ade"],
                "eval/ade_len": ade_dict["len_ade"],
            }, commit=False)
        if "action_loss" in locals() and "ce_loss" in locals():
            wandb_run.log({
                'eval/action_loss': action_loss.detach().float() if isinstance(action_loss, torch.Tensor) else 0.0,
                'eval/ce_loss': ce_loss.detach().float() if isinstance(ce_loss, torch.Tensor) else 0.0,
                'eval/vec_order_loss': vec_order_loss.detach().float() if isinstance(vec_order_loss, torch.Tensor) else 0.0,
                # 'eval/ml_bce_loss': mlbce_loss.detach().float() if 'mlbce_loss' in locals() else 0.0,
                'eval/smoothness_loss': smoothness_loss.detach().float() if isinstance(smoothness_loss, torch.Tensor) else 0.0,
            }, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False
                    
def freeze_LLM_only(model):
    """
    Freeze self-attention layers in the language_model. vision_model, multi_modal_projector, and cross-attention layers will be fine-tuned
    """
    for name, param in model.language_model.named_parameters():
                param.requires_grad = False
    for i, layer in enumerate(model.language_model.model.layers):
        if i in model.language_model.model.cross_attention_layers:
            for param in layer.parameters():
                param.requires_grad = True

def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only available in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")

def print_frozen_model_status(model, config, rank: int = 0) -> None:
    """
    Print the frozen status of the model's and the number of trainable parameters after frozen.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("After freezing the model:")
        print(f"--> {config.model_name} has {trainable_params / 1e6} Million trainable params\n")

        module_states = {}
        # Iterate over all parameters
        for name, param in model.named_parameters():
            # Extract the top-level module name (e.g., "vision_model", "language_model")
            top_module = name.split(".")[0]

            # Initialize a record for the top-level module
            if top_module not in module_states:
                module_states[top_module] = {"frozen": [], "unfrozen": []}

            # Group parameters into frozen or unfrozen
            if param.requires_grad:
                module_states[top_module]["unfrozen"].append(name)
            else:
                module_states[top_module]["frozen"].append(name)

        print("--> Model state after freezing:")
        # Analyze and print the results
        for module, states in module_states.items():
            frozen_params = states["frozen"]
            unfrozen_params = states["unfrozen"]

            if frozen_params and unfrozen_params:
                # Mixed state: both frozen and unfrozen parameters
                print(f"    {module}: Mixed")
            elif frozen_params:
                # All parameters are frozen
                print(f"    {module}: Frozen")
            else:
                # All parameters are unfrozen
                print(f"    {module}: Unfrozen")
        print("")


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (following FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
