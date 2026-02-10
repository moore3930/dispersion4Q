# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
import torch.nn as nn
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
import torch.nn.functional as F


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json


from llama_recipes.model_checkpointing import save_fsdp_model_checkpoint_full, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint, save_model_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
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


def get_value_loss(logits, scores, labels, tokenizer, xpo_hyper, beta):
    # TODO, Test if a shift is needed
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    labels[labels == -100] = tokenizer.pad_token_id
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    loss_mask = labels != tokenizer.pad_token_id
    log_prob = (per_token_logps * loss_mask).sum(-1)

    average_log_prob = beta * (log_prob / loss_mask.sum(-1) + xpo_hyper)
    predicted_probs = torch.sigmoid(average_log_prob)
    # Try
    weights = torch.where(scores == 1, torch.tensor(2.0), torch.tensor(1.0))
    scores = scores.to(predicted_probs.dtype)

    value_loss = F.binary_cross_entropy(predicted_probs, scores, weight=weights, reduction='mean')
    return value_loss, average_log_prob


def get_calibration_loss(logits, scores, labels, tokenizer, config):
    def get_pearson_correlation_per_batch(z_prob, z_score):
        # Compute covariance between z_prob and z_score for each batch (dim=1)
        mean_z_prob = z_prob.mean(dim=1, keepdim=True)
        mean_z_score = z_score.mean(dim=1, keepdim=True)

        # Compute the covariance for each batch (dim=1)
        cov_z_prob_z_score = ((z_prob - mean_z_prob) * (z_score - mean_z_score)).mean(dim=1)

        # Compute standard deviations for each batch (dim=1)
        std_z_prob = z_prob.std(dim=1)
        std_z_score = z_score.std(dim=1)

        # Pearson correlation coefficient per batch
        pearson_corr_per_batch = cov_z_prob_z_score / (std_z_prob * std_z_score + 1e-6)

        # Return the average Pearson correlation over the batch
        return pearson_corr_per_batch.mean()

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    labels[labels == -100] = tokenizer.pad_token_id
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    loss_mask = labels != tokenizer.pad_token_id
    log_prob = (per_token_logps * loss_mask).sum(-1)                            # [N]
    norm_log_prob = log_prob / loss_mask.sum(-1)                                # [N]

    list_size = config.list_size
    bon_size = config.bon_size

    norm_log_prob_grouped = norm_log_prob.view(-1, list_size)[:, :bon_size]     # [N, BoN_size]
    log_prob_grouped = log_prob.view(-1, list_size)[:, :bon_size]               # [N, BoN_size]

    # get z scores
    if config.normalize_prob:
        prob_mean = norm_log_prob_grouped.mean(dim=1, keepdim=True)             # [N, 1]
        prob_std = norm_log_prob_grouped.std(dim=1, keepdim=True) + 1e-6        # [N, 1]
        z_prob = (norm_log_prob_grouped - prob_mean) / prob_std                 # [N, BoN_size]
    else:
        prob_mean = log_prob_grouped.mean(dim=1, keepdim=True)                  # [N, 1]
        prob_std = log_prob_grouped.std(dim=1, keepdim=True) + 1e-6             # [N, 1]
        z_prob = (log_prob_grouped - prob_mean) / prob_std                      # [N, BoN_size]

    scores = scores.to(z_prob.dtype)
    scores_grouped = scores.view(-1, list_size)[:, :bon_size]                   # [N, BoN_size]
    scores_mean = scores_grouped.mean(dim=1, keepdim=True)                      # [N, 1]
    scores_std = scores_grouped.std(dim=1, keepdim=True) + 1e-6                 # [N, 1]
    z_score = (scores_grouped - scores_mean) / scores_std                       # [N, BoN_size]

    # get calibration loss
    loss = -(z_prob * z_score).mean()
    pearson = get_pearson_correlation_per_batch(z_prob, z_score)

    # Select the log_prob corresponding to the maximum z_score per batch
    max_z_score_indices = z_score.argmax(dim=1)  # Get the index of the maximum z_score for each batch
    nll_loss = -norm_log_prob_grouped.gather(1, max_z_score_indices.unsqueeze(1)).mean()  # NLL using max z_score

    min_z_score_indices = z_score.argmin(dim=1)  # Get the index of the maximum z_score for each batch
    reject_nll_loss = -norm_log_prob_grouped.gather(1, min_z_score_indices.unsqueeze(1)).mean()  # NLL using min z_score

    # CPO loss
    log_prob_grouped = log_prob.view(-1, list_size)[:, :bon_size]
    chose_nll = log_prob_grouped.gather(1, max_z_score_indices.unsqueeze(1))
    reject_nll = log_prob_grouped.gather(1, min_z_score_indices.unsqueeze(1))
    cpo_loss = -F.logsigmoid(0.1 * chose_nll - 0.1 * reject_nll).mean()
    win_rate = (chose_nll > reject_nll).float().mean().item()

    return loss, pearson, nll_loss, reject_nll_loss, cpo_loss


def get_calibration_loss_old(logits, scores, labels, tokenizer, config):
    def get_pearson_correlation_per_batch(z_prob, z_score):
        # Compute covariance between z_prob and z_score for each batch (dim=1)
        mean_z_prob = z_prob.mean(dim=1, keepdim=True)
        mean_z_score = z_score.mean(dim=1, keepdim=True)

        # Compute the covariance for each batch (dim=1)
        cov_z_prob_z_score = ((z_prob - mean_z_prob) * (z_score - mean_z_score)).mean(dim=1)

        # Compute standard deviations for each batch (dim=1)
        std_z_prob = z_prob.std(dim=1)
        std_z_score = z_score.std(dim=1)

        # Pearson correlation coefficient per batch
        pearson_corr_per_batch = cov_z_prob_z_score / (std_z_prob * std_z_score + 1e-6)

        # Return the average Pearson correlation over the batch
        return pearson_corr_per_batch.mean()

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    labels[labels == -100] = tokenizer.pad_token_id
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    loss_mask = labels != tokenizer.pad_token_id
    log_prob = (per_token_logps * loss_mask).sum(-1)    # [N]
    norm_log_prob = log_prob / loss_mask.sum(-1)        # [N]

    list_size = config.list_size
    norm_log_prob_grouped = norm_log_prob.view(-1, list_size)  # [N, 3]
    log_prob_grouped = log_prob.view(-1, list_size)     # [N, 3]

    # get z scores
    if config.normalize_prob:
        prob_mean = norm_log_prob_grouped.mean(dim=1, keepdim=True)         # [N, 1]
        prob_std = norm_log_prob_grouped.std(dim=1, keepdim=True) + 1e-6    # [N, 1]
        z_prob = (norm_log_prob_grouped - prob_mean) / prob_std             # [N, 3]
    else:
        prob_mean = log_prob_grouped.mean(dim=1, keepdim=True)         # [N, 1]
        prob_std = log_prob_grouped.std(dim=1, keepdim=True) + 1e-6    # [N, 1]
        z_prob = (log_prob_grouped - prob_mean) / prob_std             # [N, 3]

    scores = scores.to(z_prob.dtype)
    scores_grouped = scores.view(-1, list_size)                         # [N, 3]
    scores_mean = scores_grouped.mean(dim=1, keepdim=True)              # [N, 1]
    scores_std = scores_grouped.std(dim=1, keepdim=True) + 1e-6         # [N, 1]
    z_score = (scores_grouped - scores_mean) / scores_std               # [N, 3]

    # get calibration loss
    loss = -(z_prob * z_score).mean()
    pearson = get_pearson_correlation_per_batch(z_prob, z_score)

    # Select the log_prob corresponding to the maximum z_score per batch
    max_z_score_indices = z_score.argmax(dim=1)  # Get the index of the maximum z_score for each batch
    nll_loss = -norm_log_prob_grouped.gather(1, max_z_score_indices.unsqueeze(1)).mean()  # NLL using max z_score

    min_z_score_indices = z_score.argmin(dim=1)  # Get the index of the maximum z_score for each batch
    reject_nll_loss = -norm_log_prob_grouped.gather(1, min_z_score_indices.unsqueeze(1)).mean()  # NLL using min z_score

    # CPO loss
    log_prob_grouped = log_prob.view(-1, list_size)
    chose_nll = log_prob_grouped.gather(1, max_z_score_indices.unsqueeze(1))
    reject_nll = log_prob_grouped.gather(1, min_z_score_indices.unsqueeze(1))
    cpo_loss = -F.logsigmoid(0.1 * chose_nll - 0.1 * reject_nll).mean()
    win_rate = (chose_nll > reject_nll).float().mean().item()

    return loss, pearson, nll_loss, reject_nll_loss, cpo_loss


def get_listwise_loss(logits, scores, labels, tokenizer, xpo_hyper, beta, list_size):
    def calculate_kl_divergence_loss(scores, predicted_probs, temperature=1.0):
        # 检查能否 reshape 为 (N/3, 3)
        if scores.numel() % 3 != 0 or predicted_probs.numel() % 3 != 0:
            raise ValueError("scores 和 predicted_probs 的长度必须是 3 的倍数")

        # Reshape 成 (N/3, 3)
        reshaped_scores = scores.view(-1, 3)
        reshaped_predicted_probs = predicted_probs.view(-1, 3)

        # 计算 softmax 归一化
        true_probs = F.softmax(reshaped_scores / temperature, dim=-1)  # 真实分数 softmax
        pred_probs = F.softmax(reshaped_predicted_probs / temperature, dim=-1)  # 预测分数 softmax

        # 计算 KL 散度
        kl_loss = F.kl_div(pred_probs.log(), true_probs, reduction='batchmean')  # 使用 batchmean 进行归一化

        return kl_loss

    def soft_rank_loss(scores, predicted_probs, temperature=1.0):
        scores = scores.unsqueeze(0) if scores.ndim == 1 else scores
        predicted_probs = predicted_probs.unsqueeze(0) if predicted_probs.ndim == 1 else predicted_probs
        scores = scores.unsqueeze(0) if scores.ndim == 1 else scores
        pairwise_diff = scores - scores.T
        pairwise_prob = torch.sigmoid(pairwise_diff / temperature)
        soft_ranks = pairwise_prob.sum(dim=-1)

        return soft_ranks

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    labels[labels == -100] = tokenizer.pad_token_id
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    loss_mask = labels != tokenizer.pad_token_id
    log_prob = (per_token_logps * loss_mask).sum(-1)

    average_log_prob = beta * (log_prob / loss_mask.sum(-1) + xpo_hyper)
    predicted_probs = torch.sigmoid(average_log_prob)
    scores = scores.to(predicted_probs.dtype)

    value_loss = F.binary_cross_entropy(predicted_probs, scores, reduction='mean')

    # get listwise loss
    reshaped_scores = scores.view(-1, 3)
    reshaped_predicted_probs = predicted_probs.view(-1, 3)
    kl_loss = calculate_kl_divergence_loss(reshaped_scores, reshaped_predicted_probs)

    return value_loss, kl_loss, average_log_prob


class ScaledValueLoss(nn.Module):
    def __init__(self, beta_init=1.0, xpo_hyper=20.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init), requires_grad=True))
        self.xpo_hyper = nn.Parameter(torch.tensor(float(xpo_hyper), requires_grad=True))

    def forward(self, logits, scores, labels, tokenizer):
        labels[labels == -100] = tokenizer.pad_token_id
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        loss_mask = labels != tokenizer.pad_token_id
        log_prob = (per_token_logps * loss_mask).sum(-1)

        average_log_prob = self.beta * (log_prob / loss_mask.sum(-1) + self.xpo_hyper)
        predicted_probs = torch.sigmoid(average_log_prob)
        scores = scores.to(predicted_probs.dtype)
        value_loss = F.binary_cross_entropy(predicted_probs, scores, reduction='mean')

        return value_loss, predicted_probs


def get_normalized_value_loss(logits, scores, labels, tokenizer, xpo_hyper, beta):
    # TODO, Test if a shift is needed
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    labels[labels == -100] = tokenizer.pad_token_id
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    loss_mask = labels != tokenizer.pad_token_id
    log_prob = (per_token_logps * loss_mask).sum(-1)
    log_prob = log_prob / loss_mask.sum(-1)

    # get normalized scores
    log_prob_batch = log_prob.mean().detach()
    average_logits = (log_prob - log_prob_batch) * beta + xpo_hyper
    predicted_probs = torch.sigmoid(average_logits)
    scores = scores.to(predicted_probs.dtype)
    value_loss = F.binary_cross_entropy(predicted_probs, scores, reduction='mean')

    return value_loss, average_logits


def get_z_loss(logits, scores, labels, tokenizer, xpo_hyper):
    labels[labels == -100] = tokenizer.pad_token_id
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    loss_mask = labels != tokenizer.pad_token_id
    log_prob = (per_token_logps * loss_mask).sum(-1)
    log_prob = log_prob / loss_mask.sum(-1)

    # get z scores
    mean = log_prob.mean().detach()
    variance = log_prob.std().detach() + 1e-6
    z_score = (log_prob - mean) / variance
    scores = scores.to(z_score.dtype)
    print(z_score, flush=True)
    print(scores, flush=True)

    value_loss = F.mse_loss(z_score, scores, reduction='mean')

    return value_loss, z_score


def compute_value_loss(train_config, logits, scores, labels, tokenizer, xpo_hyper, beta, loss_module, config=None):
    """
    Computes the value loss based on the configuration in train_config.

    Args:
        train_config: Configuration object with loss settings.
        logits: Model logits.
        scores: Target scores.
        labels: Ground truth labels.
        tokenizer: Tokenizer instance.
        xpo_hyper: Hyperparameters for the loss computation.

    Returns:
        value_loss: Computed loss.
        avg_logits: Average logits or related metric from the chosen loss.
    """
    if train_config.normalized_value_loss:
        print("==== use normalized_value_loss ====", flush=True)
        return get_normalized_value_loss(logits, scores, labels, tokenizer, xpo_hyper, beta)
    elif train_config.z_value_loss:
        print("==== use z_value_loss ====", flush=True)
        return get_z_loss(logits, scores, labels, tokenizer, xpo_hyper)
    elif train_config.scaled_value_loss:
        print("==== use scaled value_loss ====", flush=True)
        return loss_module(logits, scores, labels, tokenizer)
    elif train_config.listwise_loss:
        print("==== use listwise_loss ====", flush=True)
        return get_calibration_loss(logits, scores, labels, tokenizer, config)
    else:
        print("==== use value_loss ====", flush=True)
        return get_value_loss(logits, scores, labels, tokenizer, xpo_hyper, beta)


def calibration_train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps,
              train_config, loss_module, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader with XPO objective.

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
        tokenizer: tokenizer used in the eval for decoding the predicitons

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
    val_loss = []

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
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch + 1}", total=total_length, dynamic_ncols=True)
            with profile(train_config, local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank == 0:
                            print("max training steps reached, stopping training, total train steps finished: ",
                                  total_train_steps - 1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            elif torch.cuda.is_available():
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        if step % 100 == 0:
                            decoded_texts = [tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in batch['input_ids']]
                            for text in decoded_texts:
                                print(text, flush=True)
                            print(batch)

                        model_output = model(**batch)
                        loss = model_output.loss
                        logits = model_output.logits
                        scores = batch['scores']
                        labels = batch['labels']
                        xpo_hyper = train_config.xpo_hyper
                        alpha = train_config.alpha
                        beta = train_config.beta
                        gama = train_config.gama

                        if not train_config.listwise_loss:
                            value_loss, avg_logits = compute_value_loss(train_config, logits, scores, labels, tokenizer,
                                                                        xpo_hyper, beta, loss_module, train_config)
                        else:
                            value_loss, pearson, chose_nll_loss, reject_nll_loss, cpo_loss = compute_value_loss(train_config, logits, scores, labels, tokenizer,
                                                                                                                xpo_hyper, beta, loss_module, train_config)

                    total_loss += loss.detach().float()
                    acc_loss = loss / gradient_accumulation_steps

                    # list-wise
                    value_acc_loss = value_loss / gradient_accumulation_steps
                    chose_nll_acc_loss = chose_nll_loss / gradient_accumulation_steps
                    cpo_acc_loss = cpo_loss / gradient_accumulation_steps

                    if train_config.listwise_loss:
                        final_loss = alpha * chose_nll_acc_loss + beta * value_acc_loss + gama * cpo_acc_loss
                    elif train_config.preferred_finetune:
                        final_loss = acc_loss
                    else:
                        final_loss = alpha * acc_loss + value_acc_loss

                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(final_loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                   train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # Optimize value loss
                        final_loss.backward()

                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                   train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank == 0:
                            if train_config.scaled_value_loss:
                                wandb_run.log({
                                    'train/epoch': epoch + 1,
                                    'train/loss': loss.detach().float(),
                                    'train/avg_logits': avg_logits.detach().float().mean().item(),
                                    'train/value_loss': value_loss.detach().float(),
                                    'train/xpo_hyper': loss_module.xpo_hyper.detach().float().item(),
                                    'train/beta': loss_module.beta.detach().float().item(),
                                })
                            elif train_config.listwise_loss:
                                wandb_run.log({
                                    'train/epoch': epoch + 1,
                                    'train/nll': loss.detach().float(),
                                    'train/chose_nll': chose_nll_loss.detach().float().mean().item(),
                                    "train/reject_nll": reject_nll_loss.detach().float().mean().item(),
                                    'train/value_loss': value_loss.detach().float(),
                                    'train/pearson': pearson.detach().float(),
                                })
                            else:
                                wandb_run.log({
                                    'train/epoch': epoch + 1,
                                    'train/loss': loss.detach().float(),
                                    'train/avg_logits': avg_logits.detach().float().mean().item(),
                                    'train/value_loss': value_loss.detach().float(),
                                })

                    pbar.set_description(
                        f"Training Epoch: {epoch + 1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)}"
                        f" completed (loss: {loss.detach().float()}, value_loss: {value_loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep,
                                     val_step_loss, val_loss, val_step_perplexity, val_prep)
                pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank == 0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        should_save_model = train_config.save_model
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = \
                calibration_evaluation(model, train_config, loss_module,
                                       eval_dataloader, local_rank, tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss

        # TODO, save each epoch
        should_save_model = True
        checkpoint_start_time = time.perf_counter()
        if should_save_model:
            if train_config.enable_fsdp:
                dist.barrier()
            if train_config.use_peft:
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"we are about to save the PEFT modules")
                else:
                    print(f"we are about to save the PEFT modules")
                save_peft_checkpoint(model, os.path.join(train_config.output_dir, str(epoch)))
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                else:
                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

            else:
                if not train_config.enable_fsdp:
                    save_model_checkpoint(model, train_config.output_dir)

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
                        save_model_and_optimizer_sharded(model, rank, train_config)

            if train_config.enable_fsdp:
                dist.barrier()
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"best eval loss on epoch {epoch + 1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch + 1} is {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(
                f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep,
                         val_step_loss, val_loss, val_step_perplexity, val_prep)

        # TODO, Early stopping
        # if len(val_loss) == 1:
        #     continue
        # elif val_loss[-1] > val_loss[-2]:
        #     print("Performance drops from Epoch {}, early stop.")
        #     break

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

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
        results["model_tflops"] = TFlops
    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def calibration_evaluation(model, train_config, loss_module, eval_dataloader, local_rank, tokenizer, wandb_run):
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
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

            # Get Eval value loss
            logits = outputs.logits
            scores = batch['scores']
            labels = batch['labels']
            xpo_hyper = train_config.xpo_hyper
            beta = train_config.beta

            if not train_config.listwise_loss:
                eval_value_loss, eval_avg_logits = compute_value_loss(train_config, logits, scores, labels, tokenizer,
                                                                      xpo_hyper, beta, loss_module, train_config)
            else:
                eval_value_loss, eval_kl_loss, eval_chose_nll, eval_reject_nll, cpo_loss = compute_value_loss(train_config, logits, scores, labels, tokenizer,
                                                                                                              xpo_hyper, beta, loss_module, train_config)

            if wandb_run:
                wandb_run.log({
                    'eval/step': total_eval_steps,
                    'eval/nll': loss.detach().float(),
                    'eval/chose_nll': eval_chose_nll.detach().float().mean().item(),
                    "eval/reject_nll": eval_reject_nll.detach().float().mean().item(),
                    'eval/value_loss': eval_value_loss.detach().float(),
                    'eval/pearson': eval_kl_loss.detach().float(),
                }, commit=False)

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
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
                    })

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity


def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps,
          train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
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
        tokenizer: tokenizer used in the eval for decoding the predicitons

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
    val_loss = []

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
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch + 1}", total=total_length, dynamic_ncols=True)
            with profile(train_config, local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank == 0:
                            print("max training steps reached, stopping training, total train steps finished: ",
                                  total_train_steps - 1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            elif torch.cuda.is_available():
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
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
                                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                   train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                   train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank == 0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(
                        f"Training Epoch: {epoch + 1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep,
                                     val_step_loss, val_loss, val_step_perplexity, val_prep)
                pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank == 0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        should_save_model = train_config.save_model
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config,
                                                                                        eval_dataloader, local_rank,
                                                                                        tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss

        # TODO, save for each epoch
        should_save_model = True
        checkpoint_start_time = time.perf_counter()
        if should_save_model:
            if train_config.enable_fsdp:
                dist.barrier()
            if train_config.use_peft:
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"we are about to save the PEFT modules")
                else:
                    print(f"we are about to save the PEFT modules")
                save_peft_checkpoint(model, os.path.join(train_config.output_dir, str(epoch)))
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                else:
                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

            else:
                if not train_config.enable_fsdp:
                    save_model_checkpoint(model, train_config.output_dir)

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
                        save_model_and_optimizer_sharded(model, rank, train_config)

            if train_config.enable_fsdp:
                dist.barrier()
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"best eval loss on epoch {epoch + 1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch + 1} is {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(
                f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep,
                         val_step_loss, val_loss, val_step_perplexity, val_prep)

        # Early stopping
        # if len(val_loss) == 1:
        #     continue
        # elif val_loss[-1] > val_loss[-2]:
        #     print("Performance drops from Epoch {}, early stop.")
        #     break

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

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
        results["model_tflops"] = TFlops
    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
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
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
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

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


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


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""


    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and torch.version.cuda >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


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
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
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
