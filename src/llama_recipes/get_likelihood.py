# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import shutil
import sys
import time

import fire
from tqdm import tqdm

import torch
import dataclasses

from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model, load_peft_model

from llama_recipes.inference.safety_utils import AgentType, get_safety_checker
from transformers import AutoTokenizer

from llama_recipes.data.concatenator import ConcatDataset

from llama_recipes.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
    get_translation_dataset,
)

from llama_recipes.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
    get_prefernce_dataset,
    get_translation_dataset,
    get_monolingual_dataset,
    get_prefernce_z_dataset,
)

from llama_recipes.configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)

from llama_recipes.utils.config_utils import (
    check_fsdp_config,
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    update_config,
)

from llama_recipes.utils.train_utils import compute_value_loss, ScaledValueLoss



def create_clean_dir(path):
    """
    Create a clean directory. If the directory exists, remove it first.
    :param path: Path of the directory to create.
    """
    # Remove the directory if it exists
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create the directory
    os.makedirs(path)


def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG

    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


def main(
    model_name,
    peft_model: str = None,
    quantization: str = None,  # Options: 4bit, 8bit
    max_new_tokens=100,  # The maximum numbers of tokens to generate
    prompt_file: str = None,
    seed: int = 42,  # seed value for reproducibility
    do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = True,
    # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 1.0,
    # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
    top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1,
    # [optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool = False,
    max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False,
    # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    share_gradio: bool = False,  # Enable endpoint creation for gradio.live
    lang_pairs: str = None,
    output_dir: str = None,
    **kwargs,
):
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    # Update the configuration for the training and sharding process
    test_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((test_config, fsdp_config), **kwargs)

    # TODO, no fsdp
    if test_config.use_wandb:
        wandb_run = setup_wandb(test_config, fsdp_config, **kwargs)
    else:
        wandb_run = None

    model = load_model(model_name, quantization, use_fast_kernels, **kwargs)

    if test_config.preload_peft_dir is not None:
        # merge peft into backbone, may not 100% aligned
        print("Load and merge peft...")
        model = load_peft_model(model, test_config.preload_peft_dir)
        model = model.merge_and_unload()

    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("All parameters:", locals())
    print(test_config)

    def calculate_log_probs(labels, logits, beta, xpo_hyper):
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        labels[labels == -100] = tokenizer.pad_token_id
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        loss_mask = labels != tokenizer.pad_token_id
        log_prob = (per_token_logps * loss_mask).sum(-1)
        norm_log_prob = log_prob / loss_mask.sum(-1)
        logits = beta * (norm_log_prob + xpo_hyper)
        predicted_probs = torch.sigmoid(logits)
        return log_prob, norm_log_prob, logits, predicted_probs

    # TODO, batch inference
    def inference_new(
            dataloader,
            pbar,
    ):
        output = []
        for step, batch in enumerate(dataloader):
            if is_xpu_available():
                batch = {k: v.to("xpu") for k, v in batch.items()}
            else:
                batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.no_grad():
                model_output = model(**batch)
                logits = model_output.logits
                labels = batch['labels']

                xpo_hyper = test_config.xpo_hyper
                beta = test_config.beta

                log_prob, norm_log_prob, logits, predicted_probs = calculate_log_probs(labels, logits, beta, xpo_hyper)

                batch_len = (labels != -100).sum(-1)
                inputs = [
                    tokenizer.decode(batch['input_ids'][i, -batch_len[i]:], skip_special_tokens=True)
                    for i in range(batch_len.size(0))
                ]

                batch_sent = [sent.replace("\n", "\t").strip() for sent in inputs]
                batch_score = ["{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(logp.item(), norm_logp.item(), logit.item(), pred.item())
                               for logp, norm_logp, logit, pred in zip(log_prob, norm_log_prob, logits, predicted_probs)]
                batch_output = ["{}\t{}".format(sent1, sent2) for sent1, sent2 in zip(batch_sent, batch_score)]

                output += batch_output

            if wandb_run:
                if not test_config.scaled_value_loss:
                    wandb_run.log({
                        'inference/epoch': 1,
                        'inference/loss': model_output.loss.detach().float().mean().item(),
                        'inference/avg_logits': logits.detach().float().mean().item(),
                        'inference/probs': predicted_probs.detach().float().mean().item(),
                    })
            pbar.update(1)
        return output

    # TODO, inference for each dataset
    output = {}

    lang_pairs = lang_pairs.split(',')

    for lang_pair in lang_pairs:
        # Get test data
        print("Processing {} ...".format(lang_pair), flush=True)
        dataset_test = get_translation_dataset(
            tokenizer,
            test_config.dataset,
            mode="eval",
            split=test_config.split,
            lang_pairs=[lang_pair]
        )
        print(f"--> Test Set Length = {len(dataset_test)}", flush=True)

        test_dl_kwargs = get_dataloader_kwargs(
            test_config, dataset_test, tokenizer, "infer"
        )

        # Create DataLoaders for inference
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            **test_dl_kwargs,
        )
        print(f"--> Num of Testing Set Batches loaded = {len(test_dataloader)}", flush=True)

        start = time.perf_counter()
        total_length = len(test_dataloader)
        pbar = tqdm(colour="blue", desc=f"Inference", total=total_length, dynamic_ncols=True)
        results = inference_new(test_dataloader, pbar=pbar)
        pbar.close()
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms", flush=True)
        output[lang_pair] = results

        # dump results
        src, tgt = lang_pair.split("-")
        create_clean_dir(os.path.join(output_dir, lang_pair))
        output_file = os.path.join(output_dir, lang_pair, "score.{}-{}.{}".format(src, tgt, tgt))
        with open(output_file, 'w') as fout:
            for line in results:
                fout.write(line.strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
