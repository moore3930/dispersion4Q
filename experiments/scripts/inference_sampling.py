# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import re
import shutil
import sys
import time
import fire
import torch
from tqdm import tqdm

from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model, load_peft_model

from llama_recipes.inference.safety_utils import AgentType, get_safety_checker
from transformers import AutoTokenizer

from llama_recipes.data.concatenator import ConcatDataset

from llama_recipes.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
    get_translation_dataset,
    get_prefernce_dataset,
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


def main(
    model_name,
    peft_model: str = None,
    quantization: str = None,  # Options: 4bit, 8bit
    max_new_tokens=1000,  # The maximum numbers of tokens to generate
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
    # dataset_config = generate_dataset_config(test_config, kwargs)

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

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # TODO, batch inference
    def inference_new(
            dataloader,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            config,
            pbar,
            **kwargs,
    ):
        output1 = []
        output2 = []
        for step, batch in enumerate(dataloader):
            if step > 10:
                break
            # TODO, dirty
            batch.pop('labels')
            if is_xpu_available():
                batch = {k: v.to("xpu") for k, v in batch.items()}
            else:
                batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.no_grad():
                batch_output = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,  # 启用采样
                    top_p=top_p,  # nucleus sampling 参数
                    temperature=temperature,  # 温度
                    min_length=min_length,  # 最小长度
                    use_cache=use_cache,  # 是否使用缓存
                    top_k=top_k,  # top-k 采样
                    repetition_penalty=repetition_penalty,  # 重复惩罚
                    length_penalty=length_penalty,  # 长度惩罚
                    num_beams=1,
                    num_return_sequences=config.num_return_samples,
                    eos_token_id=terminators,
                    **kwargs,  # 其他参数
                )

                # prompt_len = batch['attention_mask'].sum(-1)

                batch_len = batch['input_ids'].shape[-1]
                hyp_output = batch_output[:, batch_len:]
                hyp_output = [tokenizer.decode(output, skip_special_tokens=True) for output in hyp_output]

                # # replace \n with \t to read when hallucinating
                # hyp_output = [sent.replace("\n", "\t").strip() for sent in hyp_output]
                # output1 += hyp_output

                # remove \n with \t when hallucinating
                hyp_output = [re.split(r'[\t\n]', sent)[0] for sent in hyp_output]
                output1 += hyp_output

                src_input = batch_output[:, :batch_len]
                src_input = [tokenizer.decode(input, skip_special_tokens=True) for input in src_input]

                def parse_src(input):
                    match = re.search(r"Translate the following text from (.+) into (.+).\n\1: (.+)\n\2:", input)

                    if match:
                        src_lang = match.group(1)  # Extract source language
                        tgt_lang = match.group(2)  # Extract target language
                        src = match.group(3)  # Extract source text
                        return src

                src_input = [parse_src(input) for input in src_input]
                output2 += src_input

            pbar.update(1)
        return output1, output2

    # TODO, inference for each dataset
    if lang_pairs is not None:
        lang_pairs = lang_pairs.split(',')
    else:
        lang_pairs = test_config.lang_pairs

    for lang_pair in lang_pairs:
        # Get test data
        print("Processing {} ...".format(lang_pair), flush=True)
        if test_config.dataset == "haoranxu/ALMA-R-Preference":
            dataset_test = get_prefernce_dataset(
                tokenizer,
                test_config,
                test_config.dataset,
                mode="infer",
                subset_name=test_config.subset_name,
                split="train",
                lang_pairs=[lang_pair],
                filter="reference"
            )
        else:
            dataset_test = get_translation_dataset(
                tokenizer,
                test_config.dataset,
                mode="infer",
                split="test",
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
        hyp_sents, src_sents = inference_new(test_dataloader, temperature, top_p, top_k, max_new_tokens, test_config, pbar=pbar)
        pbar.close()
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms", flush=True)

        # dump results
        create_clean_dir(os.path.join(output_dir, lang_pair))
        hyp_file = os.path.join(output_dir, lang_pair, "tgt")
        src_file = os.path.join(output_dir, lang_pair, "src")
        with open(hyp_file, 'w') as h_fout, open(src_file, 'w') as s_fout:
            for hyp, src in zip(hyp_sents, src_sents):
                h_fout.write(hyp.strip() + "\n")
                s_fout.write(src.strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
