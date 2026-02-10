# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import os
import datasets
import random
from datasets import load_dataset, concatenate_datasets


from llama_recipes.configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
)

from unittest.mock import patch

random.seed(42)


@patch('builtins.input', return_value="N")
def load_bitext(dataset_name, split, lang_pairs, _):
    assert split in ["train", "valid", "test", "ref"], f"Unknown split: {split}"

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    dir_name = os.path.join(current_dir, "..", "customer_data", dataset_name)

    output_dataset = []
    for lp in lang_pairs:
        src, tgt = lp.split("-")
        pair_name = "{}-{}".format(src, tgt)
        src_name = "{}.{}-{}.{}".format(split, src, tgt, src)
        tgt_name = "{}.{}-{}.{}".format(split, src, tgt, tgt)

        with open(os.path.join(dir_name, split, pair_name, src_name)) as src_fin, \
                open(os.path.join(dir_name, split, pair_name, tgt_name)) as tgt_fin:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                src_sent = src_sent.strip()
                tgt_sent = tgt_sent.strip()
                idx_32bit = str(random.randint(-2 ** 31, 2 ** 31 - 1))
                row = {"id": idx_32bit, "src_lang": src, "tgt_lang": tgt, "src": src_sent, "tgt": tgt_sent}
                output_dataset.append(row)

    dataset = datasets.Dataset.from_list(output_dataset)
    return dataset


def get_preprocessed_bitext(tokenizer, dataset_name, mode, split, lang_pairs):

    dataset = load_bitext(dataset_name, split, lang_pairs)

    lang_name = {"en": "English", "zh": "Chinese", "ar": "Arabic", "de": "German",
                 "cs": "Czech", "ru": "Russian", "is": "Icelandic", "es": "Spanish",
                 "hi": "Hindi", "ja": "Japanese", "nl": "Dutch", "uk": "Ukrainian",
                 "fr": "French", "it": "Italian", "pt": "Portuguese", "ko": "Korean"}

    prompt = (
        # f"Translate this from {{src_lang}} to {{tgt_lang}}:\n{{src_lang}}: {{src}}\n{{tgt_lang}}:"
        f"Translate the following text from {{src_lang}} into {{tgt_lang}}.\n{{src_lang}}: {{src}}\n{{tgt_lang}}:"
    )

    def apply_prompt_template(sample):
        src = sample["src_lang"]
        tgt = sample["tgt_lang"]

        messages = [
            {"role": "user", "content": prompt.format(src_lang=lang_name[src], tgt_lang=lang_name[tgt], src=sample["src"])}
        ]

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            p = prompt.format(src_lang=lang_name[src], tgt_lang=lang_name[tgt], src=sample["src"]) # old

        return {
            "prompt": p,
            "summary": sample["tgt"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    def tokenize_prompt(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)

        sample = {
            "input_ids": prompt,
            "attention_mask": [1] * len(prompt),
            }

        return sample

    if mode == "infer":
        dataset = dataset.map(tokenize_prompt, remove_columns=list(dataset.features))
    elif mode == "eval":
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    else: # Train
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
        dataset = dataset.shuffle(seed=42)

    return dataset


def get_vllm_preprocessed_bitext(dataset_name, split, lang_pairs):

    dataset = load_bitext(dataset_name, split, lang_pairs)

    lang_name = {"en": "English", "zh": "Chinese", "ar": "Arabic", "de": "German",
                 "cs": "Czech", "ru": "Russian", "is": "Icelandic"}

    prompt = (
        f"Translate this from {{src_lang}} to {{tgt_lang}}:\n{{src_lang}}: {{src}}\n{{tgt_lang}}:"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(src_lang=lang_name[sample["src_lang"]],
                                    tgt_lang=lang_name[sample["tgt_lang"]],
                                    src=sample["src"]),
            "summary": sample["tgt"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    return dataset


# dataset = get_vllm_preprocessed_bitext('wmt22_testset', 'test', ['en-zh', 'en-de'])
#
# print(dataset)
#
# prompts = dataset['prompt']
# batch_size = 16  # Adjust based on available GPU memory
#
# # Iterate over dataset in batches
# results = []
# for i in range(0, len(prompts), batch_size):
#     batch_prompts = prompts[i:i + batch_size]
#     print(batch_prompts)
#     break