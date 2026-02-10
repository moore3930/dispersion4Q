# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import os
import datasets
import random

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
def load_text(dataset_name, split, lang_pairs, _):
    assert split in ["train", "valid", "test"], f"Unknown split: {split}"

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
                idx_32bit = str(random.randint(-2 ** 31, 2 ** 31 - 1))
                row = {"id": idx_32bit, "src_lang": src, "src": src_sent}
                output_dataset.append(row)

    dataset = datasets.Dataset.from_list(output_dataset)
    return dataset


def get_preprocessed_monolingual_data(tokenizer, dataset_config, mode, split, lang_pairs):
    dataset_name = dataset_config.dataset
    dataset = load_text(dataset_name, split, lang_pairs)

    lang_name = {"en": "English", "zh": "Chinese", "ar": "Arabic", "de": "German",
                 "cs": "Czech", "ru": "Russian", "is": "Icelandic"}

    prompt = (
        f"{{src}}"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(src=sample["src"]),
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt,
            "attention_mask": [1] * len(prompt),
            "labels": prompt,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

