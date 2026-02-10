# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import os
import datasets
import random
import pandas as pd

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
def load_preference(dataset_name, split, subset_name, lang_pairs, _):
    assert split in ["train", "valid", "test"], f"Unknown split: {split}"

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    dir_name = os.path.join(current_dir, "..", "customer_data", dataset_name, split)
    data = pd.read_csv(os.path.join(dir_name, subset_name))

    selected_data = data[data['lp'].isin(lang_pairs)]
    dataset = datasets.Dataset.from_pandas(selected_data)

    return dataset


def get_preprocessed_preference_data(tokenizer, dataset_name, subset_name, split, lang_pairs, filter):

    dataset = load_preference(dataset_name, split, subset_name, lang_pairs)

    lang_name = {"en": "English", "zh": "Chinese", "ar": "Arabic", "de": "German",
                 "cs": "Czech", "ru": "Russian", "is": "Icelandic", "es": "Spanish",
                 "hi": "Hindi", "ja": "Japanese", "nl": "Dutch", "uk": "Ukrainian",
                 "fr": "French", "it": "Italian", "pt": "Portuguese", "ko": "Korean"}

    prompt = (
        # f"Translate this from {{src_lang}} to {{tgt_lang}}:\n{{src_lang}}: {{src}}\n{{tgt_lang}}:"
        f"Translate the following text from {{src_lang}} into {{tgt_lang}}.\n{{src_lang}}: {{src}}\n{{tgt_lang}}:"
    )

    def apply_prompt_template(sample):
        lang_pair = sample["lp"]
        src, tgt = lang_pair.split('-')

        messages = [
            {"role": "user", "content": prompt.format(src_lang=lang_name[src], tgt_lang=lang_name[tgt], src=sample["src"])}
        ]

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            p = prompt.format(src_lang=lang_name[src], tgt_lang=lang_name[tgt], src=sample["src"]) # old

        return {
            "prompt": p,
            "summary": sample["mt"],
            "score": sample["score"]
        }

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] + tokenizer.eos_token, add_special_tokens=False)
        score = float(sample["score"])

        # be careful, the order of dataset.feature will be changed if set "score" instead of "scores"
        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            "scores": score,
            }

        return sample

    # Apply prompts
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.filter(lambda sample: all(value is not None for value in sample.values()))

    # Tokenize
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    # Remove long sentences for efficiency
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= 512)

    # Shuffle data
    # dataset = dataset.shuffle(seed=42)

    return dataset

# tokenizer = AutoTokenizer.from_pretrained('haoranxu/ALMA-7B-Pretrain')
# dataset = get_preprocessed_preference_data(tokenizer, 'da_dataset', 'wmt-qe-2022.train.csv', 'valid', ['zh-en'], None)
#
# print(dataset)
# print(dataset[:10])
