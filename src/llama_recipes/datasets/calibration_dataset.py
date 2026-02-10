# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
import random
import torch
import os
import itertools

from datasets import load_dataset, concatenate_datasets
from unittest.mock import patch
from transformers import AutoTokenizer
from datasets import Dataset

random.seed(42)


@patch('builtins.input', return_value="N")
def load_preference(dataset_name, subset_name, metric, split, lang_pairs, subset_size, _):

    def load_calibration_dataset(dataset_name, subset_name, metric, split, lang_pair):
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        dir_name = os.path.join(current_dir, "..", "customer_data", dataset_name, subset_name, split, lang_pair)
        data_list = []
        with open(os.path.join(dir_name, "src")) as src_fin, \
                open(os.path.join(dir_name, "tgt")) as tgt_fin, \
                open(os.path.join(dir_name, metric)) as score_fin:
            src_key = None
            tgt_list = []
            score_list = []
            for src, tgt, score in itertools.zip_longest(src_fin, tgt_fin, score_fin, fillvalue=None):
                if src is None or tgt is None or score is None:
                    break
                src = src.strip()
                tgt = tgt.strip()
                score = float(score.strip().split("score: ")[-1])
                if src_key is None or src_key == src:
                    src_key = src
                    tgt_list.append(tgt)
                    score_list.append(score)
                else:
                    cur_json = {"src": src_key, "tgt_list": tgt_list, "score_list": score_list}
                    if len(set(score_list)) != 1:
                        data_list.append(cur_json)
                    src_key = src
                    tgt_list = [tgt]
                    score_list = [score]
        return datasets.Dataset.from_list(data_list)

    def apply_process_calibration_sample(batch, lang_pair):
        # Create the lists for each column in the batch
        src_list = []
        tgt_list = []
        score_list = []
        lp_list = []

        for src, tgt_sample_list, scores_sample_list in zip(
                batch["src"], batch["tgt_list"], batch["score_list"]
        ):
            src_list.append([src] * len(tgt_sample_list))
            tgt_list.append(tgt_sample_list)
            score_list.append(scores_sample_list)
            lp_list.append([lang_pair] * len(tgt_sample_list))

        return {
            "src": src_list,
            "tgt": tgt_list,
            "score": score_list,
            "lp": lp_list,
        }

    # process each lang_pair
    multiple_datasets = []
    for lang_pair in lang_pairs:
        dataset = load_calibration_dataset(dataset_name, subset_name, metric, split, lang_pair)
        dataset = dataset.map(
            lambda row: apply_process_calibration_sample(row, lang_pair),
            batched=True,
        )

        # get subset for analysis
        if subset_size > 0:
            dataset = dataset.select(range(subset_size))

        multiple_datasets.append(dataset)

    dataset = concatenate_datasets(multiple_datasets)
    dataset = dataset.shuffle(seed=142)

    return dataset


def flatten_to_dataset(dataset):
    # Flatten the data into individual lists
    flattened_data = {
        "src": [],
        "tgt": [],
        "score": [],
        "lp": [],
    }

    for src_list, tgt_list, score_list, lp_list in zip(
            dataset['src'], dataset['tgt'], dataset['score'], dataset['lp']
    ):
        for src, tgt, score, lp in zip(src_list, tgt_list, score_list, lp_list):
            flattened_data["src"].append(src)
            flattened_data["tgt"].append(tgt)
            flattened_data["score"].append(score)
            flattened_data["lp"].append(lp)

    # Convert to a datasets.Dataset
    return Dataset.from_dict(flattened_data)


def get_calibration_data(tokenizer, dataset_name, subset_name, metric, split, lang_pairs, batch_size=18, subset_size=-1):
    dataset = load_preference(dataset_name, subset_name, metric, split, lang_pairs, subset_size)

    # Apply the flatten function
    dataset = flatten_to_dataset(dataset)

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
            "summary": sample["tgt"],
            "score": sample["score"]
        }

    def tokenize_add_label_old(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] + tokenizer.eos_token, add_special_tokens=False)
        score = float(sample["score"])
        score = float(min(max(score, 0), 1))

        return {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            "scores": score,
        }

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"], add_special_tokens=False)
        summary = summary[:200] # max len for summary
        summary.append(tokenizer.eos_token_id)
        score = float(sample["score"])
        # score = float(min(max(score, 0), 1))

        return {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            "scores": score,
        }


    def pad_to_max_length(batch):
        # Determine the maximum sequence length in the batch
        max_len = max(len(input_ids) for input_ids in batch["input_ids"])

        # Apply padding for each field
        padded_batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "scores": batch["scores"],  # No need to pad scores as it's a single value per sample
        }

        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i]
            attention_mask = batch["attention_mask"][i]
            labels = batch["labels"][i]

            pad_len = max_len - len(input_ids)

            padded_batch["input_ids"].append([2] * pad_len + input_ids)  # Left-pad with padding_id=2
            padded_batch["attention_mask"].append([0] * pad_len + attention_mask)  # Left-pad with 0
            padded_batch["labels"].append([-100] * pad_len + labels)  # Left-pad with -100 for labels

        return padded_batch

    # Apply prompts
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    # Tokenize
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    # Batch padding
    dataset = dataset.map(
        lambda batch: pad_to_max_length(batch),
        batched=True,
        batch_size=batch_size,
    )

    return dataset

# tokenizer = AutoTokenizer.from_pretrained('haoranxu/ALMA-7B-Pretrain')
# dataset = get_calibration_data(tokenizer, 'flores-gpt', 'gpt-4o-mini', ['en-de', 'en-zh'], 10)
#
# print(dataset)
# print(dataset[:10])

#
# train_dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=10,
#     num_workers=0,
#     pin_memory=True,
#     collate_fn=lambda x: {
#         "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in x], dim=0),
#         "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in x], dim=0),
#         "labels": torch.stack([torch.tensor(item["labels"]) for item in x], dim=0),
#         "scores": torch.tensor([torch.tensor(item["scores"]) for item in x], dtype=torch.float),
#     },
# )
#
# for step, batch in enumerate(train_dataloader):
#     print(batch)
#     break

# dataset = load_dataset("haoranxu/ALMA-R-Preference", "{}-{}".format("zh", "en"))['train']
# print(dataset)
#
# dataset = dataset.filter(lambda row: row["translation"]["Delta"] == 0)
# print(dataset)


# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

# import torch
# from transformers import pipeline
#
# tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")
# tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-Mistral-7B-v0.2")
#
#
# # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# messages = [
#     {"role": "user", "content": "Translate the following text from Portuguese into English.\nPortuguese: Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.\nEnglish:"},
# ]
# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#
# print(prompt)
# print(tokenizer.pad_token_id)
# print(tokenizer.pad_token)
# print(tokenizer.eos_token_id)
# print(tokenizer.eos_token)
#
# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]
#
# print(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
# print(tokenizer.convert_tokens_to_ids("\n"))
# print(tokenizer.convert_tokens_to_ids("\t"))
# print(tokenizer.convert_tokens_to_ids("de"))
#
# print("Tokenized:", tokenizer.tokenize("\t"))
# print("IDs:", tokenizer.convert_tokens_to_ids(tokenizer.tokenize("\t")))
#
# print("Tokenized:", tokenizer.tokenize("<|eot_id|>"))
# print("IDs:", tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|eot_id|>")))


