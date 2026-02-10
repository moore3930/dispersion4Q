# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.datasets import DATASET_PREPROC, DATALOADER_COLLATE_FUNC
from llama_recipes.utils.config_utils import get_dataloader_kwargs
from llama_recipes.datasets.translation_dataset import get_preprocessed_bitext, get_vllm_preprocessed_bitext
from llama_recipes.datasets.monolingual_dataset import get_preprocessed_monolingual_data
from llama_recipes.datasets.preference_dataset import get_preprocessed_preference_data
from llama_recipes.datasets.calibration_dataset import get_calibration_data
from llama_recipes.configs import train_config as TRAIN_CONFIG


def get_vllm_translation_dataset(
    dataset_name, split: str = "test", lang_pairs: list = ("en-de", "en-zh", "en-ar")
) -> torch.utils.data.Dataset:
    return get_vllm_preprocessed_bitext(dataset_name, split, lang_pairs)


def get_translation_dataset(
    tokenizer, dataset_name, mode: str = "train", split: str = "train", lang_pairs: list = ("en-de", "en-zh", "en-ar")
) -> torch.utils.data.Dataset:

    return get_preprocessed_bitext(
        tokenizer,
        dataset_name,
        mode,
        split,
        lang_pairs
    )


def get_prefernce_dataset(
    tokenizer, config, dataset_name, subset_name: str = "wmt-da-17-22.csv", split: str = "train",
        lang_pairs: list = ("en-de", "en-zh"), mode: str = None,
        filter: str = None, listwise: bool = False, batch_size: int = 15) -> torch.utils.data.Dataset:

    if dataset_name == "da_dataset":
        return get_preprocessed_preference_data(
            tokenizer,
            dataset_name,
            subset_name,
            split,
            lang_pairs,
            filter,
        )
    elif dataset_name == "calibration" and listwise:
        return get_calibration_data(
            tokenizer,
            config.dataset,
            config.subset_name,
            config.metric,
            split,
            lang_pairs,
            batch_size,
            config.subset_size,
        )
    else:
        return None


def get_monolingual_dataset(
    tokenizer, dataset_config, mode: str = "train", split: str = "train", lang_pairs: list = ("en-de", "en-zh", "en-ar")
) -> torch.utils.data.Dataset:

    return get_preprocessed_monolingual_data(
        tokenizer,
        dataset_config,
        mode,
        split,
        lang_pairs
    )


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train", lang_pairs: list = ("en-de", "en-zh", "en-ar")
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
        lang_pairs
    )


def get_custom_data_collator(
    dataset_processer, dataset_config
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATALOADER_COLLATE_FUNC:
        return None

    return DATALOADER_COLLATE_FUNC[dataset_config.dataset](
        dataset_processer,
        dataset_config
    )

def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split)
    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)
    
    if split == "train" and train_config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader
    