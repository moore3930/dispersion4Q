# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class da_dataset:
    dataset: str = "da_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    lang_pairs: list = ("en-de", "de-en", "en-zh", "zh-en", "en-ru", "ru-en", "en-cs", "cs-en")


@dataclass
class flores_dataset:
    dataset: str = "flores_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    lang_pairs: list = ("en-de", "de-en", "en-zh", "zh-en")


@dataclass
class human_written_dataset:
    dataset: str = "human_written_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    lang_pairs: list = ("en-cs", "cs-en", "en-de", "de-en", "en-is", "is-en", "en-ru", "ru-en", "en-zh", "zh-en")


@dataclass
class wmt22_testset:
    dataset: str = "wmt22_testset"
    test_split: str = "test"
    lang_pairs: list = ("cs-en", "de-en", "is-en", "ru-en", "zh-en", "en-cs", "en-de", "en-is", "en-ru", "en-zh")


@dataclass
class wmt23_testset:
    dataset: str = "wmt23_testset"
    test_split: str = "test"
    lang_pairs: list = ("cs-en", "de-en", "is-en", "ru-en", "zh-en", "en-cs", "en-de", "en-is", "en-ru", "en-zh")


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "recipes/quickstart/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""
    
@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"
