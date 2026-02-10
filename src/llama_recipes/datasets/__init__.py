# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from llama_recipes.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets.custom_dataset import get_custom_dataset,get_data_collator
from llama_recipes.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.datasets.flores_dataset import get_preprocessed_flores as get_flores_dataset
from llama_recipes.datasets.wmt22_test_dataset import get_preprocessed_wmt22_test as get_wmt22_test_dataset
from llama_recipes.datasets.translation_dataset import get_preprocessed_bitext
from llama_recipes.datasets.toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_llamaguard_toxicchat_dataset
from llama_recipes.datasets.preference_dataset import get_preprocessed_preference_data

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "flores_dataset": get_flores_dataset,
    "wmt22_testset": get_wmt22_test_dataset,
    "translation_dataset": get_preprocessed_bitext,
    "human_written_dataset": get_preprocessed_bitext,
    "da_dataset": get_preprocessed_preference_data,
}
DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_data_collator
}
