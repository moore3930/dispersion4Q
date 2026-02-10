import os
from datasets import load_dataset

# lang_pairs = ["en-de", "en-es", "en-cs", "en-ru", "en-uk", "en-is", "en-ja", "en-zh", "en-hi", "cs-uk", "ja-zh"]
lang_pairs ={"en-de": "en-de_DE", "en-es": "en-es_MX", "en-cs": "en-cs_CZ", "en-ru": "en-ru_RU",
             "en-uk": "en-uk_UA", "en-is": "en-is_IS", "en-ja": "en-ja_JP", "en-zh": "en-zh_CN",
             "en-hi": "en-hi_IN", "en-nl": "en-nl_NL", "en-pt": "en-pt_PT", "en-fr": "en-fr_FR",
             "en-it": "en-it_IT", "en-ko": "en-ko_KR"}

lang_pairs ={"en-ja": "en-ja_JP", "en-zh": "en-zh_CN",
             "en-hi": "en-hi_IN", "en-nl": "en-nl_NL", "en-pt": "en-pt_PT", "en-fr": "en-fr_FR",
             "en-it": "en-it_IT", "en-ko": "en-ko_KR"}

lang_pairs ={"en-de": "en-de_DE", "en-es": "en-es_MX", "en-cs": "en-cs_CZ", "en-ru": "en-ru_RU",
             "en-uk": "en-uk_UA"}

# lang_pairs ={"cs-uk": "", "ja-zh": ""}

# lang_pairs = {"en-pt": "en-pt_PT", "en-fr": "en-fr_FR", "en-it": "en-it_IT", "en-ko": "en-ko_KR"}
# lang_pairs = {"en-fr": "en-fr_FR"}

# Load the dataset
for lang_pair in lang_pairs:
    src, tgt = lang_pair.split("-")
    dataset = load_dataset("google/wmt24pp", lang_pairs[lang_pair])

    # Check available splits
    print(dataset)

    src_file = f"wmt24_testset_old/test/{lang_pair}/test.{lang_pair}.{src}"
    tgt_file = f"wmt24_testset_old/test/{lang_pair}/test.{lang_pair}.{tgt}"

    os.makedirs(f"wmt24_testset_old/test/{lang_pair}", exist_ok=True)

    # View an example
    with open(src_file, 'w') as src_fout, open(tgt_file, 'w') as tgt_fout:
        for line in dataset['train']:
            if not line['is_bad_source']:
                src_fout.write(line['source'].strip() + "\n")
                tgt_fout.write(line['original_target'].strip() + "\n")