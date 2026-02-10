import pandas as pd
import datasets


file_list = ["wmt-da-17-22.0.2.csv"]
lang_pairs = ["en-zh", "zh-en", "en-de", "de-en", "en-ru", "ru-en", "en-cs", "cs-en", "en-is", "is-en"]

dataframes = []

for file_name in file_list:
    # Reading a CSV file
    data = pd.read_csv(file_name)
    dataframes.append(data)

merged_data = pd.concat(dataframes, ignore_index=True)
filtered_data = merged_data[merged_data['lp'].isin(lang_pairs)]
print(len(filtered_data))

dataset = datasets.Dataset.from_pandas(filtered_data)

lang_name = {"en": "English", "zh": "Chinese", "ar": "Arabic", "de": "German",
             "cs": "Czech", "ru": "Russian", "is": "Icelandic"}

prompt = (
    f"Translate this from {{src_lang}} to {{tgt_lang}}:\n{{src_lang}}: {{src}}\n{{tgt_lang}}:"
)
def apply_prompt_template(sample):
    # Returning None will remove this row during the map operation
    if sample["src"] is None or pd.isna(sample["src"]) or \
            sample["mt"] is None or pd.isna(sample["mt"]) or \
            sample["lp"] is None or pd.isna(sample["lp"]):
        return None

    lang_pair = sample["lp"]
    src, tgt = lang_pair.split('-')

    return {
        "prompt": prompt.format(src_lang=lang_name[src],
                                tgt_lang=lang_name[tgt],
                                src=sample["src"]),
        "summary": sample["mt"],
        "score": sample["score"]
    }


for row in dataset:
    if row["mt"] is None or pd.isna(row["mt"]):
        print(row)


# dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
