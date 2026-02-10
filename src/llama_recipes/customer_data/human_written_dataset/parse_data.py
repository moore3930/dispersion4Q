import json
import os
import shutil

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

# Specify the file name
lang_pair_list = ["en-cs", "cs-en", "en-de", "de-en", "en-is", "is-en", "en-ru", "ru-en", "en-zh", "zh-en"]

for lang_pair in lang_pair_list:
    src, tgt = lang_pair.split("-")

    input_file = os.path.join("train", "{}-{}".format(src, tgt), "valid.{}-{}.json".format(src, tgt))
    create_clean_dir(os.path.join("valid", "{}-{}".format(src, tgt)))
    src_output = os.path.join("valid", "{}-{}".format(src, tgt), "valid.{}-{}.{}".format(src, tgt, src))
    tgt_output = os.path.join("valid", "{}-{}".format(src, tgt), "valid.{}-{}.{}".format(src, tgt, tgt))

    with open(os.path.join(input_file), "r", encoding="utf-8") as fin:
        with open(os.path.join("valid", "{}-{}".format(src, tgt), "valid.{}-{}.{}".format(src, tgt, src)), 'w') as src_fout, \
                open(os.path.join("valid", "{}-{}".format(src, tgt), "valid.{}-{}.{}".format(src, tgt, tgt)), 'w') as tgt_fout:
            json_data = json.load(fin)
            for idx, item in enumerate(json_data):
                is_last = idx == len(json_data) - 1
                print(item)

                # Write without a newline for the last item
                src_fout.write(item["translation"][src] + ("" if is_last else "\n"))
                tgt_fout.write(item["translation"][tgt] + ("" if is_last else "\n"))



