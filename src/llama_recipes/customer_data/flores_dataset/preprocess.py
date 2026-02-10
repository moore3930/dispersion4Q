# build pairwise data from flores
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

lang_map = {"en": "eng_Latn", "de": "deu_Latn", "ar": "ace_Arab", "zh": "zho_Hans",
            "cs": "ces_Latn", "ru": "rus_Cyrl", "is": "isl_Latn"}

lang_pairs = ["en-de", "de-en", "en-zh", "zh-en", "en-cs", "cs-en", "en-ru", "ru-en", "en-is", "is-en"]

train_dir = "train"
valid_dir = "valid"

create_clean_dir(train_dir)
create_clean_dir(valid_dir)

# get train data
for lang_pair in lang_pairs:
    src, tgt = lang_pair.split("-")
    src_lang = lang_map[src]
    tgt_lang = lang_map[tgt]

    src_file = os.path.join("dev", "{}.dev".format(src_lang))
    tgt_file = os.path.join("dev", "{}.dev".format(tgt_lang))

    src_output_dir = os.path.join("train", "{}-{}".format(src, tgt))
    tgt_output_dir = os.path.join("train", "{}-{}".format(src, tgt))
    create_clean_dir(src_output_dir)
    create_clean_dir(tgt_output_dir)

    src_output = os.path.join(src_output_dir, "train.{}-{}.{}".format(src, tgt, src))
    tgt_output = os.path.join(tgt_output_dir, "train.{}-{}.{}".format(src, tgt, tgt))

    with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
        with open(src_output, "w") as src_fout, open(tgt_output, "w") as tgt_fout:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                src_fout.write(src_sent)
                tgt_fout.write(tgt_sent)
            src_fout.flush()
            tgt_fout.flush()


# get valid data
for lang_pair in lang_pairs:
    src, tgt = lang_pair.split("-")
    src_lang = lang_map[src]
    tgt_lang = lang_map[tgt]

    src_file = os.path.join("devtest", "{}.devtest".format(src_lang))
    tgt_file = os.path.join("devtest", "{}.devtest".format(tgt_lang))

    src_output_dir = os.path.join("valid", "{}-{}".format(src, tgt))
    tgt_output_dir = os.path.join("valid", "{}-{}".format(src, tgt))
    create_clean_dir(src_output_dir)
    create_clean_dir(tgt_output_dir)

    src_output = os.path.join(src_output_dir, "valid.{}-{}.{}".format(src, tgt, src))
    tgt_output = os.path.join(tgt_output_dir, "valid.{}-{}.{}".format(src, tgt, tgt))

    with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
        with open(src_output, "w") as src_fout, open(tgt_output, "w") as tgt_fout:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                src_fout.write(src_sent)
                tgt_fout.write(tgt_sent)
            src_fout.flush()
            tgt_fout.flush()



