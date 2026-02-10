import os

train_reference_dir = "/Users/moore/workplace/projects/call_gpt/dataset/flores"
valid_reference_dir = "/Users/moore/workplace/projects/call_gpt/dataset/NTREX"
train_gpt_dir = "/Users/moore/workplace/projects/value_finetuning/src/llama_recipes/customer_data/calibration/gpt-4o-mini-16-1.0-98-new/train"
valid_gpt_dir = "/Users/moore/workplace/projects/value_finetuning/src/llama_recipes/customer_data/calibration/gpt-4o-mini-16-1.0-98-new/valid"

lang_pairs = ["en-de", "en-es", "en-ru", "en-zh", "en-fr", "en-nl", "en-it", "en-pt", "en-ko"]

for lang_pair in lang_pairs:
    reference_src_file = os.path.join(valid_reference_dir, lang_pair, "src")
    reference_tgt_file = os.path.join(valid_reference_dir, lang_pair, "tgt")

    ref_dict = {}
    with open(reference_src_file) as src_fin, open(reference_tgt_file) as tgt_fin:
        for src, tgt in zip(src_fin, tgt_fin):
            src = src.strip()
            tgt = tgt.strip()
            ref_dict[src] = tgt

    gpt_src_file = os.path.join(valid_gpt_dir, lang_pair, "src")
    gpt_tgt_file = os.path.join(valid_gpt_dir, lang_pair, "tgt")
    gpt_ref_file = os.path.join(valid_gpt_dir, lang_pair, "ref")
    with open(gpt_src_file) as gpt_src_fin, open(gpt_tgt_file) as gpt_tgt_fin:
        with open(gpt_ref_file, 'w') as gpt_ref_fout:
            for src, tgt in zip(gpt_src_fin, gpt_tgt_fin):
                ref = ref_dict[src.strip()]
                gpt_ref_fout.write(ref.strip() + "\n")
