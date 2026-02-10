from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

src_file = "../src/llama_recipes/customer_data/wmt22_testset/test/en-zh/test.en-zh.en"
tgt_file = "../src/llama_recipes/customer_data/wmt22_testset/test/en-zh/test.en-zh.zh"

nllb_src_output = "./nllb/nllb.src"
nllb_tgt_output = "./nllb/nllb.tgt"
nllb_hyp_output = "./nllb/nllb.hyp"

batch_size = 10  # 可根据显存大小调整
output = []

# Read source and target sentences
with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
    src_sentences = [line.strip() for line in src_fin]
    tgt_sentences = [line.strip() for line in tgt_fin]

# Process in batches
for i in range(0, len(src_sentences), batch_size):
    batch_src = src_sentences[i:i + batch_size]
    batch_tgt = tgt_sentences[i:i + batch_size]

    inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=100)

    with torch.no_grad():  # 关闭梯度计算，减少显存占用
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"),
            max_length=100,
            num_beams=1,  # 使用 beam search
            early_stopping=True
        )

    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    for src, tgt, hyp in zip(batch_src, batch_tgt, translations):
        output.append([src, tgt, hyp])

    if len(output) >= 100:  # 只处理 100 句
        break

# Write output
with open(nllb_src_output, 'w') as src_fout, open(nllb_tgt_output, 'w') as tgt_fout, open(nllb_hyp_output, 'w') as hyp_fout:
    for sents in output:
        src_fout.write(sents[0] + "\n")
        tgt_fout.write(sents[1] + "\n")
        hyp_fout.write(sents[2] + "\n")

print("Translation complete!")
