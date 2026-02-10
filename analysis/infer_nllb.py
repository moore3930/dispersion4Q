from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

src_file = "../src/llama_recipes/customer_data/wmt22_testset/test/en-zh/test.en-zh.en"
tgt_file = "../src/llama_recipes/customer_data/wmt22_testset/test/en-zh/test.en-zh.zh"

nllb_src_output = "./nllb.src"
nllb_tgt_output = "./nllb.tgt"
nllb_hyp_output = "./nllb.hyp"


output = []
cnt = 0

with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
    for src_sent, tgt_sent in zip(src_fin, tgt_fin):
        inputs = tokenizer(src_sent.strip(), return_tensors="pt")

        for _ in range(10):  # 进行 10 次采样
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"),  # 强制生成法语
                max_length=100,  # 设定最大生成长度
                do_sample=True,  # 启用采样
                top_k=100,  # 使用 top-k 采样
                top_p=0.98  # 使用核采样 (nucleus sampling)
            )
            translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

            cur = [src_sent.strip(), tgt_sent.strip(), translation.strip()]
            output.append(cur)

        cnt += 1
        if cnt >= 100:
            break

# with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
#     for src_sent, tgt_sent in zip(src_fin, tgt_fin):
#         inputs = tokenizer(src_sent.strip(), return_tensors="pt")
#         translated_tokens = model.generate(
#             **inputs,
#             forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"),
#             max_length=30,
#             num_return_sequences=10,  # Generate 10 different outputs
#             do_sample=True,  # Enable sampling
#             top_k=50,  # Use top-k sampling
#             top_p=0.98  # Use nucleus sampling
#         )
#         translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
#
#         for hyp in translations:
#             cur = [src_sent.strip(), tgt_sent.strip(), hyp.strip()]
#             output.append(cur)
#
#         cnt += 1
#         if cnt >= 100:
#             break

with open(nllb_src_output, 'w') as src_fout, open(nllb_tgt_output, 'w') as tgt_fout, open(nllb_hyp_output, 'w') as hyp_fout:
    for sents in output:
        src_fout.write(sents[0] + "\n")
        tgt_fout.write(sents[1] + "\n")
        hyp_fout.write(sents[2] + "\n")


