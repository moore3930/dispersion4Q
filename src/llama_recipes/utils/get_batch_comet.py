from comet import download_model, load_from_checkpoint

# 下载并加载模型
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

# 定义文件路径
src_file_path = "/home/dwu18/projects/value_finetuning/src/llama_recipes/customer_data/wmt22_testset/test/en-zh/test.en-zh.en"
tgt_file_path = "/home/dwu18/projects/value_finetuning/src/llama_recipes/customer_data/wmt22_testset/test/en-zh/test.en-zh.zh"
hyp_file_path = "/home/dwu18/projects/value_finetuning/experiments/results/sampling_results/en-zh/hyp.en-zh.zh"

# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

src_lines = read_file(src_file_path)
tgt_lines = read_file(tgt_file_path)
hyp_lines = read_file(hyp_file_path)

# 每个 src 的翻译结果数量
num_translations_per_src = 16

# 检查文件长度是否匹配
if len(hyp_lines) != len(src_lines) * num_translations_per_src:
    raise ValueError("翻译结果的行数与源句子的行数不匹配。")

# 准备所有数据进行评分
all_data = []
for i, src in enumerate(src_lines):
    translations = hyp_lines[i * num_translations_per_src: (i + 1) * num_translations_per_src]
    all_data.extend([{"src": src, "mt": mt, "ref": tgt_lines[i]} for mt in translations])

# 分批调用 predict
batch_size = 64  # 根据 GPU 的显存调整批量大小
all_scores_flat = []
for start in range(0, len(all_data), batch_size):
    batch_data = all_data[start: start + batch_size]
    batch_scores = model.predict(batch_data, batch_size=batch_size, gpus=1)
    batch_scores = batch_scores[0][1]
    print(batch_scores)
    all_scores_flat.extend(batch_scores)

# 重新计算每个 src 的得分
best_scores = []
for i in range(len(src_lines)):
    scores = all_scores_flat[i * num_translations_per_src: (i + 1) * num_translations_per_src]
    print(scores)
    best_scores.append(max(scores))

# 计算平均分数
average_best_score = sum(best_scores) / len(best_scores)
overall_average_score = sum(all_scores_flat) / len(all_scores_flat)

# 打印所有得分和平均分数
for i in range(len(src_lines)):
    scores = all_scores_flat[i * num_translations_per_src: (i + 1) * num_translations_per_src]
    print(f"源句子 {i + 1}: 所有得分 = {scores}")

print(f"平均最佳分数: {average_best_score}")
print(f"总体平均分数: {overall_average_score}")