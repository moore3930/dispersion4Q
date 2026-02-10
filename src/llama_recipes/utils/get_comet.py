from comet import download_model, load_from_checkpoint

# 下载并加载模型
model_path = download_model("Unbabel/XCOMET-XL")
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

# 准备数据进行评分
best_scores = []
all_scores = []
all_scores_flat = []
for i, src in enumerate(src_lines):
    # 获取当前 src 的 16 个翻译结果
    translations = hyp_lines[i * num_translations_per_src: (i + 1) * num_translations_per_src]

    # 构造数据格式
    data = [{"src": src, "mt": mt, "ref": tgt_lines[i]} for mt in translations]

    # 获取分数
    scores = model.predict(data, batch_size=8, gpus=1)
    all_scores.append(scores)
    all_scores_flat.extend(scores)

    # 获取当前 src 的最佳分数
    best_score = max(scores)
    best_scores.append(best_score)

# 计算平均最佳分数
average_best_score = sum(best_scores) / len(best_scores)

# 计算总体平均分数
overall_average_score = sum(all_scores_flat) / len(all_scores_flat)

# 打印所有得分和平均分数
for i, scores in enumerate(all_scores):
    print(f"源句子 {i + 1}: 所有得分 = {scores}")

print(f"平均最佳分数: {average_best_score}")
print(f"总体平均分数: {overall_average_score}")
