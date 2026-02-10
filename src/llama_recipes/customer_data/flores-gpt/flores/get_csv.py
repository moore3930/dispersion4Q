import os
import csv

# 定义文件夹路径
folders = [("gpt-4o", "gpt-4o"), ("ref", "ref"), ("alma-1", "alma-1")]

# 初始化CSV字段和数据存储
fields = ["lp", "type", "src", "mt", "raw", "score"]
data = []

# 遍历每个文件夹
for base_dir, data_type in folders:
    lang_pair = "en-de"
    lang_dir = os.path.join(base_dir, lang_pair)
    if not os.path.isdir(lang_dir):
        continue

    # 构建文件路径
    src_file = os.path.join(lang_dir, "src")
    tgt_file = os.path.join(lang_dir, "tgt")
    score_file = os.path.join(lang_dir, "score")

    # 确保文件存在
    if not all(os.path.exists(file) for file in [src_file, tgt_file, score_file]):
        print(f"文件缺失，跳过：{lang_pair} in {data_type}")
        continue

    # 读取文件内容
    with open(src_file, "r", encoding="utf-8") as src_f, \
            open(tgt_file, "r", encoding="utf-8") as tgt_f, \
            open(score_file, "r", encoding="utf-8") as score_f:

        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
        score_lines = score_f.readlines()
        score_lines = score_lines[:-1]  # 去掉最后一行（可能为空）

        # 将数据添加到列表
        for src, tgt, score in zip(src_lines, tgt_lines, score_lines):
            score_value = float(score.strip().split("score: ")[-1])
            data.append([lang_pair, data_type, src.strip(), tgt.strip(), score_value, score_value])


# 写入CSV文件
output_csv = "merged_data.{}.csv".format(lang_pair)

with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(fields)  # 写入字段名
    writer.writerows(data)  # 写入数据

print(f"数据已成功整合到 {output_csv}")
