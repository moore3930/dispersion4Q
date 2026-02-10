import os
import shutil
import random


def create_clean_dir(path):
    """
    Create a clean directory. If the directory exists, remove it first.
    :param path: Path of the directory to create.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


sample_num = 200
lang_pair = "en-ru"
src, tgt = lang_pair.split('-')

# 文件路径
a_sys_file = f"TowerInstruct-Mistral-7B-v0.2-beam5/{lang_pair}/hyp.{lang_pair}.{tgt}"
b_sys_file = f"1.0-1.0-0.0-5e-5-kiwi-xxl.score-beam5/0/{lang_pair}/hyp.{lang_pair}.{tgt}"
src_file = f"../src/llama_recipes/customer_data/wmt24_testset/test/{lang_pair}/test.{lang_pair}.{src}"

# 输出目录
output_dir = os.path.join("human_study", lang_pair)
create_clean_dir(output_dir)

# 加载数据
data = []
with open(a_sys_file, encoding='utf-8') as a_fin, \
     open(b_sys_file, encoding='utf-8') as b_fin, \
     open(src_file, encoding='utf-8') as src_fin:
    for a_sent, b_sent, src_sent in zip(a_fin, b_fin, src_fin):
        data.append((a_sent.strip(), b_sent.strip(), src_sent.strip()))

# 打乱并采样100条
random.seed(42)
random.shuffle(data)
sampled_data = data[:sample_num]

# 写入合并后的输出文件（使用 utf-8-sig 编码，避免 Excel 中文乱码）
output_file = os.path.join(output_dir, "human_annotation.tsv")
annotation_file = os.path.join(output_dir, "human_annotation_raw.tsv")
with open(output_file, 'w', encoding='utf-8-sig') as fout:
    with open(annotation_file, 'w', encoding='utf-8-sig') as fout2:
        fout.write("src\tleft\tright\tleft_label\tright_label\n")
        for a_sent, b_sent, src_sent in sampled_data:
            if random.random() < 0.5:
                left, right = a_sent, b_sent
                left_label, right_label = 'a', 'b'
            else:
                left, right = b_sent, a_sent
                left_label, right_label = 'b', 'a'
            fout.write(f"{src_sent}\t{left}\t{right}\t{left_label}\t{right_label}\n")
            fout2.write(f"{src_sent}\t{left}\t{right}\n")
