import pandas as pd
import numpy as np
import os


lang_pairs = ["en-zh", "en-de", "en-ru"]
alpha = 1
dir_name = "selected_data_{}".format(alpha)
os.makedirs(dir_name, exist_ok=True)
for subfolder in lang_pairs:
    os.makedirs(os.path.join(dir_name, subfolder), exist_ok=True)

for lang_pair in lang_pairs:

    src, tgt = lang_pair.split('-')
    file_1 = "flores/merged_data.{}.csv".format(lang_pair)
    file_2 = "flores-scores/{}/score.{}.{}".format(lang_pair, lang_pair, tgt)

    # Load the .csv file
    file_path = file_1
    df = pd.read_csv(file_path)

    data_dict = {}

    list1 = []
    list2 = []

    for index, row in df.iterrows():
        list1.append(row)

    with open(file_2, 'r') as fin:
        for line in fin:
            line_array = line.strip().split('\t')
            list2.append(line_array)

    id1 = 0
    id2 = 0
    list3 = []
    while id1 < len(list1) and id2 < len(list2):
        row1 = list1[id1]
        row2 = list2[id2]
        if row1['mt'] != row2[0]:
            id1 += 1
        else:
            list3.append(row1)
            id1 += 1
            id2 += 1


    for row1, row2 in zip(list3, list2):
        if row1['src'] in data_dict:
            cur_dict = data_dict[row1['src']]
        else:
            cur_dict = {}

        cur_dict[row1['mt']] = {"type": row1['type'], "logprob": row2[1], "score": row2[3]}
        data_dict[row1['src']] = cur_dict

    # select data
    cnt=0
    output = []
    for src in data_dict:
        cur_dict = data_dict[src]
        flag = False
        for tgt in cur_dict:
            if cur_dict[tgt]['type'] == "alma-1":
                flag = True
        if not flag:
            print(cur_dict)
            continue

        # get base score
        max_comet = -1
        for tgt in cur_dict:
            base_score = float(cur_dict[tgt]['score'])
            if base_score > max_comet:
                max_comet = base_score
            if cur_dict[tgt]['type'] == 'alma-1':
                base_score = float(cur_dict[tgt]['score'])
                base_logprob = float(cur_dict[tgt]['logprob'])

        max_rank_score = -100
        select_tgt = ""
        select_logprob = 0
        select_comet = 0
        select_rank_score = 0
        for tgt in cur_dict:
            score = float(cur_dict[tgt]['score'])
            logprob = float(cur_dict[tgt]['logprob'])
            if score < base_score:
                continue
            rank_score = (score - base_score) * (base_logprob - logprob)
            rank_score = np.power(score - base_score, alpha) * np.log(1 + np.exp(-(logprob - base_logprob)))

            if rank_score > max_rank_score:
                max_rank_score = rank_score
                select_tgt = tgt
                select_logprob = logprob
                select_comet = score
                select_rank_score = rank_score
                select_type = cur_dict[tgt]['type']

        if select_comet != max_comet:
            cnt += 1

        print("{}\t{}\t{}\t{}\t{}\t{}".format(src, select_tgt, select_type, select_rank_score, base_score, select_comet))
        output.append((src, select_tgt))
    print("Changed Numbers: {}".format(cnt))

    # write
    src, tgt = lang_pair.split('-')
    with open("./selected_data_{}/{}/train.{}.{}".format(alpha, lang_pair, lang_pair, src), 'w') as s_fout, \
            open("./selected_data_{}/{}/train.{}.{}".format(alpha, lang_pair, lang_pair, tgt), 'w') as t_fout:
        for tuple in output:
            s_sent, t_sent = tuple
            s_fout.write(s_sent + "\n")
            t_fout.write(t_sent + "\n")