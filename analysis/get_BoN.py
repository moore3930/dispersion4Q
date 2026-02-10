import argparse
import os

def process_files(input_dir, output_dir, lang_pairs, bon_size):
    os.makedirs(output_dir, exist_ok=True)

    lang_pairs = lang_pairs.strip().split(',')
    for lang_pair in lang_pairs:
        os.makedirs(os.path.join(output_dir, lang_pair), exist_ok=True)
        with open(os.path.join(input_dir, lang_pair, "src"), 'r', encoding='utf-8') as src_f, \
             open(os.path.join(input_dir, lang_pair, "tgt"), 'r', encoding='utf-8') as tgt_f, \
             open(os.path.join(input_dir, lang_pair, "kiwi-xxl.score"), 'r', encoding='utf-8') as score_f, \
             open(os.path.join(output_dir, lang_pair, "src"), 'w', encoding='utf-8') as out_src_f, \
             open(os.path.join(output_dir, lang_pair, "tgt"), 'w', encoding='utf-8') as out_tgt_f:

            src_lines = []
            tgt_lines = []
            score_lines = []

            # 先把所有 score 读出来，去掉最后多余的一行
            all_scores = score_f.readlines()
            if len(all_scores) > 0:
                all_scores = all_scores[:-1]

            for idx, (src_line, tgt_line, score_line) in enumerate(zip(src_f, tgt_f, all_scores)):
                src_lines.append(src_line.strip())
                tgt_lines.append(tgt_line.strip())
                score_lines.append(float(score_line.strip().split("score: ")[-1]))

                if len(src_lines) == 100:
                    max_index = score_lines[:bon_size].index(max(score_lines[:bon_size]))
                    out_src_f.write(src_lines[max_index] + "\n")
                    out_tgt_f.write(tgt_lines[max_index] + "\n")
                    src_lines.clear()
                    tgt_lines.clear()
                    score_lines.clear()


def main():
    parser = argparse.ArgumentParser(description="Select top-1 entry for every N lines from input files.")
    parser.add_argument('--input_dir', required=True, help='Output dir')
    parser.add_argument('--output_dir', required=True, help='Output dir')
    parser.add_argument('--lang_pairs', type=str, default='en-de,en-zh,en-es,en-ru', help='Language pairs')
    parser.add_argument('--bon_size', type=int, required=True, help='Number of lines per group')

    args = parser.parse_args()

    process_files(args.input_dir, args.output_dir, args.lang_pairs, args.bon_size)


if __name__ == "__main__":
    main()
