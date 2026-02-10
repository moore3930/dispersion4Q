import os
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Calculate correlation coefficient (Pearson or Spearman) and p-value.")
    parser.add_argument('--file1_name', type=str, required=True, default="kiwi-xxl.score", help="Path to the input1 file.")
    parser.add_argument('--file2_name', type=str, required=True, default="xcomet.score", help="Path to the input2 file.")
    parser.add_argument('--dir_name', type=str, required=True, default="gpt-4o-mini-16-1.0-98-new", help="dir name")
    parser.add_argument('--lang_pairs', type=str, required=True, default="en-zh,en-de,en-cs,en-ru,en-is,zh-en,de-en,cs-en,ru-en,is-en",
                        help="Path to the input2 file.")
    parser.add_argument('--output_file_name', type=str, required=True, help="Path to the output file.")

    # Parse arguments
    args = parser.parse_args()

    # Train set
    lang_pairs = args.lang_pairs.strip().split(',')

    for lang_pair in lang_pairs:
        file1 = os.path.join(args.dir_name, "train", lang_pair, args.file1_name)
        file2 = os.path.join(args.dir_name, "train", lang_pair, args.file2_name)
        output_file = os.path.join(args.dir_name, "train", lang_pair, args.output_file_name)

        with open(output_file, "w") as fout:
            with open(file1) as fin1, open(file2) as fin2:
                for line1, line2 in zip(fin1, fin2):
                    score1 = line1.strip().split("score: ")[-1]
                    score2 = line2.strip().split("score: ")[-1]
                    avg_score = (float(score1) + float(score2)) / 2
                    prefix = line1.strip().split("score: ")[0]
                    output_line = f"{prefix}score: {avg_score:.4f}"
                    fout.write(output_line + "\n")


    for lang_pair in lang_pairs:
        file1 = os.path.join(args.dir_name, "valid", lang_pair, args.file1_name)
        file2 = os.path.join(args.dir_name, "valid", lang_pair, args.file2_name)
        output_file = os.path.join(args.dir_name, "valid", lang_pair, args.output_file_name)

        with open(output_file, "w") as fout:
            with open(file1) as fin1, open(file2) as fin2:
                for line1, line2 in zip(fin1, fin2):
                    score1 = line1.strip().split("score: ")[-1]
                    score2 = line2.strip().split("score: ")[-1]
                    avg_score = (float(score1) + float(score2)) / 2
                    prefix = line1.strip().split("score: ")[0]
                    output_line = f"{prefix}score: {avg_score:.4f}"
                    fout.write(output_line + "\n")


if __name__ == "__main__":
    main()