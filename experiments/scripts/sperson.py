import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau


def calculate_kendall(file_path, column_x=-2, column_y=-1, delimiter='\t'):
    """
    Calculate Kendall's tau correlation coefficient and p-value from a file.
    """
    org_list = []
    tgt_list = []

    with open(file_path, 'r') as fin:
        for line in fin:
            line_array = line.strip().split(delimiter)
            org_list.append(float(line_array[column_x]))
            tgt_list.append(float(line_array[column_y]))

    # Calculate Kendall's tau correlation coefficient and p-value
    tau_corr, p_value = kendalltau(org_list, tgt_list)
    return tau_corr, p_value


def calculate_spearman(file_path, column_x=-2, column_y=-1, delimiter='\t'):
    """
    Calculate Spearman's rank correlation coefficient and p-value from a file.
    """
    org_list = []
    tgt_list = []

    with open(file_path, 'r') as fin:
        for line in fin:
            line_array = line.strip().split(delimiter)
            org_list.append(float(line_array[column_x]))
            tgt_list.append(float(line_array[column_y]))

    # Calculate Spearman's rank correlation coefficient and p-value
    spearman_corr, p_value = spearmanr(org_list, tgt_list)
    return spearman_corr, p_value

def calculate_pearson(file_path, column_x=-2, column_y=-1, delimiter='\t'):
    """
    Calculate Pearson's correlation coefficient and p-value from a file.
    """
    org_list = []
    tgt_list = []

    with open(file_path, 'r') as fin:
        for line in fin:
            line_array = line.strip().split(delimiter)
            org_list.append(float(line_array[column_x]))
            tgt_list.append(float(line_array[column_y]))

    # Calculate Pearson's correlation coefficient and p-value
    pearson_corr, p_value = pearsonr(org_list, tgt_list)
    return pearson_corr, p_value

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate correlation coefficient (Pearson or Spearman) and p-value.")
    parser.add_argument('--file_name', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--column_x', type=int, default=-5, help="Column index for the first variable (default: -2).")
    parser.add_argument('--column_y', type=int, default=-1, help="Column index for the second variable (default: -1).")
    parser.add_argument('--delimiter', type=str, default='\t', help="Delimiter used to split columns (default: '\\t').")
    parser.add_argument('--method', type=str, default='spearman', choices=['pearson', 'spearman', 'kendall'],
                        help="Method for correlation calculation (default: 'pearson').")

    # Parse arguments
    args = parser.parse_args()

    # Choose the method
    if args.method == 'pearson':
        corr, p_value = calculate_pearson(args.file_name, args.column_x, args.column_y, args.delimiter)
        corr = float(corr) * 100
        method_name = "Pearson's correlation"
    elif args.method == 'spearman':
        corr, p_value = calculate_spearman(args.file_name, args.column_x, args.column_y, args.delimiter)
        corr = float(corr) * 100
        method_name = "Spearman's rank correlation"
    elif args.method == 'kendall':
        corr, p_value = calculate_kendall(args.file_name, args.column_x, args.column_y, args.delimiter)
        corr = float(corr) * 100
        method_name = "kendall's rank correlation"

    # Print results
    print(f"{method_name} coefficient: {corr}")
    print(f"P-value: {p_value}")

if __name__ == "__main__":
    main()
