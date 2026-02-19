#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

SCORE_RE = re.compile(r"score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
DEFAULT_METRIC_ORDER = ["comet", "xcomet", "kiwi", "kiwi-xl", "kiwi-xxl"]


def parse_score_file(path: Path):
    last_score = None
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            match = SCORE_RE.search(line.strip())
            if match:
                last_score = float(match.group(1))
    return last_score


def collect_rows(exp_dir: Path):
    epoch_dirs = sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not epoch_dirs:
        epoch_dirs = [exp_dir]

    rows = []
    metric_names = set()
    for epoch_dir in epoch_dirs:
        lang_dirs = sorted([p for p in epoch_dir.iterdir() if p.is_dir() and "-" in p.name])
        for lang_dir in lang_dirs:
            metric_scores = {}
            for score_file in sorted(lang_dir.glob("*.score")):
                metric_name = score_file.stem
                metric_names.add(metric_name)
                metric_scores[metric_name] = parse_score_file(score_file)
            rows.append(
                {
                    "epoch": epoch_dir.name,
                    "lang_pair": lang_dir.name,
                    "metrics": metric_scores,
                }
            )
    return rows, metric_names


def ordered_metrics(metric_names):
    ordered = [m for m in DEFAULT_METRIC_ORDER if m in metric_names]
    ordered.extend(sorted(metric_names - set(ordered)))
    return ordered


def format_value(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def average(values):
    numeric = [v for v in values if v is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def print_table_for_experiment(exp_dir: Path):
    rows, metric_names = collect_rows(exp_dir)
    print(f"\n=== {exp_dir.name} ===")

    if not rows:
        print("No result rows found.")
        return

    metrics = ordered_metrics(metric_names)
    headers = ["epoch", "lang_pair"] + metrics
    table = []
    for row in rows:
        line = [row["epoch"], row["lang_pair"]]
        for metric in metrics:
            line.append(format_value(row["metrics"].get(metric)))
        table.append(line)

    avg_line = ["-", "AVG"]
    for metric in metrics:
        avg_line.append(format_value(average([r["metrics"].get(metric) for r in rows])))
    table.append(avg_line)

    print_markdown_table(headers, table)


def print_markdown_table(headers, rows):
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = []
    for i, header in enumerate(headers):
        max_cell = max((len(row[i]) for row in str_rows), default=0)
        widths.append(max(len(header), max_cell))

    def fmt_row(cells):
        return "| " + " | ".join(str(cells[i]).ljust(widths[i]) for i in range(len(cells))) + " |"

    print(fmt_row(headers))
    print("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    for row in str_rows:
        print(fmt_row(row))


def main():
    parser = argparse.ArgumentParser(
        description="Print score tables for experiment directories."
    )
    parser.add_argument(
        "--results-root",
        default="results/TowerInstruct-Mistral-7B-v0.2/wmt24_testset",
        help="Directory that contains experiment subdirectories.",
    )
    parser.add_argument(
        "--pattern",
        default="*-beam1*",
        help="Glob pattern used to match experiment directories.",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Results root not found: {root}")

    matched = sorted([p for p in root.glob(args.pattern) if p.is_dir()])
    if not matched:
        print(f"No experiment directories matched pattern '{args.pattern}' under {root}")
        return

    print(f"Matched {len(matched)} experiment directory(ies) under {root}")
    for exp_dir in matched:
        print_table_for_experiment(exp_dir)


if __name__ == "__main__":
    main()
