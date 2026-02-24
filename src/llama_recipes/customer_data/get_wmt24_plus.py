from __future__ import annotations

import argparse
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download google/wmt24pp and export bitext files to "
            "src/llama_recipes/customer_data/wmt24_testset/test/<lp>/."
        )
    )
    parser.add_argument(
        "--dataset",
        default="google/wmt24pp",
        help="HF dataset name (default: google/wmt24pp).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HF split to export (default: train).",
    )
    parser.add_argument(
        "--out_root",
        default="src/llama_recipes/customer_data/wmt24pp_testset/test",
        help="Output root directory for bitext files.",
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="Optional comma-separated pair list (e.g., en-de,en-zh). Empty means all pairs.",
    )
    parser.add_argument(
        "--include_bad_source",
        action="store_true",
        help="Include rows marked with is_bad_source=true.",
    )
    parser.add_argument(
        "--existing-pair-policy",
        choices=("skip", "overwrite"),
        default="overwrite",
        help=(
            "Behavior when output files for a pair already exist: "
            "'skip' leaves the pair untouched, "
            "'overwrite' rewrites from downloaded data."
        ),
    )
    parser.add_argument(
        "--target-column",
        default="target",
        help=(
            "Target text column in HF examples (default: target). "
            "If missing, the script falls back to original_target."
        ),
    )
    return parser.parse_args()


def build_pair_config_map(dataset_name: str) -> dict[str, str]:
    pair_to_config: dict[str, str] = {}
    for config_name in sorted(get_dataset_config_names(dataset_name)):
        pair = config_name.split("_", 1)[0]
        if "-" not in pair:
            continue
        if pair in pair_to_config:
            # Keep the first config if there are duplicates for the same pair.
            continue
        pair_to_config[pair] = config_name
    return pair_to_config


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    pair_to_config = build_pair_config_map(args.dataset)

    requested_pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if requested_pairs:
        missing = [p for p in requested_pairs if p not in pair_to_config]
        if missing:
            raise ValueError(
                f"Requested pair(s) not found in {args.dataset}: {', '.join(missing)}"
            )
        pairs = {pair: pair_to_config[pair] for pair in requested_pairs}
    else:
        pairs = pair_to_config

    print(f"Found {len(pair_to_config)} pair->config mappings in {args.dataset}")
    print(f"Exporting {len(pairs)} pairs to {out_root}")

    for pair, config_name in pairs.items():
        src, tgt = pair.split("-", 1)
        out_dir = out_root / pair
        src_file = out_dir / f"test.{pair}.{src}"
        tgt_file = out_dir / f"test.{pair}.{tgt}"

        if (
            args.existing_pair_policy == "skip"
            and src_file.exists()
            and tgt_file.exists()
        ):
            print(f"{pair} <- {config_name}: skipped (existing files)")
            continue

        ds = load_dataset(args.dataset, config_name, split=args.split)

        out_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        with src_file.open("w", encoding="utf-8") as src_fout, tgt_file.open(
            "w", encoding="utf-8"
        ) as tgt_fout:
            for row in ds:
                if not args.include_bad_source and row.get("is_bad_source", False):
                    continue
                src_text = str(row.get("source", "")).strip()
                tgt_value = row.get(args.target_column)
                if tgt_value is None:
                    tgt_value = row.get("original_target", "")
                tgt_text = str(tgt_value).strip()
                if not src_text or not tgt_text:
                    continue
                src_fout.write(src_text + "\n")
                tgt_fout.write(tgt_text + "\n")
                kept += 1

        print(f"{pair} <- {config_name}: wrote {kept} lines")


if __name__ == "__main__":
    main()
