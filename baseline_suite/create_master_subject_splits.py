#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from split_utils import create_master_split_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one reusable subject-level fit/calibration/test split table "
            "for every predictive baseline."
        )
    )
    parser.add_argument("--folds-dir", default="./data/folds")
    parser.add_argument("--output", default="./data/master_subject_splits.csv")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    table = create_master_split_table(
        folds_dir=Path(args.folds_dir),
        n_folds=args.n_folds,
        fold_start=args.fold_start,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    table.to_csv(output, index=False)

    summary = (
        table.groupby(["fold", "partition"])["id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    metadata = {
        "folds_dir": args.folds_dir,
        "n_folds": args.n_folds,
        "fold_start": args.fold_start,
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
        "split_algorithm": (
            "For fold j, np.random.default_rng(seed+j) selects round(fraction*N) "
            "positions without replacement from the original fold-training ID order."
        ),
    }
    metadata_path = output.with_name(output.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved master split table: {output.resolve()}")
    print(f"Saved split summary: {summary_path.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
