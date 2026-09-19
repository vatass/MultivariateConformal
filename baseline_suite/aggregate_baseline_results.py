#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate completed multivariate baseline results."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-prefix", default="all_models")
    return parser.parse_args()


def read_model_files(results_dir: Path, filename: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for model_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        path = model_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            if "model" not in frame.columns:
                frame.insert(0, "model", model_dir.name)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No {filename!r} files were found under {results_dir}."
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise NotADirectoryError(results_dir)

    overall = read_model_files(results_dir, "overall_metrics.csv")
    fold_metrics = read_model_files(results_dir, "fold_metrics.csv")
    runtimes = read_model_files(results_dir, "training_runtime_and_counts.csv")

    overall_path = results_dir / f"{args.output_prefix}_overall_metrics.csv"
    fold_path = results_dir / f"{args.output_prefix}_fold_metrics.csv"
    runtime_path = results_dir / f"{args.output_prefix}_runtime_and_counts.csv"
    overall.to_csv(overall_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    runtimes.to_csv(runtime_path, index=False)

    joint = overall[
        (overall["scope"].astype(str) == "joint")
        | (overall["biomarker"].astype(str) == "ALL_JOINT")
    ].copy()
    joint = joint.sort_values(
        ["method", "coverage_mean", "normalized_width_mean"],
        ascending=[True, False, True],
    )
    joint_path = results_dir / f"{args.output_prefix}_joint_summary.csv"
    joint.to_csv(joint_path, index=False)

    print(f"Saved: {overall_path.resolve()}")
    print(f"Saved: {fold_path.resolve()}")
    print(f"Saved: {runtime_path.resolve()}")
    print(f"Saved: {joint_path.resolve()}")
    print("\nJoint summary:")
    print(joint.to_string(index=False))


if __name__ == "__main__":
    main()
