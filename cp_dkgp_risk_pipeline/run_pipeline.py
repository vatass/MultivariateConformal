#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from conformal_features import (
    add_joint_intervals,
    build_subject_biomarker_scores,
    calculate_joint_qhat,
    derive_roc_features,
    feature_sets,
    prediction_frame,
    summarize_joint_coverage,
)
from data_pipeline import (
    Partition,
    count_unique_times,
    finite_target_view,
    load_binary_labels,
    load_biomarkers,
    load_dataset,
    make_fixed_grid,
    prepare_three_partitions,
    validate_target_indices,
)
from dkgp_core import (
    DKGPConfig,
    predict_dkgp,
    save_checkpoint,
    set_seed,
    train_dkgp,
)
from lr_modeling import run_three_lr_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end CP-DKGP high-risk classification pipeline with an untouched "
            "outer test set, five-fold development cross-fitting, joint multivariate "
            "conformal bands, RoC/RoCB feature construction, and three logistic "
            "regression experiments."
        )
    )

    # Data and labels.
    parser.add_argument("--file", required=True, help="Trajectory CSV with X and Y.")
    parser.add_argument("--labels-file", required=True, help="One row per eligible subject.")
    parser.add_argument("--subject-column", default="anon_id")
    parser.add_argument("--label-id-column", default="anon_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--positive-value",
        default=None,
        help="Optional positive label value when labels are not already numeric 0/1.",
    )
    parser.add_argument("--biomarkers-json", required=True)
    parser.add_argument("--output-dir", required=True)

    # Subject-level design.
    parser.add_argument("--outer-test-fraction", type=float, default=0.20)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--outer-split-file", default=None)
    parser.add_argument("--seed", type=int, default=42)

    # CP-DKGP.
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--kernel", default="RBF")
    parser.add_argument("--mean", default="Constant")
    parser.add_argument(
        "--uncertainty",
        choices=("epistemic", "predictive"),
        default="epistemic",
    )
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--save-models", action="store_true")

    # RoC feature construction.
    parser.add_argument(
        "--feature-time-mode",
        choices=("observed", "fixed_grid"),
        default="observed",
        help=(
            "observed uses each subject's available random visit times. fixed_grid "
            "repeats baseline inputs on a common time grid; this avoids using visit "
            "schedule as a classifier input but does not directly inherit observed-time "
            "coverage without an additional grid-validity argument."
        ),
    )
    parser.add_argument(
        "--feature-grid",
        default=None,
        help='Comma-separated grid, for example "0,12,24,36".',
    )
    parser.add_argument("--allow-time-varying-inputs", action="store_true")
    parser.add_argument("--invariant-tolerance", type=float, default=1e-6)
    parser.add_argument("--roc-min-time", type=float, default=None)
    parser.add_argument("--roc-max-time", type=float, default=None)
    parser.add_argument(
        "--time-scale-factor",
        type=float,
        default=1.0,
        help="Multiply slopes by this value; use 12 for annual slopes when time is months.",
    )

    # Logistic regression.
    parser.add_argument("--lr-c-grid", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--lr-cv-folds", type=int, default=5)
    parser.add_argument(
        "--lr-class-weight",
        choices=("balanced", "none"),
        default="balanced",
    )
    parser.add_argument(
        "--threshold-rule",
        choices=("youden", "0.5"),
        default="youden",
    )
    parser.add_argument("--lr-max-iter", type=int, default=5000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed cross-fit/final stage outputs when present.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.outer_test_fraction < 1:
        raise ValueError("outer-test-fraction must be in (0, 1).")
    if args.crossfit_folds < 2:
        raise ValueError("crossfit-folds must be at least 2.")
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("calibration-fraction must be in (0, 1).")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if args.time_scale_factor <= 0:
        raise ValueError("time-scale-factor must be positive.")
    if args.invariant_tolerance < 0:
        raise ValueError("invariant-tolerance cannot be negative.")
    if args.feature_time_mode == "fixed_grid" and not args.feature_grid:
        raise ValueError("--feature-grid is required for fixed_grid mode.")


def parse_float_list(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def split_calibration_subjects(
    subject_ids: Sequence[str],
    fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    unique = list(dict.fromkeys(str(value) for value in subject_ids))
    if len(unique) < 2:
        raise ValueError("At least two eligible subjects are required for fit/calibration.")
    n_cal = int(round(fraction * len(unique)))
    n_cal = max(1, min(len(unique) - 1, n_cal))
    rng = np.random.default_rng(seed)
    positions = set(rng.choice(len(unique), size=n_cal, replace=False).tolist())
    calibration = [value for index, value in enumerate(unique) if index in positions]
    remaining = [value for index, value in enumerate(unique) if index not in positions]
    return remaining, calibration


def make_or_load_outer_split(
    *,
    labels: pd.DataFrame,
    args: argparse.Namespace,
    split_dir: Path,
) -> pd.DataFrame:
    split_dir.mkdir(parents=True, exist_ok=True)
    saved = split_dir / "outer_subject_split.csv"

    candidate = Path(args.outer_split_file) if args.outer_split_file else None
    if candidate is not None:
        split = pd.read_csv(candidate)
    elif args.resume and saved.exists():
        split = pd.read_csv(saved)
    else:
        development_ids, final_test_ids = train_test_split(
            labels["id"].to_numpy(dtype=str),
            test_size=args.outer_test_fraction,
            random_state=args.seed,
            shuffle=True,
            stratify=labels["label"].to_numpy(dtype=int),
        )
        split = labels.copy()
        final_set = set(final_test_ids)
        split["outer_role"] = np.where(
            split["id"].isin(final_set), "final_test", "development"
        )

    required = {"id", "outer_role"}
    missing = required.difference(split.columns)
    if missing:
        raise KeyError(f"Outer split file is missing columns: {sorted(missing)}")
    split = split.copy()
    split["id"] = split["id"].astype(str)
    split = split.merge(labels, on="id", how="inner", suffixes=("", "_label"))
    if "label_label" in split.columns:
        if "label" in split.columns and not np.array_equal(
            split["label"].to_numpy(), split["label_label"].to_numpy()
        ):
            raise ValueError("Outer split labels disagree with the supplied label file.")
        split = split.drop(columns=["label_label"])
    if not set(split["outer_role"].unique()).issubset({"development", "final_test"}):
        raise ValueError("outer_role must be development or final_test.")
    if split["id"].duplicated().any():
        raise ValueError("Outer split contains duplicated subject IDs.")
    expected_ids = set(labels["id"].astype(str))
    split_ids = set(split["id"].astype(str))
    if split_ids != expected_ids:
        missing_ids = sorted(expected_ids.difference(split_ids))[:10]
        extra_ids = sorted(split_ids.difference(expected_ids))[:10]
        raise ValueError(
            "Outer split must contain every currently eligible labeled subject exactly "
            f"once. Missing examples: {missing_ids}; extra examples: {extra_ids}."
        )
    for role, group in split.groupby("outer_role"):
        if group["label"].nunique() != 2:
            raise ValueError(f"Both classes must be represented in {role}.")
    split.to_csv(saved, index=False)
    return split.sort_values(["outer_role", "id"]).reset_index(drop=True)


def save_split_manifest(
    *,
    path: Path,
    stage: str,
    fold: int,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    target_ids: Sequence[str],
    target_role: str,
) -> None:
    rows = []
    for role, ids in (
        ("dkgp_fit", fit_ids),
        ("cp_calibration", calibration_ids),
        (target_role, target_ids),
    ):
        rows.extend(
            {
                "stage": stage,
                "fold": fold,
                "role": role,
                "id": str(subject_id),
            }
            for subject_id in ids
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_cp_dkgp_stage(
    *,
    stage: str,
    fold: int,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    target_ids: Sequence[str],
    data: pd.DataFrame,
    subject_column: str,
    biomarkers: Mapping[str, int],
    alpha: float,
    device: torch.device,
    config: DKGPConfig,
    output_dir: Path,
    seed: int,
    feature_time_mode: str,
    fixed_grid_times: np.ndarray | None,
    allow_time_varying_inputs: bool,
    invariant_tolerance: float,
    roc_min_time: float | None,
    roc_max_time: float | None,
    time_scale_factor: float,
    save_models: bool,
    resume: bool,
) -> Dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features_wide.csv"
    required_outputs = [
        feature_path,
        output_dir / "target_observed_intervals_long.csv",
        output_dir / "conformal_quantile.csv",
    ]
    if resume and all(path.exists() for path in required_outputs):
        print(f"Reusing completed stage: {output_dir}")
        result = {
            "features_wide": pd.read_csv(feature_path),
            "target_observed_intervals": pd.read_csv(
                output_dir / "target_observed_intervals_long.csv"
            ),
            "qhat": pd.read_csv(output_dir / "conformal_quantile.csv"),
        }
        optional = {
            "feature_intervals": output_dir / "feature_intervals_long.csv",
            "roc_features_long": output_dir / "roc_features_long.csv",
            "target_joint_coverage": output_dir / "target_joint_coverage_by_subject.csv",
        }
        for key, path in optional.items():
            result[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return result

    save_split_manifest(
        path=output_dir / "subject_roles.csv",
        stage=stage,
        fold=fold,
        fit_ids=fit_ids,
        calibration_ids=calibration_ids,
        target_ids=target_ids,
        target_role="feature_holdout" if stage == "development_crossfit" else "final_test",
    )

    fit_partition, calibration_partition, target_partition = prepare_three_partitions(
        data=data,
        fit_ids=fit_ids,
        calibration_ids=calibration_ids,
        target_ids=target_ids,
        subject_column=subject_column,
    )
    validate_target_indices(fit_partition.y, biomarkers)

    if feature_time_mode == "fixed_grid":
        assert fixed_grid_times is not None
        grid_x, grid_ids, grid_times = make_fixed_grid(
            partition=target_partition,
            times=fixed_grid_times,
            invariant_tolerance=invariant_tolerance,
            allow_time_varying_inputs=allow_time_varying_inputs,
        )
    else:
        grid_x = None
        grid_ids = None
        grid_times = None

    calibration_frames: List[pd.DataFrame] = []
    target_observed_frames: List[pd.DataFrame] = []
    feature_grid_frames: List[pd.DataFrame] = []
    history_records: List[dict] = []

    for position, (biomarker, target_index) in enumerate(biomarkers.items(), start=1):
        print(
            f"  [{position}/{len(biomarkers)}] {biomarker} "
            f"(target index {target_index})"
        )
        model_seed = seed + fold * 10_000 + position * 100
        set_seed(model_seed)

        fit_x, fit_y, fit_subject_rows, _ = finite_target_view(
            fit_partition, target_index
        )
        cal_x, cal_y, cal_subject_rows, _ = finite_target_view(
            calibration_partition, target_index
        )
        target_x, target_y, target_subject_rows, _ = finite_target_view(
            target_partition, target_index
        )

        model, likelihood, losses = train_dkgp(
            train_x_cpu=fit_x,
            train_y_cpu=fit_y,
            device=device,
            config=config,
            progress_description=f"{stage}:{fold}:{biomarker}",
        )

        cal_prediction = predict_dkgp(
            model=model,
            likelihood=likelihood,
            x_cpu=cal_x,
            device=device,
            config=config,
        )
        target_prediction = predict_dkgp(
            model=model,
            likelihood=likelihood,
            x_cpu=target_x,
            device=device,
            config=config,
        )

        calibration_frames.append(
            prediction_frame(
                stage=stage,
                fold=fold,
                split="calibration",
                subject_ids=cal_subject_rows,
                biomarker=biomarker,
                target_index=target_index,
                times=cal_x[:, -1].numpy(),
                y=cal_y.numpy(),
                **cal_prediction,
            )
        )
        target_observed_frames.append(
            prediction_frame(
                stage=stage,
                fold=fold,
                split="target_observed",
                subject_ids=target_subject_rows,
                biomarker=biomarker,
                target_index=target_index,
                times=target_x[:, -1].numpy(),
                y=target_y.numpy(),
                **target_prediction,
            )
        )

        if grid_x is not None:
            grid_prediction = predict_dkgp(
                model=model,
                likelihood=likelihood,
                x_cpu=grid_x,
                device=device,
                config=config,
            )
            feature_grid_frames.append(
                prediction_frame(
                    stage=stage,
                    fold=fold,
                    split="target_fixed_grid",
                    subject_ids=grid_ids,
                    biomarker=biomarker,
                    target_index=target_index,
                    times=grid_times,
                    y=None,
                    **grid_prediction,
                )
            )

        history_records.extend(
            {
                "stage": stage,
                "fold": fold,
                "biomarker": biomarker,
                "target_index": target_index,
                "iteration": iteration + 1,
                "negative_mll": loss,
                "fit_observations": len(fit_y),
                "fit_subjects": len(np.unique(fit_subject_rows)),
            }
            for iteration, loss in enumerate(losses)
        )

        if save_models:
            save_checkpoint(
                path=output_dir / "models" / f"{biomarker}.pt",
                model=model,
                likelihood=likelihood,
                train_x_cpu=fit_x,
                train_y_cpu=fit_y,
                biomarker=biomarker,
                target_index=target_index,
                config=config,
            )

        del model, likelihood
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    calibration_predictions = pd.concat(calibration_frames, ignore_index=True)
    target_observed_predictions = pd.concat(target_observed_frames, ignore_index=True)
    calibration_predictions.to_csv(
        output_dir / "calibration_predictions_long.csv", index=False
    )
    target_observed_predictions.to_csv(
        output_dir / "target_observed_predictions_long.csv", index=False
    )
    pd.DataFrame(history_records).to_csv(
        output_dir / "training_history.csv", index=False
    )

    by_biomarker_scores, common_scores = build_subject_biomarker_scores(
        calibration_predictions,
        list(biomarkers.keys()),
    )
    by_biomarker_scores.to_csv(
        output_dir / "calibration_scores_by_subject_biomarker.csv", index=False
    )
    common_scores.to_csv(
        output_dir / "calibration_joint_scores_common_subjects.csv", index=False
    )

    qhat, qhat_table = calculate_joint_qhat(
        common_scores=common_scores,
        alpha=alpha,
        stage=stage,
        fold=fold,
    )
    qhat_table.to_csv(output_dir / "conformal_quantile.csv", index=False)

    target_observed_intervals = add_joint_intervals(
        target_observed_predictions, qhat
    )
    target_observed_intervals.to_csv(
        output_dir / "target_observed_intervals_long.csv", index=False
    )

    trajectory_coverage, joint_coverage = summarize_joint_coverage(
        target_observed_intervals, list(biomarkers.keys())
    )
    trajectory_coverage.to_csv(
        output_dir / "target_trajectory_coverage_by_biomarker.csv", index=False
    )
    joint_coverage.to_csv(
        output_dir / "target_joint_coverage_by_subject.csv", index=False
    )

    if feature_time_mode == "observed":
        feature_intervals = target_observed_intervals.copy()
    else:
        feature_grid_predictions = pd.concat(feature_grid_frames, ignore_index=True)
        feature_grid_predictions.to_csv(
            output_dir / "feature_grid_predictions_long.csv", index=False
        )
        feature_intervals = add_joint_intervals(feature_grid_predictions, qhat)
    feature_intervals.to_csv(
        output_dir / "feature_intervals_long.csv", index=False
    )

    roc_long, features_wide = derive_roc_features(
        intervals=feature_intervals,
        biomarker_names=list(biomarkers.keys()),
        time_scale_factor=time_scale_factor,
        minimum_time=roc_min_time,
        maximum_time=roc_max_time,
    )
    roc_long.to_csv(output_dir / "roc_features_long.csv", index=False)
    features_wide.to_csv(feature_path, index=False)

    return {
        "features_wide": features_wide,
        "target_observed_intervals": target_observed_intervals,
        "feature_intervals": feature_intervals,
        "roc_features_long": roc_long,
        "qhat": qhat_table,
        "target_joint_coverage": joint_coverage,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    split_dir = output_dir / "splits"
    crossfit_dir = output_dir / "development_crossfit"
    final_dir = output_dir / "final_model"
    lr_dir = output_dir / "logistic_regression"
    for path in (output_dir, split_dir, crossfit_dir, final_dir, lr_dir):
        path.mkdir(parents=True, exist_ok=True)

    data = load_dataset(Path(args.file), args.subject_column)
    biomarkers = load_biomarkers(Path(args.biomarkers_json))
    labels = load_binary_labels(
        labels_path=Path(args.labels_file),
        data=data,
        subject_column=args.subject_column,
        label_id_column=args.label_id_column,
        label_column=args.label_column,
        positive_value=args.positive_value,
    )

    if args.feature_time_mode == "observed":
        time_counts = count_unique_times(
            data=data,
            subject_ids=labels["id"].tolist(),
            subject_column=args.subject_column,
        )
        insufficient = time_counts[time_counts < 2].index
        if len(insufficient):
            print(
                f"Excluding {len(insufficient)} labeled subjects with fewer than two "
                "unique observed times because an observed-time RoC is undefined."
            )
            labels = labels[~labels["id"].isin(insufficient)].copy()
        if labels["label"].nunique() != 2:
            raise ValueError("Both classes must remain after the observed-time filter.")

    fixed_grid_times = (
        np.asarray(parse_float_list(args.feature_grid), dtype=float)
        if args.feature_time_mode == "fixed_grid"
        else None
    )
    c_grid = parse_float_list(args.lr_c_grid)

    outer_split = make_or_load_outer_split(
        labels=labels,
        args=args,
        split_dir=split_dir,
    )
    development = outer_split[outer_split["outer_role"] == "development"].copy()
    final_test = outer_split[outer_split["outer_role"] == "final_test"].copy()

    minimum_class_count = int(development["label"].value_counts().min())
    effective_crossfit_folds = min(args.crossfit_folds, minimum_class_count)
    if effective_crossfit_folds < 2:
        raise ValueError("Not enough development subjects per class for cross-fitting.")

    all_data_subjects = set(data[args.subject_column].astype(str).unique())
    final_test_ids = set(final_test["id"].astype(str))
    development_ids = development["id"].to_numpy(dtype=str)

    device = torch.device(
        f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu"
    )
    config = DKGPConfig(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        activation=args.activation,
        kernel=args.kernel,
        mean=args.mean,
        uncertainty=args.uncertainty,
        std_floor=args.std_floor,
        prediction_batch_size=args.prediction_batch_size,
    )
    config.validate()

    print(f"Device: {device}")
    print(f"Eligible labeled subjects: {len(labels)}")
    print(f"Development subjects: {len(development)}")
    print(f"Untouched final-test subjects: {len(final_test)}")
    print(f"Cross-fitting folds: {effective_crossfit_folds}")
    print(f"Biomarkers: {biomarkers}")

    started = time.time()
    crossfit = StratifiedKFold(
        n_splits=effective_crossfit_folds,
        shuffle=True,
        random_state=args.seed + 1,
    )

    dev_feature_frames: List[pd.DataFrame] = []
    dev_interval_frames: List[pd.DataFrame] = []
    dev_qhat_frames: List[pd.DataFrame] = []
    dev_coverage_frames: List[pd.DataFrame] = []
    crossfit_manifest_rows: List[dict] = []

    for fold, (_, holdout_positions) in enumerate(
        crossfit.split(development_ids, development["label"].to_numpy(dtype=int))
    ):
        print("\n" + "=" * 92)
        print(f"DEVELOPMENT CROSS-FIT FOLD {fold}")
        print("=" * 92)

        holdout_ids = development_ids[holdout_positions].tolist()
        remaining_eligible = [
            subject_id
            for subject_id in development_ids
            if subject_id not in set(holdout_ids)
        ]
        _, calibration_ids = split_calibration_subjects(
            remaining_eligible,
            args.calibration_fraction,
            args.seed + 100 + fold,
        )
        excluded = final_test_ids | set(holdout_ids) | set(calibration_ids)
        fit_ids = sorted(all_data_subjects.difference(excluded))

        if set(fit_ids) & set(calibration_ids):
            raise RuntimeError("Fit/calibration overlap detected.")
        if set(fit_ids) & set(holdout_ids):
            raise RuntimeError("Fit/holdout overlap detected.")
        if final_test_ids & (set(fit_ids) | set(calibration_ids) | set(holdout_ids)):
            raise RuntimeError("Final-test subject leaked into a development stage.")

        crossfit_manifest_rows.extend(
            {
                "fold": fold,
                "id": subject_id,
                "role": role,
            }
            for role, ids in (
                ("dkgp_fit", fit_ids),
                ("cp_calibration", calibration_ids),
                ("feature_holdout", holdout_ids),
            )
            for subject_id in ids
        )

        result = run_cp_dkgp_stage(
            stage="development_crossfit",
            fold=fold,
            fit_ids=fit_ids,
            calibration_ids=calibration_ids,
            target_ids=holdout_ids,
            data=data,
            subject_column=args.subject_column,
            biomarkers=biomarkers,
            alpha=args.alpha,
            device=device,
            config=config,
            output_dir=crossfit_dir / f"fold_{fold}",
            seed=args.seed,
            feature_time_mode=args.feature_time_mode,
            fixed_grid_times=fixed_grid_times,
            allow_time_varying_inputs=args.allow_time_varying_inputs,
            invariant_tolerance=args.invariant_tolerance,
            roc_min_time=args.roc_min_time,
            roc_max_time=args.roc_max_time,
            time_scale_factor=args.time_scale_factor,
            save_models=args.save_models,
            resume=args.resume,
        )
        dev_feature_frames.append(result["features_wide"])
        dev_interval_frames.append(result["target_observed_intervals"])
        dev_qhat_frames.append(result["qhat"])
        if not result["target_joint_coverage"].empty:
            dev_coverage_frames.append(result["target_joint_coverage"])

    pd.DataFrame(crossfit_manifest_rows).to_csv(
        split_dir / "development_crossfit_subject_roles.csv", index=False
    )

    development_features = pd.concat(dev_feature_frames, ignore_index=True)
    if development_features["id"].duplicated().any():
        duplicates = development_features.loc[
            development_features["id"].duplicated(), "id"
        ].tolist()
        raise RuntimeError(f"Development feature rows are duplicated: {duplicates[:10]}")

    expected_development = set(development["id"])
    observed_development = set(development_features["id"].astype(str))
    missing_development = expected_development.difference(observed_development)
    if missing_development:
        raise ValueError(
            f"{len(missing_development)} development subjects lack complete RoC/RoCB "
            f"features; examples: {sorted(missing_development)[:10]}. Use a fixed grid "
            "or restrict the label file to subjects with sufficient complete trajectories."
        )

    development_features = development_features.merge(
        development[["id", "label"]], on="id", how="inner", validate="one_to_one"
    )
    development_features.to_csv(
        output_dir / "development_oof_roc_rocb_features.csv", index=False
    )
    pd.concat(dev_interval_frames, ignore_index=True).to_csv(
        output_dir / "development_oof_observed_intervals_long.csv", index=False
    )
    pd.concat(dev_qhat_frames, ignore_index=True).to_csv(
        output_dir / "development_crossfit_qhats.csv", index=False
    )
    if dev_coverage_frames:
        pd.concat(dev_coverage_frames, ignore_index=True).to_csv(
            output_dir / "development_oof_joint_coverage_by_subject.csv", index=False
        )

    print("\n" + "=" * 92)
    print("FINAL CP-DKGP MODEL FOR THE UNTOUCHED TEST SET")
    print("=" * 92)
    _, final_calibration_ids = split_calibration_subjects(
        development["id"].tolist(),
        args.calibration_fraction,
        args.seed + 50_000,
    )
    final_fit_ids = sorted(
        all_data_subjects.difference(final_test_ids | set(final_calibration_ids))
    )
    save_split_manifest(
        path=split_dir / "final_model_subject_roles.csv",
        stage="final_model",
        fold=-1,
        fit_ids=final_fit_ids,
        calibration_ids=final_calibration_ids,
        target_ids=sorted(final_test_ids),
        target_role="final_test",
    )

    final_result = run_cp_dkgp_stage(
        stage="final_model",
        fold=-1,
        fit_ids=final_fit_ids,
        calibration_ids=final_calibration_ids,
        target_ids=sorted(final_test_ids),
        data=data,
        subject_column=args.subject_column,
        biomarkers=biomarkers,
        alpha=args.alpha,
        device=device,
        config=config,
        output_dir=final_dir,
        seed=args.seed + 60_000,
        feature_time_mode=args.feature_time_mode,
        fixed_grid_times=fixed_grid_times,
        allow_time_varying_inputs=args.allow_time_varying_inputs,
        invariant_tolerance=args.invariant_tolerance,
        roc_min_time=args.roc_min_time,
        roc_max_time=args.roc_max_time,
        time_scale_factor=args.time_scale_factor,
        save_models=args.save_models,
        resume=args.resume,
    )
    final_features = final_result["features_wide"]
    missing_final = final_test_ids.difference(set(final_features["id"].astype(str)))
    if missing_final:
        raise ValueError(
            f"{len(missing_final)} final-test subjects lack complete features; "
            f"examples: {sorted(missing_final)[:10]}."
        )
    final_features = final_features.merge(
        final_test[["id", "label"]], on="id", how="inner", validate="one_to_one"
    )
    final_features.to_csv(
        output_dir / "final_test_roc_rocb_features.csv", index=False
    )
    final_result["target_observed_intervals"].to_csv(
        output_dir / "final_test_observed_intervals_long.csv", index=False
    )

    selected_feature_sets = feature_sets(list(biomarkers.keys()))
    class_weight = None if args.lr_class_weight == "none" else args.lr_class_weight
    metrics = run_three_lr_experiments(
        development_features=development_features,
        final_test_features=final_features,
        feature_sets=selected_feature_sets,
        label_column="label",
        id_column="id",
        output_dir=lr_dir,
        c_grid=c_grid,
        cv_folds=args.lr_cv_folds,
        class_weight=class_weight,
        threshold_rule=args.threshold_rule,
        seed=args.seed + 70_000,
        max_iter=args.lr_max_iter,
        bootstrap_replicates=args.bootstrap_replicates,
    )

    metadata = {
        "data_file": str(Path(args.file).resolve()),
        "labels_file": str(Path(args.labels_file).resolve()),
        "biomarkers": biomarkers,
        "alpha": args.alpha,
        "outer_test_fraction": args.outer_test_fraction,
        "crossfit_folds_requested": args.crossfit_folds,
        "crossfit_folds_used": effective_crossfit_folds,
        "calibration_fraction": args.calibration_fraction,
        "feature_time_mode": args.feature_time_mode,
        "feature_grid": fixed_grid_times.tolist() if fixed_grid_times is not None else None,
        "roc_min_time": args.roc_min_time,
        "roc_max_time": args.roc_max_time,
        "time_scale_factor": args.time_scale_factor,
        "development_subjects": len(development),
        "final_test_subjects": len(final_test),
        "development_positive": int(development["label"].sum()),
        "final_test_positive": int(final_test["label"].sum()),
        "dkgp_config": asdict(config),
        "lr_feature_sets": selected_feature_sets,
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "seed": args.seed,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nCompleted the full CP-DKGP + logistic-regression pipeline.")
    print(f"Results: {output_dir.resolve()}")
    print("\nFinal untouched-test results:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
