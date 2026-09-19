#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from baseline_models import build_model, set_seed
from conformal_utils import (
    build_subject_scores,
    calculate_qhats,
    construct_intervals,
    summarize_intervals,
)
from data_utils import (
    finite_target_view,
    load_biomarkers,
    load_dataset,
    prediction_frame,
    prepare_partitions,
    validate_target_indices,
)
from split_utils import ids_for_fold, load_master_split_table


SUPPORTED_MODELS = (
    "mlp",
    "drmc",
    "bootstrap",
    "dqr",
    "exact_gp",
    "dmegp",
    "lmm",
    "gam",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train longitudinal predictive baselines for multiple biomarkers and "
            "apply individual, Bonferroni, and joint subject-level conformal calibration."
        )
    )
    parser.add_argument("--file", default="./data/data.csv")
    parser.add_argument("--subject-splits", required=True)
    parser.add_argument("--biomarkers-json", required=True)
    parser.add_argument("--output-dir", default="./results/multivariate_baselines")
    parser.add_argument(
        "--models",
        default=",".join(SUPPORTED_MODELS),
        help=f"Comma-separated subset of: {','.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument("--fold-start", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--continue-on-error", action="store_true")

    # Shared neural-network parameters.
    parser.add_argument("--hidden-dim-1", type=int, default=128)
    parser.add_argument("--hidden-dim-2", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.01)
    parser.add_argument("--mlp-weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)

    # DRMC and bootstrap.
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--bootstrap-models", type=int, default=10)

    # DQR. Defaults give a native 90% interval when alpha=0.10.
    parser.add_argument("--dqr-lower", type=float, default=0.05)
    parser.add_argument("--dqr-upper", type=float, default=0.95)

    # GP settings.
    parser.add_argument(
        "--uncertainty",
        choices=("epistemic", "predictive"),
        default="epistemic",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--exact-gp-iterations", type=int, default=100)
    parser.add_argument("--exact-gp-learning-rate", type=float, default=0.10)
    parser.add_argument("--max-cholesky-size", type=int, default=1000)

    # DMEGP.
    parser.add_argument("--dmegp-latent-dim", type=int, default=64)
    parser.add_argument("--dmegp-inducing-points", type=int, default=256)
    parser.add_argument("--dmegp-epochs", type=int, default=50)
    parser.add_argument("--dmegp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--dmegp-weight-decay", type=float, default=1e-3)

    # LMM.
    parser.add_argument("--lmm-max-fixed-features", type=int, default=20)
    parser.add_argument("--lmm-maxiter", type=int, default=500)

    # GAM-style partially linear spline model.
    parser.add_argument("--gam-knots", type=int, default=8)
    parser.add_argument("--gam-degree", type=int, default=3)
    parser.add_argument("--gam-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--gam-max-linear-features", type=int, default=30)
    return parser.parse_args()


def parse_model_names(raw: str) -> List[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one model must be requested.")
    unknown = set(names).difference(SUPPORTED_MODELS)
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
    return list(dict.fromkeys(names))


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_one_model(
    *,
    model_name: str,
    folds: Sequence[int],
    args: argparse.Namespace,
    data: pd.DataFrame,
    split_table: pd.DataFrame,
    biomarkers: Dict[str, int],
    device: torch.device,
    model_output_dir: Path,
) -> None:
    biomarker_names = list(biomarkers.keys())
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale aggregate files for this requested run.
    aggregate_names = (
        "calibration_predictions_long.csv",
        "test_predictions_long.csv",
        "calibration_scores_by_subject_biomarker.csv",
        "calibration_scores_common_subjects_wide.csv",
        "conformal_quantiles.csv",
        "test_intervals_long.csv",
        "test_trajectory_coverage_by_biomarker.csv",
        "test_joint_coverage_by_subject.csv",
        "fold_metrics.csv",
        "overall_metrics.csv",
    )
    for filename in aggregate_names:
        path = model_output_dir / filename
        if path.exists():
            path.unlink()

    all_intervals: List[pd.DataFrame] = []
    all_trajectory: List[pd.DataFrame] = []
    all_joint_subject: List[pd.DataFrame] = []
    all_fold_metrics: List[pd.DataFrame] = []
    training_records: List[dict] = []

    for fold in folds:
        fold_started = time.time()
        print("\n" + "=" * 88)
        print(f"MODEL={model_name} | FOLD={fold}")
        print("=" * 88)

        split_ids = ids_for_fold(split_table, fold)
        fit_partition, calibration_partition, test_partition = prepare_partitions(
            data=data,
            fit_ids=split_ids["fit"],
            calibration_ids=split_ids["calibration"],
            test_ids=split_ids["test"],
        )
        validate_target_indices(fit_partition.y, biomarkers)

        calibration_frames: List[pd.DataFrame] = []
        test_frames: List[pd.DataFrame] = []

        for biomarker_position, (biomarker, target_index) in enumerate(
            biomarkers.items(), start=1
        ):
            print(
                f"[{biomarker_position}/{len(biomarkers)}] {biomarker} "
                f"(Y index {target_index})"
            )
            model_seed = args.seed + fold * 10_000 + biomarker_position * 100
            set_seed(model_seed)

            fit_x, fit_y, fit_subject_ids = finite_target_view(
                fit_partition, target_index
            )
            cal_x, cal_y, cal_subject_ids = finite_target_view(
                calibration_partition, target_index
            )
            test_x, test_y, test_subject_ids = finite_target_view(
                test_partition, target_index
            )
            outcome_scale = float(np.std(fit_y))
            if not np.isfinite(outcome_scale) or outcome_scale < args.std_floor:
                outcome_scale = args.std_floor

            baseline = build_model(
                model_name,
                seed=model_seed,
                device=device,
                args=args,
            )
            train_started = time.time()
            baseline.fit(fit_x, fit_y, fit_subject_ids)
            train_seconds = time.time() - train_started

            cal_prediction = baseline.predict(cal_x, alpha=args.alpha)
            test_prediction = baseline.predict(test_x, alpha=args.alpha)

            cal_frame = prediction_frame(
                model_name=model_name,
                fold=fold,
                split="calibration",
                biomarker=biomarker,
                target_index=target_index,
                x=cal_x,
                y=cal_y,
                subject_ids=cal_subject_ids,
                prediction=cal_prediction,
            )
            test_frame = prediction_frame(
                model_name=model_name,
                fold=fold,
                split="test",
                biomarker=biomarker,
                target_index=target_index,
                x=test_x,
                y=test_y,
                subject_ids=test_subject_ids,
                prediction=test_prediction,
            )
            cal_frame["outcome_scale"] = outcome_scale
            test_frame["outcome_scale"] = outcome_scale
            calibration_frames.append(cal_frame)
            test_frames.append(test_frame)

            training_records.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "biomarker": biomarker,
                    "target_index": target_index,
                    "fit_observations": len(fit_y),
                    "fit_subjects": len(np.unique(fit_subject_ids)),
                    "calibration_observations": len(cal_y),
                    "calibration_subjects_with_biomarker": len(np.unique(cal_subject_ids)),
                    "test_observations": len(test_y),
                    "test_subjects_with_biomarker": len(np.unique(test_subject_ids)),
                    "outcome_scale": outcome_scale,
                    "training_seconds": train_seconds,
                }
            )

            del baseline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        calibration_predictions = pd.concat(calibration_frames, ignore_index=True)
        test_predictions = pd.concat(test_frames, ignore_index=True)

        per_subject_scores, common_scores, common_ids = build_subject_scores(
            calibration_predictions=calibration_predictions,
            biomarker_names=biomarker_names,
        )
        single_qhat, bonf_qhat, joint_qhat, qhat_table = calculate_qhats(
            common_scores=common_scores,
            biomarker_names=biomarker_names,
            alpha=args.alpha,
        )
        intervals = construct_intervals(
            test_predictions=test_predictions,
            single_qhat=single_qhat,
            bonferroni_qhat=bonf_qhat,
            joint_qhat=joint_qhat,
        )
        trajectory, joint_subject, fold_metrics, _ = summarize_intervals(
            intervals=intervals,
            biomarker_names=biomarker_names,
        )

        append_csv(
            calibration_predictions,
            model_output_dir / "calibration_predictions_long.csv",
        )
        append_csv(test_predictions, model_output_dir / "test_predictions_long.csv")
        append_csv(
            per_subject_scores,
            model_output_dir / "calibration_scores_by_subject_biomarker.csv",
        )
        append_csv(
            common_scores,
            model_output_dir / "calibration_scores_common_subjects_wide.csv",
        )
        append_csv(qhat_table, model_output_dir / "conformal_quantiles.csv")
        append_csv(intervals, model_output_dir / "test_intervals_long.csv")
        append_csv(
            trajectory,
            model_output_dir / "test_trajectory_coverage_by_biomarker.csv",
        )
        append_csv(
            joint_subject,
            model_output_dir / "test_joint_coverage_by_subject.csv",
        )
        append_csv(fold_metrics, model_output_dir / "fold_metrics.csv")

        all_intervals.append(intervals)
        all_trajectory.append(trajectory)
        all_joint_subject.append(joint_subject)
        all_fold_metrics.append(fold_metrics)

        print(
            f"Common calibration subjects across all biomarkers: {len(common_ids)}"
        )
        print(qhat_table[["method", "biomarker", "qhat", "rank", "n_calibration_subjects"]].to_string(index=False))
        joint_display = fold_metrics[fold_metrics["biomarker"] == "ALL_JOINT"]
        print("\nJoint test coverage:")
        print(joint_display[["method", "coverage", "mean_normalized_width", "n_test_subjects"]].to_string(index=False))
        print(f"Fold runtime: {time.time() - fold_started:.1f} seconds")

    fold_metrics_all = pd.concat(all_fold_metrics, ignore_index=True)
    overall = (
        fold_metrics_all.groupby(
            ["model", "method", "biomarker", "scope"],
            observed=True,
            as_index=False,
        )
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            normalized_width_mean=("mean_normalized_width", "mean"),
            normalized_width_std=("mean_normalized_width", "std"),
            folds=("fold", "nunique"),
        )
    )
    overall.to_csv(model_output_dir / "overall_metrics.csv", index=False)
    pd.DataFrame.from_records(training_records).to_csv(
        model_output_dir / "training_runtime_and_counts.csv", index=False
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be strictly between 0 and 1.")
    if not 0.0 < args.dqr_lower < 0.5 < args.dqr_upper < 1.0:
        raise ValueError("DQR quantiles must satisfy 0 < lower < 0.5 < upper < 1.")

    models = parse_model_names(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    biomarkers = load_biomarkers(Path(args.biomarkers_json))
    data = load_dataset(Path(args.file))
    split_table = load_master_split_table(Path(args.subject_splits))

    available_folds = sorted(split_table["fold"].unique().tolist())
    if args.fold_start is None:
        fold_start = available_folds[0]
    else:
        fold_start = args.fold_start
    if args.n_folds is None:
        folds = [fold for fold in available_folds if fold >= fold_start]
    else:
        folds = list(range(fold_start, fold_start + args.n_folds))
        unavailable = set(folds).difference(available_folds)
        if unavailable:
            raise KeyError(f"Requested folds absent from split table: {sorted(unavailable)}")

    device = torch.device(
        f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Models: {models}")
    print(f"Biomarkers: {biomarkers}")
    print(f"Folds: {folds}")
    print(f"Master subject splits: {Path(args.subject_splits).resolve()}")

    run_metadata = vars(args).copy()
    run_metadata.update(
        {
            "resolved_models": models,
            "resolved_folds": folds,
            "biomarkers": biomarkers,
            "device": str(device),
        }
    )
    (output_dir / "run_configuration.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    failures: List[dict] = []
    for model_name in models:
        model_output_dir = output_dir / model_name
        try:
            run_one_model(
                model_name=model_name,
                folds=folds,
                args=args,
                data=data,
                split_table=split_table,
                biomarkers=biomarkers,
                device=device,
                model_output_dir=model_output_dir,
            )
        except Exception as exc:
            failure = {
                "model": model_name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            (model_output_dir / "FAILED.txt").parent.mkdir(parents=True, exist_ok=True)
            (model_output_dir / "FAILED.txt").write_text(
                failure["traceback"], encoding="utf-8"
            )
            if not args.continue_on_error:
                raise
            print(f"MODEL FAILED: {model_name}: {exc}")

    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        print(f"Completed with {len(failures)} failed model(s).")
        sys.exit(1)
    else:
        print("All requested baselines completed successfully.")


if __name__ == "__main__":
    main()
