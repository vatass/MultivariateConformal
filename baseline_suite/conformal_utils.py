from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def exact_split_conformal_quantile(
    scores: Iterable[float],
    alpha: float,
) -> Tuple[float, int, int]:
    values = np.asarray(list(scores), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite calibration scores were supplied.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")

    n = int(values.size)
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf"), rank, n
    return float(np.partition(values, rank - 1)[rank - 1]), rank, n


def add_pointwise_nonconformity(calibration_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = calibration_predictions.copy()
    if frame["score_type"].nunique() != 1:
        raise ValueError("One model/fold conformalization call must have one score_type.")
    score_type = str(frame["score_type"].iloc[0])

    scale = frame["conformal_scale"].to_numpy(dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Every conformal scale must be finite and positive.")

    if score_type == "scaled_residual":
        raw = np.abs(frame["y"].to_numpy() - frame["mean"].to_numpy())
    elif score_type == "cqr":
        raw = np.maximum.reduce(
            [
                frame["native_lower"].to_numpy() - frame["y"].to_numpy(),
                frame["y"].to_numpy() - frame["native_upper"].to_numpy(),
                np.zeros(len(frame), dtype=float),
            ]
        )
    else:
        raise ValueError(f"Unsupported score_type: {score_type}")

    frame["pointwise_nonconformity"] = raw / scale
    return frame


def build_subject_scores(
    calibration_predictions: pd.DataFrame,
    biomarker_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    scored = add_pointwise_nonconformity(calibration_predictions)
    per_subject_biomarker = (
        scored.groupby(["model", "fold", "id", "biomarker"], as_index=False)
        .agg(
            conformal_score=("pointwise_nonconformity", "max"),
            n_timepoints=("pointwise_nonconformity", "size"),
        )
    )

    pivot = per_subject_biomarker.pivot(
        index="id", columns="biomarker", values="conformal_score"
    )
    common_ids = pivot.dropna(subset=list(biomarker_names)).index.astype(str).tolist()
    if not common_ids:
        raise ValueError(
            "No calibration subject has finite observations for every requested biomarker."
        )

    excluded = pivot.shape[0] - len(common_ids)
    if excluded:
        warnings.warn(
            f"Excluded {excluded} calibration subjects from joint calibration because "
            "at least one requested biomarker was missing."
        )

    common = pivot.loc[common_ids, list(biomarker_names)].copy()
    common["joint_score"] = common.max(axis=1)
    common = common.reset_index()
    common.insert(0, "fold", int(calibration_predictions["fold"].iloc[0]))
    common.insert(0, "model", str(calibration_predictions["model"].iloc[0]))
    return per_subject_biomarker, common, common_ids


def calculate_qhats(
    common_scores: pd.DataFrame,
    biomarker_names: Sequence[str],
    alpha: float,
) -> Tuple[Dict[str, float], Dict[str, float], float, pd.DataFrame]:
    model = str(common_scores["model"].iloc[0])
    fold = int(common_scores["fold"].iloc[0])
    k = len(biomarker_names)
    bonf_alpha = alpha / k

    single: Dict[str, float] = {}
    bonferroni: Dict[str, float] = {}
    records: List[dict] = []

    for biomarker in biomarker_names:
        q_single, rank_single, n_single = exact_split_conformal_quantile(
            common_scores[biomarker], alpha
        )
        q_bonf, rank_bonf, n_bonf = exact_split_conformal_quantile(
            common_scores[biomarker], bonf_alpha
        )
        single[biomarker] = q_single
        bonferroni[biomarker] = q_bonf
        records.extend(
            [
                {
                    "model": model,
                    "fold": fold,
                    "method": "single_cp",
                    "biomarker": biomarker,
                    "alpha_used": alpha,
                    "qhat": q_single,
                    "rank": rank_single,
                    "n_calibration_subjects": n_single,
                },
                {
                    "model": model,
                    "fold": fold,
                    "method": "bonferroni_cp",
                    "biomarker": biomarker,
                    "alpha_used": bonf_alpha,
                    "qhat": q_bonf,
                    "rank": rank_bonf,
                    "n_calibration_subjects": n_bonf,
                },
            ]
        )

    q_joint, rank_joint, n_joint = exact_split_conformal_quantile(
        common_scores["joint_score"], alpha
    )
    records.append(
        {
            "model": model,
            "fold": fold,
            "method": "joint_cp",
            "biomarker": "ALL",
            "alpha_used": alpha,
            "qhat": q_joint,
            "rank": rank_joint,
            "n_calibration_subjects": n_joint,
        }
    )
    return single, bonferroni, q_joint, pd.DataFrame.from_records(records)


def construct_intervals(
    test_predictions: pd.DataFrame,
    single_qhat: Mapping[str, float],
    bonferroni_qhat: Mapping[str, float],
    joint_qhat: float,
) -> pd.DataFrame:
    score_type = str(test_predictions["score_type"].iloc[0])
    factors = {
        "single_cp": test_predictions["biomarker"].map(single_qhat).to_numpy(dtype=float),
        "bonferroni_cp": test_predictions["biomarker"].map(bonferroni_qhat).to_numpy(dtype=float),
        "joint_cp": np.full(len(test_predictions), joint_qhat, dtype=float),
    }

    raw = test_predictions.copy()
    raw["method"] = "raw"
    raw["qhat"] = np.nan
    raw["lower"] = raw["native_lower"]
    raw["upper"] = raw["native_upper"]
    frames = [raw]

    scale = test_predictions["conformal_scale"].to_numpy(dtype=float)
    for method, q in factors.items():
        frame = test_predictions.copy()
        frame["method"] = method
        frame["qhat"] = q
        adjustment = q * scale
        if score_type == "scaled_residual":
            frame["lower"] = frame["mean"].to_numpy() - adjustment
            frame["upper"] = frame["mean"].to_numpy() + adjustment
        elif score_type == "cqr":
            frame["lower"] = frame["native_lower"].to_numpy() - adjustment
            frame["upper"] = frame["native_upper"].to_numpy() + adjustment
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")
        frames.append(frame)

    intervals = pd.concat(frames, ignore_index=True)
    intervals["width"] = intervals["upper"] - intervals["lower"]
    intervals["normalized_width"] = intervals["width"] / intervals["outcome_scale"]
    intervals["point_covered"] = (
        (intervals["y"] >= intervals["lower"])
        & (intervals["y"] <= intervals["upper"])
    )
    intervals["absolute_error"] = np.abs(intervals["y"] - intervals["mean"])
    return intervals


def summarize_intervals(
    intervals: pd.DataFrame,
    biomarker_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectory = (
        intervals.groupby(
            ["model", "fold", "method", "id", "biomarker"],
            observed=True,
            as_index=False,
        )
        .agg(
            trajectory_covered=("point_covered", "all"),
            pointwise_coverage=("point_covered", "mean"),
            n_timepoints=("point_covered", "size"),
            mean_width=("width", "mean"),
            mean_normalized_width=("normalized_width", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            maximum_time=("time", "max"),
        )
    )

    joint_records: List[dict] = []
    for (model, fold, method), group in trajectory.groupby(["model", "fold", "method"]):
        coverage = group.pivot(index="id", columns="biomarker", values="trajectory_covered")
        width = group.pivot(index="id", columns="biomarker", values="mean_normalized_width")
        eligible = coverage.dropna(subset=list(biomarker_names)).index
        for subject_id in eligible:
            joint_records.append(
                {
                    "model": model,
                    "fold": fold,
                    "method": method,
                    "id": subject_id,
                    "joint_covered": bool(
                        coverage.loc[subject_id, list(biomarker_names)].astype(bool).all()
                    ),
                    "mean_normalized_width": float(
                        width.loc[subject_id, list(biomarker_names)].mean()
                    ),
                }
            )
    joint_subject = pd.DataFrame.from_records(joint_records)

    biomarker_metrics = (
        trajectory.groupby(
            ["model", "fold", "method", "biomarker"],
            observed=True,
            as_index=False,
        )
        .agg(
            coverage=("trajectory_covered", "mean"),
            pointwise_coverage=("pointwise_coverage", "mean"),
            mean_width=("mean_width", "mean"),
            mean_normalized_width=("mean_normalized_width", "mean"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            n_test_subjects=("id", "nunique"),
        )
    )
    biomarker_metrics["scope"] = "biomarker"

    joint_metrics = (
        joint_subject.groupby(["model", "fold", "method"], as_index=False)
        .agg(
            coverage=("joint_covered", "mean"),
            mean_normalized_width=("mean_normalized_width", "mean"),
            n_test_subjects=("id", "nunique"),
        )
    )
    joint_metrics["biomarker"] = "ALL_JOINT"
    joint_metrics["pointwise_coverage"] = np.nan
    joint_metrics["mean_width"] = np.nan
    joint_metrics["mean_absolute_error"] = np.nan
    joint_metrics["scope"] = "joint"

    columns = [
        "model",
        "fold",
        "method",
        "biomarker",
        "scope",
        "coverage",
        "pointwise_coverage",
        "mean_width",
        "mean_normalized_width",
        "mean_absolute_error",
        "n_test_subjects",
    ]
    fold_metrics = pd.concat(
        [biomarker_metrics[columns], joint_metrics[columns]], ignore_index=True
    )

    overall = (
        fold_metrics.groupby(
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
    return trajectory, joint_subject, fold_metrics, overall
