from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def prediction_frame(
    *,
    stage: str,
    fold: int,
    split: str,
    subject_ids: np.ndarray,
    biomarker: str,
    target_index: int,
    times: np.ndarray,
    y: np.ndarray | None,
    mean: np.ndarray,
    variance: np.ndarray,
    std: np.ndarray,
) -> pd.DataFrame:
    n = len(subject_ids)
    vectors = {
        "times": np.asarray(times).reshape(-1),
        "mean": np.asarray(mean).reshape(-1),
        "variance": np.asarray(variance).reshape(-1),
        "std": np.asarray(std).reshape(-1),
    }
    for name, value in vectors.items():
        if len(value) != n:
            raise ValueError(f"{name} has length {len(value)}; expected {n}.")
    if y is None:
        y_vector = np.full(n, np.nan, dtype=float)
    else:
        y_vector = np.asarray(y).reshape(-1)
        if len(y_vector) != n:
            raise ValueError(f"y has length {len(y_vector)}; expected {n}.")

    frame = pd.DataFrame(
        {
            "stage": stage,
            "fold": fold,
            "split": split,
            "id": np.asarray(subject_ids, dtype=str),
            "biomarker": biomarker,
            "target_index": target_index,
            "time": vectors["times"].astype(float),
            "y": y_vector.astype(float),
            "mean": vectors["mean"].astype(float),
            "variance": vectors["variance"].astype(float),
            "std": vectors["std"].astype(float),
        }
    )
    required = frame[["time", "mean", "variance", "std"]].to_numpy()
    if not np.isfinite(required).all():
        raise FloatingPointError("Prediction frame contains non-finite required values.")
    return frame


def build_subject_biomarker_scores(
    calibration_predictions: pd.DataFrame,
    biomarker_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"stage", "fold", "id", "biomarker", "y", "mean", "std"}
    missing = required.difference(calibration_predictions.columns)
    if missing:
        raise KeyError(f"Calibration predictions are missing: {sorted(missing)}")
    if calibration_predictions["y"].isna().any():
        raise ValueError("Calibration outcomes cannot be missing.")

    frame = calibration_predictions.copy()
    frame["normalized_residual"] = (
        np.abs(frame["y"] - frame["mean"]) / frame["std"]
    )
    if not np.isfinite(frame["normalized_residual"]).all():
        raise FloatingPointError("Non-finite calibration score encountered.")

    by_biomarker = (
        frame.groupby(["stage", "fold", "id", "biomarker"], as_index=False)
        .agg(
            conformal_score=("normalized_residual", "max"),
            n_observed_times=("normalized_residual", "size"),
            maximum_time=("time", "max"),
            minimum_time=("time", "min"),
        )
    )

    index_columns = ["stage", "fold", "id"]
    wide = by_biomarker.pivot(
        index=index_columns,
        columns="biomarker",
        values="conformal_score",
    )
    complete = wide.dropna(subset=list(biomarker_names)).copy()
    if complete.empty:
        raise ValueError(
            "No calibration subject has finite trajectories for every biomarker."
        )
    complete["joint_score"] = complete[list(biomarker_names)].max(axis=1)
    complete = complete.reset_index()
    return by_biomarker, complete


def exact_split_conformal_quantile(
    scores: Iterable[float],
    alpha: float,
) -> Tuple[float, int, int]:
    values = np.asarray(list(scores), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite calibration scores were supplied.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")

    n = int(values.size)
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf"), rank, n
    qhat = float(np.partition(values, rank - 1)[rank - 1])
    return qhat, rank, n


def calculate_joint_qhat(
    *,
    common_scores: pd.DataFrame,
    alpha: float,
    stage: str,
    fold: int,
) -> Tuple[float, pd.DataFrame]:
    qhat, rank, n = exact_split_conformal_quantile(
        common_scores["joint_score"].to_numpy(), alpha
    )
    if not np.isfinite(qhat):
        raise ValueError(
            "The exact joint conformal quantile is infinite. Increase the number "
            "of common calibration subjects or increase alpha."
        )
    table = pd.DataFrame(
        [
            {
                "stage": stage,
                "fold": fold,
                "method": "joint_cp",
                "alpha": alpha,
                "qhat": qhat,
                "rank": rank,
                "n_common_calibration_subjects": n,
            }
        ]
    )
    return qhat, table


def add_joint_intervals(predictions: pd.DataFrame, qhat: float) -> pd.DataFrame:
    result = predictions.copy()
    result["method"] = "joint_cp"
    result["qhat"] = float(qhat)
    result["lower"] = result["mean"] - qhat * result["std"]
    result["upper"] = result["mean"] + qhat * result["std"]
    result["width"] = result["upper"] - result["lower"]
    if result["y"].notna().any():
        result["point_covered"] = np.where(
            result["y"].notna(),
            (result["y"] >= result["lower"]) & (result["y"] <= result["upper"]),
            np.nan,
        )
    else:
        result["point_covered"] = np.nan
    return result


def summarize_joint_coverage(
    intervals: pd.DataFrame,
    biomarker_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = intervals[intervals["y"].notna()].copy()
    if observed.empty:
        return pd.DataFrame(), pd.DataFrame()

    trajectory = (
        observed.groupby(
            ["stage", "fold", "id", "biomarker"],
            as_index=False,
            observed=True,
        )
        .agg(
            trajectory_covered=("point_covered", "all"),
            n_timepoints=("point_covered", "size"),
            mean_width=("width", "mean"),
            minimum_time=("time", "min"),
            maximum_time=("time", "max"),
        )
    )

    joint_rows: List[dict] = []
    for (stage, fold), fold_df in trajectory.groupby(["stage", "fold"]):
        coverage_wide = fold_df.pivot(
            index="id", columns="biomarker", values="trajectory_covered"
        )
        eligible = coverage_wide.dropna(subset=list(biomarker_names)).index
        for subject_id in eligible:
            values = coverage_wide.loc[subject_id, list(biomarker_names)].astype(bool)
            joint_rows.append(
                {
                    "stage": stage,
                    "fold": fold,
                    "id": subject_id,
                    "joint_covered": bool(values.all()),
                    "n_failed_biomarkers": int((~values).sum()),
                }
            )
    joint = pd.DataFrame(joint_rows)
    return trajectory, joint


def _slope_linear_functional(
    time: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Dict[str, float]:
    order = np.argsort(time)
    time = np.asarray(time, dtype=float)[order]
    point = np.asarray(point, dtype=float)[order]
    lower = np.asarray(lower, dtype=float)[order]
    upper = np.asarray(upper, dtype=float)[order]

    finite = np.isfinite(time) & np.isfinite(point) & np.isfinite(lower) & np.isfinite(upper)
    time, point, lower, upper = time[finite], point[finite], lower[finite], upper[finite]

    unique_times = np.unique(time)
    if unique_times.size < 2:
        return {
            "roc_point": np.nan,
            "rocb_lower": np.nan,
            "rocb_upper": np.nan,
            "n_timepoints": int(len(time)),
            "followup": 0.0,
        }

    centered = time - time.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        raise ValueError("Cannot calculate a slope from identical times.")
    weights = centered / denominator

    roc_point = float(np.sum(weights * point))
    rocb_lower = float(
        np.sum(np.where(weights >= 0, weights * lower, weights * upper))
    )
    rocb_upper = float(
        np.sum(np.where(weights >= 0, weights * upper, weights * lower))
    )
    if rocb_lower > rocb_upper + 1e-10:
        raise RuntimeError("Calculated RoCB lower endpoint exceeds upper endpoint.")

    return {
        "roc_point": roc_point,
        "rocb_lower": rocb_lower,
        "rocb_upper": rocb_upper,
        "n_timepoints": int(len(time)),
        "followup": float(time.max() - time.min()),
    }


def derive_roc_features(
    *,
    intervals: pd.DataFrame,
    biomarker_names: Sequence[str],
    time_scale_factor: float,
    minimum_time: float | None,
    maximum_time: float | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Derive point RoC and exact rectangle-induced RoC bounds."""
    if time_scale_factor <= 0:
        raise ValueError("time_scale_factor must be positive.")

    working = intervals.copy()
    if minimum_time is not None:
        working = working[working["time"] >= minimum_time].copy()
    if maximum_time is not None:
        working = working[working["time"] <= maximum_time].copy()
    if working.empty:
        raise ValueError("No interval rows remain after applying the RoC time window.")

    records: List[dict] = []
    group_columns = ["stage", "fold", "id", "biomarker"]
    for keys, group in working.groupby(group_columns, observed=True):
        stage, fold, subject_id, biomarker = keys
        result = _slope_linear_functional(
            group["time"].to_numpy(),
            group["mean"].to_numpy(),
            group["lower"].to_numpy(),
            group["upper"].to_numpy(),
        )
        result.update(
            {
                "stage": stage,
                "fold": fold,
                "id": subject_id,
                "biomarker": biomarker,
            }
        )
        for column in ("roc_point", "rocb_lower", "rocb_upper"):
            result[column] *= time_scale_factor
        records.append(result)

    long = pd.DataFrame(records)
    complete = long.pivot_table(
        index=["stage", "fold", "id"],
        columns="biomarker",
        values=["roc_point", "rocb_lower", "rocb_upper"],
        aggfunc="first",
    )
    required_columns = [
        (stat, biomarker)
        for stat in ("roc_point", "rocb_lower", "rocb_upper")
        for biomarker in biomarker_names
    ]
    missing_columns = [column for column in required_columns if column not in complete.columns]
    if missing_columns:
        raise ValueError(f"Feature table is missing columns: {missing_columns}")
    complete = complete.dropna(subset=required_columns).copy()
    complete = complete[required_columns]
    complete.columns = [
        f"{biomarker}__{stat}" for stat, biomarker in complete.columns
    ]
    wide = complete.reset_index()
    return long, wide


def feature_sets(biomarker_names: Sequence[str]) -> Mapping[str, List[str]]:
    point = [f"{biomarker}__roc_point" for biomarker in biomarker_names]
    bounds = [
        name
        for biomarker in biomarker_names
        for name in (
            f"{biomarker}__rocb_lower",
            f"{biomarker}__rocb_upper",
        )
    ]
    return {
        "point_roc": point,
        "rocb": bounds,
        "point_roc_plus_rocb": point + bounds,
    }
