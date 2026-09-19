#!/usr/bin/env python3
"""
Train separate DKGP models for several biomarkers and compare four interval methods:

1. raw_dkgp:
   Uncalibrated Gaussian interval based on the selected DKGP uncertainty.

2. single_cp:
   One trajectory-wise conformal calibration per biomarker:
       R_i,k = max_t |Y_i,t,k - mu_i,t,k| / sigma_i,t,k
   Each biomarker is calibrated at miscoverage alpha.

3. bonferroni_cp:
   The same biomarker-specific scores, but each biomarker is calibrated at
   alpha / K. By the union bound, this targets simultaneous coverage at 1-alpha.

4. joint_cp:
   One subject-level joint score:
       R_i,joint = max_k max_t |Y_i,t,k - mu_i,t,k| / sigma_i,t,k
   A single quantile at miscoverage alpha is shared across biomarkers.

The script preserves the subject-level fold split and selects one common
calibration-subject set before fitting any biomarker model.

Expected project modules:
    from functions import process_temporal_singletask_data
    from models import SingleTaskDeepKernel

Expected CSV columns:
    anon_id : subject identifier
    X       : serialized input vector
    Y       : serialized target vector

Important:
- With --uncertainty epistemic, sigma is sqrt(model(x).variance).
- With --uncertainty predictive, sigma is
  sqrt(likelihood(model(x)).variance), which includes observation noise.
- Exact split-conformal quantiles are used. If the requested quantile rank is
  n_cal + 1, qhat is mathematically infinite rather than clipped to the maximum.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import gpytorch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from functions import process_temporal_singletask_data
from models import SingleTaskDeepKernel


@dataclass(frozen=True)
class ModelConfig:
    iterations: int
    learning_rate: float
    weight_decay: float
    dropout: float
    activation: str
    kernel: str
    mean: str


@dataclass
class Partition:
    x: torch.Tensor
    y: torch.Tensor
    subject_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate DKGP predictors with individual, Bonferroni, and "
            "joint multivariate conformal bands."
        )
    )
    parser.add_argument("--file", default="./data/data.csv")
    parser.add_argument("--folds-dir", default="./data/folds")
    parser.add_argument(
        "--subject-splits",
        default=None,
        help=(
            "Optional master CSV with columns fold, partition, id. When supplied, "
            "the script uses exactly those fit/calibration/test subjects and ignores "
            "random calibration splitting."
        ),
    )
    parser.add_argument("--output-dir", default="./results/multivariate_cp_dkgp")
    parser.add_argument(
        "--biomarkers-json",
        required=True,
        help=(
            "JSON file mapping biomarker names to target-vector indices. "
            'Example: {"hippocampus_right": 13, "hippocampus_left": 14}'
        ),
    )
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--kernel", default="RBF")
    parser.add_argument("--mean", default="Constant")
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--uncertainty",
        choices=("epistemic", "predictive"),
        default="epistemic",
        help=(
            "epistemic: latent GP posterior variance; "
            "predictive: latent variance plus Gaussian likelihood noise."
        ),
    )
    parser.add_argument(
        "--std-floor",
        type=float,
        default=1e-6,
        help="Lower bound applied to the DKGP standard deviation.",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save each fitted model checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_biomarkers(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict) or not raw:
        raise ValueError("--biomarkers-json must contain a non-empty JSON object.")

    biomarkers: Dict[str, int] = {}
    for name, index in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid biomarker name: {name!r}")
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Invalid target index for {name}: {index!r}")
        biomarkers[name] = index

    if len(set(biomarkers.values())) != len(biomarkers):
        raise ValueError("Every biomarker must have a distinct target index.")
    return biomarkers


def load_master_subject_splits(path: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"id": str})
    required = {"fold", "partition", "id"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Master split CSV lacks columns: {sorted(missing)}")
    table["fold"] = table["fold"].astype(int)
    table["partition"] = table["partition"].astype(str)
    table["id"] = table["id"].astype(str)
    invalid = set(table["partition"].unique()).difference({"fit", "calibration", "test"})
    if invalid:
        raise ValueError(f"Invalid partitions in master split CSV: {sorted(invalid)}")
    if table.duplicated(["fold", "id"]).any():
        raise ValueError("A subject appears in multiple partitions within a fold.")
    return table


def master_ids_for_fold(table: pd.DataFrame, fold: int) -> Tuple[List[str], List[str], List[str]]:
    fold_table = table[table["fold"] == fold]
    if fold_table.empty:
        raise KeyError(f"Fold {fold} is absent from the master split CSV.")
    partitions = {
        name: fold_table.loc[fold_table["partition"] == name, "id"].tolist()
        for name in ("fit", "calibration", "test")
    }
    if any(not partitions[name] for name in partitions):
        raise ValueError(f"Fold {fold} does not contain all three non-empty partitions.")
    return partitions["fit"], partitions["calibration"], partitions["test"]


def load_pickle_ids(path: Path) -> List:
    objects: List = []
    with path.open("rb") as handle:
        while True:
            try:
                objects.append(pickle.load(handle))
            except EOFError:
                break

    if not objects:
        raise ValueError(f"No IDs were found in {path}.")

    if len(objects) == 1 and isinstance(
        objects[0], (list, tuple, set, np.ndarray, pd.Series)
    ):
        return list(objects[0])

    flattened: List = []
    for obj in objects:
        if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
            flattened.extend(list(obj))
        else:
            flattened.append(obj)
    return flattened


def split_fit_calibration(
    fold_train_ids: Sequence,
    fraction: float,
    seed: int,
) -> Tuple[List, List]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--calibration-fraction must be strictly between 0 and 1.")
    if len(fold_train_ids) < 2:
        raise ValueError("At least two training subjects are required.")

    n_cal = int(round(fraction * len(fold_train_ids)))
    n_cal = max(1, min(len(fold_train_ids) - 1, n_cal))

    rng = np.random.default_rng(seed)
    selected_positions = set(
        rng.choice(len(fold_train_ids), size=n_cal, replace=False).tolist()
    )
    calibration_ids = [
        subject_id
        for position, subject_id in enumerate(fold_train_ids)
        if position in selected_positions
    ]
    fit_ids = [
        subject_id
        for position, subject_id in enumerate(fold_train_ids)
        if position not in selected_positions
    ]
    return fit_ids, calibration_ids


def _rows_for_ids(data: pd.DataFrame, subject_ids: Sequence) -> pd.DataFrame:
    rows = data[data["anon_id"].isin(subject_ids)].copy()
    if rows.empty:
        raise ValueError("A requested data partition contains no rows.")
    return rows


def prepare_partitions(
    data: pd.DataFrame,
    fit_ids: Sequence,
    calibration_ids: Sequence,
    test_ids: Sequence,
) -> Tuple[Partition, Partition, Partition]:
    fit_rows = _rows_for_ids(data, fit_ids)
    cal_rows = _rows_for_ids(data, calibration_ids)
    test_rows = _rows_for_ids(data, test_ids)

    fit_x, fit_y, cal_x, cal_y = process_temporal_singletask_data(
        train_x=fit_rows["X"],
        train_y=fit_rows["Y"],
        test_x=cal_rows["X"],
        test_y=cal_rows["Y"],
    )
    test_x, test_y, _, _ = process_temporal_singletask_data(
        train_x=test_rows["X"],
        train_y=test_rows["Y"],
        test_x=test_rows["X"],
        test_y=test_rows["Y"],
    )

    fit_subject_ids = fit_rows["anon_id"].to_numpy()
    cal_subject_ids = cal_rows["anon_id"].to_numpy()
    test_subject_ids = test_rows["anon_id"].to_numpy()

    if len(fit_subject_ids) != fit_x.shape[0]:
        raise RuntimeError("Fit subject IDs are not aligned with fit rows.")
    if len(cal_subject_ids) != cal_x.shape[0]:
        raise RuntimeError("Calibration subject IDs are not aligned with calibration rows.")
    if len(test_subject_ids) != test_x.shape[0]:
        raise RuntimeError("Test subject IDs are not aligned with test rows.")

    return (
        Partition(fit_x, fit_y, fit_subject_ids),
        Partition(cal_x, cal_y, cal_subject_ids),
        Partition(test_x, test_y, test_subject_ids),
    )


def validate_target_indices(y: torch.Tensor, biomarkers: Mapping[str, int]) -> None:
    if y.ndim != 2:
        raise ValueError(f"Expected a 2D target tensor, obtained shape {tuple(y.shape)}.")
    maximum_index = max(biomarkers.values())
    if maximum_index >= y.shape[1]:
        raise IndexError(
            f"The largest requested target index is {maximum_index}, "
            f"but Y contains only {y.shape[1]} targets."
        )


def train_model(
    train_x_cpu: torch.Tensor,
    train_y_cpu: torch.Tensor,
    device: torch.device,
    config: ModelConfig,
) -> Tuple[SingleTaskDeepKernel, gpytorch.likelihoods.GaussianLikelihood, List[float]]:
    train_x = train_x_cpu.to(device)
    train_y = train_y_cpu.to(device).reshape(-1)

    latent_dim = max(1, int(train_x.shape[1] / 2))
    depth = [(train_x.shape[1], latent_dim)]

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SingleTaskDeepKernel(
        input_dim=train_x.shape[1],
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        depth=depth,
        dropout=config.dropout,
        activation=config.activation,
        kernel_choice=config.kernel,
        mean=config.mean,
        pretrained=False,
        feature_extractor=None,
        latent_dim=latent_dim,
        gphyper=None,
    ).to(device)

    model.train()
    likelihood.train()
    model.feature_extractor.train()

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.feature_extractor.parameters(),
                "lr": config.learning_rate,
            },
            {"params": model.covar_module.parameters(), "lr": config.learning_rate},
            {"params": model.mean_module.parameters(), "lr": config.learning_rate},
            {"params": likelihood.parameters(), "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses: List[float] = []
    for _ in tqdm(range(config.iterations), leave=False):
        optimizer.zero_grad(set_to_none=True)
        output = model(train_x)
        loss = -mll(output, train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DKGP loss encountered: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    likelihood.eval()
    model.feature_extractor.eval()
    return model, likelihood, losses


@torch.no_grad()
def predict_in_batches(
    model: SingleTaskDeepKernel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_cpu: torch.Tensor,
    device: torch.device,
    uncertainty: str,
    std_floor: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: List[np.ndarray] = []
    variances: List[np.ndarray] = []

    model.eval()
    likelihood.eval()

    for start in range(0, x_cpu.shape[0], batch_size):
        stop = min(start + batch_size, x_cpu.shape[0])
        x_batch = x_cpu[start:stop].to(device)

        with gpytorch.settings.fast_pred_var():
            latent_distribution = model(x_batch)
            if uncertainty == "epistemic":
                selected_distribution = latent_distribution
            else:
                selected_distribution = likelihood(latent_distribution)

        means.append(selected_distribution.mean.detach().cpu().numpy())
        variances.append(selected_distribution.variance.detach().cpu().numpy())

    mean = np.concatenate(means).reshape(-1)
    variance = np.concatenate(variances).reshape(-1)
    variance = np.maximum(variance, 0.0)
    std = np.maximum(np.sqrt(variance), std_floor)
    return mean, variance, std


def predict_partition_for_biomarker(
    partition: Partition,
    target_index: int,
    biomarker: str,
    model: SingleTaskDeepKernel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    device: torch.device,
    uncertainty: str,
    std_floor: float,
    batch_size: int,
    fold: int,
    split_name: str,
) -> pd.DataFrame:
    target = partition.y[:, target_index].detach().cpu().numpy().reshape(-1)
    finite_mask = np.isfinite(target)

    if not finite_mask.any():
        raise ValueError(
            f"No finite {biomarker} targets are available in the {split_name} partition."
        )

    x_valid = partition.x[finite_mask]
    y_valid = target[finite_mask]
    ids_valid = partition.subject_ids[finite_mask]
    times_valid = x_valid[:, -1].detach().cpu().numpy().reshape(-1)

    mean, variance, std = predict_in_batches(
        model=model,
        likelihood=likelihood,
        x_cpu=x_valid,
        device=device,
        uncertainty=uncertainty,
        std_floor=std_floor,
        batch_size=batch_size,
    )

    return pd.DataFrame(
        {
            "fold": fold,
            "split": split_name,
            "id": ids_valid,
            "biomarker": biomarker,
            "target_index": target_index,
            "time": times_valid,
            "y": y_valid,
            "mean": mean,
            "variance": variance,
            "std": std,
            "absolute_error": np.abs(y_valid - mean),
        }
    )


def exact_split_conformal_quantile(
    scores: Iterable[float],
    alpha: float,
) -> Tuple[float, int, int]:
    """
    Return the exact split-conformal order statistic.

    rank = ceil((n + 1) * (1 - alpha)), using one-based indexing.

    If rank == n + 1, the exact quantile is +infinity. Clipping the rank to n
    would change the finite-sample guarantee.
    """
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

    qhat = float(np.partition(values, rank - 1)[rank - 1])
    return qhat, rank, n


def build_calibration_scores(
    calibration_predictions: pd.DataFrame,
    biomarker_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    scores = calibration_predictions.copy()
    scores["normalized_residual"] = (
        np.abs(scores["y"] - scores["mean"]) / scores["std"]
    )

    per_subject_biomarker = (
        scores.groupby(["fold", "id", "biomarker"], as_index=False)
        .agg(
            conformal_score=("normalized_residual", "max"),
            n_timepoints=("normalized_residual", "size"),
        )
    )

    if per_subject_biomarker["fold"].nunique() != 1:
        raise ValueError("build_calibration_scores expects one fold at a time.")

    pivot = per_subject_biomarker.pivot(
        index="id", columns="biomarker", values="conformal_score"
    )
    common_subject_ids = pivot.dropna(subset=list(biomarker_names)).index.tolist()

    if not common_subject_ids:
        raise ValueError(
            "No calibration subject has finite observations for every requested biomarker."
        )

    missing_count = pivot.shape[0] - len(common_subject_ids)
    if missing_count > 0:
        warnings.warn(
            f"{missing_count} selected calibration subjects were excluded from "
            "joint calibration because at least one biomarker was missing."
        )

    common_scores = pivot.loc[common_subject_ids, list(biomarker_names)].copy()
    common_scores["joint_score"] = common_scores.max(axis=1)
    common_scores = common_scores.reset_index()
    return per_subject_biomarker, common_scores, common_subject_ids


def calculate_qhats(
    common_scores: pd.DataFrame,
    biomarker_names: Sequence[str],
    alpha: float,
    fold: int,
) -> Tuple[Dict[str, float], Dict[str, float], float, pd.DataFrame]:
    k_biomarkers = len(biomarker_names)
    bonferroni_alpha = alpha / k_biomarkers

    single_qhat: Dict[str, float] = {}
    bonferroni_qhat: Dict[str, float] = {}
    records: List[dict] = []

    for biomarker in biomarker_names:
        q_single, rank_single, n_single = exact_split_conformal_quantile(
            common_scores[biomarker].to_numpy(), alpha
        )
        q_bonf, rank_bonf, n_bonf = exact_split_conformal_quantile(
            common_scores[biomarker].to_numpy(), bonferroni_alpha
        )

        single_qhat[biomarker] = q_single
        bonferroni_qhat[biomarker] = q_bonf

        records.extend(
            [
                {
                    "fold": fold,
                    "method": "single_cp",
                    "biomarker": biomarker,
                    "alpha_used": alpha,
                    "qhat": q_single,
                    "rank": rank_single,
                    "n_calibration_subjects": n_single,
                },
                {
                    "fold": fold,
                    "method": "bonferroni_cp",
                    "biomarker": biomarker,
                    "alpha_used": bonferroni_alpha,
                    "qhat": q_bonf,
                    "rank": rank_bonf,
                    "n_calibration_subjects": n_bonf,
                },
            ]
        )

    joint_qhat, joint_rank, joint_n = exact_split_conformal_quantile(
        common_scores["joint_score"].to_numpy(), alpha
    )
    records.append(
        {
            "fold": fold,
            "method": "joint_cp",
            "biomarker": "ALL",
            "alpha_used": alpha,
            "qhat": joint_qhat,
            "rank": joint_rank,
            "n_calibration_subjects": joint_n,
        }
    )

    return (
        single_qhat,
        bonferroni_qhat,
        joint_qhat,
        pd.DataFrame.from_records(records),
    )


def construct_intervals(
    test_predictions: pd.DataFrame,
    biomarker_names: Sequence[str],
    alpha: float,
    single_qhat: Mapping[str, float],
    bonferroni_qhat: Mapping[str, float],
    joint_qhat: float,
) -> pd.DataFrame:
    gaussian_factor = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    method_factors = {
        "raw_dkgp": pd.Series(
            gaussian_factor, index=test_predictions.index, dtype=float
        ),
        "single_cp": test_predictions["biomarker"].map(single_qhat).astype(float),
        "bonferroni_cp": test_predictions["biomarker"]
        .map(bonferroni_qhat)
        .astype(float),
        "joint_cp": pd.Series(joint_qhat, index=test_predictions.index, dtype=float),
    }

    interval_frames: List[pd.DataFrame] = []
    for method, factor in method_factors.items():
        frame = test_predictions.copy()
        frame["method"] = method
        frame["interval_factor"] = factor.to_numpy()
        frame["lower"] = frame["mean"] - frame["interval_factor"] * frame["std"]
        frame["upper"] = frame["mean"] + frame["interval_factor"] * frame["std"]
        frame["width"] = frame["upper"] - frame["lower"]
        frame["point_covered"] = (frame["y"] >= frame["lower"]) & (
            frame["y"] <= frame["upper"]
        )
        interval_frames.append(frame)

    intervals = pd.concat(interval_frames, ignore_index=True)
    intervals["biomarker"] = pd.Categorical(
        intervals["biomarker"], categories=list(biomarker_names), ordered=True
    )
    return intervals.sort_values(
        ["fold", "method", "id", "biomarker", "time"]
    ).reset_index(drop=True)


def summarize_intervals(
    intervals: pd.DataFrame,
    biomarker_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectory = (
        intervals.groupby(
            ["fold", "method", "id", "biomarker"],
            observed=True,
            as_index=False,
        )
        .agg(
            trajectory_covered=("point_covered", "all"),
            n_timepoints=("point_covered", "size"),
            mean_width=("width", "mean"),
            median_width=("width", "median"),
            mean_absolute_error=("absolute_error", "mean"),
        )
    )

    joint_rows: List[dict] = []
    for (fold, method), method_df in trajectory.groupby(["fold", "method"]):
        coverage_pivot = method_df.pivot(
            index="id", columns="biomarker", values="trajectory_covered"
        )
        width_pivot = method_df.pivot(
            index="id", columns="biomarker", values="mean_width"
        )

        eligible_ids = coverage_pivot.dropna(
            subset=list(biomarker_names)
        ).index
        coverage_complete = coverage_pivot.loc[eligible_ids, list(biomarker_names)]
        width_complete = width_pivot.loc[eligible_ids, list(biomarker_names)]

        for subject_id in eligible_ids:
            joint_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "id": subject_id,
                    "joint_covered": bool(
                        coverage_complete.loc[subject_id].astype(bool).all()
                    ),
                    "mean_width_across_biomarkers": float(
                        width_complete.loc[subject_id].mean()
                    ),
                    "n_biomarkers": len(biomarker_names),
                }
            )

    joint_subject = pd.DataFrame.from_records(joint_rows)

    biomarker_metrics = (
        trajectory.groupby(["fold", "method", "biomarker"], observed=True, as_index=False)
        .agg(
            trajectory_coverage=("trajectory_covered", "mean"),
            mean_width=("mean_width", "mean"),
            median_width=("median_width", "median"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            n_test_subjects=("id", "nunique"),
        )
    )
    biomarker_metrics["metric_scope"] = "biomarker"

    joint_metrics = (
        joint_subject.groupby(["fold", "method"], as_index=False)
        .agg(
            trajectory_coverage=("joint_covered", "mean"),
            mean_width=("mean_width_across_biomarkers", "mean"),
            n_test_subjects=("id", "nunique"),
        )
    )
    joint_metrics["biomarker"] = "ALL_JOINT"
    joint_metrics["median_width"] = np.nan
    joint_metrics["mean_absolute_error"] = np.nan
    joint_metrics["metric_scope"] = "joint"

    fold_metrics = pd.concat(
        [
            biomarker_metrics[
                [
                    "fold",
                    "method",
                    "biomarker",
                    "metric_scope",
                    "trajectory_coverage",
                    "mean_width",
                    "median_width",
                    "mean_absolute_error",
                    "n_test_subjects",
                ]
            ],
            joint_metrics[
                [
                    "fold",
                    "method",
                    "biomarker",
                    "metric_scope",
                    "trajectory_coverage",
                    "mean_width",
                    "median_width",
                    "mean_absolute_error",
                    "n_test_subjects",
                ]
            ],
        ],
        ignore_index=True,
    )

    overall_metrics = (
        fold_metrics.groupby(
            ["method", "biomarker", "metric_scope"], observed=True, as_index=False
        )
        .agg(
            coverage_mean=("trajectory_coverage", "mean"),
            coverage_std=("trajectory_coverage", "std"),
            width_mean=("mean_width", "mean"),
            width_std=("mean_width", "std"),
            folds=("fold", "nunique"),
        )
    )

    return trajectory, joint_subject, fold_metrics, overall_metrics


def save_checkpoint(
    path: Path,
    model: SingleTaskDeepKernel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    biomarker: str,
    target_index: int,
    model_config: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "biomarker": biomarker,
            "target_index": target_index,
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "train_x": train_x.detach().cpu(),
            "train_y": train_y.detach().cpu(),
            "model_config": asdict(model_config),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be strictly between 0 and 1.")
    if args.std_floor <= 0.0:
        raise ValueError("--std-floor must be positive.")
    if args.prediction_batch_size < 1:
        raise ValueError("--prediction-batch-size must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    biomarkers = load_biomarkers(args.biomarkers_json)
    biomarker_names = list(biomarkers.keys())

    device = torch.device(
        f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu"
    )
    model_config = ModelConfig(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        activation=args.activation,
        kernel=args.kernel,
        mean=args.mean,
    )

    print(f"Device: {device}")
    print(f"Biomarkers: {biomarkers}")
    print(f"Uncertainty normalizer: {args.uncertainty}")
    print(f"Calibration fraction: {args.calibration_fraction}")
    print(f"Target simultaneous coverage: {1.0 - args.alpha:.3f}")

    data = pd.read_csv(args.file)
    data["anon_id"] = data["anon_id"].astype(str)
    required_columns = {"anon_id", "X", "Y"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise KeyError(f"Missing required data columns: {sorted(missing_columns)}")

    all_calibration_predictions: List[pd.DataFrame] = []
    all_test_intervals: List[pd.DataFrame] = []
    all_calibration_scores: List[pd.DataFrame] = []
    all_common_scores: List[pd.DataFrame] = []
    all_qhats: List[pd.DataFrame] = []
    all_training_history: List[dict] = []
    all_split_records: List[dict] = []

    folds_dir = Path(args.folds_dir)
    master_splits = (
        load_master_subject_splits(args.subject_splits)
        if args.subject_splits is not None
        else None
    )
    if master_splits is not None:
        print(f"Using master subject splits: {Path(args.subject_splits).resolve()}")
    start_time = time.time()

    for fold in range(args.fold_start, args.fold_start + args.n_folds):
        print("\n" + "=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)

        if master_splits is not None:
            fit_ids, calibration_ids, test_ids = master_ids_for_fold(master_splits, fold)
        else:
            fold_train_ids = [str(x) for x in load_pickle_ids(folds_dir / f"fold_{fold}_train.pkl")]
            test_ids = [str(x) for x in load_pickle_ids(folds_dir / f"fold_{fold}_test.pkl")]

            overlap = set(fold_train_ids).intersection(test_ids)
            if overlap:
                raise ValueError(
                    f"Fold {fold}: {len(overlap)} subjects occur in both train and test."
                )

            fit_ids, calibration_ids = split_fit_calibration(
                fold_train_ids=fold_train_ids,
                fraction=args.calibration_fraction,
                seed=args.seed + fold,
            )

        print(
            f"Fit subjects: {len(fit_ids)} | "
            f"Calibration subjects selected a priori: {len(calibration_ids)} | "
            f"Test subjects: {len(test_ids)}"
        )

        for partition_name, subject_ids in (
            ("fit", fit_ids),
            ("calibration", calibration_ids),
            ("test", test_ids),
        ):
            all_split_records.extend(
                {
                    "fold": fold,
                    "partition": partition_name,
                    "id": subject_id,
                }
                for subject_id in subject_ids
            )

        fit_partition, cal_partition, test_partition = prepare_partitions(
            data=data,
            fit_ids=fit_ids,
            calibration_ids=calibration_ids,
            test_ids=test_ids,
        )
        validate_target_indices(fit_partition.y, biomarkers)

        fold_calibration_predictions: List[pd.DataFrame] = []
        fold_test_predictions: List[pd.DataFrame] = []

        for biomarker_position, (biomarker, target_index) in enumerate(
            biomarkers.items(), start=1
        ):
            print(
                f"\n[{biomarker_position}/{len(biomarkers)}] "
                f"Training {biomarker} (Y index {target_index})"
            )
            set_seed(args.seed + 10_000 * fold + biomarker_position)

            train_target = fit_partition.y[:, target_index].reshape(-1)
            train_mask = torch.isfinite(train_target)
            if not bool(train_mask.any()):
                raise ValueError(
                    f"No finite fitting targets are available for {biomarker}."
                )

            biomarker_train_x = fit_partition.x[train_mask].detach().cpu()
            biomarker_train_y = train_target[train_mask].detach().cpu()
            print(f"Training observations: {biomarker_train_x.shape[0]}")

            model, likelihood, losses = train_model(
                train_x_cpu=biomarker_train_x,
                train_y_cpu=biomarker_train_y,
                device=device,
                config=model_config,
            )

            all_training_history.extend(
                {
                    "fold": fold,
                    "biomarker": biomarker,
                    "target_index": target_index,
                    "iteration": iteration + 1,
                    "negative_mll": loss,
                }
                for iteration, loss in enumerate(losses)
            )

            cal_predictions = predict_partition_for_biomarker(
                partition=cal_partition,
                target_index=target_index,
                biomarker=biomarker,
                model=model,
                likelihood=likelihood,
                device=device,
                uncertainty=args.uncertainty,
                std_floor=args.std_floor,
                batch_size=args.prediction_batch_size,
                fold=fold,
                split_name="calibration",
            )
            test_predictions = predict_partition_for_biomarker(
                partition=test_partition,
                target_index=target_index,
                biomarker=biomarker,
                model=model,
                likelihood=likelihood,
                device=device,
                uncertainty=args.uncertainty,
                std_floor=args.std_floor,
                batch_size=args.prediction_batch_size,
                fold=fold,
                split_name="test",
            )

            fold_calibration_predictions.append(cal_predictions)
            fold_test_predictions.append(test_predictions)

            if args.save_models:
                save_checkpoint(
                    path=output_dir
                    / "models"
                    / f"fold_{fold}"
                    / f"{biomarker}.pt",
                    model=model,
                    likelihood=likelihood,
                    train_x=biomarker_train_x,
                    train_y=biomarker_train_y,
                    biomarker=biomarker,
                    target_index=target_index,
                    model_config=model_config,
                )

            del model, likelihood, biomarker_train_x, biomarker_train_y
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        calibration_predictions = pd.concat(
            fold_calibration_predictions, ignore_index=True
        )
        test_predictions = pd.concat(fold_test_predictions, ignore_index=True)

        (
            per_subject_biomarker_scores,
            common_scores,
            common_calibration_ids,
        ) = build_calibration_scores(
            calibration_predictions=calibration_predictions,
            biomarker_names=biomarker_names,
        )
        common_scores.insert(0, "fold", fold)

        print(
            "Effective common calibration subjects with all biomarkers: "
            f"{len(common_calibration_ids)}"
        )

        (
            single_qhat,
            bonferroni_qhat,
            joint_qhat,
            qhat_table,
        ) = calculate_qhats(
            common_scores=common_scores,
            biomarker_names=biomarker_names,
            alpha=args.alpha,
            fold=fold,
        )

        if np.isinf(qhat_table["qhat"]).any():
            infinite_rows = qhat_table[np.isinf(qhat_table["qhat"])]
            warnings.warn(
                "At least one exact conformal threshold is infinite. This occurs "
                "when the common calibration set is too small for the requested "
                "miscoverage. Affected rows:\n"
                + infinite_rows[
                    [
                        "method",
                        "biomarker",
                        "alpha_used",
                        "rank",
                        "n_calibration_subjects",
                    ]
                ].to_string(index=False)
            )

        intervals = construct_intervals(
            test_predictions=test_predictions,
            biomarker_names=biomarker_names,
            alpha=args.alpha,
            single_qhat=single_qhat,
            bonferroni_qhat=bonferroni_qhat,
            joint_qhat=joint_qhat,
        )

        all_calibration_predictions.append(calibration_predictions)
        all_test_intervals.append(intervals)
        all_calibration_scores.append(per_subject_biomarker_scores)
        all_common_scores.append(common_scores)
        all_qhats.append(qhat_table)

    calibration_predictions_df = pd.concat(
        all_calibration_predictions, ignore_index=True
    )
    intervals_df = pd.concat(all_test_intervals, ignore_index=True)
    calibration_scores_df = pd.concat(all_calibration_scores, ignore_index=True)
    common_scores_df = pd.concat(all_common_scores, ignore_index=True)
    qhats_df = pd.concat(all_qhats, ignore_index=True)
    training_history_df = pd.DataFrame.from_records(all_training_history)
    splits_df = pd.DataFrame.from_records(all_split_records)

    (
        trajectory_coverage_df,
        joint_subject_coverage_df,
        fold_metrics_df,
        overall_metrics_df,
    ) = summarize_intervals(intervals_df, biomarker_names)

    calibration_predictions_df.to_csv(
        output_dir / "calibration_predictions_long.csv", index=False
    )
    calibration_scores_df.to_csv(
        output_dir / "calibration_scores_by_subject_biomarker.csv", index=False
    )
    common_scores_df.to_csv(
        output_dir / "calibration_scores_common_subjects_wide.csv", index=False
    )
    qhats_df.to_csv(output_dir / "conformal_quantiles.csv", index=False)
    intervals_df.to_csv(output_dir / "test_intervals_long.csv", index=False)
    trajectory_coverage_df.to_csv(
        output_dir / "test_trajectory_coverage_by_biomarker.csv", index=False
    )
    joint_subject_coverage_df.to_csv(
        output_dir / "test_joint_coverage_by_subject.csv", index=False
    )
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    overall_metrics_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    training_history_df.to_csv(output_dir / "training_history.csv", index=False)
    splits_df.to_csv(output_dir / "subject_splits.csv", index=False)

    metadata = {
        "data_file": args.file,
        "folds_dir": args.folds_dir,
        "subject_splits": args.subject_splits,
        "biomarkers": biomarkers,
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "bonferroni_alpha_per_biomarker": args.alpha / len(biomarkers),
        "calibration_fraction": args.calibration_fraction,
        "uncertainty": args.uncertainty,
        "std_floor": args.std_floor,
        "seed": args.seed,
        "fold_start": args.fold_start,
        "n_folds": args.n_folds,
        "model_config": asdict(model_config),
        "device": str(device),
        "runtime_seconds": time.time() - start_time,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nCompleted.")
    print(f"Results written to: {output_dir.resolve()}")
    print("\nJoint coverage summary:")
    print(
        overall_metrics_df[
            overall_metrics_df["biomarker"].astype(str) == "ALL_JOINT"
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
