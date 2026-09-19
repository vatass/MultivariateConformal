from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def _import_preprocessor():
    try:
        from functions import process_temporal_singletask_data
        return process_temporal_singletask_data
    except ImportError:
        cwd = str(Path.cwd().resolve())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            from functions import process_temporal_singletask_data
            return process_temporal_singletask_data
        except ImportError as exc:
            raise ImportError(
                "Could not import process_temporal_singletask_data from functions.py. "
                "Run from the project root or add the project root to PYTHONPATH."
            ) from exc


process_temporal_singletask_data = _import_preprocessor()


@dataclass
class Partition:
    x: torch.Tensor
    y: torch.Tensor
    subject_ids: np.ndarray
    row_order: np.ndarray


def _to_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def load_biomarkers(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Biomarker JSON must contain a non-empty object.")

    biomarkers: Dict[str, int] = {}
    for name, index in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid biomarker name: {name!r}")
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Invalid target index for {name}: {index!r}")
        biomarkers[name] = index
    if len(set(biomarkers.values())) != len(biomarkers):
        raise ValueError("Biomarker target indices must be distinct.")
    return biomarkers


def load_dataset(path: Path, subject_column: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {subject_column, "X", "Y"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data.copy()
    data[subject_column] = data[subject_column].astype(str)
    data["_row_order"] = np.arange(len(data), dtype=int)
    return data


def load_binary_labels(
    *,
    labels_path: Path | None,
    data: pd.DataFrame,
    subject_column: str,
    label_id_column: str,
    label_column: str,
    positive_value: str | None,
) -> pd.DataFrame:
    """
    Return one row per labeled subject with columns: id, label.

    The label file should contain only subjects eligible for the high-risk versus
    low-risk experiment, for example baseline-MCI subjects. This keeps cohort
    inclusion separate from the longitudinal DKGP training population.
    """
    source = data.copy() if labels_path is None else pd.read_csv(labels_path)
    if label_id_column not in source.columns or label_column not in source.columns:
        raise KeyError(
            f"Label source must contain {label_id_column!r} and {label_column!r}."
        )

    labels = source[[label_id_column, label_column]].copy()
    labels[label_id_column] = labels[label_id_column].astype(str)
    labels = labels.dropna(subset=[label_id_column, label_column])

    duplicated = labels.groupby(label_id_column)[label_column].nunique(dropna=False)
    conflicting = duplicated[duplicated > 1]
    if not conflicting.empty:
        raise ValueError(
            f"{len(conflicting)} subjects have conflicting downstream labels."
        )
    labels = labels.drop_duplicates(subset=[label_id_column], keep="first")

    values = list(pd.unique(labels[label_column]))
    if positive_value is None:
        numeric = pd.to_numeric(labels[label_column], errors="coerce")
        if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
            raise ValueError(
                "Labels are not binary 0/1. Supply --positive-value to map a "
                "two-class string or numeric label to class 1."
            )
        labels["label"] = numeric.astype(int)
    else:
        if len(values) != 2:
            raise ValueError(
                "--positive-value requires exactly two observed label values; "
                f"found {values}."
            )
        matches = labels[label_column].astype(str).eq(str(positive_value))
        if not matches.any():
            raise ValueError(
                f"Positive label value {positive_value!r} was not found in {values}."
            )
        labels["label"] = matches.astype(int)

    labels = labels.rename(columns={label_id_column: "id"})[["id", "label"]]

    available = set(data[subject_column].astype(str).unique())
    labels = labels[labels["id"].isin(available)].copy()
    if labels.empty:
        raise ValueError("No labeled subjects were found in the trajectory dataset.")
    if labels["label"].nunique() != 2:
        raise ValueError("Both downstream classes must be represented.")
    return labels.sort_values("id").reset_index(drop=True)


def rows_for_subjects(
    data: pd.DataFrame,
    subject_ids: Sequence[str],
    subject_column: str,
) -> pd.DataFrame:
    requested = {str(subject_id) for subject_id in subject_ids}
    rows = (
        data[data[subject_column].isin(requested)]
        .sort_values("_row_order")
        .copy()
    )
    if rows.empty:
        raise ValueError("A requested subject partition has no rows.")
    missing = requested.difference(set(rows[subject_column].unique()))
    if missing:
        raise KeyError(
            f"{len(missing)} requested subjects are absent from the dataset; "
            f"examples: {sorted(missing)[:10]}"
        )
    return rows


def _preprocess_pair(
    fit_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fit_x, fit_y, target_x, target_y = process_temporal_singletask_data(
        train_x=fit_rows["X"],
        train_y=fit_rows["Y"],
        test_x=target_rows["X"],
        test_y=target_rows["Y"],
    )
    return (
        _to_tensor(fit_x),
        _to_tensor(fit_y),
        _to_tensor(target_x),
        _to_tensor(target_y),
    )


def prepare_three_partitions(
    *,
    data: pd.DataFrame,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    target_ids: Sequence[str],
    subject_column: str,
) -> Tuple[Partition, Partition, Partition]:
    fit_rows = rows_for_subjects(data, fit_ids, subject_column)
    calibration_rows = rows_for_subjects(data, calibration_ids, subject_column)
    target_rows = rows_for_subjects(data, target_ids, subject_column)

    fit_x_cal, fit_y_cal, calibration_x, calibration_y = _preprocess_pair(
        fit_rows, calibration_rows
    )
    fit_x_target, fit_y_target, target_x, target_y = _preprocess_pair(
        fit_rows, target_rows
    )

    if fit_x_cal.shape != fit_x_target.shape or fit_y_cal.shape != fit_y_target.shape:
        raise RuntimeError(
            "Fit partition shape changed across preprocessing calls. The project "
            "preprocessor may be applying target-dependent transformations."
        )
    if not torch.allclose(fit_x_cal, fit_x_target, equal_nan=True):
        raise RuntimeError(
            "Fit features changed across preprocessing calls. Ensure preprocessing "
            "statistics are derived only from the DKGP fitting set."
        )
    if not torch.allclose(fit_y_cal, fit_y_target, equal_nan=True):
        raise RuntimeError("Fit targets changed across preprocessing calls.")

    fit_ids_rows = fit_rows[subject_column].to_numpy(dtype=str)
    cal_ids_rows = calibration_rows[subject_column].to_numpy(dtype=str)
    target_ids_rows = target_rows[subject_column].to_numpy(dtype=str)

    expected = [
        ("fit", len(fit_ids_rows), fit_x_cal.shape[0], fit_y_cal.shape[0]),
        (
            "calibration",
            len(cal_ids_rows),
            calibration_x.shape[0],
            calibration_y.shape[0],
        ),
        ("target", len(target_ids_rows), target_x.shape[0], target_y.shape[0]),
    ]
    for name, n_ids, n_x, n_y in expected:
        if not (n_ids == n_x == n_y):
            raise RuntimeError(
                f"{name} row alignment failed: IDs={n_ids}, X={n_x}, Y={n_y}."
            )

    return (
        Partition(
            x=fit_x_cal,
            y=fit_y_cal,
            subject_ids=fit_ids_rows,
            row_order=fit_rows["_row_order"].to_numpy(dtype=int),
        ),
        Partition(
            x=calibration_x,
            y=calibration_y,
            subject_ids=cal_ids_rows,
            row_order=calibration_rows["_row_order"].to_numpy(dtype=int),
        ),
        Partition(
            x=target_x,
            y=target_y,
            subject_ids=target_ids_rows,
            row_order=target_rows["_row_order"].to_numpy(dtype=int),
        ),
    )


def validate_target_indices(y: torch.Tensor, biomarkers: Mapping[str, int]) -> None:
    if y.ndim != 2:
        raise ValueError(f"Expected a 2D target matrix, obtained {tuple(y.shape)}.")
    maximum = max(biomarkers.values())
    if maximum >= y.shape[1]:
        raise IndexError(
            f"Largest target index is {maximum}, but Y has only {y.shape[1]} columns."
        )


def finite_target_view(
    partition: Partition,
    target_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    target = partition.y[:, target_index].reshape(-1)
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        raise ValueError(f"No finite target values for index {target_index}.")
    mask_np = mask.detach().cpu().numpy().astype(bool)
    return (
        partition.x[mask].detach().cpu(),
        target[mask].detach().cpu(),
        partition.subject_ids[mask_np],
        partition.row_order[mask_np],
    )


def count_unique_times(
    *,
    data: pd.DataFrame,
    subject_ids: Sequence[str],
    subject_column: str,
) -> pd.Series:
    rows = rows_for_subjects(data, subject_ids, subject_column)
    x, _, _, _ = _preprocess_pair(rows, rows)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("Parsed X must be a 2D matrix with time in its last column.")
    frame = pd.DataFrame(
        {
            "id": rows[subject_column].to_numpy(dtype=str),
            "time": x[:, -1].detach().cpu().numpy(),
        }
    )
    return frame.groupby("id")["time"].nunique()


def make_fixed_grid(
    *,
    partition: Partition,
    times: np.ndarray,
    invariant_tolerance: float,
    allow_time_varying_inputs: bool,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Build one prediction grid per subject from the earliest available X row.

    This assumes the last X column is time and all preceding columns are baseline
    predictors. The function checks that non-time predictors are invariant within
    subject unless explicitly overridden.
    """
    if times.ndim != 1 or times.size < 2:
        raise ValueError("A fixed feature grid requires at least two time points.")
    if not np.all(np.diff(times) > 0):
        raise ValueError("Fixed-grid times must be strictly increasing.")

    records_x = []
    records_id = []
    records_time = []

    for subject_id in pd.unique(partition.subject_ids):
        mask = partition.subject_ids == subject_id
        subject_x = partition.x[mask].detach().cpu().numpy()
        subject_times = subject_x[:, -1]
        baseline_position = int(np.argmin(subject_times))
        baseline_x = subject_x[baseline_position].copy()

        if subject_x.shape[0] > 1:
            max_deviation = float(
                np.nanmax(np.abs(subject_x[:, :-1] - baseline_x[None, :-1]))
            )
            if max_deviation > invariant_tolerance and not allow_time_varying_inputs:
                raise ValueError(
                    f"Subject {subject_id} has time-varying non-time inputs "
                    f"(maximum deviation {max_deviation:.6g}). Fixed-grid prediction "
                    "would require defining how those covariates evolve. Use observed "
                    "times or pass --allow-time-varying-inputs only if this variation "
                    "is known to be harmless."
                )

        grid_x = np.repeat(baseline_x[None, :], len(times), axis=0)
        grid_x[:, -1] = times
        records_x.append(grid_x)
        records_id.extend([subject_id] * len(times))
        records_time.extend(times.tolist())

    return (
        torch.as_tensor(np.vstack(records_x), dtype=torch.float32),
        np.asarray(records_id, dtype=str),
        np.asarray(records_time, dtype=float),
    )
