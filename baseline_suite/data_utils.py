from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import torch


def _import_preprocessor():
    """Import the project's parser/preprocessor from the current project root."""
    try:
        from functions import process_temporal_singletask_data
        return process_temporal_singletask_data
    except ImportError as first_error:
        cwd = str(Path.cwd().resolve())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            from functions import process_temporal_singletask_data
            return process_temporal_singletask_data
        except ImportError as second_error:
            raise ImportError(
                "Could not import process_temporal_singletask_data from functions.py. "
                "Run the baseline script from the project root or set PYTHONPATH to "
                "the directory containing functions.py."
            ) from second_error


process_temporal_singletask_data = _import_preprocessor()


@dataclass
class Partition:
    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_biomarkers(path: Path) -> Dict[str, int]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("The biomarker configuration must be a non-empty JSON object.")

    result: Dict[str, int] = {}
    for name, index in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid biomarker name: {name!r}")
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Invalid target index for {name}: {index!r}")
        result[name] = index
    if len(set(result.values())) != len(result):
        raise ValueError("Biomarker target indices must be distinct.")
    return result


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"anon_id", "X", "Y"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data.copy()
    data["anon_id"] = data["anon_id"].astype(str)
    data["_row_order"] = np.arange(len(data), dtype=int)
    return data


def _rows_for_subjects(data: pd.DataFrame, subject_ids) -> pd.DataFrame:
    requested = {str(subject_id) for subject_id in subject_ids}
    rows = data[data["anon_id"].isin(requested)].sort_values("_row_order").copy()
    if rows.empty:
        raise ValueError("A requested partition contains no dataset rows.")

    found = set(rows["anon_id"].unique())
    missing = requested.difference(found)
    if missing:
        preview = sorted(missing)[:10]
        raise KeyError(
            f"{len(missing)} requested subject IDs were absent from the dataset; "
            f"examples: {preview}"
        )
    return rows


def _call_preprocessor(train_rows: pd.DataFrame, test_rows: pd.DataFrame):
    train_x, train_y, test_x, test_y = process_temporal_singletask_data(
        train_x=train_rows["X"],
        train_y=train_rows["Y"],
        test_x=test_rows["X"],
        test_y=test_rows["Y"],
    )
    train_x = _to_numpy(train_x).astype(np.float32, copy=False)
    train_y = _to_numpy(train_y).astype(np.float32, copy=False)
    test_x = _to_numpy(test_x).astype(np.float32, copy=False)
    test_y = _to_numpy(test_y).astype(np.float32, copy=False)
    return train_x, train_y, test_x, test_y


def prepare_partitions(
    *,
    data: pd.DataFrame,
    fit_ids,
    calibration_ids,
    test_ids,
) -> Tuple[Partition, Partition, Partition]:
    fit_rows = _rows_for_subjects(data, fit_ids)
    calibration_rows = _rows_for_subjects(data, calibration_ids)
    test_rows = _rows_for_subjects(data, test_ids)

    fit_x_cal, fit_y_cal, calibration_x, calibration_y = _call_preprocessor(
        fit_rows, calibration_rows
    )
    fit_x_test, fit_y_test, test_x, test_y = _call_preprocessor(fit_rows, test_rows)

    if fit_x_cal.shape != fit_x_test.shape or fit_y_cal.shape != fit_y_test.shape:
        raise RuntimeError(
            "The fit partition changed when preprocessing calibration versus test data."
        )
    if not np.allclose(fit_x_cal, fit_x_test, equal_nan=True):
        raise RuntimeError(
            "Fit features are not deterministic across preprocessing calls. "
            "The project preprocessor may be fitting transformations separately."
        )
    if not np.allclose(fit_y_cal, fit_y_test, equal_nan=True):
        raise RuntimeError("Fit targets changed across preprocessing calls.")

    fit_subject_ids = fit_rows["anon_id"].to_numpy(dtype=str)
    calibration_subject_ids = calibration_rows["anon_id"].to_numpy(dtype=str)
    test_subject_ids = test_rows["anon_id"].to_numpy(dtype=str)

    expected = [
        ("fit", len(fit_subject_ids), fit_x_cal.shape[0], fit_y_cal.shape[0]),
        (
            "calibration",
            len(calibration_subject_ids),
            calibration_x.shape[0],
            calibration_y.shape[0],
        ),
        ("test", len(test_subject_ids), test_x.shape[0], test_y.shape[0]),
    ]
    for name, n_ids, n_x, n_y in expected:
        if not (n_ids == n_x == n_y):
            raise RuntimeError(
                f"{name} row alignment failed: IDs={n_ids}, X={n_x}, Y={n_y}."
            )

    return (
        Partition(fit_x_cal, fit_y_cal, fit_subject_ids),
        Partition(calibration_x, calibration_y, calibration_subject_ids),
        Partition(test_x, test_y, test_subject_ids),
    )


def validate_target_indices(y: np.ndarray, biomarkers: Mapping[str, int]) -> None:
    if y.ndim != 2:
        raise ValueError(f"Expected a 2D target matrix, received shape {y.shape}.")
    maximum = max(biomarkers.values())
    if maximum >= y.shape[1]:
        raise IndexError(
            f"Largest requested target index is {maximum}, but Y has {y.shape[1]} columns."
        )


def finite_target_view(
    partition: Partition,
    target_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(partition.y[:, target_index], dtype=np.float32).reshape(-1)
    mask = np.isfinite(target)
    if not mask.any():
        raise ValueError(f"No finite observations exist for target index {target_index}.")
    x = np.asarray(partition.x[mask], dtype=np.float32)
    y = target[mask]
    ids = np.asarray(partition.subject_ids[mask], dtype=str)
    return x, y, ids


def prediction_frame(
    *,
    model_name: str,
    fold: int,
    split: str,
    biomarker: str,
    target_index: int,
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    prediction: Mapping[str, object],
) -> pd.DataFrame:
    n = len(y)

    def vector(name: str, *, required: bool = True, default=np.nan) -> np.ndarray:
        if name not in prediction:
            if required:
                raise KeyError(f"Prediction output is missing required key {name!r}.")
            return np.full(n, default)
        value = np.asarray(prediction[name])
        if value.ndim == 0:
            value = np.full(n, value.item())
        value = value.reshape(-1)
        if len(value) != n:
            raise ValueError(
                f"Prediction field {name!r} has length {len(value)}; expected {n}."
            )
        return value

    score_type = prediction.get("score_type")
    if not isinstance(score_type, str):
        raise TypeError("Prediction output must contain scalar string 'score_type'.")

    frame = pd.DataFrame(
        {
            "model": model_name,
            "fold": fold,
            "split": split,
            "id": np.asarray(subject_ids, dtype=str),
            "biomarker": biomarker,
            "target_index": target_index,
            "time": np.asarray(x[:, -1], dtype=float).reshape(-1),
            "y": np.asarray(y, dtype=float).reshape(-1),
            "mean": vector("mean").astype(float),
            "std": vector("std").astype(float),
            "native_lower": vector("native_lower").astype(float),
            "native_upper": vector("native_upper").astype(float),
            "conformal_scale": vector("conformal_scale").astype(float),
            "score_type": score_type,
        }
    )

    for optional_name in ("q_lower", "q_median", "q_upper"):
        if optional_name in prediction:
            frame[optional_name] = vector(optional_name).astype(float)

    numeric = [
        "time",
        "y",
        "mean",
        "std",
        "native_lower",
        "native_upper",
        "conformal_scale",
    ]
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise FloatingPointError(
            f"Non-finite prediction values were produced by {model_name} for {biomarker}."
        )
    if (frame["conformal_scale"] <= 0).any():
        raise ValueError("Every conformal scale must be strictly positive.")
    return frame
