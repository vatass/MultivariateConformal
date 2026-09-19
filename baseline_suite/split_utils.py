from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_PARTITIONS = ("fit", "calibration", "test")


def _load_pickle_ids(path: Path) -> List[str]:
    objects: List[object] = []
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
        values = list(objects[0])
    else:
        values: List[object] = []
        for obj in objects:
            if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
                values.extend(list(obj))
            else:
                values.append(obj)
    return [str(value) for value in values]


def _validate_fold_partition_ids(
    fold: int,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    test_ids: Sequence[str],
) -> None:
    groups = {
        "fit": list(map(str, fit_ids)),
        "calibration": list(map(str, calibration_ids)),
        "test": list(map(str, test_ids)),
    }
    for name, values in groups.items():
        if not values:
            raise ValueError(f"Fold {fold}: partition {name!r} is empty.")
        if len(values) != len(set(values)):
            raise ValueError(f"Fold {fold}: duplicate IDs occur within {name!r}.")

    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = set(groups[left]).intersection(groups[right])
            if overlap:
                raise ValueError(
                    f"Fold {fold}: {len(overlap)} IDs overlap between {left} and {right}."
                )


def load_master_split_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"fold", "partition", "id"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Split table is missing columns: {sorted(missing)}")

    table = table.loc[:, ["fold", "partition", "id"]].copy()
    table["fold"] = pd.to_numeric(table["fold"], errors="raise").astype(int)
    table["partition"] = table["partition"].astype(str).str.lower().str.strip()
    table["id"] = table["id"].astype(str)

    unknown = set(table["partition"].unique()).difference(REQUIRED_PARTITIONS)
    if unknown:
        raise ValueError(f"Unknown partition labels: {sorted(unknown)}")

    duplicates = table.duplicated(["fold", "partition", "id"])
    if duplicates.any():
        raise ValueError(
            f"The split table contains {int(duplicates.sum())} duplicate rows."
        )

    for fold, group in table.groupby("fold"):
        labels = set(group["partition"].unique())
        missing_labels = set(REQUIRED_PARTITIONS).difference(labels)
        if missing_labels:
            raise ValueError(
                f"Fold {fold} is missing partitions: {sorted(missing_labels)}"
            )
        ids = {
            partition: group.loc[group["partition"] == partition, "id"].tolist()
            for partition in REQUIRED_PARTITIONS
        }
        _validate_fold_partition_ids(fold, **{
            "fit_ids": ids["fit"],
            "calibration_ids": ids["calibration"],
            "test_ids": ids["test"],
        })

    return table.sort_values(["fold", "partition", "id"]).reset_index(drop=True)


def ids_for_fold(table: pd.DataFrame, fold: int) -> Dict[str, List[str]]:
    fold_table = table[table["fold"] == fold]
    if fold_table.empty:
        raise KeyError(f"Fold {fold} is absent from the master split table.")
    result = {
        partition: fold_table.loc[
            fold_table["partition"] == partition, "id"
        ].astype(str).tolist()
        for partition in REQUIRED_PARTITIONS
    }
    _validate_fold_partition_ids(
        fold,
        fit_ids=result["fit"],
        calibration_ids=result["calibration"],
        test_ids=result["test"],
    )
    return result


def create_master_split_table(
    *,
    folds_dir: Path,
    n_folds: int,
    fold_start: int,
    calibration_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1.")

    records: List[dict] = []
    for fold in range(fold_start, fold_start + n_folds):
        train_ids = _load_pickle_ids(folds_dir / f"fold_{fold}_train.pkl")
        test_ids = _load_pickle_ids(folds_dir / f"fold_{fold}_test.pkl")
        if len(train_ids) < 2:
            raise ValueError(f"Fold {fold} has fewer than two non-test subjects.")

        n_calibration = int(round(calibration_fraction * len(train_ids)))
        n_calibration = max(1, min(len(train_ids) - 1, n_calibration))
        rng = np.random.default_rng(seed + fold)
        chosen_positions = set(
            rng.choice(len(train_ids), size=n_calibration, replace=False).tolist()
        )
        calibration_ids = [
            subject_id
            for position, subject_id in enumerate(train_ids)
            if position in chosen_positions
        ]
        fit_ids = [
            subject_id
            for position, subject_id in enumerate(train_ids)
            if position not in chosen_positions
        ]
        _validate_fold_partition_ids(
            fold,
            fit_ids=fit_ids,
            calibration_ids=calibration_ids,
            test_ids=test_ids,
        )
        for partition, values in (
            ("fit", fit_ids),
            ("calibration", calibration_ids),
            ("test", test_ids),
        ):
            records.extend(
                {"fold": fold, "partition": partition, "id": subject_id}
                for subject_id in values
            )
    return pd.DataFrame.from_records(records)
