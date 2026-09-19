from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_lr_pipeline(*, class_weight: str | None, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=0,
                ),
            ),
        ]
    )


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    rule: str,
) -> float:
    if rule == "0.5":
        return 0.5
    if rule != "youden":
        raise ValueError("threshold_rule must be '0.5' or 'youden'.")

    candidates = np.unique(np.concatenate([[0.0], probabilities, [1.0]]))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        prediction = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        score = sensitivity + specificity - 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    prediction = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    sensitivity = tp / (tp + fn) if tp + fn else np.nan

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }


def bootstrap_metric_intervals(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> pd.DataFrame:
    if replicates < 1:
        return pd.DataFrame()
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]
    if positive_indices.size == 0 or negative_indices.size == 0:
        raise ValueError("Bootstrap CIs require both classes in the final test set.")

    for replicate in range(replicates):
        sampled_positive = rng.choice(
            positive_indices, size=positive_indices.size, replace=True
        )
        sampled_negative = rng.choice(
            negative_indices, size=negative_indices.size, replace=True
        )
        sampled = np.concatenate([sampled_positive, sampled_negative])
        rng.shuffle(sampled)
        metrics = classification_metrics(
            y_true[sampled], probabilities[sampled], threshold
        )
        metrics["replicate"] = replicate
        rows.append(metrics)

    bootstrap = pd.DataFrame(rows)
    alpha = 1.0 - confidence
    interval_rows = []
    metric_names = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
    ]
    for metric in metric_names:
        values = bootstrap[metric].dropna().to_numpy()
        interval_rows.append(
            {
                "metric": metric,
                "lower": float(np.quantile(values, alpha / 2)),
                "upper": float(np.quantile(values, 1 - alpha / 2)),
                "bootstrap_mean": float(np.mean(values)),
                "replicates": int(len(values)),
                "confidence": confidence,
            }
        )
    return pd.DataFrame(interval_rows)


def train_and_evaluate_feature_set(
    *,
    name: str,
    development: pd.DataFrame,
    final_test: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str,
    id_column: str,
    c_grid: Sequence[float],
    cv_folds: int,
    class_weight: str | None,
    threshold_rule: str,
    seed: int,
    max_iter: int,
    bootstrap_replicates: int,
) -> Dict[str, object]:
    missing_dev = set(feature_columns).difference(development.columns)
    missing_test = set(feature_columns).difference(final_test.columns)
    if missing_dev or missing_test:
        raise KeyError(
            f"Feature set {name} is missing columns. "
            f"Development: {sorted(missing_dev)}; test: {sorted(missing_test)}"
        )

    x_dev = development[list(feature_columns)].to_numpy(dtype=float)
    y_dev = development[label_column].to_numpy(dtype=int)
    x_test = final_test[list(feature_columns)].to_numpy(dtype=float)
    y_test = final_test[label_column].to_numpy(dtype=int)

    minimum_class_count = int(np.bincount(y_dev).min())
    effective_cv_folds = min(cv_folds, minimum_class_count)
    if effective_cv_folds < 2:
        raise ValueError(
            f"Feature set {name}: not enough subjects per class for LR cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=effective_cv_folds,
        shuffle=True,
        random_state=seed,
    )
    pipeline = make_lr_pipeline(class_weight=class_weight, max_iter=max_iter)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"logistic__C": list(c_grid)},
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(x_dev, y_dev)

    best_model = search.best_estimator_
    oof_probability = cross_val_predict(
        clone(best_model),
        x_dev,
        y_dev,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold = select_threshold(y_dev, oof_probability, threshold_rule)

    best_model.fit(x_dev, y_dev)
    test_probability = best_model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, test_probability, threshold)
    metrics.update(
        {
            "feature_set": name,
            "best_C": float(search.best_params_["logistic__C"]),
            "development_cv_roc_auc": float(search.best_score_),
            "n_features": len(feature_columns),
            "effective_lr_cv_folds": effective_cv_folds,
        }
    )

    predictions = final_test[[id_column, label_column]].copy()
    predictions["feature_set"] = name
    predictions["probability"] = test_probability
    predictions["threshold"] = threshold
    predictions["prediction"] = (test_probability >= threshold).astype(int)

    oof_predictions = development[[id_column, label_column]].copy()
    oof_predictions["feature_set"] = name
    oof_predictions["oof_probability"] = oof_probability
    oof_predictions["threshold"] = threshold
    oof_predictions["oof_prediction"] = (oof_probability >= threshold).astype(int)

    logistic = best_model.named_steps["logistic"]
    coefficients = pd.DataFrame(
        {
            "feature_set": name,
            "feature": list(feature_columns),
            "coefficient_standardized": logistic.coef_.reshape(-1),
        }
    )
    coefficients["intercept"] = float(logistic.intercept_[0])

    cv_results = pd.DataFrame(search.cv_results_)
    keep = [
        "param_logistic__C",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    cv_results = cv_results[keep].copy()
    cv_results.insert(0, "feature_set", name)

    ci = bootstrap_metric_intervals(
        y_true=y_test,
        probabilities=test_probability,
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed + 10_000,
    )
    if not ci.empty:
        ci.insert(0, "feature_set", name)

    return {
        "model": best_model,
        "metrics": metrics,
        "predictions": predictions,
        "development_oof_predictions": oof_predictions,
        "coefficients": coefficients,
        "cv_results": cv_results,
        "bootstrap_intervals": ci,
    }


def run_three_lr_experiments(
    *,
    development_features: pd.DataFrame,
    final_test_features: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    label_column: str,
    id_column: str,
    output_dir: Path,
    c_grid: Sequence[float],
    cv_folds: int,
    class_weight: str | None,
    threshold_rule: str,
    seed: int,
    max_iter: int,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[dict] = []
    all_predictions: List[pd.DataFrame] = []
    all_dev_predictions: List[pd.DataFrame] = []
    all_coefficients: List[pd.DataFrame] = []
    all_cv_results: List[pd.DataFrame] = []
    all_ci: List[pd.DataFrame] = []

    import joblib

    for offset, (name, columns) in enumerate(feature_sets.items()):
        result = train_and_evaluate_feature_set(
            name=name,
            development=development_features,
            final_test=final_test_features,
            feature_columns=columns,
            label_column=label_column,
            id_column=id_column,
            c_grid=c_grid,
            cv_folds=cv_folds,
            class_weight=class_weight,
            threshold_rule=threshold_rule,
            seed=seed + offset,
            max_iter=max_iter,
            bootstrap_replicates=bootstrap_replicates,
        )
        metrics_rows.append(result["metrics"])
        all_predictions.append(result["predictions"])
        all_dev_predictions.append(result["development_oof_predictions"])
        all_coefficients.append(result["coefficients"])
        all_cv_results.append(result["cv_results"])
        if not result["bootstrap_intervals"].empty:
            all_ci.append(result["bootstrap_intervals"])
        joblib.dump(result["model"], output_dir / f"logistic_{name}.joblib")

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "final_test_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        output_dir / "final_test_predictions.csv", index=False
    )
    pd.concat(all_dev_predictions, ignore_index=True).to_csv(
        output_dir / "development_lr_oof_predictions.csv", index=False
    )
    pd.concat(all_coefficients, ignore_index=True).to_csv(
        output_dir / "logistic_coefficients.csv", index=False
    )
    pd.concat(all_cv_results, ignore_index=True).to_csv(
        output_dir / "logistic_cv_results.csv", index=False
    )
    if all_ci:
        pd.concat(all_ci, ignore_index=True).to_csv(
            output_dir / "final_test_bootstrap_intervals.csv", index=False
        )
    return metrics
