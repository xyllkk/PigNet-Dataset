# -*- coding: utf-8 -*-
"""Leakage-free nested cross-validation runner for tree-based baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid

from pignet_common import (
    SEED,
    FoldAssignment,
    WindowSample,
    assert_nested_partition,
    evaluate_fold_predictions,
    flatten_tree_samples,
    load_or_create_fold_manifest,
    make_inner_folds,
    prepare_experiment,
    set_global_seed,
    subset_by_pigs,
    write_experiment_workbook,
)

EstimatorBuilder = Callable[[Mapping[str, object]], RegressorMixin]


class IndependentMultiOutputRegressor(BaseEstimator, RegressorMixin):
    """Mean-imputed independent regressors for direct multi-output forecasting."""

    def __init__(self, base_estimator: RegressorMixin) -> None:
        self.base_estimator = base_estimator

    def fit(self, x, y):
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim != 2:
            raise ValueError("The target matrix must have shape [samples, horizons].")
        self.imputer_ = SimpleImputer(strategy="mean")
        x_imputed = self.imputer_.fit_transform(x)
        self.estimators_ = []
        for horizon_index in range(y_array.shape[1]):
            estimator = clone(self.base_estimator)
            estimator.fit(x_imputed, y_array[:, horizon_index])
            self.estimators_.append(estimator)
        self.n_outputs_ = y_array.shape[1]
        return self

    def predict(self, x) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("The estimator must be fitted before prediction.")
        x_imputed = self.imputer_.transform(x)
        columns = [
            np.asarray(estimator.predict(x_imputed), dtype=float).reshape(-1)
            for estimator in self.estimators_
        ]
        return np.column_stack(columns)


def pig_macro_rmse(
    samples: Sequence[WindowSample], predictions: np.ndarray
) -> float:
    _, pig_metrics, _ = evaluate_fold_predictions(
        samples,
        predictions,
        fold=0,
        set_name="inner_validation",
    )
    return float(pig_metrics["RMSE"].mean())


def exhaustive_grouped_search(
    outer_assignment: FoldAssignment,
    outer_training_samples: Sequence[WindowSample],
    feature_columns: Sequence[str],
    parameter_grid: Mapping[str, Sequence[object]],
    estimator_builder: EstimatorBuilder,
    cache_path: Path,
    outer_fold: int,
    inner_splits: int = 5,
    seed: int = SEED,
) -> tuple[dict, pd.DataFrame]:
    """Tune exclusively within one outer-training partition."""
    candidates = list(ParameterGrid(parameter_grid))
    assignments = make_inner_folds(
        outer_training_samples,
        outer_fold=outer_fold,
        n_splits=inner_splits,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict] = {}
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        required = {"outer_fold", "parameter_json", "mean_macro_RMSE"}
        if required.issubset(cached.columns):
            cached = cached[cached["outer_fold"] == int(outer_fold)]
            for row in cached.to_dict("records"):
                completed[str(row["parameter_json"])] = row

    result_rows: list[dict] = list(completed.values())
    for candidate_index, parameters in enumerate(candidates, start=1):
        parameter_json = json.dumps(
            parameters,
            sort_keys=True,
            ensure_ascii=False,
        )
        if parameter_json in completed:
            print(
                f"Outer fold {outer_fold:02d}, candidate "
                f"{candidate_index}/{len(candidates)}: cached"
            )
            continue

        start_time = time.perf_counter()
        fold_scores: list[float] = []
        for inner_assignment in assignments:
            assert_nested_partition(outer_assignment, inner_assignment)
            inner_training_samples = subset_by_pigs(
                outer_training_samples,
                inner_assignment.train_pigs,
            )
            inner_validation_samples = subset_by_pigs(
                outer_training_samples,
                inner_assignment.validation_pigs,
            )
            x_train, y_train, _ = flatten_tree_samples(
                inner_training_samples,
                feature_columns,
            )
            x_validation, _, _ = flatten_tree_samples(
                inner_validation_samples,
                feature_columns,
            )
            estimator = estimator_builder(parameters)
            estimator.fit(x_train, y_train)
            predictions = np.asarray(
                estimator.predict(x_validation),
                dtype=float,
            )
            fold_scores.append(
                pig_macro_rmse(inner_validation_samples, predictions)
            )

        elapsed = time.perf_counter() - start_time
        row = {
            "outer_fold": int(outer_fold),
            "candidate_index": int(candidate_index),
            "parameter_json": parameter_json,
            "mean_macro_RMSE": float(np.mean(fold_scores)),
            "std_macro_RMSE": (
                float(np.std(fold_scores, ddof=1))
                if len(fold_scores) > 1
                else 0.0
            ),
            "elapsed_seconds": float(elapsed),
        }
        for inner_fold, score in enumerate(fold_scores, start=1):
            row[f"inner_fold{inner_fold}_macro_RMSE"] = float(score)
        row.update(parameters)
        result_rows.append(row)

        pd.DataFrame(result_rows).sort_values("candidate_index").to_csv(
            cache_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"Outer fold {outer_fold:02d}, candidate "
            f"{candidate_index}/{len(candidates)}: "
            f"macro RMSE={row['mean_macro_RMSE']:.6f}, "
            f"time={elapsed:.1f}s"
        )

    results = pd.DataFrame(result_rows).sort_values(
        ["mean_macro_RMSE", "std_macro_RMSE", "candidate_index"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    if results.empty:
        raise RuntimeError(f"Outer fold {outer_fold}: no tuning result was produced.")
    best_parameters = json.loads(str(results.iloc[0]["parameter_json"]))
    return best_parameters, results


def extract_native_feature_importance(
    estimator: RegressorMixin,
    feature_names: Sequence[str],
    fold: int,
) -> pd.DataFrame:
    """Average native feature importance over the seven output regressors."""
    importances: list[np.ndarray] = []
    for output_estimator in getattr(estimator, "estimators_", []):
        fitted = output_estimator
        if hasattr(fitted, "named_steps"):
            fitted = fitted.named_steps.get("regressor", fitted)
        values = getattr(fitted, "feature_importances_", None)
        if values is not None:
            importances.append(np.asarray(values, dtype=float))
    if not importances:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "fold": fold,
            "feature": list(feature_names),
            "importance": np.mean(np.stack(importances), axis=0),
        }
    ).sort_values("importance", ascending=False)


def run_tree_experiment(
    model_name: str,
    data_path: str | Path,
    output_dir: str | Path,
    canonical_manifest_path: str | Path,
    parameter_grid: Mapping[str, Sequence[object]],
    estimator_builder: EstimatorBuilder,
    expected_grid_size: int,
    outer_splits: int = 10,
    inner_splits: int = 5,
    seed: int = SEED,
) -> None:
    """Run strict outer 10-fold and inner 5-fold nested cross-validation."""
    set_global_seed(seed)
    candidates = list(ParameterGrid(parameter_grid))
    if len(candidates) != int(expected_grid_size):
        raise RuntimeError(
            f"{model_name} grid contains {len(candidates)} candidates; "
            f"expected {expected_grid_size}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, feature_columns, samples = prepare_experiment(data_path)
    outer_assignments = load_or_create_fold_manifest(
        samples,
        path=canonical_manifest_path,
        n_splits=outer_splits,
        seed=seed,
    )

    prediction_frames: list[pd.DataFrame] = []
    pig_metric_frames: list[pd.DataFrame] = []
    horizon_metric_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    selected_by_fold: dict[str, dict] = {}

    for outer_assignment in outer_assignments:
        outer_training_samples = subset_by_pigs(
            samples,
            outer_assignment.train_pigs,
        )
        outer_validation_samples = subset_by_pigs(
            samples,
            outer_assignment.validation_pigs,
        )

        best_parameters, tuning_results = exhaustive_grouped_search(
            outer_assignment=outer_assignment,
            outer_training_samples=outer_training_samples,
            feature_columns=feature_columns,
            parameter_grid=parameter_grid,
            estimator_builder=estimator_builder,
            cache_path=(
                output_dir
                / "tuning_cache"
                / f"outer_fold_{outer_assignment.fold:02d}.csv"
            ),
            outer_fold=outer_assignment.fold,
            inner_splits=inner_splits,
            seed=seed,
        )
        tuning_frames.append(tuning_results)
        selected_by_fold[str(outer_assignment.fold)] = best_parameters

        x_train, y_train, _ = flatten_tree_samples(
            outer_training_samples,
            feature_columns,
        )
        x_validation, _, _ = flatten_tree_samples(
            outer_validation_samples,
            feature_columns,
        )
        estimator = estimator_builder(best_parameters)
        estimator.fit(x_train, y_train)
        predictions = np.asarray(
            estimator.predict(x_validation),
            dtype=float,
        )
        prediction_frame, pig_metrics, horizon_metrics = evaluate_fold_predictions(
            outer_validation_samples,
            predictions,
            fold=outer_assignment.fold,
            set_name="outer_validation",
        )
        importance = extract_native_feature_importance(
            estimator,
            x_train.columns,
            fold=outer_assignment.fold,
        )

        prediction_frames.append(prediction_frame)
        pig_metric_frames.append(pig_metrics)
        horizon_metric_frames.append(horizon_metrics)
        if not importance.empty:
            importance_frames.append(importance)
        selected_rows.append(
            {
                "outer_fold": outer_assignment.fold,
                "inner_seed": seed + outer_assignment.fold,
                **best_parameters,
            }
        )
        print(
            f"Outer fold {outer_assignment.fold:02d}: "
            f"selected={best_parameters}, "
            f"outer macro RMSE={pig_metrics['RMSE'].mean():.6f}"
        )

    (output_dir / "selected_parameters_by_outer_fold.json").write_text(
        json.dumps(selected_by_fold, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    configuration = {
        "model": model_name,
        "seed": seed,
        "outer_folds": len(outer_assignments),
        "inner_tuning_folds": inner_splits,
        "protocol": "strict nested pig-level cross-validation",
        "outer_validation_usage": "final evaluation only",
        "window": 14,
        "horizon": 7,
        "features": feature_columns,
        "parameter_grid": dict(parameter_grid),
        "number_of_candidates_per_outer_fold": len(candidates),
        "selection_metric": "pig-level macro-averaged RMSE",
        "missing_value_processing": "training-partition mean imputation",
        "scaling": "none",
    }
    output_excel = output_dir / f"{model_name}_W14_H7_NestedOuter10CV.xlsx"
    write_experiment_workbook(
        output_excel,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        pig_metrics=pd.concat(pig_metric_frames, ignore_index=True),
        horizon_metrics=pd.concat(horizon_metric_frames, ignore_index=True),
        assignments=outer_assignments,
        tuning_results=pd.concat(tuning_frames, ignore_index=True),
        selected_parameters=pd.DataFrame(selected_rows),
        feature_importance=(
            pd.concat(importance_frames, ignore_index=True)
            if importance_frames
            else None
        ),
        configuration=configuration,
    )
    print(f"Results written to {output_excel}")
