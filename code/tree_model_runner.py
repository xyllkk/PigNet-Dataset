# -*- coding: utf-8 -*-
"""Shared exhaustive-search runner for the PigNet tree-based baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import ParameterGrid

from pignet_common import (
    HORIZON,
    SEED,
    WindowSample,
    evaluate_fold_predictions,
    flatten_tree_samples,
    make_pig_folds,
    prepare_experiment,
    save_fold_manifest,
    set_global_seed,
    subset_by_pigs,
    write_experiment_workbook,
)

EstimatorBuilder = Callable[[Mapping[str, object]], RegressorMixin]


def pig_macro_rmse(
    samples: Sequence[WindowSample], predictions: np.ndarray
) -> float:
    _, pig_metrics, _ = evaluate_fold_predictions(
        samples, predictions, fold=0, set_name="tuning"
    )
    return float(pig_metrics["RMSE"].mean())


def exhaustive_grouped_search(
    samples: Sequence[WindowSample],
    feature_columns: Sequence[str],
    parameter_grid: Mapping[str, Sequence[object]],
    estimator_builder: EstimatorBuilder,
    cache_path: Path,
    inner_splits: int = 5,
    seed: int = SEED,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate every Cartesian-grid candidate using canonical pig-level folds."""
    candidates = list(ParameterGrid(parameter_grid))
    assignments = make_pig_folds(samples, n_splits=inner_splits, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict] = {}
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        for row in cached.to_dict("records"):
            completed[str(row["parameter_json"])] = row

    result_rows: list[dict] = list(completed.values())
    for candidate_index, parameters in enumerate(candidates, start=1):
        parameter_json = json.dumps(parameters, sort_keys=True, ensure_ascii=False)
        if parameter_json in completed:
            print(f"Candidate {candidate_index}/{len(candidates)}: cached")
            continue

        start_time = time.perf_counter()
        fold_scores: list[float] = []
        for assignment in assignments:
            training_samples = subset_by_pigs(samples, assignment.train_pigs)
            validation_samples = subset_by_pigs(samples, assignment.validation_pigs)
            x_train, y_train, _ = flatten_tree_samples(training_samples, feature_columns)
            x_validation, _, _ = flatten_tree_samples(validation_samples, feature_columns)

            estimator = estimator_builder(parameters)
            estimator.fit(x_train, y_train)
            predictions = np.asarray(estimator.predict(x_validation), dtype=float)
            fold_scores.append(pig_macro_rmse(validation_samples, predictions))

        elapsed = time.perf_counter() - start_time
        row = {
            "candidate_index": candidate_index,
            "parameter_json": parameter_json,
            "mean_macro_RMSE": float(np.mean(fold_scores)),
            "std_macro_RMSE": float(np.std(fold_scores, ddof=1))
            if len(fold_scores) > 1
            else 0.0,
            "elapsed_seconds": elapsed,
        }
        for fold_index, score in enumerate(fold_scores, start=1):
            row[f"fold{fold_index}_macro_RMSE"] = score
        row.update(parameters)
        result_rows.append(row)
        pd.DataFrame(result_rows).sort_values("candidate_index").to_csv(
            cache_path, index=False, encoding="utf-8-sig"
        )
        print(
            f"Candidate {candidate_index}/{len(candidates)}: "
            f"macro RMSE={row['mean_macro_RMSE']:.6f}, "
            f"time={elapsed:.1f}s"
        )

    results = pd.DataFrame(result_rows).sort_values(
        ["mean_macro_RMSE", "std_macro_RMSE", "candidate_index"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best_parameters = json.loads(str(results.iloc[0]["parameter_json"]))
    return best_parameters, results


def extract_native_feature_importance(
    estimator: RegressorMixin,
    feature_names: Sequence[str],
    fold: int,
) -> pd.DataFrame:
    """Average native feature importances over the seven output estimators."""
    importances: list[np.ndarray] = []
    estimators = getattr(estimator, "estimators_", [])
    for output_estimator in estimators:
        fitted = output_estimator
        if hasattr(fitted, "named_steps"):
            fitted = fitted.named_steps.get("regressor", fitted)
        values = getattr(fitted, "feature_importances_", None)
        if values is not None:
            importances.append(np.asarray(values, dtype=float))
    if not importances:
        return pd.DataFrame()
    mean_importance = np.mean(np.stack(importances), axis=0)
    return pd.DataFrame(
        {
            "fold": fold,
            "feature": list(feature_names),
            "importance": mean_importance,
        }
    ).sort_values("importance", ascending=False)


def run_tree_experiment(
    model_name: str,
    data_path: str | Path,
    output_dir: str | Path,
    parameter_grid: Mapping[str, Sequence[object]],
    estimator_builder: EstimatorBuilder,
    expected_grid_size: int,
    outer_splits: int = 10,
    inner_splits: int = 5,
    seed: int = SEED,
) -> None:
    """Run exhaustive tuning followed by canonical outer grouped evaluation."""
    set_global_seed(seed)
    candidates = list(ParameterGrid(parameter_grid))
    if len(candidates) != expected_grid_size:
        raise RuntimeError(
            f"{model_name} grid contains {len(candidates)} candidates; "
            f"expected {expected_grid_size}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, feature_columns, samples = prepare_experiment(data_path)
    outer_assignments = make_pig_folds(samples, n_splits=outer_splits, seed=seed)
    save_fold_manifest(output_dir / "fold_assignments.json", outer_assignments, seed=seed)

    best_parameters, tuning_results = exhaustive_grouped_search(
        samples=samples,
        feature_columns=feature_columns,
        parameter_grid=parameter_grid,
        estimator_builder=estimator_builder,
        cache_path=output_dir / "tuning_progress.csv",
        inner_splits=inner_splits,
        seed=seed,
    )
    (output_dir / "selected_parameters.json").write_text(
        json.dumps(best_parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Selected parameters: {best_parameters}")

    prediction_frames: list[pd.DataFrame] = []
    pig_metric_frames: list[pd.DataFrame] = []
    horizon_metric_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    selected_rows: list[dict] = []

    for assignment in outer_assignments:
        training_samples = subset_by_pigs(samples, assignment.train_pigs)
        validation_samples = subset_by_pigs(samples, assignment.validation_pigs)
        x_train, y_train, _ = flatten_tree_samples(training_samples, feature_columns)
        x_validation, _, _ = flatten_tree_samples(validation_samples, feature_columns)

        estimator = estimator_builder(best_parameters)
        estimator.fit(x_train, y_train)
        predictions = np.asarray(estimator.predict(x_validation), dtype=float)
        prediction_frame, pig_metrics, horizon_metrics = evaluate_fold_predictions(
            validation_samples,
            predictions,
            fold=assignment.fold,
        )
        importance = extract_native_feature_importance(
            estimator, x_train.columns, fold=assignment.fold
        )

        prediction_frames.append(prediction_frame)
        pig_metric_frames.append(pig_metrics)
        horizon_metric_frames.append(horizon_metrics)
        if not importance.empty:
            importance_frames.append(importance)
        selected_rows.append({"fold": assignment.fold, **best_parameters})
        print(
            f"Outer fold {assignment.fold:02d}: "
            f"macro RMSE={pig_metrics['RMSE'].mean():.6f}"
        )

    configuration = {
        "model": model_name,
        "seed": seed,
        "outer_folds": len(outer_assignments),
        "inner_tuning_folds": inner_splits,
        "window": 14,
        "horizon": HORIZON,
        "features": feature_columns,
        "parameter_grid": dict(parameter_grid),
        "number_of_candidates": len(candidates),
        "selection_metric": "pig-level macro-averaged RMSE",
        "missing_value_processing": "training-fold mean imputation",
        "scaling": "none",
    }
    output_excel = output_dir / f"{model_name}_W14_H7_Outer10CV.xlsx"
    write_experiment_workbook(
        output_excel,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        pig_metrics=pd.concat(pig_metric_frames, ignore_index=True),
        horizon_metrics=pd.concat(horizon_metric_frames, ignore_index=True),
        assignments=outer_assignments,
        tuning_results=tuning_results,
        selected_parameters=pd.DataFrame(selected_rows),
        feature_importance=(
            pd.concat(importance_frames, ignore_index=True)
            if importance_frames
            else None
        ),
        configuration=configuration,
    )
    print(f"Results written to {output_excel}")
