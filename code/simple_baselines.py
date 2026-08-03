# -*- coding: utf-8 -*-
"""Linear and quadratic growth baselines using the canonical outer folds."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from pignet_common import (
    HORIZON,
    SEED,
    WindowSample,
    evaluate_fold_predictions,
    load_or_create_fold_manifest,
    prepare_experiment,
    set_global_seed,
    subset_by_pigs,
    write_experiment_workbook,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "SimpleBaselines_nested_10_fold_CV_win14"
CANONICAL_FOLD_MANIFEST = ROOT / "results" / "canonical_outer_folds_W14_H7_seed42.json"

EstimatorFactory = Callable[[], Pipeline]


def build_design_matrix(
    samples: Sequence[WindowSample],
    target_age_lookup: Mapping[tuple[str, pd.Timestamp], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create horizon-level age and initial-body-weight predictors."""
    rows: list[list[float]] = []
    targets: list[float] = []
    for sample in samples:
        for horizon_index, target_date in enumerate(sample.target_dates):
            key = (sample.pig_id, pd.Timestamp(target_date))
            if key not in target_age_lookup:
                raise KeyError(f"Missing target age for pig={sample.pig_id}, date={target_date}.")
            rows.append(
                [
                    float(target_age_lookup[key]),
                    float(sample.initial_weight),
                ]
            )
            targets.append(float(sample.y[horizon_index]))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def predict_windows(
    estimator: Pipeline,
    samples: Sequence[WindowSample],
    target_age_lookup: Mapping[tuple[str, pd.Timestamp], float],
) -> np.ndarray:
    design, _ = build_design_matrix(samples, target_age_lookup)
    predictions = np.asarray(estimator.predict(design), dtype=float)
    return predictions.reshape(len(samples), HORIZON)


def linear_estimator() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("regressor", LinearRegression()),
        ]
    )


def quadratic_estimator() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("regressor", LinearRegression()),
        ]
    )


def run_baseline(
    model_name: str,
    estimator_factory: EstimatorFactory,
    samples: Sequence[WindowSample],
    target_age_lookup: Mapping[tuple[str, pd.Timestamp], float],
    assignments,
) -> None:
    prediction_frames: list[pd.DataFrame] = []
    pig_metric_frames: list[pd.DataFrame] = []
    horizon_metric_frames: list[pd.DataFrame] = []

    for assignment in assignments:
        training_samples = subset_by_pigs(samples, assignment.train_pigs)
        validation_samples = subset_by_pigs(samples, assignment.validation_pigs)
        x_train, y_train = build_design_matrix(training_samples, target_age_lookup)

        estimator = estimator_factory()
        estimator.fit(x_train, y_train)
        predictions = predict_windows(
            estimator,
            validation_samples,
            target_age_lookup=target_age_lookup,
        )
        prediction_frame, pig_metrics, horizon_metrics = evaluate_fold_predictions(
            validation_samples,
            predictions,
            fold=assignment.fold,
            set_name="outer_validation",
        )
        prediction_frames.append(prediction_frame)
        pig_metric_frames.append(pig_metrics)
        horizon_metric_frames.append(horizon_metrics)

    output_path = OUTPUT_DIR / f"{model_name}_W14_H7_Outer10CV.xlsx"
    write_experiment_workbook(
        output_path,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        pig_metrics=pd.concat(pig_metric_frames, ignore_index=True),
        horizon_metrics=pd.concat(horizon_metric_frames, ignore_index=True),
        assignments=assignments,
        configuration={
            "model": model_name,
            "seed": SEED,
            "outer_folds": len(assignments),
            "outer_validation_usage": "final evaluation only",
            "input_features": ["target_age", "initial_body_weight"],
            "window": 14,
            "horizon": HORIZON,
            "hyperparameter_selection": "not applicable",
        },
    )
    print(f"Results written to {output_path}")


def main() -> None:
    set_global_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    normalized, feature_columns, samples = prepare_experiment(DATA_PATH)
    if "日龄" not in feature_columns:
        raise ValueError("The age column '日龄' is required by the simple baselines.")
    target_age_lookup = {
        (str(row["耳缺号"]), pd.Timestamp(row["日期"])): float(row["日龄"])
        for _, row in normalized.dropna(subset=["日龄"]).iterrows()
    }
    assignments = load_or_create_fold_manifest(
        samples,
        path=CANONICAL_FOLD_MANIFEST,
        n_splits=10,
        seed=SEED,
    )

    run_baseline(
        "Linear",
        linear_estimator,
        samples,
        target_age_lookup,
        assignments,
    )
    run_baseline(
        "Quadratic",
        quadratic_estimator,
        samples,
        target_age_lookup,
        assignments,
    )


if __name__ == "__main__":
    main()
