# -*- coding: utf-8 -*-
"""LightGBM baseline with exhaustive grouped hyperparameter search."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from pignet_common import SEED
from tree_model_runner import run_tree_experiment

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "LightGBM_10_fold_CV_win14"

PARAMETER_GRID = {
    "n_estimators": [800, 1000, 1200],
    "learning_rate": [0.03, 0.05],
    "max_depth": [4, 6, 8],
    "num_leaves": [16, 64, 256],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_samples": [1, 3],
    "reg_lambda": [1.0, 1.5],
}
EXPECTED_GRID_SIZE = 864


def build_estimator(parameters: Mapping[str, object]) -> MultiOutputRegressor:
    regressor = LGBMRegressor(
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
        subsample_freq=1,
        verbosity=-1,
        **dict(parameters),
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("regressor", regressor),
        ]
    )
    return MultiOutputRegressor(pipeline, n_jobs=1)


def main() -> None:
    run_tree_experiment(
        model_name="LightGBM",
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        parameter_grid=PARAMETER_GRID,
        estimator_builder=build_estimator,
        expected_grid_size=EXPECTED_GRID_SIZE,
        outer_splits=10,
        inner_splits=5,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
