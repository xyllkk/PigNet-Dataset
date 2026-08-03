# -*- coding: utf-8 -*-
"""CatBoost baseline with exhaustive grouped hyperparameter search."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from pignet_common import SEED
from tree_model_runner import run_tree_experiment

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "CatBoost_10_fold_CV_win14"

PARAMETER_GRID = {
    "iterations": [800, 1000, 1200],
    "learning_rate": [0.03, 0.05],
    "depth": [4, 6, 8],
    "subsample": [0.8, 0.9],
    "rsm": [0.8, 0.9],
    "min_data_in_leaf": [1, 3],
    "l2_leaf_reg": [1.0, 1.5],
}
EXPECTED_GRID_SIZE = 288


def build_estimator(parameters: Mapping[str, object]) -> MultiOutputRegressor:
    regressor = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=SEED,
        bootstrap_type="Bernoulli",
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
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
        model_name="CatBoost",
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
