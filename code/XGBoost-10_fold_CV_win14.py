# -*- coding: utf-8 -*-
"""XGBoost baseline with exhaustive grouped hyperparameter search."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from xgboost import XGBRegressor

from pignet_common import SEED
from tree_model_runner import IndependentMultiOutputRegressor, run_tree_experiment

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "XGBoost_nested_10_fold_CV_win14"
CANONICAL_FOLD_MANIFEST = ROOT / "results" / "canonical_outer_folds_W14_H7_seed42.json"

PARAMETER_GRID = {
    "n_estimators": [800, 1000, 1200],
    "learning_rate": [0.03, 0.05],
    "max_depth": [4, 6, 8],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_weight": [1, 3],
    "reg_lambda": [1.0, 1.5],
}
EXPECTED_GRID_SIZE = 288


def build_estimator(parameters: Mapping[str, object]) -> IndependentMultiOutputRegressor:
    regressor = XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
        **dict(parameters),
    )
    return IndependentMultiOutputRegressor(regressor)


def main() -> None:
    run_tree_experiment(
        model_name="XGBoost",
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        canonical_manifest_path=CANONICAL_FOLD_MANIFEST,
        parameter_grid=PARAMETER_GRID,
        estimator_builder=build_estimator,
        expected_grid_size=EXPECTED_GRID_SIZE,
        outer_splits=10,
        inner_splits=5,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
