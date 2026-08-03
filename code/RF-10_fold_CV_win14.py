# -*- coding: utf-8 -*-
"""Random Forest baseline with exhaustive grouped hyperparameter search."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from sklearn.ensemble import RandomForestRegressor

from pignet_common import SEED
from tree_model_runner import IndependentMultiOutputRegressor, run_tree_experiment

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "RandomForest_nested_10_fold_CV_win14"
CANONICAL_FOLD_MANIFEST = ROOT / "results" / "canonical_outer_folds_W14_H7_seed42.json"

PARAMETER_GRID = {
    "n_estimators": [800, 1000, 1200],
    "max_depth": [4, 6, 8],
    "max_features": [0.8, 0.9],
    "max_samples": [0.8, 0.9],
    "min_samples_leaf": [1, 3],
}
EXPECTED_GRID_SIZE = 72


def build_estimator(parameters: Mapping[str, object]) -> IndependentMultiOutputRegressor:
    regressor = RandomForestRegressor(
        random_state=SEED,
        n_jobs=-1,
        bootstrap=True,
        oob_score=False,
        **dict(parameters),
    )
    return IndependentMultiOutputRegressor(regressor)


def main() -> None:
    run_tree_experiment(
        model_name="RandomForest",
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
