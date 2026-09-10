# -*- coding: utf-8 -*-
"""Static and synthetic checks for the unified nested cross-validation protocol."""

from __future__ import annotations

import ast
import importlib.util
import py_compile
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from deep_model_runner import (  # noqa: E402
    TrainingConfig,
    select_training_epoch_across_inner_folds,
)
from pignet_common import (  # noqa: E402
    FoldAssignment,
    WindowSample,
    assert_nested_partition,
    build_window_samples,
    load_or_create_fold_manifest,
    make_inner_folds,
    make_pig_folds,
    subset_by_pigs,
    validate_fold_assignments,
)
from tree_model_runner import exhaustive_grouped_search  # noqa: E402

MODEL_FILES = (
    "PigNet.py",
    "TimesNet.py",
    "LSTM-10_fold_CV_win14.py",
    "LightTS.py",
    "XGBoost-10_fold_CV_win14.py",
    "Light-10_fold_CV_win14.py",
    "RF-10_fold_CV_win14.py",
    "Cat-10_fold_CV_win14.py",
    "simple_baselines.py",
)
SUPPORT_FILES = (
    "pignet_common.py",
    "deep_model_runner.py",
    "tree_model_runner.py",
)
DEEP_FILES = MODEL_FILES[:4]
TREE_FILES = MODEL_FILES[4:8]
EXPECTED_GRID_SIZES = {
    "XGBoost-10_fold_CV_win14.py": 288,
    "Light-10_fold_CV_win14.py": 864,
    "RF-10_fold_CV_win14.py": 72,
    "Cat-10_fold_CV_win14.py": 288,
}
MANIFEST_EXPRESSION = (
    'ROOT / "results" / "canonical_outer_folds_W14_H7_seed42.json"'
)


def synthetic_samples(number_of_pigs: int = 20) -> list[WindowSample]:
    rng = np.random.default_rng(42)
    samples: list[WindowSample] = []
    for pig_index in range(number_of_pigs):
        pig_id = f"P{pig_index:03d}"
        for window_index in range(2):
            x = rng.normal(size=(14, 4)).astype(float)
            baseline = 40.0 + pig_index * 0.5 + window_index * 0.1
            y = baseline + np.arange(1, 8, dtype=float) * 0.2
            start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=window_index)
            target_dates = tuple(
                start + pd.Timedelta(days=14 + horizon)
                for horizon in range(1, 8)
            )
            samples.append(
                WindowSample(
                    pig_id=pig_id,
                    breed="B",
                    station="S1",
                    x=x,
                    y=y,
                    target_dates=target_dates,
                    initial_weight=baseline - 1.0,
                )
            )
    return samples


def compile_all_files() -> None:
    for name in (*MODEL_FILES, *SUPPORT_FILES, "verify_revised_code.py"):
        path = CODE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        py_compile.compile(str(path), doraise=True)


def check_shared_manifest_expression() -> None:
    for name in MODEL_FILES:
        source = (CODE_DIR / name).read_text(encoding="utf-8")
        if MANIFEST_EXPRESSION not in source:
            raise RuntimeError(f"{name} does not use the canonical outer manifest.")
    for name in DEEP_FILES:
        source = (CODE_DIR / name).read_text(encoding="utf-8")
        if "run_nested_deep_experiment" not in source:
            raise RuntimeError(f"{name} does not use the nested deep runner.")
    for name in TREE_FILES:
        source = (CODE_DIR / name).read_text(encoding="utf-8")
        if "run_tree_experiment" not in source:
            raise RuntimeError(f"{name} does not use the nested tree runner.")


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} was not found in {path.name}.")


def check_grid_sizes() -> None:
    for file_name, expected in EXPECTED_GRID_SIZES.items():
        grid = literal_assignment(CODE_DIR / file_name, "PARAMETER_GRID")
        size = int(np.prod([len(values) for values in grid.values()]))
        if size != expected:
            raise RuntimeError(
                f"{file_name}: grid size={size}, expected={expected}."
            )


def check_outer_and_inner_partitions() -> None:
    samples = synthetic_samples()
    pig_ids = sorted({sample.pig_id for sample in samples})
    assignments = make_pig_folds(samples, n_splits=10, seed=42)
    validate_fold_assignments(assignments, pig_ids, n_splits=10)

    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "folds.json"
        first = load_or_create_fold_manifest(
            samples, manifest, n_splits=10, seed=42
        )
        second = load_or_create_fold_manifest(
            list(reversed(samples)), manifest, n_splits=10, seed=42
        )
        if first != second:
            raise RuntimeError("Repeated model runs do not recover identical outer folds.")

    for outer in assignments:
        outer_training = subset_by_pigs(samples, outer.train_pigs)
        inner_assignments = make_inner_folds(
            outer_training,
            outer_fold=outer.fold,
            n_splits=5,
            seed=42,
        )
        for inner in inner_assignments:
            assert_nested_partition(outer, inner)


def check_calendar_continuity() -> None:
    dates = pd.date_range("2025-01-01", periods=21, freq="D")
    frame = pd.DataFrame(
        {
            "日期": dates,
            "耳缺号": "P001",
            "体重": np.linspace(40.0, 44.0, len(dates)),
            "breed": "Duroc",
            "station": "S1",
            "feature": np.arange(len(dates), dtype=float),
        }
    )
    samples = build_window_samples(
        frame,
        feature_columns=["feature"],
        breed_col="breed",
        station_col="station",
    )
    if len(samples) != 1:
        raise RuntimeError("A continuous 21-day interval should yield one sample.")

    frame.loc[10:, "日期"] = frame.loc[10:, "日期"] + pd.Timedelta(days=1)
    try:
        build_window_samples(
            frame,
            feature_columns=["feature"],
            breed_col="breed",
            station_col="station",
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("A calendar gap was incorrectly accepted as continuous.")


class TinyForecaster(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(input_dim, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x.mean(dim=1))


def check_deep_inner_selection() -> None:
    samples = synthetic_samples(number_of_pigs=10)
    outer = make_pig_folds(samples, n_splits=2, seed=42)[0]
    outer_training = subset_by_pigs(samples, outer.train_pigs)
    inner_assignments = make_inner_folds(
        outer_training,
        outer_fold=outer.fold,
        n_splits=2,
        seed=42,
    )
    selected_epoch, schedule, history, summary = (
        select_training_epoch_across_inner_folds(
            model_factory=lambda input_dim: TinyForecaster(input_dim),
            outer_assignment=outer,
            outer_training_samples=outer_training,
            inner_assignments=inner_assignments,
            config=TrainingConfig(
                maximum_epochs=2,
                batch_size=8,
                early_stopping_patience=2,
                scheduler_patience=1,
                inner_splits=2,
            ),
            seed=42,
        )
    )
    if selected_epoch < 1 or len(schedule) != selected_epoch:
        raise RuntimeError("Deep inner-fold epoch selection failed.")
    if set(summary["inner_fold"]) != {1, 2}:
        raise RuntimeError("Deep epoch selection did not use every inner fold.")
    if set(history["stage"]) != {"inner_selection"}:
        raise RuntimeError("Unexpected stage in inner training history.")


def check_tree_nested_search() -> None:
    samples = synthetic_samples(number_of_pigs=10)
    outer = make_pig_folds(samples, n_splits=2, seed=42)[0]
    outer_training = subset_by_pigs(samples, outer.train_pigs)

    def estimator_builder(_parameters):
        return MultiOutputRegressor(LinearRegression())

    with tempfile.TemporaryDirectory() as directory:
        best, results = exhaustive_grouped_search(
            outer_assignment=outer,
            outer_training_samples=outer_training,
            feature_columns=["f1", "f2", "f3", "f4"],
            parameter_grid={"variant": [0, 1]},
            estimator_builder=estimator_builder,
            cache_path=Path(directory) / "cache.csv",
            outer_fold=outer.fold,
            inner_splits=2,
            seed=42,
        )
    if len(results) != 2 or "variant" not in best:
        raise RuntimeError("Nested tree-model search failed.")


def import_file(file_name: str):
    module_name = file_name.replace("-", "_").replace(".", "_")
    specification = importlib.util.spec_from_file_location(
        module_name, CODE_DIR / file_name
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {file_name}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def check_deep_forward_shapes() -> None:
    for file_name in DEEP_FILES:
        module = import_file(file_name)
        model = module.build_model(6)
        output = model(torch.randn(3, 14, 6))
        if tuple(output.shape) != (3, 7):
            raise RuntimeError(
                f"{file_name}: expected output shape (3, 7), got {tuple(output.shape)}."
            )


def main() -> None:
    compile_all_files()
    check_shared_manifest_expression()
    check_grid_sizes()
    check_outer_and_inner_partitions()
    check_calendar_continuity()
    check_deep_inner_selection()
    check_tree_nested_search()
    check_deep_forward_shapes()
    print("PASS: all nested cross-validation protocol checks completed successfully.")


if __name__ == "__main__":
    main()
