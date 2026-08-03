# -*- coding: utf-8 -*-
"""Static validation for the revised PigNet baseline scripts."""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path

from sklearn.model_selection import ParameterGrid

EXPECTED_GRID_SIZES = {
    "XGBoost-10_fold_CV_win14.py": 288,
    "Light-10_fold_CV_win14.py": 864,
    "RF-10_fold_CV_win14.py": 72,
    "Cat-10_fold_CV_win14.py": 288,
}


def read_parameter_grid(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "PARAMETER_GRID"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise TypeError(f"PARAMETER_GRID in {path.name} is not a dictionary.")
            return value
    raise RuntimeError(f"PARAMETER_GRID was not found in {path.name}.")


def main() -> None:
    root = Path(__file__).resolve().parent
    python_files = sorted(root.glob("*.py"))
    for path in python_files:
        py_compile.compile(str(path), doraise=True)
        print(f"Syntax PASS: {path.name}")

    for file_name, expected_size in EXPECTED_GRID_SIZES.items():
        grid = read_parameter_grid(root / file_name)
        actual_size = len(list(ParameterGrid(grid)))
        if actual_size != expected_size:
            raise RuntimeError(
                f"{file_name}: expected {expected_size} combinations, got {actual_size}."
            )
        print(f"Grid PASS: {file_name} ({actual_size} combinations)")

    print("All static validation checks passed.")


if __name__ == "__main__":
    main()
