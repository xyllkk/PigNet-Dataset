"""Aggregate historical tree-model SHAP outputs at the original-variable level.

The historical CV scripts explain each of seven independently fitted horizon
estimators with ``shap.TreeExplainer``.  This public entry point applies the
same validation-window sampling and aggregation to serialized outer-fold
estimators supplied by the user.  Model files are intentionally not bundled
with the repository.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_NAMES = ("RF", "XGBoost", "LightGBM", "CatBoost")
DEFAULT_HORIZON = 7
DEFAULT_MAX_WINDOWS_PER_PIG = 20
DEFAULT_SAMPLING_SEED = 42


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel validation-window table."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def find_column(frame: pd.DataFrame, names: tuple[str, ...], label: str) -> str:
    """Find a column using case-insensitive aliases."""
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    raise ValueError(f"Could not find {label} column; tried {names}.")


def balanced_sample(
    frame: pd.DataFrame,
    group_column: str,
    max_per_group: int,
    seed: int,
) -> pd.DataFrame:
    """Sample at most ``max_per_group`` windows per pig, preserving order."""
    if max_per_group <= 0:
        raise ValueError("max_per_pig must be positive.")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for _, group in frame.groupby(group_column, sort=False):
        indices = group.index.to_numpy()
        if len(indices) > max_per_group:
            indices = rng.choice(indices, size=max_per_group, replace=False)
        selected.extend(indices.tolist())
    return frame.loc[sorted(set(selected))]


def strip_lag(name: str) -> str:
    """Map ``<variable>_lag1`` ... ``_lag14`` to the variable family."""
    if name in {"init_weight", "initial_weight"}:
        return "Initial body weight"
    base = re.sub(r"_lag\d+$", "", name)
    return base


def unwrap_estimator(estimator: Any) -> Any:
    """Unwrap common pipeline and LightGBM booster wrappers."""
    if hasattr(estimator, "named_steps"):
        estimator = estimator.named_steps.get("regressor", estimator)
    if hasattr(estimator, "booster_"):
        estimator = estimator.booster_
    return estimator


def load_model(path: Path) -> Any:
    """Load a user-supplied joblib or pickle model."""
    if path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        try:
            return joblib.load(path)
        except Exception:
            with path.open("rb") as handle:
                return pickle.load(handle)
    raise ValueError(f"Unsupported model format: {path.name}")


def locate_model(model_dir: Path, model_name: str, fold: int) -> Path:
    """Locate one serialized multi-output estimator for a model and fold."""
    candidates = []
    for suffix in (".joblib", ".pkl", ".pickle"):
        candidates.extend(
            [
                model_dir / f"{model_name}_fold{fold:02d}{suffix}",
                model_dir / f"{model_name}_fold_{fold:02d}{suffix}",
                model_dir / f"{model_name}_F{fold}{suffix}",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No serialized estimator found for {model_name} fold {fold}; "
        f"expected one of {[candidate.name for candidate in candidates]}."
    )


def shap_matrix(explainer: Any, features: pd.DataFrame) -> np.ndarray:
    """Return a two-dimensional SHAP matrix for a regression estimator."""
    values = explainer.shap_values(features)
    if isinstance(values, list):
        values = values[0]
    if hasattr(values, "values"):
        values = values.values
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D SHAP matrix, received shape {matrix.shape}.")
    return matrix


def aggregate_fold(
    model: Any,
    validation: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
    horizon: int,
    max_per_pig: int,
    sampling_seed: int,
    model_name: str,
    fold: int,
) -> pd.DataFrame:
    """Compute lag-summed and horizon-summed importance for one fold."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "The SHAP analysis requires the pinned shap dependency from requirements.txt."
        ) from exc
    sampled = balanced_sample(validation, group_column, max_per_pig, sampling_seed)
    features = sampled[feature_columns]
    estimators = getattr(model, "estimators_", None)
    if estimators is None or len(estimators) < horizon:
        raise ValueError(
            f"{model_name} fold {fold} must expose {horizon} estimators_ entries."
        )

    per_horizon: list[pd.DataFrame] = []
    for horizon_index in range(horizon):
        estimator = unwrap_estimator(estimators[horizon_index])
        explainer = shap.TreeExplainer(estimator)
        values = shap_matrix(explainer, features)
        if values.shape[1] != len(feature_columns):
            raise ValueError(
                f"{model_name} fold {fold} feature count mismatch: "
                f"model returned {values.shape[1]}, data has {len(feature_columns)}."
            )
        per_feature = pd.DataFrame(
            {
                "feature": feature_columns,
                "abs_shap_mean": np.abs(values).mean(axis=0),
                "horizon": horizon_index + 1,
            }
        )
        per_feature["feature_family"] = per_feature["feature"].map(strip_lag)
        per_horizon.append(
            per_feature.groupby("feature_family", as_index=False)["abs_shap_mean"]
            .sum()
            .assign(horizon=horizon_index + 1)
        )

    family_horizon = pd.concat(per_horizon, ignore_index=True)
    totals = (
        family_horizon.groupby("feature_family", as_index=False)["abs_shap_mean"]
        .sum()
        .rename(columns={"abs_shap_mean": "total_shap"})
    )
    totals.insert(0, "model", model_name)
    totals.insert(1, "fold", fold)
    return totals


def run(args: argparse.Namespace) -> None:
    """Run the historical aggregation without fitting any model."""
    data = read_table(args.data)
    fold_column = find_column(data, ("fold", "outer_fold", "Fold"), "fold")
    group_column = find_column(
        data, ("pig_id", "pig", "Pig", "耳标号", "猪耳号"), "pig identifier"
    )
    excluded = {fold_column, group_column}
    numeric_columns = [
        str(column)
        for column in data.columns
        if str(column) not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]
    feature_columns = [
        column
        for column in numeric_columns
        if re.search(r"_lag\d+$", column) or column in {"init_weight", "initial_weight"}
    ]
    if not feature_columns:
        raise ValueError(
            "The validation table must contain lagged columns and an initial-weight column."
        )
    folds = sorted(pd.to_numeric(data[fold_column], errors="raise").astype(int).unique())
    if not folds:
        raise ValueError("No outer folds were found in the validation table.")

    fold_rows: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        for fold in folds:
            model_path = locate_model(args.model_dir, model_name, fold)
            model = load_model(model_path)
            validation = data[
                pd.to_numeric(data[fold_column], errors="raise").astype(int) == fold
            ]
            fold_rows.append(
                aggregate_fold(
                    model,
                    validation,
                    feature_columns,
                    group_column,
                    args.horizon,
                    args.max_windows_per_pig,
                    args.sampling_seed,
                    model_name,
                    fold,
                )
            )

    fold_totals = pd.concat(fold_rows, ignore_index=True)
    summary = (
        fold_totals.groupby(["model", "feature_family"], as_index=False)["total_shap"]
        .agg(mean_total_shap="mean", sd_total_shap=lambda values: values.std(ddof=1))
    )
    counts = fold_totals.groupby(["model", "feature_family"]).size().rename("n_folds")
    summary = summary.join(counts, on=["model", "feature_family"])
    summary["sem_total_shap"] = summary["sd_total_shap"] / np.sqrt(summary["n_folds"])
    summary = summary.rename(columns={"feature_family": "feature"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_totals.to_csv(args.output_dir / "shap_feature_importance_by_fold.csv", index=False)
    summary[
        ["model", "feature", "mean_total_shap", "sem_total_shap"]
    ].to_csv(args.output_dir / "shap_feature_importance.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Validation-window CSV/XLSX with fold and pig columns.")
    parser.add_argument(
        "--model-dir",
        "--checkpoint-dir",
        dest="model_dir",
        type=Path,
        required=True,
        help="Directory containing serialized model files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for fold and summary CSV files.")
    parser.add_argument("--max-windows-per-pig", type=int, default=DEFAULT_MAX_WINDOWS_PER_PIG)
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
