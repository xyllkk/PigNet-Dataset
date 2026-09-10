# -*- coding: utf-8 -*-
"""Shared data, cross-validation, evaluation, and reporting utilities for PigNet baselines."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

SEED = 42
DATE_COL = "日期"
PIG_COL = "耳缺号"
WEIGHT_COL = "体重"
BREED_COL: str | None = None
STATION_COL: str | None = None
WINDOW = 14
HORIZON = 7
EXCLUDE_FEATURE_KEYWORDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowSample:
    pig_id: str
    breed: str
    station: str
    x: np.ndarray
    y: np.ndarray
    target_dates: tuple[pd.Timestamp, ...]
    initial_weight: float


@dataclass(frozen=True)
class FoldAssignment:
    fold: int
    train_pigs: tuple[str, ...]
    validation_pigs: tuple[str, ...]


def set_global_seed(seed: int = SEED, deterministic_torch: bool = True) -> None:
    """Reset Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def load_daily_table(path: str | Path) -> pd.DataFrame:
    """Read one table or all supported tables in a directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")

    if path.is_file():
        frame = _read_table(path)
        frame["__source__"] = path.name
        return frame

    files = sorted([*path.glob("*.xlsx"), *path.glob("*.xls"), *path.glob("*.csv")])
    frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            frame = _read_table(file_path)
        except Exception as exc:
            print(f"Skipping unreadable file {file_path.name}: {exc}")
            continue
        frame["__source__"] = file_path.name
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No readable xlsx, xls, or csv files were found in {path}")
    return pd.concat(frames, ignore_index=True)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Normalize identifiers, dates, durations, breed, station, and row ordering."""
    required = {DATE_COL, PIG_COL, WEIGHT_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = df.copy()
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce")
    frame[PIG_COL] = frame[PIG_COL].astype(str)
    frame[WEIGHT_COL] = pd.to_numeric(frame[WEIGHT_COL], errors="coerce")
    if "日龄" in frame.columns:
        frame["日龄"] = pd.to_numeric(frame["日龄"], errors="coerce")

    for duration_col in ("采食时间", "采食时长", "总采食时长"):
        if duration_col not in frame.columns:
            continue
        series = frame[duration_col]
        output_col = f"{duration_col}_s"
        if pd.api.types.is_numeric_dtype(series):
            frame[output_col] = pd.to_numeric(series, errors="coerce") * 86400.0
        elif pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            frame[output_col] = (
                parsed.dt.hour * 3600 + parsed.dt.minute * 60 + parsed.dt.second
            ).astype(float)
        else:
            text = series.astype(str).str.replace("：", ":", regex=False).str.strip()
            frame[output_col] = pd.to_timedelta(text, errors="coerce").dt.total_seconds()

    breed_col = BREED_COL or "_breed_"
    if BREED_COL and BREED_COL in frame.columns:
        frame[breed_col] = frame[BREED_COL].astype(str)
    else:
        frame[breed_col] = frame[PIG_COL].map(_derive_breed)

    station_col = STATION_COL or "_station_"
    if STATION_COL and STATION_COL in frame.columns:
        frame[station_col] = frame[STATION_COL].astype(str)
    else:
        from_pig = frame[PIG_COL].map(_station_from_pig_id)
        from_source = (
            frame["__source__"].map(_station_from_source)
            if "__source__" in frame.columns
            else pd.Series(index=frame.index, dtype=object)
        )
        frame[station_col] = from_pig.fillna(from_source).fillna("S?")

    frame = frame.dropna(subset=[DATE_COL, PIG_COL])
    frame = frame.drop_duplicates(subset=[DATE_COL, PIG_COL], keep="first")
    frame = frame.sort_values([PIG_COL, DATE_COL]).reset_index(drop=True)
    return frame, breed_col, station_col


def _derive_breed(value: str) -> str:
    match = re.match(r"^[A-Za-z\u4e00-\u9fa5]+", str(value))
    return match.group(0) if match else "UNK"


def _station_from_pig_id(value: str) -> str | None:
    match = re.search(r"(\d+)[\-_]\d+", str(value))
    return f"S{match.group(1)}" if match else None


def _station_from_source(value: str) -> str | None:
    text = str(value)
    match = re.search(r"(\d+)[\-_]\d+", text)
    if match:
        return f"S{match.group(1)}"
    match = re.search(
        r"(?:站|room|pen|house|测定站)\s*([0-9一二三四五六七八九]+)",
        text.lower(),
    )
    return f"S{match.group(1)}" if match else None


def select_feature_columns(df: pd.DataFrame, breed_col: str, station_col: str) -> list[str]:
    """Select numeric predictors while excluding identifiers and the response."""
    excluded = {DATE_COL, PIG_COL, WEIGHT_COL, breed_col, station_col, "__source__"}
    candidates = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    candidates = [
        col
        for col in candidates
        if not any(token.lower() in str(col).lower() for token in EXCLUDE_FEATURE_KEYWORDS)
    ]
    if not candidates:
        raise ValueError("No numeric input features were found.")

    priority = [
        "采食量",
        "采食量(kg)",
        "总采食量",
        "采食次数",
        "采食时间_s",
        "采食时长_s",
        "总采食时长_s",
        "采食时间",
        "采食时长",
        "总采食时长",
        "日龄",
    ]
    ordered = [col for col in priority if col in candidates]
    ordered.extend(col for col in candidates if col not in ordered)
    return ordered


def build_window_samples(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    breed_col: str,
    station_col: str,
    window: int = WINDOW,
    horizon: int = HORIZON,
) -> list[WindowSample]:
    """Create direct multi-output windows independently for each pig."""
    samples: list[WindowSample] = []
    retained_columns = [
        DATE_COL,
        *feature_columns,
        WEIGHT_COL,
        breed_col,
        station_col,
    ]

    for pig_id, group in df.groupby(PIG_COL, sort=True):
        group = group[retained_columns].sort_values(DATE_COL).reset_index(drop=True)
        if len(group) < window + horizon:
            continue
        initial_series = group[WEIGHT_COL].dropna()
        if initial_series.empty:
            continue
        initial_weight = float(initial_series.iloc[0])
        features = group[list(feature_columns)].apply(pd.to_numeric, errors="coerce")

        for endpoint in range(window - 1, len(group) - horizon):
            interval_dates = group[DATE_COL].iloc[
                endpoint - window + 1 : endpoint + horizon + 1
            ]
            day_steps = np.diff(
                interval_dates.dt.normalize().to_numpy(dtype="datetime64[D]")
            )
            if len(interval_dates) != window + horizon or not np.all(
                day_steps == np.timedelta64(1, "D")
            ):
                continue
            target = group[WEIGHT_COL].iloc[endpoint + 1 : endpoint + horizon + 1]
            if len(target) != horizon or target.isna().any():
                continue
            x = features.iloc[endpoint - window + 1 : endpoint + 1].to_numpy(dtype=float)
            y = target.to_numpy(dtype=float)
            dates = tuple(group[DATE_COL].iloc[endpoint + 1 : endpoint + horizon + 1])
            samples.append(
                WindowSample(
                    pig_id=str(pig_id),
                    breed=str(group[breed_col].iloc[0]),
                    station=str(group[station_col].iloc[0]),
                    x=x,
                    y=y,
                    target_dates=dates,
                    initial_weight=initial_weight,
                )
            )
    if not samples:
        raise RuntimeError("No valid forecasting windows were generated.")
    return samples


def make_pig_folds(
    samples: Sequence[WindowSample],
    n_splits: int,
    seed: int = SEED,
) -> list[FoldAssignment]:
    """Construct one canonical, seed-controlled fold assignment at pig level."""
    pig_ids = np.array(sorted({sample.pig_id for sample in samples}), dtype=object)
    if len(pig_ids) < 2:
        raise ValueError("At least two pigs are required for grouped cross-validation.")
    n_splits = min(int(n_splits), len(pig_ids))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments: list[FoldAssignment] = []
    for fold, (train_index, validation_index) in enumerate(splitter.split(pig_ids), start=1):
        assignments.append(
            FoldAssignment(
                fold=fold,
                train_pigs=tuple(str(value) for value in pig_ids[train_index]),
                validation_pigs=tuple(str(value) for value in pig_ids[validation_index]),
            )
        )
    return assignments


def validate_fold_assignments(
    assignments: Sequence[FoldAssignment],
    pig_ids: Iterable[str],
    n_splits: int | None = None,
) -> None:
    """Validate complete, disjoint, one-time outer validation coverage."""
    expected = set(map(str, pig_ids))
    if n_splits is not None and len(assignments) != int(n_splits):
        raise ValueError(
            f"Expected {n_splits} folds, but the manifest contains {len(assignments)}."
        )

    validation_counts = {pig_id: 0 for pig_id in expected}
    for assignment in assignments:
        train = set(assignment.train_pigs)
        validation = set(assignment.validation_pigs)
        if train & validation:
            overlap = sorted(train & validation)
            raise ValueError(
                f"Fold {assignment.fold} contains train/validation overlap: {overlap}"
            )
        if train | validation != expected:
            missing = sorted(expected.difference(train | validation))
            extra = sorted((train | validation).difference(expected))
            raise ValueError(
                f"Fold {assignment.fold} does not cover the canonical pig set; "
                f"missing={missing}, extra={extra}."
            )
        for pig_id in validation:
            validation_counts[pig_id] += 1

    invalid = {pig_id: count for pig_id, count in validation_counts.items() if count != 1}
    if invalid:
        raise ValueError(
            "Every pig must occur in the outer validation set exactly once; "
            f"invalid counts={invalid}."
        )


def load_fold_manifest(path: str | Path) -> tuple[int, list[FoldAssignment]]:
    """Load a previously generated canonical fold manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignments = [
        FoldAssignment(
            fold=int(item["fold"]),
            train_pigs=tuple(map(str, item["train_pigs"])),
            validation_pigs=tuple(map(str, item["validation_pigs"])),
        )
        for item in payload["folds"]
    ]
    return int(payload["seed"]), assignments


def load_or_create_fold_manifest(
    samples: Sequence[WindowSample],
    path: str | Path,
    n_splits: int = 10,
    seed: int = SEED,
) -> list[FoldAssignment]:
    """Create once and subsequently reuse the exact same outer fold assignment."""
    path = Path(path)
    pig_ids = sorted({sample.pig_id for sample in samples})
    if path.exists():
        stored_seed, assignments = load_fold_manifest(path)
        if stored_seed != int(seed):
            raise ValueError(
                f"Fold manifest seed={stored_seed} does not match requested seed={seed}."
            )
        validate_fold_assignments(assignments, pig_ids, n_splits=min(n_splits, len(pig_ids)))
        return assignments

    assignments = make_pig_folds(samples, n_splits=n_splits, seed=seed)
    validate_fold_assignments(assignments, pig_ids, n_splits=len(assignments))
    save_fold_manifest(path, assignments, seed=seed)
    return assignments


def make_inner_folds(
    outer_training_samples: Sequence[WindowSample],
    outer_fold: int,
    n_splits: int = 5,
    seed: int = SEED,
) -> list[FoldAssignment]:
    """Create deterministic inner folds using only the current outer-training pigs."""
    inner_seed = int(seed) + int(outer_fold)
    assignments = make_pig_folds(
        outer_training_samples, n_splits=n_splits, seed=inner_seed
    )
    pig_ids = sorted({sample.pig_id for sample in outer_training_samples})
    validate_fold_assignments(assignments, pig_ids, n_splits=len(assignments))
    return assignments


def assert_nested_partition(
    outer_assignment: FoldAssignment,
    inner_assignment: FoldAssignment,
) -> None:
    """Assert that inner selection uses only pigs from the outer training set."""
    outer_train = set(outer_assignment.train_pigs)
    outer_validation = set(outer_assignment.validation_pigs)
    inner_train = set(inner_assignment.train_pigs)
    inner_validation = set(inner_assignment.validation_pigs)
    if inner_train & inner_validation:
        raise ValueError("Inner training and validation pigs overlap.")
    if not (inner_train | inner_validation).issubset(outer_train):
        raise ValueError("An inner split contains pigs outside the outer training set.")
    if (inner_train | inner_validation) & outer_validation:
        raise ValueError("Outer validation pigs leaked into inner model selection.")


def subset_by_pigs(samples: Sequence[WindowSample], pigs: Iterable[str]) -> list[WindowSample]:
    allowed = set(map(str, pigs))
    return [sample for sample in samples if sample.pig_id in allowed]


def fold_assignments_frame(assignments: Sequence[FoldAssignment]) -> pd.DataFrame:
    rows: list[dict] = []
    for assignment in assignments:
        rows.extend(
            {"fold": assignment.fold, "set": "train", "pig_id": pig}
            for pig in assignment.train_pigs
        )
        rows.extend(
            {"fold": assignment.fold, "set": "validation", "pig_id": pig}
            for pig in assignment.validation_pigs
        )
    return pd.DataFrame(rows)


def stack_deep_samples(samples: Sequence[WindowSample]) -> tuple[np.ndarray, np.ndarray]:
    """Stack temporal features and repeat initial body weight across time steps."""
    if not samples:
        return np.empty((0, WINDOW, 0)), np.empty((0, HORIZON))
    x_parts = []
    y_parts = []
    for sample in samples:
        initial = np.full((sample.x.shape[0], 1), sample.initial_weight, dtype=float)
        x_parts.append(np.concatenate([sample.x, initial], axis=1))
        y_parts.append(sample.y)
    return np.stack(x_parts), np.stack(y_parts)


def flatten_tree_samples(
    samples: Sequence[WindowSample], feature_columns: Sequence[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Flatten temporal inputs into lagged tabular predictors."""
    if not samples:
        return pd.DataFrame(), np.empty((0, HORIZON)), np.empty((0,), dtype=object)

    columns: list[str] = []
    for lag_index in range(WINDOW):
        lag = WINDOW - lag_index
        columns.extend(f"{feature}_lag{lag}" for feature in feature_columns)
    columns.append("initial_weight")

    rows = []
    targets = []
    groups = []
    for sample in samples:
        row = np.concatenate([sample.x.reshape(-1), [sample.initial_weight]])
        rows.append(row)
        targets.append(sample.y)
        groups.append(sample.pig_id)
    return (
        pd.DataFrame(np.asarray(rows, dtype=float), columns=columns),
        np.asarray(targets, dtype=float),
        np.asarray(groups, dtype=object),
    )


class SequenceStandardizer:
    """Training-fold mean imputation followed by feature-wise z-score scaling."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "SequenceStandardizer":
        flattened = x.reshape(-1, x.shape[-1])
        self.mean_ = np.nanmean(flattened, axis=0)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        self.std_ = np.nanstd(flattened, axis=0)
        self.std_ = np.where(np.isfinite(self.std_) & (self.std_ >= 1e-12), self.std_, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceStandardizer must be fitted before transformation.")
        output = np.asarray(x, dtype=float).copy()
        missing = np.isnan(output)
        if missing.any():
            feature_indices = np.where(missing)[2]
            output[missing] = np.take(self.mean_, feature_indices)
        return (output - self.mean_) / self.std_


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_fold_predictions(
    samples: Sequence[WindowSample],
    predictions: np.ndarray,
    fold: int,
    set_name: str = "validation",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return window-level predictions, pig-level metrics, and horizon-level metrics."""
    if len(samples) != len(predictions):
        raise ValueError("The sample and prediction counts differ.")

    prediction_rows: list[dict] = []
    for sample, prediction in zip(samples, predictions):
        for horizon_index, (date, observed, estimated) in enumerate(
            zip(sample.target_dates, sample.y, prediction), start=1
        ):
            prediction_rows.append(
                {
                    "fold": fold,
                    "set": set_name,
                    "pig_id": sample.pig_id,
                    "breed": sample.breed,
                    "station": sample.station,
                    "horizon": horizon_index,
                    "target_date": date,
                    "observed": float(observed),
                    "predicted": float(estimated),
                    "error": float(estimated - observed),
                }
            )
    prediction_frame = pd.DataFrame(prediction_rows)

    pig_rows: list[dict] = []
    horizon_rows: list[dict] = []
    for pig_id, pig_frame in prediction_frame.groupby("pig_id", sort=True):
        pig_rmse: list[float] = []
        pig_r2: list[float] = []
        metadata = pig_frame.iloc[0]
        record = {
            "fold": fold,
            "set": set_name,
            "pig_id": pig_id,
            "breed": metadata["breed"],
            "station": metadata["station"],
            "n_windows": int(len(pig_frame) // HORIZON),
        }
        for horizon, horizon_frame in pig_frame.groupby("horizon", sort=True):
            observed = horizon_frame["observed"].to_numpy(dtype=float)
            predicted = horizon_frame["predicted"].to_numpy(dtype=float)
            value_rmse = rmse(observed, predicted)
            value_r2 = r2_score(observed, predicted) if np.unique(observed).size > 1 else np.nan
            record[f"RMSE_h{int(horizon)}"] = value_rmse
            record[f"R2_h{int(horizon)}"] = value_r2
            pig_rmse.append(value_rmse)
            pig_r2.append(value_r2)
            horizon_rows.append(
                {
                    "fold": fold,
                    "set": set_name,
                    "pig_id": pig_id,
                    "horizon": int(horizon),
                    "RMSE": value_rmse,
                    "R2": value_r2,
                    "n": int(len(observed)),
                }
            )
        record["RMSE"] = float(np.nanmean(pig_rmse))
        record["R2"] = float(np.nanmean(pig_r2)) if np.isfinite(pig_r2).any() else np.nan
        pig_rows.append(record)

    return prediction_frame, pd.DataFrame(pig_rows), pd.DataFrame(horizon_rows)


def summarize_pig_metrics(pig_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize fold-level macro metrics and overall stability."""
    fold_rows: list[dict] = []
    for fold, frame in pig_metrics.groupby("fold", sort=True):
        fold_rows.append(
            {
                "fold": int(fold),
                "macro_RMSE": float(frame["RMSE"].mean()),
                "macro_R2": float(frame["R2"].mean()),
                "n_pigs": int(frame["pig_id"].nunique()),
            }
        )
    fold_summary = pd.DataFrame(fold_rows)
    stability_rows = []
    for metric in ("macro_RMSE", "macro_R2"):
        values = fold_summary[metric].to_numpy(dtype=float)
        mean, std, low, high, cv = bootstrap_mean_ci(values, seed=SEED)
        stability_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "CI95_low": low,
                "CI95_high": high,
                "CV": cv,
                "n_folds": int(len(values)),
            }
        )
    return fold_summary, pd.DataFrame(stability_rows)


def bootstrap_mean_ci(
    values: Sequence[float],
    repetitions: int = 2000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (np.nan,) * 5
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size == 1:
        return mean, std, mean, mean, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(repetitions, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
    cv = float(std / mean) if abs(mean) > 1e-12 else np.nan
    return mean, std, float(low), float(high), cv


def write_experiment_workbook(
    output_path: str | Path,
    predictions: pd.DataFrame,
    pig_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    assignments: Sequence[FoldAssignment],
    training_history: pd.DataFrame | None = None,
    tuning_results: pd.DataFrame | None = None,
    selected_parameters: pd.DataFrame | None = None,
    feature_importance: pd.DataFrame | None = None,
    configuration: dict | None = None,
) -> None:
    """Write all reproducibility outputs to one Excel workbook."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fold_summary, stability = summarize_pig_metrics(pig_metrics)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        pig_metrics.to_excel(writer, sheet_name="PigMetrics", index=False)
        horizon_metrics.to_excel(writer, sheet_name="HorizonMetrics", index=False)
        fold_summary.to_excel(writer, sheet_name="FoldSummary", index=False)
        stability.to_excel(writer, sheet_name="Stability", index=False)
        fold_assignments_frame(assignments).to_excel(
            writer, sheet_name="FoldAssignments", index=False
        )
        if training_history is not None and not training_history.empty:
            training_history.to_excel(writer, sheet_name="TrainingHistory", index=False)
        if tuning_results is not None and not tuning_results.empty:
            tuning_results.to_excel(writer, sheet_name="TuningResults", index=False)
        if selected_parameters is not None and not selected_parameters.empty:
            selected_parameters.to_excel(writer, sheet_name="SelectedParameters", index=False)
        if feature_importance is not None and not feature_importance.empty:
            feature_importance.to_excel(writer, sheet_name="FeatureImportance", index=False)
        if configuration:
            pd.DataFrame(
                [{"key": key, "value": json.dumps(value, ensure_ascii=False, default=str)}
                 for key, value in configuration.items()]
            ).to_excel(writer, sheet_name="Configuration", index=False)


def save_fold_manifest(
    path: str | Path,
    assignments: Sequence[FoldAssignment],
    seed: int = SEED,
) -> None:
    payload = {
        "seed": seed,
        "folds": [asdict(assignment) for assignment in assignments],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_experiment(data_path: str | Path) -> tuple[pd.DataFrame, list[str], list[WindowSample]]:
    raw = load_daily_table(data_path)
    normalized, breed_col, station_col = normalize_dataframe(raw)
    features = select_feature_columns(normalized, breed_col, station_col)
    samples = build_window_samples(normalized, features, breed_col, station_col)
    return normalized, features, samples
