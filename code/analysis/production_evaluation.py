"""Evaluate saved out-of-fold forecasts using the manuscript's operational metrics.

The script does not fit models.  It consumes a table of rolling seven-day
out-of-fold predictions and reports pooled fold RMSE, 90/100-kg milestone
metrics, and trajectory-level growth-rate metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Model",
    "Fold",
    "Pig",
    "Window",
    "Horizon",
    "Observed_BW",
    "Predicted_BW",
}
RUNNER_COLUMNS = {
    "fold",
    "set",
    "pig_id",
    "horizon",
    "target_date",
    "observed",
    "predicted",
}
HELD_OUT_SET_VALUES = frozenset({"outer_validation", "outer_test", "test"})
KNOWN_NON_HELD_OUT_SET_VALUES = frozenset(
    {"inner_validation", "validation", "fine_tune", "train"}
)
THRESHOLDS = (90.0, 100.0)


def _normalise_horizon(values: pd.Series, column: str) -> pd.Series:
    """Return integer forecast horizons and reject malformed values."""
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or (~np.isfinite(numeric)).any():
        raise ValueError(f"{column} contains missing or non-finite values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column} must contain integer horizons.")
    horizon = numeric.astype(int)
    if not horizon.between(1, 7).all():
        raise ValueError(f"{column} must be in the range 1--7.")
    return horizon


def _validate_canonical_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the legacy canonical schema without changing its semantics."""
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    out = frame.copy()
    for column in ("Model", "Fold", "Pig"):
        if out[column].isna().any() or out[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must not be empty.")
    out["Horizon"] = _normalise_horizon(out["Horizon"], "Horizon")
    for column in ("Observed_BW", "Predicted_BW"):
        out[column] = pd.to_numeric(out[column], errors="raise")
        if out[column].isna().any() or (~np.isfinite(out[column])).any():
            raise ValueError(f"{column} contains missing or non-finite values.")
    return out


def _normalise_runner_predictions(
    frame: pd.DataFrame, model_name: str | None
) -> pd.DataFrame:
    """Map a model-runner ``Predictions`` table to the canonical schema."""
    missing = sorted(RUNNER_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Model-runner prediction table is missing columns: {missing}")
    if "Model" in frame.columns:
        models = frame["Model"].copy()
    elif model_name is not None and str(model_name).strip():
        models = pd.Series(str(model_name).strip(), index=frame.index)
    else:
        raise ValueError(
            "The Predictions schema has no Model column; provide --model-name."
        )

    out = pd.DataFrame(index=frame.index)
    out["Model"] = models
    if out["Model"].isna().any() or out["Model"].astype(str).str.strip().eq("").any():
        raise ValueError("Model must not be empty.")
    out["Fold"] = frame["fold"]
    out["Pig"] = frame["pig_id"]
    out["Horizon"] = _normalise_horizon(frame["horizon"], "horizon")
    out["Observed_BW"] = pd.to_numeric(frame["observed"], errors="raise")
    out["Predicted_BW"] = pd.to_numeric(frame["predicted"], errors="raise")
    for column in ("Observed_BW", "Predicted_BW"):
        if out[column].isna().any() or (~np.isfinite(out[column])).any():
            raise ValueError(f"{column} contains missing or non-finite values.")
    if out["Fold"].isna().any() or out["Pig"].isna().any():
        raise ValueError("Fold and Pig must not be empty.")

    set_values = frame["set"].astype(str).str.strip().str.casefold()
    known_values = HELD_OUT_SET_VALUES | KNOWN_NON_HELD_OUT_SET_VALUES
    unexpected = sorted(set(set_values.dropna()) - known_values)
    if unexpected:
        raise ValueError(
            "Unexpected set values; expected held-out or known training values, "
            f"received {unexpected}."
        )
    held_out = set_values.isin(HELD_OUT_SET_VALUES)
    if not held_out.any():
        raise ValueError(
            "No held-out predictions found. Expected set values: "
            f"{sorted(HELD_OUT_SET_VALUES)}; received {sorted(set(set_values))}."
        )
    out = out.loc[held_out].copy()

    target_dates = pd.to_datetime(frame.loc[held_out, "target_date"], errors="raise")
    if target_dates.isna().any():
        raise ValueError("target_date contains missing or unparseable values.")
    forecast_origins = target_dates - pd.to_timedelta(out["Horizon"] - 1, unit="D")
    out["target_date"] = target_dates
    out["forecast_origin_date"] = forecast_origins
    group_keys = ["Model", "Fold", "Pig"]
    first_origin = out.groupby(group_keys, sort=False)["forecast_origin_date"].transform("min")
    out["Window"] = (out["forecast_origin_date"] - first_origin).dt.days + 1
    out["Window"] = out["Window"].astype(int)

    duplicate_keys = ["Model", "Fold", "Pig", "Window", "Horizon"]
    if out.duplicated(duplicate_keys).any():
        raise ValueError(
            "A reconstructed window contains more than one row for a horizon: "
            f"{duplicate_keys}."
        )
    return out.reset_index(drop=True)


def normalize_prediction_schema(
    frame: pd.DataFrame, model_name: str | None = None
) -> pd.DataFrame:
    """Normalize canonical or model-runner predictions to one schema."""
    if REQUIRED_COLUMNS.issubset(frame.columns):
        return _validate_canonical_predictions(frame)
    if RUNNER_COLUMNS.issubset(frame.columns):
        return _normalise_runner_predictions(frame, model_name=model_name)
    raise ValueError(
        "Unsupported prediction schema. Expected canonical columns "
        f"{sorted(REQUIRED_COLUMNS)} or model-runner columns {sorted(RUNNER_COLUMNS)}."
    )


def read_predictions(
    path: Path, sheet: str | None = None, model_name: str | None = None
) -> pd.DataFrame:
    """Read and normalize a prediction table from CSV or Excel."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        if sheet is None:
            candidates = [
                name
                for name in ("Window_predictions", "Predictions")
                if name in workbook.sheet_names
            ]
            if not candidates:
                raise ValueError(
                    "Workbook contains neither 'Window_predictions' nor 'Predictions'. "
                    f"Available sheets: {workbook.sheet_names}"
                )
            sheet = candidates[0]
        elif sheet not in workbook.sheet_names:
            raise ValueError(
                f"Sheet {sheet!r} was not found. Available sheets: {workbook.sheet_names}"
            )
        frame = pd.read_excel(workbook, sheet_name=sheet)
    else:
        raise ValueError(f"Unsupported prediction format: {path.suffix}")
    return normalize_prediction_schema(frame, model_name=model_name)


def pooled_rmse(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute one pooled RMSE per model/fold, then summarize across folds."""
    rows: list[dict[str, float | int | str]] = []
    for (model, fold), group in predictions.groupby(["Model", "Fold"], sort=True):
        error = group["Predicted_BW"].to_numpy(float) - group["Observed_BW"].to_numpy(float)
        rows.append({"Model": model, "Fold": int(fold), "RMSE": float(np.sqrt(np.mean(error**2)))})
    fold_metrics = pd.DataFrame(rows)
    summary = fold_metrics.groupby("Model", sort=False)["RMSE"].agg(
        mean="mean", sd=lambda values: values.std(ddof=1), n_folds="count"
    ).reset_index()
    summary["metric"] = "rolling_pooled_RMSE"
    summary = summary.rename(columns={"mean": "value", "sd": "sd_across_folds"})
    return summary[["Model", "metric", "value", "sd_across_folds", "n_folds"]]


def milestone_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute event F1 and paired crossing-day MAE at both thresholds."""
    rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        for (model, fold), group in predictions.groupby(["Model", "Fold"], sort=True):
            windows = group.pivot_table(
                index=["Pig", "Window"],
                columns="Horizon",
                values=["Observed_BW", "Predicted_BW"],
                aggfunc="first",
            ).dropna(how="all")
            observed = windows["Observed_BW"].to_numpy(float)
            predicted = windows["Predicted_BW"].to_numpy(float)
            observed_event = np.any(observed >= threshold, axis=1)
            predicted_event = np.any(predicted >= threshold, axis=1)
            true_positive = observed_event & predicted_event
            false_positive = ~observed_event & predicted_event
            false_negative = observed_event & ~predicted_event
            denominator = 2 * int(true_positive.sum()) + int(false_positive.sum()) + int(false_negative.sum())
            f1 = float(2 * true_positive.sum() / denominator) if denominator else np.nan
            observed_day = np.argmax(observed >= threshold, axis=1) + 1
            predicted_day = np.argmax(predicted >= threshold, axis=1) + 1
            paired = observed_event & predicted_event
            crossing_mae = (
                float(np.mean(np.abs(predicted_day[paired] - observed_day[paired])))
                if paired.any()
                else np.nan
            )
            rows.append(
                {
                    "Model": model,
                    "Fold": int(fold),
                    "Threshold_kg": threshold,
                    "F1": f1,
                    "crossing_day_MAE": crossing_mae,
                    "n_windows": int(len(windows)),
                    "timing_pairs": int(paired.sum()),
                }
            )
    fold_metrics = pd.DataFrame(rows)
    summary = (
        fold_metrics.groupby(["Model", "Threshold_kg"], sort=False)[
            ["F1", "crossing_day_MAE"]
        ]
        .agg(["mean", lambda values: values.std(ddof=1)])
        .reset_index()
    )
    summary.columns = [
        "Model",
        "Threshold_kg",
        "F1_mean",
        "F1_sd",
        "crossing_day_MAE_mean",
        "crossing_day_MAE_sd",
    ]
    return summary


def trajectory_growth(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fit observed/predicted slopes per pig and summarize MAE/RMSE/rank."""
    pig_rows: list[dict[str, float | int | str]] = []
    for (model, fold, pig), group in predictions.groupby(["Model", "Fold", "Pig"], sort=True):
        x = group["Window"].to_numpy(float) + group["Horizon"].to_numpy(float)
        observed = group["Observed_BW"].to_numpy(float)
        predicted = group["Predicted_BW"].to_numpy(float)
        if len(x) < 2 or np.ptp(x) == 0:
            continue
        observed_slope = float(np.polyfit(x, observed, 1)[0])
        predicted_slope = float(np.polyfit(x, predicted, 1)[0])
        pig_rows.append(
            {
                "Model": model,
                "Fold": int(fold),
                "Pig": pig,
                "Observed_stage_ADG": observed_slope,
                "Predicted_stage_ADG": predicted_slope,
            }
        )
    pig_metrics = pd.DataFrame(pig_rows)
    fold_rows: list[dict[str, float | int | str]] = []
    for (model, fold), group in pig_metrics.groupby(["Model", "Fold"], sort=True):
        error = group["Predicted_stage_ADG"] - group["Observed_stage_ADG"]
        fold_rows.append(
            {
                "Model": model,
                "Fold": int(fold),
                "ADG_MAE": float(np.mean(np.abs(error))),
                "ADG_RMSE": float(np.sqrt(np.mean(error**2))),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    summary = fold_metrics.groupby("Model", sort=False)[["ADG_MAE", "ADG_RMSE"]].agg(
        ["mean", lambda values: values.std(ddof=1)]
    ).reset_index()
    summary.columns = ["Model", "ADG_MAE_mean", "ADG_MAE_sd", "ADG_RMSE_mean", "ADG_RMSE_sd"]
    pooled_rows: list[dict[str, float | int | str]] = []
    for model, group in pig_metrics.groupby("Model", sort=False):
        rho = group["Observed_stage_ADG"].rank().corr(group["Predicted_stage_ADG"].rank())
        pooled_rows.append({"Model": model, "pooled_Spearman_rho": float(rho), "n_pigs": int(len(group))})
    return summary.merge(pd.DataFrame(pooled_rows), on="Model", how="left")


def run(args: argparse.Namespace) -> None:
    """Compute and write operational metric summaries."""
    predictions = read_predictions(args.predictions, args.sheet, model_name=args.model_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled_rmse(predictions).to_csv(args.output_dir / "rolling_pooled_rmse.csv", index=False)
    milestone_metrics(predictions).to_csv(args.output_dir / "milestone_metrics.csv", index=False)
    trajectory_growth(predictions).to_csv(args.output_dir / "growth_rate_metrics.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Saved out-of-fold prediction CSV/XLSX.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metric summaries.")
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet containing prediction rows (default: Window_predictions, then Predictions).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model label required for model-runner Predictions tables without a Model column.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
