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
THRESHOLDS = (90.0, 100.0)


def read_predictions(path: Path, sheet: str) -> pd.DataFrame:
    """Read a prediction table from CSV or Excel."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet)
    else:
        raise ValueError(f"Unsupported prediction format: {path.suffix}")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    return frame


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
    predictions = read_predictions(args.predictions, args.sheet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled_rmse(predictions).to_csv(args.output_dir / "rolling_pooled_rmse.csv", index=False)
    milestone_metrics(predictions).to_csv(args.output_dir / "milestone_metrics.csv", index=False)
    trajectory_growth(predictions).to_csv(args.output_dir / "growth_rate_metrics.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Saved out-of-fold prediction CSV/XLSX.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metric summaries.")
    parser.add_argument("--sheet", default="Window_predictions", help="Excel sheet containing prediction rows.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
