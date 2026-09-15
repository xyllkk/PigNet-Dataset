"""Select the look-back window from saved inner-validation metrics.

The historical window-screening workflow evaluated W=7--21 using validation
data from outer-training pigs.  This utility performs the final selection step
from a saved ranking table and does not retrain any model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


WINDOW_MIN = 7
WINDOW_MAX = 21


def read_metrics(path: Path) -> pd.DataFrame:
    """Read a saved window-ranking table."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported metrics format: {path.suffix}")
    required = {"Window", "RMSE_Hmean7"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Window-ranking table is missing columns: {missing}")
    frame = frame.copy()
    frame["Window"] = pd.to_numeric(frame["Window"], errors="raise").astype(int)
    frame["RMSE_Hmean7"] = pd.to_numeric(frame["RMSE_Hmean7"], errors="raise")
    frame = frame[frame["Window"].between(WINDOW_MIN, WINDOW_MAX)]
    if frame.empty:
        raise ValueError("No candidate windows in the historical range 7--21.")
    return frame.sort_values(["RMSE_Hmean7", "Window"], kind="stable").reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    """Write the selected window and complete ranking."""
    ranking = read_metrics(args.metrics)
    selected = int(ranking.iloc[0]["Window"])
    result = ranking.assign(selected_window=selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Selected window: W={selected}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="Saved inner-validation window ranking CSV/XLSX.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV containing the selected window.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
