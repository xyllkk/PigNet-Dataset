"""Summarize fold-level PigNet metrics for the four QC sensitivity variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CONFIGURATIONS = ("QC-REF", "QC-STRICT", "QC-RELAXED", "QC-ROBUST")


def summarize(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for name, path in zip(CONFIGURATIONS, paths):
        frame = pd.read_csv(path)
        missing = {"RMSE", "R2"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows.append(
            {
                "QC configuration": name,
                "RMSE mean": frame["RMSE"].mean(),
                "RMSE SD": frame["RMSE"].std(ddof=1),
                "R2 mean": frame["R2"].mean(),
                "R2 SD": frame["R2"].std(ddof=1),
                "n_folds": len(frame),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the supplementary QC sensitivity analysis."
    )
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--relaxed", type=Path, required=True)
    parser.add_argument("--robust", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = summarize(
        [arguments.ref, arguments.strict, arguments.relaxed, arguments.robust]
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(arguments.output, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
