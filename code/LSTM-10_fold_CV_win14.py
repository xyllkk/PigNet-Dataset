# -*- coding: utf-8 -*-
"""LSTM baseline with canonical outer folds and leakage-free inner early stopping."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from deep_model_runner import TrainingConfig, run_nested_deep_experiment
from pignet_common import HORIZON, SEED

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "LSTM_nested_10_fold_CV_win14"
CANONICAL_FOLD_MANIFEST = ROOT / "results" / "canonical_outer_folds_W14_H7_seed42.json"

HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.25

TRAINING = TrainingConfig(
    maximum_epochs=400,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=3e-4,
    huber_beta=1.0,
    gradient_clip_norm=1.0,
    scheduler_factor=0.5,
    scheduler_patience=8,
    early_stopping_patience=25,
    minimum_improvement=1e-6,
    smooth_lambda=0.0,
    inner_splits=5,
)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE // 2, HORIZON),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1, :])


def build_model(input_dim: int) -> nn.Module:
    return LSTMForecaster(input_dim=input_dim)


def main() -> None:
    run_nested_deep_experiment(
        model_name="LSTM",
        model_factory=build_model,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        canonical_manifest_path=CANONICAL_FOLD_MANIFEST,
        training_config=TRAINING,
        model_configuration={
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        outer_splits=10,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
