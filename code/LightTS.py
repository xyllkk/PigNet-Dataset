# -*- coding: utf-8 -*-
"""LightTS/IEBlock baseline with canonical outer folds and leakage-free inner early stopping."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from deep_model_runner import TrainingConfig, run_nested_deep_experiment
from pignet_common import HORIZON, SEED, WINDOW

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "LightTS_nested_10_fold_CV_win14"
CANONICAL_FOLD_MANIFEST = ROOT / "results" / "canonical_outer_folds_W14_H7_seed42.json"

D_MODEL = 128
DROPOUT = 0.25
CHUNK_SIZE = 7

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


class IEBlock(nn.Module):
    """Information-exchange block used in the LightTS architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_nodes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        bottleneck = max(hidden_dim // 4, 4)
        self.spatial_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck),
        )
        self.channel_projection = nn.Linear(num_nodes, num_nodes, bias=True)
        nn.init.eye_(self.channel_projection.weight)
        nn.init.zeros_(self.channel_projection.bias)
        self.output_projection = nn.Linear(bottleneck, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_first = x.transpose(1, 2)
        spatial = self.spatial_projection(node_first)
        exchanged = self.channel_projection(spatial.transpose(1, 2)).transpose(1, 2)
        output = self.output_projection(spatial + exchanged)
        return output.transpose(1, 2)


class LightTSForecaster(nn.Module):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        if WINDOW % CHUNK_SIZE != 0:
            raise ValueError("WINDOW must be divisible by CHUNK_SIZE.")
        self.input_features = input_features
        self.number_of_chunks = WINDOW // CHUNK_SIZE
        branch_dim = D_MODEL // 4

        self.continuous_block = IEBlock(
            input_dim=CHUNK_SIZE,
            hidden_dim=D_MODEL,
            output_dim=branch_dim,
            num_nodes=self.number_of_chunks * input_features,
            dropout=DROPOUT,
        )
        self.interval_block = IEBlock(
            input_dim=self.number_of_chunks,
            hidden_dim=D_MODEL,
            output_dim=branch_dim,
            num_nodes=CHUNK_SIZE * input_features,
            dropout=DROPOUT,
        )
        self.output_block = IEBlock(
            input_dim=2 * branch_dim,
            hidden_dim=D_MODEL // 2,
            output_dim=HORIZON,
            num_nodes=input_features,
            dropout=DROPOUT,
        )
        self.autoregressive_projection = nn.Linear(WINDOW, HORIZON)
        self.feature_readout = nn.Linear(input_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, feature_count = x.shape
        if sequence_length != WINDOW or feature_count != self.input_features:
            raise ValueError(
                f"Expected [batch, {WINDOW}, {self.input_features}], got {tuple(x.shape)}."
            )

        blocks = x.reshape(
            batch_size,
            self.number_of_chunks,
            CHUNK_SIZE,
            feature_count,
        )
        continuous = blocks.permute(0, 2, 1, 3).reshape(
            batch_size,
            CHUNK_SIZE,
            self.number_of_chunks * feature_count,
        )
        continuous = self.continuous_block(continuous)
        continuous = continuous.reshape(
            batch_size,
            -1,
            self.number_of_chunks,
            feature_count,
        ).mean(dim=2)

        interval = blocks.permute(0, 1, 3, 2).reshape(
            batch_size,
            self.number_of_chunks,
            feature_count * CHUNK_SIZE,
        )
        interval = self.interval_block(interval)
        interval = interval.reshape(
            batch_size,
            -1,
            feature_count,
            CHUNK_SIZE,
        ).mean(dim=3)

        representation = torch.cat([continuous, interval], dim=1)
        nonlinear = self.output_block(representation)
        nonlinear = self.feature_readout(nonlinear).squeeze(-1)

        autoregressive = self.autoregressive_projection(x.transpose(1, 2)).transpose(1, 2)
        autoregressive = self.feature_readout(autoregressive).squeeze(-1)
        return nonlinear + autoregressive


def build_model(input_dim: int) -> nn.Module:
    return LightTSForecaster(input_features=input_dim)


def main() -> None:
    run_nested_deep_experiment(
        model_name="LightTS",
        model_factory=build_model,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        canonical_manifest_path=CANONICAL_FOLD_MANIFEST,
        training_config=TRAINING,
        model_configuration={
            "d_model": D_MODEL,
            "dropout": DROPOUT,
            "chunk_size": CHUNK_SIZE,
        },
        outer_splits=10,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
