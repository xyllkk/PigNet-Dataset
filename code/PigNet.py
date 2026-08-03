# -*- coding: utf-8 -*-
"""PigNet with canonical outer folds and leakage-free inner early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_model_runner import TrainingConfig, run_nested_deep_experiment
from pignet_common import HORIZON, SEED

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "PigNet_nested_10_fold_CV_win14"
CANONICAL_FOLD_MANIFEST = ROOT / "results" / "canonical_outer_folds_W14_H7_seed42.json"

HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.25
USE_TCN_FRONTEND = False
TCN_DILATIONS = (1, 2, 4)
TCN_KERNEL = 3
TCN_DROPOUT = 0.15
USE_FILM_INIT = True
SE_REDUCTION = 4
USE_HORIZON_QUERY_ATTENTION = True
ATTENTION_HEADS = 2
ATTENTION_DROPOUT = 0.10
FUSE_LAST_STATE = True
USE_HORIZON_CONVOLUTION = True
HORIZON_CONVOLUTION_KERNEL = 3
HORIZON_CONVOLUTION_DROPOUT = 0.15

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
    smooth_lambda=0.02,
    inner_splits=5,
)


class CausalConvBlock(nn.Module):
    """Residual causal convolution over the input sequence."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2.")
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.normalization = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = F.pad(x, (self.left_padding, 0))
        x = self.convolution(x).transpose(1, 2)
        x = self.normalization(x)
        x = self.dropout(F.gelu(x))
        return residual + x


class TCNFrontend(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        dilations: Sequence[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(
                    channels=feature_dim,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class HorizonConvRefiner(nn.Module):
    """Depthwise-separable convolution along the forecast-horizon axis."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        refined = context.transpose(1, 2)
        refined = self.pointwise(self.depthwise(refined)).transpose(1, 2)
        return context + self.dropout(refined)


class PigNetForecaster(nn.Module):
    """LSTM encoder with FiLM, channel recalibration, horizon queries, and H-Conv."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.hidden_size = HIDDEN_SIZE
        self.horizon = HORIZON
        self.tcn = (
            TCNFrontend(
                feature_dim=input_dim,
                dilations=TCN_DILATIONS,
                kernel_size=TCN_KERNEL,
                dropout=TCN_DROPOUT,
            )
            if USE_TCN_FRONTEND
            else None
        )
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.sequence_dropout = nn.Dropout(DROPOUT)

        self.film = (
            nn.Sequential(
                nn.Linear(1, 64),
                nn.GELU(),
                nn.Linear(64, 2 * HIDDEN_SIZE),
            )
            if USE_FILM_INIT
            else None
        )
        reduced = max(HIDDEN_SIZE // SE_REDUCTION, 1)
        self.se_reduction = nn.Linear(HIDDEN_SIZE, reduced)
        self.se_expansion = nn.Linear(reduced, HIDDEN_SIZE)

        if USE_HORIZON_QUERY_ATTENTION:
            if HIDDEN_SIZE % ATTENTION_HEADS != 0:
                raise ValueError("HIDDEN_SIZE must be divisible by ATTENTION_HEADS.")
            self.horizon_queries = nn.Parameter(
                torch.randn(HORIZON, HIDDEN_SIZE)
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=HIDDEN_SIZE,
                num_heads=ATTENTION_HEADS,
                dropout=ATTENTION_DROPOUT,
                batch_first=True,
            )
        else:
            self.register_parameter("horizon_queries", None)
            self.attention = None

        self.horizon_refiner = (
            HorizonConvRefiner(
                channels=HIDDEN_SIZE,
                kernel_size=HORIZON_CONVOLUTION_KERNEL,
                dropout=HORIZON_CONVOLUTION_DROPOUT,
            )
            if USE_HORIZON_CONVOLUTION
            else None
        )
        self.context_normalization = nn.LayerNorm(HIDDEN_SIZE)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tcn is not None:
            x = self.tcn(x)
        encoded, _ = self.encoder(x)
        encoded = self.sequence_dropout(encoded)

        if self.film is not None:
            initial_weight = x[:, 0, -1].unsqueeze(-1)
            gamma, beta = self.film(initial_weight).chunk(2, dim=-1)
            gamma = 1.0 + 0.2 * torch.tanh(gamma)
            beta = 0.2 * torch.tanh(beta)
            encoded = gamma.unsqueeze(1) * encoded + beta.unsqueeze(1)

        channel_gate = encoded.mean(dim=1)
        channel_gate = F.gelu(self.se_reduction(channel_gate))
        channel_gate = torch.sigmoid(self.se_expansion(channel_gate))
        encoded = encoded * channel_gate.unsqueeze(1)

        if self.attention is not None and self.horizon_queries is not None:
            queries = self.horizon_queries.unsqueeze(0).expand(
                encoded.shape[0], -1, -1
            )
            context, _ = self.attention(queries, encoded, encoded)
        else:
            context = encoded.mean(dim=1, keepdim=True).repeat(1, HORIZON, 1)

        if FUSE_LAST_STATE:
            context = context + encoded[:, -1, :].unsqueeze(1)
        if self.horizon_refiner is not None:
            context = self.horizon_refiner(context)
        context = self.context_normalization(context)
        prediction = self.head(context.reshape(-1, HIDDEN_SIZE))
        return prediction.view(x.shape[0], HORIZON)


def build_model(input_dim: int) -> nn.Module:
    return PigNetForecaster(input_dim=input_dim)


def main() -> None:
    run_nested_deep_experiment(
        model_name="PigNet",
        model_factory=build_model,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        canonical_manifest_path=CANONICAL_FOLD_MANIFEST,
        training_config=TRAINING,
        model_configuration={
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "use_tcn_frontend": USE_TCN_FRONTEND,
            "use_film_initial_weight": USE_FILM_INIT,
            "se_reduction": SE_REDUCTION,
            "use_horizon_query_attention": USE_HORIZON_QUERY_ATTENTION,
            "attention_heads": ATTENTION_HEADS,
            "attention_dropout": ATTENTION_DROPOUT,
            "fuse_last_state": FUSE_LAST_STATE,
            "use_horizon_convolution": USE_HORIZON_CONVOLUTION,
            "horizon_convolution_kernel": HORIZON_CONVOLUTION_KERNEL,
            "horizon_convolution_dropout": HORIZON_CONVOLUTION_DROPOUT,
        },
        outer_splits=10,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
