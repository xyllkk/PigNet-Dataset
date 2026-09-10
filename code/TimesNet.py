# -*- coding: utf-8 -*-
"""TimesNet baseline with canonical outer folds and leakage-free inner early stopping."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_model_runner import TrainingConfig, run_nested_deep_experiment
from pignet_common import HORIZON, SEED, WINDOW

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "TimesNet_nested_10_fold_CV_win14"
CANONICAL_FOLD_MANIFEST = ROOT / "results" / "canonical_outer_folds_W14_H7_seed42.json"

D_MODEL = 128
ENCODER_LAYERS = 1
DROPOUT = 0.25
TOP_K = 3
NUMBER_OF_KERNELS = 3

TRAINING = TrainingConfig(
    maximum_epochs=400,
    batch_size=128,
    learning_rate=3e-4,
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


def sinusoidal_position_encoding(
    length: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    indices = torch.arange(dimension, device=device, dtype=dtype).unsqueeze(0)
    rates = 1.0 / (10000 ** (((indices // 2) * 2) / dimension))
    angles = positions * rates
    encoding = torch.zeros(length, dimension, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(angles[:, 0::2])
    encoding[:, 1::2] = torch.cos(angles[:, 1::2])
    return encoding.unsqueeze(0)


class TimesBlock(nn.Module):
    """Frequency-period decomposition followed by multi-kernel 2-D convolution."""

    def __init__(
        self,
        d_model: int,
        top_k: int,
        number_of_kernels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.top_k = max(1, int(top_k))
        kernel_sizes = [3, 5, 7, 11][: max(1, int(number_of_kernels))]
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(
                    d_model,
                    d_model,
                    kernel_size=(1, kernel_size),
                    padding=(0, kernel_size // 2),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        residual = x
        channel_first = x.permute(0, 2, 1)
        spectrum = torch.fft.rfft(channel_first, dim=-1)
        amplitude = spectrum.abs().mean(dim=(0, 1))
        if amplitude.shape[0] <= 1:
            return self.normalization(residual)

        amplitude = amplitude.clone()
        amplitude[0] = 0
        k = min(self.top_k, amplitude.shape[0] - 1)
        values, indices = torch.topk(amplitude, k=k)
        weights = torch.softmax(values, dim=0)

        period_outputs: list[torch.Tensor] = []
        for frequency_index in indices.tolist():
            period = max(1, int(sequence_length // max(1, frequency_index)))
            padding = (period - sequence_length % period) % period
            padded = (
                F.pad(channel_first, (0, padding))
                if padding > 0
                else channel_first
            )
            rows = padded.shape[-1] // period
            two_dimensional = padded.view(
                batch_size, channels, rows, period
            )
            convolved = sum(
                F.gelu(convolution(two_dimensional))
                for convolution in self.convolutions
            ) / len(self.convolutions)
            convolved = self.dropout(convolved)
            period_outputs.append(
                convolved.reshape(batch_size, channels, -1)[
                    :, :, :sequence_length
                ]
            )

        combined = torch.zeros_like(channel_first)
        for weight, output in zip(weights, period_outputs):
            combined = combined + weight * output
        return self.normalization(combined.permute(0, 2, 1) + residual)


class TimesNetForecaster(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, D_MODEL)
        self.embedding_dropout = nn.Dropout(DROPOUT)
        self.temporal_projection = nn.Linear(WINDOW, WINDOW + HORIZON)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=D_MODEL,
                    top_k=TOP_K,
                    number_of_kernels=NUMBER_OF_KERNELS,
                    dropout=DROPOUT,
                )
                for _ in range(max(1, ENCODER_LAYERS))
            ]
        )
        self.final_normalization = nn.LayerNorm(D_MODEL)
        self.output_projection = nn.Linear(D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != WINDOW:
            raise ValueError(
                f"Expected sequence length {WINDOW}, received {x.shape[1]}."
            )
        representation = self.embedding_dropout(self.value_embedding(x))
        representation = self.temporal_projection(
            representation.permute(0, 2, 1)
        ).permute(0, 2, 1)
        representation = representation + sinusoidal_position_encoding(
            representation.shape[1],
            D_MODEL,
            representation.device,
            representation.dtype,
        )
        for block in self.blocks:
            representation = block(representation)
        representation = self.final_normalization(representation)
        output = self.output_projection(representation).squeeze(-1)
        return output[:, -HORIZON:]


def build_model(input_dim: int) -> nn.Module:
    return TimesNetForecaster(input_dim=input_dim)


def main() -> None:
    run_nested_deep_experiment(
        model_name="TimesNet",
        model_factory=build_model,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        canonical_manifest_path=CANONICAL_FOLD_MANIFEST,
        training_config=TRAINING,
        model_configuration={
            "d_model": D_MODEL,
            "encoder_layers": ENCODER_LAYERS,
            "dropout": DROPOUT,
            "top_k": TOP_K,
            "number_of_kernels": NUMBER_OF_KERNELS,
        },
        outer_splits=10,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
