# -*- coding: utf-8 -*-
"""LightTS/IEBlock baseline with canonical pig-level 10-fold cross-validation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pignet_common import (
    HORIZON,
    SEED,
    WINDOW,
    SequenceStandardizer,
    evaluate_fold_predictions,
    make_pig_folds,
    prepare_experiment,
    save_fold_manifest,
    set_global_seed,
    stack_deep_samples,
    subset_by_pigs,
    write_experiment_workbook,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "main_cohort"
OUTPUT_DIR = ROOT / "results" / "LightTS_10_fold_CV_win14"
OUTPUT_EXCEL = OUTPUT_DIR / "LightTS_W14_H7_Outer10CV.xlsx"

N_SPLITS = 10
MAX_EPOCHS = 400
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 3e-4
D_MODEL = 128
DROPOUT = 0.25
CHUNK_SIZE = 7
HUBER_BETA = 1.0
GRADIENT_CLIP_NORM = 1.0
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 8
EARLY_STOPPING_PATIENCE = 25
MINIMUM_IMPROVEMENT = 1e-6


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class IEBlock(nn.Module):
    """Information-exchange block used by the LightTS forecasting architecture."""

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
        # x: [batch, temporal_dimension, nodes]
        node_first = x.transpose(1, 2)
        spatial = self.spatial_projection(node_first)
        exchanged = self.channel_projection(spatial.transpose(1, 2)).transpose(1, 2)
        output = self.output_projection(spatial + exchanged)
        return output.transpose(1, 2)


class LightTSForecaster(nn.Module):
    """Continuous- and interval-sampling LightTS forecaster."""

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
                f"Expected [batch, {WINDOW}, {self.input_features}], got {tuple(x.shape)}"
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


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        SequenceDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def validation_rmse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    observed_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            prediction = model(x_batch.to(device))
            observed_parts.append(y_batch.numpy())
            predicted_parts.append(prediction.cpu().numpy())
    observed = np.concatenate(observed_parts, axis=0)
    predicted = np.concatenate(predicted_parts, axis=0)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def train_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    fold: int,
) -> tuple[LightTSForecaster, SequenceStandardizer, pd.DataFrame, float]:
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standardizer = SequenceStandardizer().fit(x_train)
    x_train_scaled = standardizer.transform(x_train)
    x_validation_scaled = standardizer.transform(x_validation)

    model = LightTSForecaster(input_features=x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.SmoothL1Loss(beta=HUBER_BETA)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    train_loader = make_loader(x_train_scaled, y_train, shuffle=True, seed=SEED)
    validation_loader = make_loader(
        x_validation_scaled, y_validation, shuffle=False, seed=SEED
    )

    best_value = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        current_value = validation_rmse(model, validation_loader, device)
        scheduler.step(current_value)
        history_rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_RMSE": current_value,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if current_value < best_value - MINIMUM_IMPROVEMENT:
            best_value = current_value
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was produced.")
    model.load_state_dict(best_state)
    return model, standardizer, pd.DataFrame(history_rows), float(best_value)


def predict(
    model: LightTSForecaster,
    standardizer: SequenceStandardizer,
    x: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    scaled = standardizer.transform(x)
    loader = make_loader(
        scaled,
        np.zeros((len(scaled), HORIZON), dtype=np.float32),
        shuffle=False,
        seed=SEED,
    )
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            output.append(model(x_batch.to(device)).cpu().numpy())
    return np.concatenate(output, axis=0)


def main() -> None:
    set_global_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, feature_columns, samples = prepare_experiment(DATA_PATH)
    assignments = make_pig_folds(samples, n_splits=N_SPLITS, seed=SEED)
    save_fold_manifest(OUTPUT_DIR / "fold_assignments.json", assignments, seed=SEED)

    prediction_frames: list[pd.DataFrame] = []
    pig_metric_frames: list[pd.DataFrame] = []
    horizon_metric_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    selected_rows: list[dict] = []

    for assignment in assignments:
        training_samples = subset_by_pigs(samples, assignment.train_pigs)
        validation_samples = subset_by_pigs(samples, assignment.validation_pigs)
        x_train, y_train = stack_deep_samples(training_samples)
        x_validation, y_validation = stack_deep_samples(validation_samples)

        model, standardizer, history, best_value = train_fold(
            x_train,
            y_train,
            x_validation,
            y_validation,
            fold=assignment.fold,
        )
        predictions = predict(model, standardizer, x_validation)
        prediction_frame, pig_metrics, horizon_metrics = evaluate_fold_predictions(
            validation_samples,
            predictions,
            fold=assignment.fold,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_mean": standardizer.mean_,
                "scaler_std": standardizer.std_,
                "feature_columns": feature_columns,
                "fold": assignment.fold,
                "seed": SEED,
            },
            OUTPUT_DIR / f"LightTS_fold{assignment.fold}.pt",
        )
        prediction_frames.append(prediction_frame)
        pig_metric_frames.append(pig_metrics)
        horizon_metric_frames.append(horizon_metrics)
        history_frames.append(history)
        selected_rows.append(
            {
                "fold": assignment.fold,
                "best_validation_RMSE": best_value,
                "epochs_trained": int(history["epoch"].max()),
                "d_model": D_MODEL,
                "chunk_size": CHUNK_SIZE,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        )
        print(
            f"Fold {assignment.fold:02d}: "
            f"best validation RMSE={best_value:.6f}, epochs={len(history)}"
        )

    configuration = {
        "model": "LightTS",
        "seed": SEED,
        "outer_folds": len(assignments),
        "window": WINDOW,
        "horizon": HORIZON,
        "features": feature_columns,
        "maximum_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": LR_SCHEDULER_FACTOR,
        "scheduler_patience": LR_SCHEDULER_PATIENCE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "minimum_improvement": MINIMUM_IMPROVEMENT,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
    }
    write_experiment_workbook(
        OUTPUT_EXCEL,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        pig_metrics=pd.concat(pig_metric_frames, ignore_index=True),
        horizon_metrics=pd.concat(horizon_metric_frames, ignore_index=True),
        assignments=assignments,
        training_history=pd.concat(history_frames, ignore_index=True),
        selected_parameters=pd.DataFrame(selected_rows),
        configuration=configuration,
    )
    print(f"Results written to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
