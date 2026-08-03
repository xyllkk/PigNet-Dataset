# -*- coding: utf-8 -*-
"""LSTM baseline with canonical pig-level 10-fold cross-validation."""

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
OUTPUT_DIR = ROOT / "results" / "LSTM_10_fold_CV_win14"
OUTPUT_EXCEL = OUTPUT_DIR / "LSTM_W14_H7_Outer10CV.xlsx"

N_SPLITS = 10
MAX_EPOCHS = 400
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 3e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.25
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


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SequenceDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    observed_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            prediction = model(x_batch)
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
) -> tuple[LSTMForecaster, SequenceStandardizer, pd.DataFrame, float]:
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    standardizer = SequenceStandardizer().fit(x_train)
    x_train_scaled = standardizer.transform(x_train)
    x_validation_scaled = standardizer.transform(x_validation)

    model = LSTMForecaster(input_dim=x_train.shape[-1]).to(device)
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

    best_validation_rmse = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        validation_rmse = validate(model, validation_loader, device)
        scheduler.step(validation_rmse)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "training_loss": float(np.mean(batch_losses)),
                "validation_RMSE": validation_rmse,
                "learning_rate": learning_rate,
            }
        )

        if validation_rmse < best_validation_rmse - MINIMUM_IMPROVEMENT:
            best_validation_rmse = validation_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was produced.")
    model.load_state_dict(best_state)
    return model, standardizer, pd.DataFrame(history_rows), float(best_validation_rmse)


def predict(
    model: LSTMForecaster,
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
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            predictions.append(model(x_batch.to(device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


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
        train_samples = subset_by_pigs(samples, assignment.train_pigs)
        validation_samples = subset_by_pigs(samples, assignment.validation_pigs)
        x_train, y_train = stack_deep_samples(train_samples)
        x_validation, y_validation = stack_deep_samples(validation_samples)

        model, standardizer, history, best_rmse = train_fold(
            x_train,
            y_train,
            x_validation,
            y_validation,
            fold=assignment.fold,
        )
        validation_predictions = predict(model, standardizer, x_validation)
        prediction_frame, pig_metrics, horizon_metrics = evaluate_fold_predictions(
            validation_samples,
            validation_predictions,
            fold=assignment.fold,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler_mean": standardizer.mean_,
            "scaler_std": standardizer.std_,
            "feature_columns": feature_columns,
            "fold": assignment.fold,
            "seed": SEED,
        }
        torch.save(checkpoint, OUTPUT_DIR / f"LSTM_fold{assignment.fold}.pt")

        prediction_frames.append(prediction_frame)
        pig_metric_frames.append(pig_metrics)
        horizon_metric_frames.append(horizon_metrics)
        history_frames.append(history)
        selected_rows.append(
            {
                "fold": assignment.fold,
                "best_validation_RMSE": best_rmse,
                "epochs_trained": int(history["epoch"].max()),
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        )
        print(
            f"Fold {assignment.fold:02d}: "
            f"best validation RMSE={best_rmse:.6f}, epochs={len(history)}"
        )

    configuration = {
        "model": "LSTM",
        "seed": SEED,
        "outer_folds": len(assignments),
        "window": 14,
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
