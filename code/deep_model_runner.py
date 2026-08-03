# -*- coding: utf-8 -*-
"""Leakage-free nested cross-validation runner for neural forecasting models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pignet_common import (
    HORIZON,
    SEED,
    FoldAssignment,
    SequenceStandardizer,
    WindowSample,
    assert_nested_partition,
    evaluate_fold_predictions,
    load_or_create_fold_manifest,
    make_inner_folds,
    prepare_experiment,
    set_global_seed,
    stack_deep_samples,
    subset_by_pigs,
    write_experiment_workbook,
)

ModelFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class TrainingConfig:
    maximum_epochs: int = 400
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 3e-4
    huber_beta: float = 1.0
    gradient_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    early_stopping_patience: int = 25
    minimum_improvement: float = 1e-6
    smooth_lambda: float = 0.0
    inner_splits: int = 5


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def _make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        SequenceDataset(x, y),
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def _loss_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    smooth_lambda: float,
) -> torch.Tensor:
    loss = criterion(prediction, target)
    if smooth_lambda > 0.0 and prediction.shape[1] > 1:
        smoothness = torch.mean(torch.abs(prediction[:, 1:] - prediction[:, :-1]))
        loss = loss + float(smooth_lambda) * smoothness
    return loss


def _predict_scaled(
    model: nn.Module,
    x_scaled: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    loader = _make_loader(
        x_scaled,
        np.zeros((len(x_scaled), HORIZON), dtype=np.float32),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            parts.append(model(x_batch.to(device)).cpu().numpy())
    if not parts:
        return np.empty((0, HORIZON), dtype=float)
    return np.concatenate(parts, axis=0)


def _pig_macro_rmse(
    validation_samples: Sequence[WindowSample],
    predictions: np.ndarray,
) -> float:
    _, pig_metrics, _ = evaluate_fold_predictions(
        validation_samples,
        predictions,
        fold=0,
        set_name="inner_validation",
    )
    return float(pig_metrics["RMSE"].mean())


def select_training_epoch_on_inner_fold(
    model_factory: ModelFactory,
    inner_training_samples: Sequence[WindowSample],
    inner_validation_samples: Sequence[WindowSample],
    outer_fold: int,
    inner_fold: int,
    config: TrainingConfig,
    seed: int,
) -> tuple[int, list[float], pd.DataFrame, float]:
    """Select one best epoch using one inner validation fold only."""
    x_train, y_train = stack_deep_samples(inner_training_samples)
    x_validation, _ = stack_deep_samples(inner_validation_samples)
    if len(x_train) == 0 or len(x_validation) == 0:
        raise RuntimeError(
            f"Outer fold {outer_fold}, inner fold {inner_fold}: "
            "empty inner training or validation data."
        )

    run_seed = int(seed) + 1000 * int(outer_fold) + int(inner_fold)
    set_global_seed(run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    standardizer = SequenceStandardizer().fit(x_train)
    x_train_scaled = standardizer.transform(x_train)
    x_validation_scaled = standardizer.transform(x_validation)

    model = model_factory(x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.SmoothL1Loss(beta=config.huber_beta)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    train_loader = _make_loader(
        x_train_scaled,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=run_seed,
    )

    best_epoch = 0
    best_inner_rmse = np.inf
    epochs_without_improvement = 0
    learning_rate_schedule: list[float] = []
    history_rows: list[dict] = []

    for epoch in range(1, config.maximum_epochs + 1):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        learning_rate_schedule.append(epoch_learning_rate)
        model.train()
        losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = _loss_value(
                prediction,
                y_batch,
                criterion=criterion,
                smooth_lambda=config.smooth_lambda,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_predictions = _predict_scaled(
            model,
            x_validation_scaled,
            batch_size=config.batch_size,
            seed=run_seed,
        )
        inner_macro_rmse = _pig_macro_rmse(
            inner_validation_samples, validation_predictions
        )
        scheduler.step(inner_macro_rmse)
        history_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "stage": "inner_selection",
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "inner_macro_RMSE": inner_macro_rmse,
                "learning_rate_used": epoch_learning_rate,
                "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if inner_macro_rmse < best_inner_rmse - config.minimum_improvement:
            best_inner_rmse = inner_macro_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_epoch < 1:
        raise RuntimeError(
            f"Outer fold {outer_fold}, inner fold {inner_fold}: "
            "no valid inner checkpoint."
        )
    return (
        best_epoch,
        learning_rate_schedule,
        pd.DataFrame(history_rows),
        float(best_inner_rmse),
    )


def _aggregate_learning_rate_schedule(
    schedules: Sequence[Sequence[float]],
    selected_epoch: int,
) -> list[float]:
    """Aggregate inner-fold schedules by the epoch-wise median learning rate."""
    if not schedules:
        raise ValueError("At least one inner learning-rate schedule is required.")
    output: list[float] = []
    for epoch_index in range(int(selected_epoch)):
        values = [
            float(schedule[min(epoch_index, len(schedule) - 1)])
            for schedule in schedules
            if len(schedule) > 0
        ]
        if not values:
            raise RuntimeError("An empty learning-rate schedule was encountered.")
        output.append(float(np.median(values)))
    return output


def select_training_epoch_across_inner_folds(
    model_factory: ModelFactory,
    outer_assignment: FoldAssignment,
    outer_training_samples: Sequence[WindowSample],
    inner_assignments: Sequence[FoldAssignment],
    config: TrainingConfig,
    seed: int,
) -> tuple[int, list[float], pd.DataFrame, pd.DataFrame]:
    """Select the retraining epoch by grouped inner five-fold validation."""
    best_epochs: list[int] = []
    best_scores: list[float] = []
    schedules: list[list[float]] = []
    histories: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for inner_assignment in inner_assignments:
        assert_nested_partition(outer_assignment, inner_assignment)
        inner_training_samples = subset_by_pigs(
            outer_training_samples, inner_assignment.train_pigs
        )
        inner_validation_samples = subset_by_pigs(
            outer_training_samples, inner_assignment.validation_pigs
        )
        best_epoch, schedule, history, best_rmse = (
            select_training_epoch_on_inner_fold(
                model_factory=model_factory,
                inner_training_samples=inner_training_samples,
                inner_validation_samples=inner_validation_samples,
                outer_fold=outer_assignment.fold,
                inner_fold=inner_assignment.fold,
                config=config,
                seed=seed,
            )
        )
        best_epochs.append(best_epoch)
        best_scores.append(best_rmse)
        schedules.append(schedule)
        histories.append(history)
        summary_rows.append(
            {
                "outer_fold": outer_assignment.fold,
                "inner_fold": inner_assignment.fold,
                "best_epoch": best_epoch,
                "best_inner_macro_RMSE": best_rmse,
                "inner_training_pigs": ",".join(inner_assignment.train_pigs),
                "inner_validation_pigs": ",".join(
                    inner_assignment.validation_pigs
                ),
            }
        )

    selected_epoch = int(np.clip(np.rint(np.median(best_epochs)), 1, config.maximum_epochs))
    selected_schedule = _aggregate_learning_rate_schedule(
        schedules, selected_epoch=selected_epoch
    )
    return (
        selected_epoch,
        selected_schedule,
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def retrain_on_outer_training_set(
    model_factory: ModelFactory,
    outer_training_samples: Sequence[WindowSample],
    selected_epoch: int,
    learning_rate_schedule: Sequence[float],
    outer_fold: int,
    config: TrainingConfig,
    seed: int,
) -> tuple[nn.Module, SequenceStandardizer, pd.DataFrame]:
    """Retrain from scratch on all outer-training pigs for the selected epochs."""
    x_train, y_train = stack_deep_samples(outer_training_samples)
    if len(x_train) == 0:
        raise RuntimeError(f"Outer fold {outer_fold}: empty outer training data.")
    if selected_epoch > len(learning_rate_schedule):
        raise ValueError("The replay learning-rate schedule is shorter than selected_epoch.")

    run_seed = int(seed) + 100_000 + 1000 * int(outer_fold)
    set_global_seed(run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standardizer = SequenceStandardizer().fit(x_train)
    x_train_scaled = standardizer.transform(x_train)

    model = model_factory(x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate_schedule[0]),
        weight_decay=config.weight_decay,
    )
    criterion = nn.SmoothL1Loss(beta=config.huber_beta)
    train_loader = _make_loader(
        x_train_scaled,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=run_seed,
    )

    history_rows: list[dict] = []
    for epoch in range(1, selected_epoch + 1):
        replay_lr = float(learning_rate_schedule[epoch - 1])
        for group in optimizer.param_groups:
            group["lr"] = replay_lr
        model.train()
        losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = _loss_value(
                prediction,
                y_batch,
                criterion=criterion,
                smooth_lambda=config.smooth_lambda,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": np.nan,
                "stage": "outer_retrain",
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "inner_macro_RMSE": np.nan,
                "learning_rate_used": replay_lr,
                "learning_rate_next": replay_lr,
            }
        )
    return model, standardizer, pd.DataFrame(history_rows)


def predict(
    model: nn.Module,
    standardizer: SequenceStandardizer,
    x: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    return _predict_scaled(
        model,
        standardizer.transform(x),
        batch_size=batch_size,
        seed=seed,
    )


def run_nested_deep_experiment(
    model_name: str,
    model_factory: ModelFactory,
    data_path: str | Path,
    output_dir: str | Path,
    canonical_manifest_path: str | Path,
    training_config: TrainingConfig,
    model_configuration: Mapping[str, object],
    outer_splits: int = 10,
    seed: int = SEED,
) -> None:
    """Run outer evaluation with inner five-fold epoch selection and refitting."""
    set_global_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, feature_columns, samples = prepare_experiment(data_path)
    outer_assignments = load_or_create_fold_manifest(
        samples,
        path=canonical_manifest_path,
        n_splits=outer_splits,
        seed=seed,
    )

    prediction_frames: list[pd.DataFrame] = []
    pig_metric_frames: list[pd.DataFrame] = []
    horizon_metric_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    inner_summary_frames: list[pd.DataFrame] = []

    for outer_assignment in outer_assignments:
        outer_training_samples = subset_by_pigs(
            samples, outer_assignment.train_pigs
        )
        outer_validation_samples = subset_by_pigs(
            samples, outer_assignment.validation_pigs
        )
        inner_assignments = make_inner_folds(
            outer_training_samples,
            outer_fold=outer_assignment.fold,
            n_splits=training_config.inner_splits,
            seed=seed,
        )

        selected_epoch, lr_schedule, selection_history, inner_summary = (
            select_training_epoch_across_inner_folds(
                model_factory=model_factory,
                outer_assignment=outer_assignment,
                outer_training_samples=outer_training_samples,
                inner_assignments=inner_assignments,
                config=training_config,
                seed=seed,
            )
        )
        model, standardizer, retraining_history = retrain_on_outer_training_set(
            model_factory=model_factory,
            outer_training_samples=outer_training_samples,
            selected_epoch=selected_epoch,
            learning_rate_schedule=lr_schedule,
            outer_fold=outer_assignment.fold,
            config=training_config,
            seed=seed,
        )

        x_outer_validation, _ = stack_deep_samples(outer_validation_samples)
        outer_predictions = predict(
            model,
            standardizer,
            x_outer_validation,
            batch_size=training_config.batch_size,
            seed=seed + outer_assignment.fold,
        )
        prediction_frame, pig_metrics, horizon_metrics = evaluate_fold_predictions(
            outer_validation_samples,
            outer_predictions,
            fold=outer_assignment.fold,
            set_name="outer_validation",
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler_mean": standardizer.mean_,
            "scaler_std": standardizer.std_,
            "feature_columns": feature_columns,
            "outer_fold": outer_assignment.fold,
            "outer_training_pigs": outer_assignment.train_pigs,
            "outer_validation_pigs": outer_assignment.validation_pigs,
            "inner_folds": [
                {
                    "fold": assignment.fold,
                    "training_pigs": assignment.train_pigs,
                    "validation_pigs": assignment.validation_pigs,
                }
                for assignment in inner_assignments
            ],
            "selected_epoch": selected_epoch,
            "inner_fold_best_epochs": inner_summary["best_epoch"].tolist(),
            "inner_fold_best_macro_RMSE": inner_summary[
                "best_inner_macro_RMSE"
            ].tolist(),
            "seed": seed,
        }
        torch.save(
            checkpoint,
            output_dir / f"{model_name}_outer_fold{outer_assignment.fold}.pt",
        )

        prediction_frames.append(prediction_frame)
        pig_metric_frames.append(pig_metrics)
        horizon_metric_frames.append(horizon_metrics)
        history_frames.extend([selection_history, retraining_history])
        inner_summary_frames.append(inner_summary)
        selected_rows.append(
            {
                "outer_fold": outer_assignment.fold,
                "selected_epoch": selected_epoch,
                "mean_best_inner_macro_RMSE": float(
                    inner_summary["best_inner_macro_RMSE"].mean()
                ),
                "std_best_inner_macro_RMSE": float(
                    inner_summary["best_inner_macro_RMSE"].std(ddof=1)
                ),
                "inner_seed": seed + outer_assignment.fold,
                **dict(model_configuration),
                **asdict(training_config),
            }
        )
        print(
            f"Outer fold {outer_assignment.fold:02d}: "
            f"selected epoch={selected_epoch}, "
            f"mean inner macro RMSE="
            f"{inner_summary['best_inner_macro_RMSE'].mean():.6f}, "
            f"outer macro RMSE={pig_metrics['RMSE'].mean():.6f}"
        )

    configuration = {
        "model": model_name,
        "seed": seed,
        "outer_folds": len(outer_assignments),
        "inner_selection_folds": training_config.inner_splits,
        "inner_selection_rule": (
            "run grouped inner five-fold early stopping independently and use "
            "the rounded median of the five best epochs"
        ),
        "outer_validation_usage": "final evaluation only",
        "retraining_rule": (
            "retrain from scratch on all outer-training pigs for the selected "
            "epoch count; replay the epoch-wise median inner learning-rate schedule"
        ),
        "window": 14,
        "horizon": HORIZON,
        "features": feature_columns,
        "training": asdict(training_config),
        "model_configuration": dict(model_configuration),
        "selection_metric": "pig-level macro-averaged RMSE",
    }
    output_excel = output_dir / f"{model_name}_W14_H7_NestedOuter10CV.xlsx"
    write_experiment_workbook(
        output_excel,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        pig_metrics=pd.concat(pig_metric_frames, ignore_index=True),
        horizon_metrics=pd.concat(horizon_metric_frames, ignore_index=True),
        assignments=outer_assignments,
        training_history=pd.concat(history_frames, ignore_index=True),
        tuning_results=pd.concat(inner_summary_frames, ignore_index=True),
        selected_parameters=pd.DataFrame(selected_rows),
        configuration=configuration,
    )
    print(f"Results written to {output_excel}")
