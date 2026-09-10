"""Shared independent-cohort adaptation used by the three published entry points.

This module is a public, path-independent consolidation of the original PigNet,
TimesNet, and LSTM target-cohort scripts. It preserves their model state-dict
names, source-scaler preprocessing, pig-level split logic, full-parameter Adam
fine-tuning, and per-pig/per-horizon evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CODE_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from pignet_common import (  # noqa: E402
    HORIZON,
    WINDOW,
    WindowSample,
    evaluate_fold_predictions,
    prepare_experiment,
    stack_deep_samples,
)

TARGET_SPLIT_SEEDS = (497867277, 584623948, 774988242, 817703552, 874869484)
FINE_TUNE_RATIO = 0.1
FINE_TUNE_EPOCHS = 120
FINE_TUNE_BATCH_SIZE = 128
FINE_TUNE_LEARNING_RATE = 3e-4
FINE_TUNE_WEIGHT_DECAY = 3e-4
EXPECTED_TARGET_PIGS = 33
EXPECTED_FINE_TUNE_PIGS = 3
EXPECTED_TEST_PIGS = 30


class HorizonConvRefiner(nn.Module):
    """Depthwise/pointwise refinement along the seven-step horizon."""

    def __init__(self, channels: int, kernel: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=channels,
            bias=True,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        refined = self.pw(self.dw(context.transpose(1, 2))).transpose(1, 2)
        return context + self.drop(refined)


class LSTM2MultiHPro(nn.Module):
    """Checkpoint-compatible PigNet/LSTM architecture from the original runs."""

    def __init__(
        self,
        input_dim: int,
        output_horizon: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_film: bool,
        use_horizon_attention: bool,
        fuse_last: bool,
        use_horizon_convolution: bool,
    ) -> None:
        super().__init__()
        self.out_h = output_horizon
        self.hidden_size = hidden_size
        self.use_hq_attn = use_horizon_attention
        self.fuse_last = fuse_last

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.film_mlp = (
            nn.Sequential(
                nn.Linear(1, 64),
                nn.GELU(),
                nn.Linear(64, hidden_size * 2),
            )
            if use_film
            else None
        )
        self.se_fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.se_fc2 = nn.Linear(hidden_size // 4, hidden_size)
        if use_horizon_attention:
            self.h_queries = nn.Parameter(torch.randn(output_horizon, hidden_size))
            self.mha = nn.MultiheadAttention(
                hidden_size,
                num_heads=2,
                dropout=0.10,
                batch_first=True,
            )
        self.ln_ctx = nn.LayerNorm(hidden_size)
        self.hconv = (
            HorizonConvRefiner(hidden_size, kernel=3, dropout=0.15)
            if use_horizon_convolution
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        encoded, _ = self.lstm(x)
        encoded = self.dropout(encoded)
        if self.film_mlp is not None:
            initial_weight = x[:, 0, -1].unsqueeze(-1)
            gamma, beta = self.film_mlp(initial_weight).split(
                self.hidden_size, dim=-1
            )
            gamma = 1.0 + 0.2 * torch.tanh(gamma)
            beta = 0.2 * torch.tanh(beta)
            encoded = gamma.unsqueeze(1) * encoded + beta.unsqueeze(1)

        channel_weights = encoded.mean(dim=1)
        channel_weights = F.gelu(self.se_fc1(channel_weights))
        channel_weights = torch.sigmoid(self.se_fc2(channel_weights))
        encoded = encoded * channel_weights.unsqueeze(1)

        if self.use_hq_attn:
            queries = self.h_queries.unsqueeze(0).expand(
                batch_size, self.out_h, self.hidden_size
            )
            context, _ = self.mha(queries, encoded, encoded)
            if self.fuse_last:
                context = context + encoded[:, -1, :].unsqueeze(1)
        else:
            context = encoded.mean(dim=1, keepdim=True).repeat(1, self.out_h, 1)
            if self.fuse_last:
                context = context + encoded[:, -1, :].unsqueeze(1)

        if self.hconv is not None:
            context = self.hconv(context)
        context = self.ln_ctx(context)
        output = self.head(context.reshape(batch_size * self.out_h, self.hidden_size))
        return output.view(batch_size, self.out_h)


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
    """Checkpoint-compatible TimesNet frequency block."""

    def __init__(
        self,
        d_model: int,
        top_k: int = 3,
        number_of_kernels: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.top_k = max(1, top_k)
        kernel_sizes = [3, 5, 7, 11][: max(1, number_of_kernels)]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    d_model,
                    d_model,
                    kernel_size=(1, kernel),
                    padding=(0, kernel // 2),
                )
                for kernel in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        residual = x
        channel_first = x.permute(0, 2, 1)
        amplitude = torch.fft.rfft(channel_first, dim=-1).abs().mean(dim=(0, 1))
        if amplitude.shape[0] <= 1:
            return self.ln(residual)
        amplitude = amplitude.clone()
        amplitude[0] = 0
        count = min(self.top_k, amplitude.shape[0] - 1)
        values, indices = torch.topk(amplitude, k=count)
        weights = torch.softmax(values, dim=0)

        period_outputs = []
        for frequency_index in indices.tolist():
            period = max(1, int(sequence_length // max(1, frequency_index)))
            padding = (period - sequence_length % period) % period
            padded = F.pad(channel_first, (0, padding)) if padding else channel_first
            rows = padded.shape[-1] // period
            two_dimensional = padded.view(batch_size, channels, rows, period)
            convolved = sum(F.gelu(layer(two_dimensional)) for layer in self.convs)
            convolved = self.dropout(convolved / len(self.convs))
            period_outputs.append(
                convolved.reshape(batch_size, channels, -1)[:, :, :sequence_length]
            )

        combined = torch.zeros_like(channel_first)
        for weight, output in zip(weights, period_outputs):
            combined = combined + weight * output
        return self.ln(combined.permute(0, 2, 1) + residual)


class TimesNetForecaster(nn.Module):
    """Checkpoint-compatible TimesNet model from the source-cohort runs."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        prediction_length: int,
        d_model: int,
        encoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = sequence_length
        self.pred_len = prediction_length
        self.d_model = d_model
        self.value_emb = nn.Linear(input_dim, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.predict_linear = nn.Linear(sequence_length, sequence_length + prediction_length)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=d_model,
                    top_k=3,
                    number_of_kernels=3,
                    dropout=dropout,
                )
                for _ in range(max(1, encoder_layers))
            ]
        )
        self.proj = nn.Linear(d_model, 1)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, received {x.shape[1]}."
            )
        representation = self.emb_drop(self.value_emb(x))
        representation = self.predict_linear(
            representation.permute(0, 2, 1)
        ).permute(0, 2, 1)
        representation = representation + sinusoidal_position_encoding(
            representation.shape[1],
            self.d_model,
            representation.device,
            representation.dtype,
        )
        for block in self.blocks:
            representation = block(representation)
        output = self.proj(self.final_ln(representation)).squeeze(-1)
        return output[:, -self.pred_len :]


def build_model(
    model_name: str,
    input_dim: int,
    output_horizon: int,
    parameters: dict,
) -> nn.Module:
    hidden_size = int(parameters.get("hidden_size", 128))
    num_layers = int(parameters.get("num_layers", 1))
    dropout = float(parameters.get("dropout", 0.25))
    if model_name == "timesnet":
        return TimesNetForecaster(
            input_dim=input_dim,
            sequence_length=WINDOW,
            prediction_length=output_horizon,
            d_model=hidden_size,
            encoder_layers=num_layers,
            dropout=dropout,
        )
    return LSTM2MultiHPro(
        input_dim=input_dim,
        output_horizon=output_horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_film=model_name == "pignet",
        use_horizon_attention=model_name == "pignet",
        fuse_last=model_name == "pignet",
        use_horizon_convolution=model_name == "pignet",
    )


def split_target_pigs(
    samples: Sequence[WindowSample], seed: int
) -> tuple[set[str], set[str]]:
    """Reproduce the original breed-stratified 10% pig-level split."""
    generator = np.random.default_rng(seed)
    all_pigs = sorted({sample.pig_id for sample in samples})
    train_count = max(1, min(round(len(all_pigs) * FINE_TUNE_RATIO), len(all_pigs) - 1))
    breed_to_pigs: dict[str, list[str]] = {}
    for sample in samples:
        breed_to_pigs.setdefault(sample.breed, [])
        if sample.pig_id not in breed_to_pigs[sample.breed]:
            breed_to_pigs[sample.breed].append(sample.pig_id)
    for pigs in breed_to_pigs.values():
        pigs.sort()

    total = sum(len(pigs) for pigs in breed_to_pigs.values())
    allocation = {
        breed: round(len(pigs) / total * train_count)
        for breed, pigs in breed_to_pigs.items()
    }
    difference = train_count - sum(allocation.values())
    breeds = list(allocation)
    index = 0
    while difference:
        breed = breeds[index % len(breeds)]
        if difference > 0 and allocation[breed] < len(breed_to_pigs[breed]):
            allocation[breed] += 1
            difference -= 1
        elif difference < 0 and allocation[breed] > 0:
            allocation[breed] -= 1
            difference += 1
        index += 1

    fine_tune_pigs: set[str] = set()
    for breed, count in allocation.items():
        pigs = breed_to_pigs[breed].copy()
        generator.shuffle(pigs)
        fine_tune_pigs.update(pigs[:count])
    if len(fine_tune_pigs) < train_count:
        remaining = [pig for pig in all_pigs if pig not in fine_tune_pigs]
        generator.shuffle(remaining)
        fine_tune_pigs.update(remaining[: train_count - len(fine_tune_pigs)])
    test_pigs = set(all_pigs) - fine_tune_pigs
    return fine_tune_pigs, test_pigs


def _normalize_with_source_scaler(
    values: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray
) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    missing = np.isnan(output)
    if missing.any():
        feature_indices = np.where(missing)[2]
        output[missing] = np.take(mean, feature_indices)
    return (output - mean) / standard_deviation


def _load_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    expected_fold: str,
    input_dim: int,
    output_horizon: int,
    device: torch.device,
) -> tuple[dict, nn.Module, np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    fold_tag = str(checkpoint.get("fold_tag", ""))
    if fold_tag != expected_fold:
        raise ValueError(
            f"Expected source checkpoint {expected_fold}, but {checkpoint_path} "
            f"records {fold_tag or 'no fold tag'}."
        )
    parameters = dict(checkpoint["best_params"])
    model = build_model(model_name, input_dim, output_horizon, parameters).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    mean = checkpoint["scaler_mean"].detach().cpu().numpy().astype(float)
    standard_deviation = checkpoint["scaler_std"].detach().cpu().numpy().astype(float)
    standard_deviation = np.where(standard_deviation < 1e-12, 1.0, standard_deviation)
    return checkpoint, model, mean, standard_deviation


def _fine_tune(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    seed: int,
    device: torch.device,
) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=FINE_TUNE_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINE_TUNE_LEARNING_RATE,
        weight_decay=FINE_TUNE_WEIGHT_DECAY,
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    use_smoothing = model_name in {"pignet", "timesnet"}

    for _ in range(FINE_TUNE_EPOCHS):
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            if use_smoothing:
                first_differences = predictions[:, 1:] - predictions[:, :-1]
                loss = loss + 0.02 * torch.mean(torch.abs(first_differences))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def _predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            inputs = torch.tensor(x[start : start + 1024], dtype=torch.float32, device=device)
            batches.append(model(inputs).cpu().numpy())
    return np.vstack(batches)


def _write_split_workbook(
    output_path: Path,
    predictions: pd.DataFrame,
    pig_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    split_record: dict,
    configuration: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        pig_metrics.to_excel(writer, sheet_name="PigMetrics", index=False)
        horizon_metrics.to_excel(writer, sheet_name="HorizonMetrics", index=False)
        pd.DataFrame([split_record]).to_excel(writer, sheet_name="SplitInfo", index=False)
        pd.DataFrame(
            [
                {"setting": key, "value": json.dumps(value, ensure_ascii=False)}
                for key, value in configuration.items()
            ]
        ).to_excel(writer, sheet_name="Configuration", index=False)


def run(
    model_name: str,
    display_name: str,
    source_fold: str,
    data_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
    device_name: str | None,
    verify_only: bool,
) -> None:
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _, feature_columns, samples = prepare_experiment(data_path)
    all_pigs = {sample.pig_id for sample in samples}
    if len(all_pigs) != EXPECTED_TARGET_PIGS:
        raise ValueError(
            f"Published protocol requires {EXPECTED_TARGET_PIGS} target pigs; "
            f"found {len(all_pigs)}."
        )

    checkpoint, _, _, _ = _load_checkpoint(
        checkpoint_path,
        model_name,
        source_fold,
        input_dim=len(feature_columns) + 1,
        output_horizon=HORIZON,
        device=device,
    )
    checkpoint_features = list(checkpoint.get("feat_cols", []))
    if checkpoint_features and checkpoint_features != feature_columns:
        raise ValueError(
            "Target feature order does not match the source checkpoint: "
            f"target={feature_columns}, source={checkpoint_features}."
        )

    split_rows = []
    for split_index, seed in enumerate(seeds, start=1):
        fine_tune_pigs, test_pigs = split_target_pigs(samples, int(seed))
        if (
            len(fine_tune_pigs) != EXPECTED_FINE_TUNE_PIGS
            or len(test_pigs) != EXPECTED_TEST_PIGS
        ):
            raise ValueError(
                "Published split must contain 3 fine-tuning pigs and 30 test pigs; "
                f"found {len(fine_tune_pigs)} and {len(test_pigs)}."
            )
        split_record = {
            "split": split_index,
            "seed": int(seed),
            "n_pigs_total": len(all_pigs),
            "n_fine_tune_pigs": len(fine_tune_pigs),
            "n_test_pigs": len(test_pigs),
            "fine_tune_pigs": ",".join(sorted(fine_tune_pigs)),
            "test_pigs": ",".join(sorted(test_pigs)),
        }
        print(
            f"{display_name} seed={seed}: fine_tune={split_record['fine_tune_pigs']} "
            f"test_count={len(test_pigs)}"
        )
        if verify_only:
            continue

        fine_tune_samples = [sample for sample in samples if sample.pig_id in fine_tune_pigs]
        test_samples = [sample for sample in samples if sample.pig_id in test_pigs]
        x_fine_tune, y_fine_tune = stack_deep_samples(fine_tune_samples)
        x_test, _ = stack_deep_samples(test_samples)
        checkpoint, model, mean, standard_deviation = _load_checkpoint(
            checkpoint_path,
            model_name,
            source_fold,
            input_dim=x_fine_tune.shape[-1],
            output_horizon=y_fine_tune.shape[-1],
            device=device,
        )
        x_fine_tune = _normalize_with_source_scaler(
            x_fine_tune, mean, standard_deviation
        )
        x_test = _normalize_with_source_scaler(x_test, mean, standard_deviation)
        _fine_tune(model, x_fine_tune, y_fine_tune, model_name, int(seed), device)

        fine_tune_predictions = _predict(model, x_fine_tune, device)
        test_predictions = _predict(model, x_test, device)
        ft_frames = evaluate_fold_predictions(
            fine_tune_samples,
            fine_tune_predictions,
            fold=split_index,
            set_name="fine_tune",
        )
        test_frames = evaluate_fold_predictions(
            test_samples,
            test_predictions,
            fold=split_index,
            set_name="test",
        )
        predictions = pd.concat([ft_frames[0], test_frames[0]], ignore_index=True)
        pig_metrics = pd.concat([ft_frames[1], test_frames[1]], ignore_index=True)
        horizon_metrics = pd.concat([ft_frames[2], test_frames[2]], ignore_index=True)
        test_pig_metrics = test_frames[1]
        split_record["test_macro_RMSE"] = float(test_pig_metrics["RMSE"].mean())
        split_record["test_macro_R2"] = float(test_pig_metrics["R2"].mean())
        split_rows.append(split_record)

        _write_split_workbook(
            output_dir / f"{model_name}_independent_seed_{seed}.xlsx",
            predictions,
            pig_metrics,
            horizon_metrics,
            split_record,
            {
                "model": display_name,
                "source_checkpoint": checkpoint_path.name,
                "source_fold": source_fold,
                "window": WINDOW,
                "horizon": HORIZON,
                "fine_tune_ratio": FINE_TUNE_RATIO,
                "fine_tune_epochs": FINE_TUNE_EPOCHS,
                "fine_tune_batch_size": FINE_TUNE_BATCH_SIZE,
                "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE,
                "fine_tune_weight_decay": FINE_TUNE_WEIGHT_DECAY,
                "target_validation": False,
                "early_stopping": False,
                "feature_columns": feature_columns,
            },
        )

    if not verify_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(split_rows).to_csv(
            output_dir / f"{model_name}_independent_summary.csv", index=False
        )


def main(model_name: str, display_name: str, source_fold: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Run the published {display_name} independent-cohort protocol."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "second_batch",
        help="Target-cohort workbook or directory of workbooks/CSV files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "weights" / f"{display_name}.pt",
        help=f"Source-domain {source_fold} checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "independent_cohort" / model_name,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(TARGET_SPLIT_SEEDS),
        help="Pig-level target-cohort split seeds.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Validate checkpoint compatibility and print splits without training "
            "or writing files."
        ),
    )
    arguments = parser.parse_args()
    run(
        model_name=model_name,
        display_name=display_name,
        source_fold=source_fold,
        data_path=arguments.data_path,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        seeds=arguments.seeds,
        device_name=arguments.device,
        verify_only=arguments.verify_only,
    )
