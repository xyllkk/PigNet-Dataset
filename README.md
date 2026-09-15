# PigNet

**Short-term individual pig body-weight forecasting using multi-source data
without continuous historical body-weight inputs**

PigNet forecasts the next seven daily body weights from 14 consecutive days of
feeding, age, and environmental observations plus one initial body-weight
anchor. The repository contains the released cohorts, model and baseline code,
source checkpoints, independent-cohort adaptation scripts, and compact
manuscript-aligned reference results.

## Repository structure

| Path | Contents |
|---|---|
| `code/` | PigNet, neural baselines, tree baselines, shared evaluation code, and verification utilities |
| `code/independent_cohort/` | Executable PigNet, TimesNet, and LSTM target-cohort adaptation pipelines |
| `code/analysis/` | QC sensitivity, SHAP, and production-analysis utilities |
| `data/` | Main and independent cohorts; see [`data/README.md`](data/README.md) |
| `weights/` | Source-domain checkpoints used for independent-cohort adaptation |
| `results/reference_metrics/` | Final compact benchmark, QC, and split records |
| `docs/reproducibility.md` | Full protocol, settings, seeds, and checkpoint provenance |

## Task definition

For each pig, the input is a calendar-continuous 14-day sequence. Predictors
include daily feeding behaviour, age, and environmental measurements; the first
non-missing weight in that pig's series is repeated as a static anchor. The
output is a direct seven-value forecast for days 1 through 7. All 21 dates
forming an input/target pair must be consecutive calendar days.

## Model

PigNet uses a one-layer LSTM (hidden size 128) followed by initial-weight FiLM
conditioning, squeeze-and-excitation channel recalibration, seven learned
horizon queries attending to shared encoded keys and values, residual fusion
with the last encoded state, and depthwise/pointwise horizon refinement. Its
training loss is Huber loss plus a first-order adjacent-horizon difference
penalty with coefficient 0.02.

## Evaluation design

Main-cohort comparisons use the same subject-level outer 10-fold assignment for
all models. Neural training duration is selected only inside each outer-training
partition using deterministic five-fold pig-level validation; tree-model search
is likewise restricted to outer-training pigs. Outer-validation pigs are used
once for final evaluation. Missing predictors and standardization parameters are
estimated from the applicable training partition only.

Per-pig RMSE is averaged across all seven forecast horizons. Pig-level R² is
averaged only across horizons for which R² is defined, and fold-level R² is then
macro-averaged across pigs with defined pig-level values. See
[`docs/reproducibility.md`](docs/reproducibility.md) for the complete procedure.

## Main results

| Model | RMSE (kg) | R² |
|---|---:|---:|
| PigNet | 2.28 | 0.958 |
| TimesNet | 2.62 | 0.944 |
| LSTM | 3.21 | 0.919 |

PigNet reduced RMSE by 13.0% relative to TimesNet (2.28 vs. 2.62 kg), and the
fold-level difference remained significant after Benjamini-Hochberg
multiple-comparison correction (BH-adjusted *p* = 0.037109; reported as 0.037
in the manuscript). Pairwise differences in fold-level overall RMSE between
PigNet and each baseline were assessed using two-sided exact Wilcoxon
signed-rank tests. Because seven PigNet-versus-baseline comparisons were
performed, *p*-values were adjusted using the Benjamini-Hochberg procedure to
control the false discovery rate; statistical significance was defined as a
BH-adjusted *p*-value < 0.05. Values are also recorded
in [`results/reference_metrics/main_results.csv`](results/reference_metrics/main_results.csv)
and [`results/reference_metrics/multiple_comparison_results.csv`](results/reference_metrics/multiple_comparison_results.csv).

## Independent-cohort evaluation

The independent cohort contains 33 retained Duroc pigs. For each of five fixed
pig-level splits, 3 pigs are used for full-parameter fine-tuning and the
remaining 30 pigs are used only for testing. The three models use identical
split assignments:

`497867277`, `584623948`, `774988242`, `817703552`, `874869484`

Source checkpoints were selected from source-cohort outer-fold models before
target-cohort evaluation: PigNet F4, TimesNet F6, and LSTM F7. The files in
`weights/` are those exact checkpoints. Adaptation uses Adam, learning rate
3e-4, weight decay 3e-4, batch size 128, and 120 epochs, with no target-domain
validation and no early stopping.

## Reproduction

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the main-cohort neural models from the repository root:

```bash
python code/PigNet.py
python code/TimesNet.py
python code/LSTM-10_fold_CV_win14.py
python code/LightTS.py
```

Run all five independent-cohort splits for each model:

```bash
python code/independent_cohort/pignet_independent_cohort.py
python code/independent_cohort/timesnet_independent_cohort.py
python code/independent_cohort/lstm_independent_cohort.py
```

Add `--verify-only` to an independent-cohort command to check checkpoint
compatibility and print all split assignments without training or writing
results. Use `--help` for path and device options.

## Reproducing manuscript analyses

Tree-model feature importance can be summarized without retraining when the
corresponding serialized outer-fold estimators and validation-window table are
supplied:

```bash
python code/analysis/shap_feature_importance.py --data VALIDATION.csv --model-dir TREE_MODELS --output-dir shap_results
```

It uses `shap.TreeExplainer`, pig-balanced validation subsampling (at most 20
windows per pig, seed 42), lag-family summation, horizon summation, and fold
mean ± SEM. Provide one MultiOutputRegressor-compatible joblib/pickle per model
and outer fold (for example, `RF_fold01.joblib`).

### Production-oriented evaluation

`code/analysis/production_evaluation.py` implements the rolling-forecast RMSE,
milestone-attainment, crossing-day, and trajectory-level growth-rate metrics
described in the manuscript. The script operates on out-of-fold prediction
files generated by the corresponding final model runs. It accepts both the
canonical `Window_predictions` table and model-runner `Predictions` output:

```bash
python code/analysis/production_evaluation.py --predictions PREDICTIONS.xlsx --model-name TimesNet --output-dir production_results
```

Window screening can be finalized from a saved inner-validation ranking table:

```bash
python code/analysis/window_selection.py --metrics WINDOW_RANKING.csv --output window_selection.csv
```

## QC sensitivity analysis

The supplementary QC analysis compares QC-REF, QC-STRICT, QC-RELAXED, and
QC-ROBUST. The largest mean-RMSE change from QC-REF was 0.131 kg (5.9%). The
reference summary is in
[`results/reference_metrics/qc_sensitivity.csv`](results/reference_metrics/qc_sensitivity.csv),
and `python code/analysis/qc_sensitivity.py --help` documents how to regenerate a
summary from four fold-level CSV files.

## Requirements

Python package versions used by the public scripts are pinned in
[`requirements.txt`](requirements.txt). CUDA is optional; PyTorch automatically
uses CPU when CUDA is unavailable.

## Citation

Citation metadata and the confirmed manuscript author list are provided in
[`CITATION.cff`](CITATION.cff).

## License

The source code in this repository is licensed under the MIT License. See
[`LICENSE`](LICENSE).

Unless otherwise stated, the datasets, model weights, documentation, and
research result files are licensed under the Creative Commons Attribution 4.0
International (CC BY 4.0) License. See [`LICENSE-DATA`](LICENSE-DATA).

If you use PigNet, the released datasets, or model weights in academic work,
please cite the associated paper and this repository. Citation metadata are
provided in [`CITATION.cff`](CITATION.cff).
