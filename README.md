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
| `code/analysis/` | Supplementary QC sensitivity summarization |
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

Per-pig RMSE and R² are calculated separately at each forecast horizon, averaged
across the seven horizons within pig, and then macro-averaged across pigs. See
[`docs/reproducibility.md`](docs/reproducibility.md) for the complete procedure.

## Main results

| Model | RMSE (kg) | R² |
|---|---:|---:|
| PigNet | 2.28 | 0.958 |
| TimesNet | 2.62 | 0.944 |
| LSTM | 3.21 | 0.919 |

PigNet reduced RMSE by approximately 13.0% relative to TimesNet. Exact two-sided
Wilcoxon signed-rank tests were applied to paired outer-fold RMSE values. The exact
two-sided Wilcoxon signed-rank comparison gave *p* = 0.03711 (reported as
0.037); no multiple-comparison correction was applied. Values are also recorded
in [`results/reference_metrics/main_results.csv`](results/reference_metrics/main_results.csv).

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

No software or data license has yet been specified for this repository. No
license was inferred or added during repository preparation.
