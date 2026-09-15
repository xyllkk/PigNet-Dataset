# Reproducibility protocol

This document records the settings that correspond to the final PigNet
manuscript and supplementary material. Paths are repository-relative and all
commands are run from the repository root.

## Forecasting task

Each example belongs to one pig and spans 21 consecutive calendar days:

1. days 1–14 provide feeding, age, and environmental predictors;
2. the first non-missing weight in the pig's chronologically sorted record is
   repeated as a static initial body-weight anchor;
3. days 15–21 provide seven direct body-weight targets.

No time-varying body-weight measurement from the input interval is used. A
candidate is discarded if any adjacent pair of dates in the 21-day interval is
not separated by exactly one calendar day or if a target is missing.

## Preprocessing

The original data schema is preserved; `日期`, `耳缺号`, and `体重` are the
required date, pig-identifier, and body-weight fields. Numeric feeding, age, and
environmental columns are predictors. Duration fields are converted to seconds.
Records are de-duplicated by pig/date and ordered chronologically within pig.

For main-cohort cross-validation, predictor means and standard deviations are
estimated from the current training partition. Missing predictor values are
replaced with those training means before z-score standardization. Validation
and test records never contribute preprocessing statistics.

For independent-cohort adaptation, the source-checkpoint feature order, means,
and standard deviations are retained for both fine-tuning and test records.

## Main-cohort nested cross-validation

- grouping unit: pig;
- outer evaluation: 10 folds generated with seed 42;
- one canonical outer assignment is shared by every model;
- neural inner selection: five pig-level folds drawn only from the current
  outer-training pigs, with inner seed `42 + outer_fold`;
- tree-model search: performed independently inside each outer-training
  partition;
- outer-validation pigs: used only for the final fold evaluation.

Neural inner runs use pig-level macro RMSE for learning-rate scheduling and
early stopping. The rounded median of the five inner best epochs determines the
fixed training duration used to refit on all outer-training pigs. The outer
validation partition does not select epochs or hyperparameters.

### Neural defaults

| Model | Maximum epochs | Batch size | Learning rate | Weight decay | Smoothing coefficient |
|---|---:|---:|---:|---:|---:|
| PigNet | 400 | 128 | 1e-3 | 3e-4 | 0.02 |
| TimesNet | 400 | 128 | 3e-4 | 3e-4 | 0.02 |
| LSTM | 400 | 128 | 1e-3 | 3e-4 | 0 |
| LightTS | 400 | 128 | 1e-3 | 3e-4 | 0 |

PigNet uses a one-layer LSTM with hidden size 128 and dropout 0.25. Initial
weight drives FiLM conditioning. Squeeze-and-excitation recalibrates encoded
channels. Seven learned horizon queries attend to shared encoded keys and
values, after which the last encoded state is added. A depthwise/pointwise
one-dimensional convolution refines neighbouring horizon representations and
uses a residual connection.

The PigNet smoothness term is the mean absolute first difference between
adjacent predictions: `mean(abs(y_hat[:, 1:] - y_hat[:, :-1]))`. It is not a
higher-order difference penalty.

## Metrics and statistical comparison

For each pig, RMSE and R² are computed separately for forecast horizons 1–7.
Per-pig RMSE is averaged across all seven forecast horizons. Pig-level R² is
averaged only across horizons for which R² is defined, and fold-level R² is then
macro-averaged across pigs with defined pig-level values. Fold-level values
support stability and paired comparisons.

Final main-cohort reference values are:

| Model | RMSE (kg) | R² |
|---|---:|---:|
| PigNet | 2.28 | 0.958 |
| TimesNet | 2.62 | 0.944 |
| LSTM | 3.21 | 0.919 |

PigNet's RMSE reduction relative to TimesNet is approximately 13.0%. The exact
two-sided Wilcoxon signed-rank result is *p* = 0.03711 (displayed as 0.037).
No multiple-comparison correction was applied.

## Manuscript analysis reproduction

### SHAP feature importance

The historical tree-model analysis uses `shap.TreeExplainer` for RF, XGBoost,
LightGBM, and CatBoost. Each outer fold explains seven independently fitted
horizon estimators on validation windows sampled in a pig-balanced manner (at
most 20 windows per pig, sampling seed 42). For each lagged feature, the
per-window absolute SHAP values are averaged; suffixes `_lag1` through `_lag14`
are then mapped to the original variable family and summed. Initial body weight
is a single static feature and is not lag-expanded. Family values are summed
across horizons within each fold, and Fig. 11 reports the mean and SEM across
the ten outer-fold totals. The public entry point is
`code/analysis/shap_feature_importance.py`; it requires the serialized
outer-fold tree estimators and validation-window table, which are not bundled.
Because neither the historical SHAP aggregation workbook nor those serialized
tree estimators is public, this repository does not claim a numerical reprint
of Fig. 11.
The exact historical SHAP package version was not retained; the public script
uses the compatible version pinned in `requirements.txt`.

### Production-oriented evaluation

`code/analysis/production_evaluation.py` consumes saved out-of-fold rolling
predictions and computes the operational metrics without fitting models. For
each outer fold, rolling pooled RMSE pools all retained test-pig window-horizon
points and is then averaged arithmetically across folds. Milestone events are
defined as any observed or predicted value reaching 90 or 100 kg within the
seven-day horizon; crossing-day MAE is calculated only for paired-positive
windows. Growth-rate metrics fit ordinary least-squares body-weight slopes
against relative day (`Window + Horizon`) for each pig, then report fold-level
MAE/RMSE and pooled Spearman correlation across pig trajectories.

The historical window-screening workflow evaluated W=7--21 using validation
data from outer-training pigs and selected W=14. The public
`code/analysis/window_selection.py` utility finalizes that choice from a saved
inner-validation ranking table; it does not retrain the candidate windows.
No recoverable historical implementation was found for the input-availability
analysis, so no public entry point is advertised for that analysis.

## Independent-cohort protocol

The target cohort contains 33 retained Duroc pigs. Each repeated split uses 3 of 33
pigs (9.1%) for fine-tuning and 30 of 33 pigs (90.9%) for testing. The historical
splitter included breed-stratification logic. Because all 33 pigs in the target
cohort were Duroc, this was equivalent to pig-level random sampling in the
independent-cohort evaluation. The test pigs do not provide validation signals
and do not influence training duration.

All model parameters are updated for 120 epochs using Adam, learning rate 3e-4,
weight decay 3e-4, and batch size 128. There is no target-domain validation and
no early stopping. PigNet and TimesNet retain their first-difference penalty;
LSTM uses only Huber loss.

### Source checkpoints

For each model, the designated source-cohort checkpoint was selected before
target evaluation from its ten outer-fold source models according to the
source-cohort validation criterion.

| Model | Source fold | Repository file | SHA-256 |
|---|---|---|---|
| PigNet | CV10_F4 | `weights/PigNet.pt` | `f3545ac9a12b1f8b26d0ca15de9a5b1dabcd65502899766a2cf9e768c65710b3` |
| TimesNet | CV10_F6 | `weights/TimesNet.pt` | `5d01ba03b7b2fc9733908f2390e0e788c5034753a12099976e86261d3ba274c9` |
| LSTM | CV10_F7 | `weights/LSTM.pt` | `cfe074560d1d35aeb1d835f26793d3df4639338255798d4fb89171956214101d` |

### Repeated split assignments

The same five seeds and pig assignments are used for PigNet, TimesNet, and
LSTM. Full test-pig lists are stored in
`results/reference_metrics/independent_cohort_splits.csv`.

| Seed | Fine-tuning pigs |
|---:|---|
| 497867277 | 202701, 203103, 203203 |
| 584623948 | 202705, 202711, 203803 |
| 774988242 | 202705, 202805, 204305 |
| 817703552 | 202705, 203203, 204107 |
| 874869484 | 202705, 202709, 203307 |

The executable entry points are:

```bash
python code/independent_cohort/pignet_independent_cohort.py
python code/independent_cohort/timesnet_independent_cohort.py
python code/independent_cohort/lstm_independent_cohort.py
```

Each command runs the five seeds by default. `--verify-only` checks the
checkpoint architecture and prints the assignments without training or writing
files. Generated workbooks record predictions, pig-level metrics,
horizon-level metrics, split membership, and the effective configuration.

## Supplementary QC sensitivity analysis

The four PigNet QC variants produced the following 10-fold mean ± standard
deviation RMSE values:

| Variant | RMSE (kg) |
|---|---:|
| QC-REF | 2.227598 ± 0.309600 |
| QC-STRICT | 2.358958 ± 0.322775 |
| QC-RELAXED | 2.340590 ± 0.242429 |
| QC-ROBUST | 2.343769 ± 0.293516 |

The maximum mean-RMSE change from QC-REF is 0.131 kg, or 5.9%. The public
summary utility accepts four fold-level CSV files with `RMSE` and `R2` columns:

```bash
python code/analysis/qc_sensitivity.py --ref REF.csv --strict STRICT.csv --relaxed RELAXED.csv --robust ROBUST.csv --output results/qc_sensitivity_summary.csv
```

The compact manuscript-aligned reference table is stored in
`results/reference_metrics/qc_sensitivity.csv`.
