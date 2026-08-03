PigNet-Dataset

This repository provides the dataset, source code, shared evaluation utilities, and representative trained model weights for the manuscript:

PigNet: Multi-step forecasting of pig body weight from multi-source data using an LSTM with time-aware attention and depthwise separable convolutional smoothing

Overview

PigNet-Dataset supports reproducible research on short-term individual pig body-weight forecasting from daily multi-source records. The main task uses a 14-day look-back window to predict body weight over the following 7 days.

The repository includes:

main-cohort data used for model development and evaluation;

second-batch data used for independent-cohort analysis;

source code for PigNet and all comparison models;

shared code for preprocessing, canonical fold assignment, nested cross-validation, evaluation, and result export;

representative trained weights for selected neural models.

The revised main-cohort evaluation follows a strict pig-level nested cross-validation protocol:

all models reuse the same canonical outer 10-fold pig assignment;

neural models select the training duration and learning-rate schedule using inner 5-fold pig-level validation;

tree-based models perform exhaustive hyperparameter search independently within each outer-training partition;

the outer-validation pigs are used only for final evaluation.

Repository structure

PigNet-Dataset/
├── code/
│   ├── PigNet.py
│   ├── TimesNet.py
│   ├── LSTM-10_fold_CV_win14.py
│   ├── LightTS.py
│   ├── XGBoost-10_fold_CV_win14.py
│   ├── Light-10_fold_CV_win14.py
│   ├── RF-10_fold_CV_win14.py
│   ├── Cat-10_fold_CV_win14.py
│   ├── simple_baselines.py
│   ├── pignet_common.py
│   ├── deep_model_runner.py
│   ├── tree_model_runner.py
│   └── verify_revised_code.py
├── data/
│   ├── main_cohort/
│   └── second_batch/
├── weights/
├── results/                         # generated after running the scripts
├── column_mapping.md
├── requirements.txt
└── README.md

Data

1. Main cohort

The data/main_cohort/ directory contains the main experimental Excel files used for model development and evaluation.

The revised scripts use this directory for:

model training and comparison;

pig-level outer 10-fold evaluation;

inner model selection and hyperparameter tuning;

fold-to-fold variability analysis;

prediction, pig-level, and forecast-horizon-level result export.

2. Second batch

The data/second_batch/ directory contains independently collected second-batch records used for the independent-cohort analysis described in the manuscript.

The current scripts in code/ are centered on the main-cohort nested cross-validation benchmark. The second-batch data are retained to support the corresponding independent-cohort experiment and future transfer-evaluation scripts.

3. Data format

The raw Excel files retain the original Chinese column names used during data collection and analysis. The code reads these original fields directly.

An English reference for the column names is provided in column_mapping.md.

The required core columns include:

日期: date;

耳缺号: pig identifier;

体重: body weight.

Additional numerical fields may include feeding amount, feeding frequency, feeding duration, age, temperature, relative humidity, ammonia, carbon dioxide, and other available environmental measurements.

Code

Model scripts

File

Description

PigNet.py

Proposed PigNet model with an LSTM encoder, FiLM conditioning, channel recalibration, horizon-query attention, and depthwise-separable horizon smoothing.

TimesNet.py

TimesNet forecasting baseline.

LSTM-10_fold_CV_win14.py

Standard LSTM forecasting baseline.

LightTS.py

LightTS/IEBlock forecasting baseline.

XGBoost-10_fold_CV_win14.py

XGBoost baseline with exhaustive inner hyperparameter search.

Light-10_fold_CV_win14.py

LightGBM baseline with exhaustive inner hyperparameter search.

RF-10_fold_CV_win14.py

Random Forest baseline with exhaustive inner hyperparameter search.

Cat-10_fold_CV_win14.py

CatBoost baseline with exhaustive inner hyperparameter search.

simple_baselines.py

Linear and quadratic limited-input growth baselines.

Shared protocol and verification scripts

File

Description

pignet_common.py

Shared data loading, preprocessing, sliding-window construction, canonical pig-level fold assignment, scaling, evaluation, and Excel reporting utilities.

deep_model_runner.py

Leakage-free nested cross-validation runner for PigNet, TimesNet, LSTM, and LightTS.

tree_model_runner.py

Leakage-free nested cross-validation and exhaustive hyperparameter-search runner for the four tree-based models.

verify_revised_code.py

Static and synthetic checks for syntax, shared fold usage, grid sizes, nested partitioning, neural output shapes, and simplified nested-CV execution.

Experimental protocol

Forecasting task

Data granularity: daily;

look-back window: 14 days;

forecast horizon: 7 days;

prediction type: direct multi-step forecasting;

grouping unit: individual pig identified by ear tag.

Canonical outer data split

All models use the same outer fold manifest:

results/canonical_outer_folds_W14_H7_seed42.json

The manifest is generated from the sorted set of eligible pig identifiers using:

KFold(n_splits=10, shuffle=True, random_state=42)

The first executed model creates the manifest. All subsequent models load and validate the same file.

The code verifies that:

training and validation pigs do not overlap within a fold;

each fold covers the complete eligible pig set;

every pig appears in the outer-validation set exactly once;

all windows belonging to one pig remain in the same partition.

Do not edit or delete the canonical fold manifest between model runs. If the dataset or window-generation logic is intentionally changed, delete the old manifest and regenerate it for the new eligible pig set.

Neural-model selection

PigNet, TimesNet, LSTM, and LightTS use strict nested pig-level validation.

For each outer fold:

only the outer-training pigs enter inner model selection;

deterministic inner 5-fold pig-level validation is created;

each inner fold trains for at most 400 epochs;

pig-level macro-averaged RMSE controls learning-rate reduction and early stopping;

the rounded median of the five best epochs is selected;

the model is reinitialized and retrained from scratch on all outer-training pigs;

the outer-validation pigs are evaluated only after final training.

The common training settings are:

optimizer: Adam;

initial learning rate: 1e-3;

weight decay: 3e-4;

batch size: 128;

maximum epochs: 400;

loss: SmoothL1 with beta=1.0;

gradient clipping norm: 1.0;

learning-rate scheduler: ReduceLROnPlateau;

scheduler factor: 0.5;

scheduler patience: 8;

early-stopping patience: 25;

minimum improvement: 1e-6;

inner folds: 5.

PigNet and TimesNet additionally use a forecast-smoothness loss weight of 0.02. LSTM and LightTS use no additional smoothness penalty.

Tree-model selection

XGBoost, LightGBM, Random Forest, and CatBoost use strict outer 10-fold and inner 5-fold nested cross-validation.

For each outer fold:

exhaustive Cartesian-grid search is performed using only the outer-training pigs;

candidate configurations are ranked by mean inner-fold pig-level macro RMSE;

the selected configuration is refitted on all outer-training pigs;

the outer-validation pigs are used once for final evaluation.

The numbers of candidate configurations per outer fold are:

Model

Candidates

XGBoost

288

LightGBM

864

Random Forest

72

CatBoost

288

Each tree model trains one independent regressor for each of the seven forecast horizons. Missing values are imputed using means estimated only from the relevant training partition. Tree inputs are not standardized.

Preprocessing

The shared preprocessing code:

parses dates and pig identifiers;

converts body weight and age to numerical values;

removes duplicate pig-date records;

sorts records by pig and date;

converts feeding-duration fields to seconds when present;

derives breed and station labels when explicit fields are unavailable;

selects numerical predictors while excluding identifiers and the response;

constructs 14-day input and 7-day output windows independently for each pig.

For neural models, missing values are filled using training-partition feature means, followed by feature-wise z-score standardization using training-partition statistics. The same fitted statistics are then applied to validation data.

Evaluation

The primary model-selection statistic is pig-level macro-averaged RMSE, which gives equal weight to each pig rather than to each sliding window.

The exported results include:

window-level observations and predictions;

pig-level RMSE and R²;

forecast-horizon-level RMSE and R²;

outer fold assignments;

training histories or tuning results;

selected epochs or hyperparameters;

model and protocol configuration.

Reference software environment

The public reproducibility environment uses Python 3.11 and core stable package releases available no later than 30 September 2025:

numpy==2.3.3
pandas==2.3.3
scikit-learn==1.7.2
torch==2.8.0
openpyxl==3.1.5
XlsxWriter==3.2.9
xgboost==3.0.5
lightgbm==4.6.0
catboost==1.2.8

The corresponding requirements.txt file should contain the same pinned versions.

For a standard environment:

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

Activate the environment on Linux or macOS:

source .venv/bin/activate

Install the dependencies:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

For CUDA-specific PyTorch wheels, install the appropriate PyTorch 2.8.0 build for the local CUDA platform using the official PyTorch installation instructions, and then install the remaining packages from requirements.txt.

Running the code

All commands below are executed from the repository root.

1. Verify the revised protocol

python code/verify_revised_code.py

A successful check ends with:

PASS: all nested cross-validation protocol checks completed successfully.

This verification script performs static and small synthetic checks. It does not replace full training on the released dataset.

2. Generate the canonical fold manifest

The manifest is created automatically by the first model that is run. A convenient first command is the relatively fast simple-baseline script:

python code/simple_baselines.py

This creates:

results/canonical_outer_folds_W14_H7_seed42.json

The same file is then reused by every other model.

3. Run the neural models

python code/PigNet.py
python code/TimesNet.py
python code/LSTM-10_fold_CV_win14.py
python code/LightTS.py

Each neural script performs inner model selection, retrains a new model for each outer fold, saves fold-specific checkpoints, and exports an Excel workbook.

Example PigNet output directory:

results/PigNet_nested_10_fold_CV_win14/

4. Run the tree-based models

python code/XGBoost-10_fold_CV_win14.py
python code/Light-10_fold_CV_win14.py
python code/RF-10_fold_CV_win14.py
python code/Cat-10_fold_CV_win14.py

Each tree-model directory contains:

one tuning-cache CSV file per outer fold;

the selected parameters for each outer fold;

feature-importance results when supported;

the final nested-cross-validation Excel workbook.

The tuning caches allow completed candidate evaluations to be reused after an interrupted run.

5. Run the simple baselines

python code/simple_baselines.py

This runs both the linear and quadratic growth baselines using the same canonical outer folds.

Output files

The scripts write generated files under results/.

Typical outputs include:

results/
├── canonical_outer_folds_W14_H7_seed42.json
├── PigNet_nested_10_fold_CV_win14/
│   ├── PigNet_outer_fold1.pt
│   ├── ...
│   └── PigNet_W14_H7_NestedOuter10CV.xlsx
├── TimesNet_nested_10_fold_CV_win14/
├── LSTM_nested_10_fold_CV_win14/
├── LightTS_nested_10_fold_CV_win14/
├── XGBoost_nested_10_fold_CV_win14/
│   ├── tuning_cache/
│   ├── selected_parameters_by_outer_fold.json
│   └── XGBoost_W14_H7_NestedOuter10CV.xlsx
├── LightGBM_nested_10_fold_CV_win14/
├── RandomForest_nested_10_fold_CV_win14/
├── CatBoost_nested_10_fold_CV_win14/
└── SimpleBaselines_nested_10_fold_CV_win14/

Computational considerations

The revised protocol is computationally more demanding than a single 10-fold evaluation because model selection is repeated independently inside every outer-training partition.

In particular:

neural models require five inner training runs plus one final retraining run for every outer fold;

XGBoost evaluates 288 candidates per outer fold;

LightGBM evaluates 864 candidates per outer fold;

Random Forest evaluates 72 candidates per outer fold;

CatBoost evaluates 288 candidates per outer fold;

every tree candidate is evaluated over five inner folds and seven independent forecast-horizon regressors.

Run the exhaustive tree-model experiments on a machine with sufficient CPU time and memory. The per-fold tuning cache supports recovery after interruption.

Trained weights

The weights/ directory contains representative pretrained weights:

weights/
├── PigNet.pt
├── LSTM.pt
└── TimesNet.pt

These files are provided for result inspection and architecture-level verification. The nested cross-validation scripts do not automatically load them. Instead, they train fold-specific models from scratch and save the resulting checkpoints under results/.

Reproducibility notes

The global base random seed is 42.

Python, NumPy, PyTorch, and CUDA random generators are initialized by the shared utilities.

Deterministic PyTorch behavior is requested with cudnn.benchmark=False, cudnn.deterministic=True, and deterministic algorithms where supported.

Exact numerical identity across different operating systems, hardware architectures, CUDA versions, or third-party library builds is not guaranteed.

The canonical outer fold manifest should be archived together with the reported results.

The Excel workbooks and tuning caches should be retained as the provenance records for the final tables.

Scope and limitations

The data were collected under the measuring-station-based experimental setting described in the manuscript.

The repository supports the forecasting task studied in the paper and should not be interpreted as direct validation under truly low-cost sensing conditions. Performance under different farms, breeds, sensing systems, management protocols, and missing-data patterns requires separate external validation.

Citation

If you use this repository, please cite the corresponding manuscript:

PigNet: Multi-step forecasting of pig body weight from multi-source data using
an LSTM with time-aware attention and depthwise separable convolutional smoothing

A complete bibliographic entry will be added after publication.

Contact

For questions regarding the dataset or code, please contact:

Zhenlong WuChina Agricultural UniversityE-mail: wuzhenlong@cau.edu.cn
