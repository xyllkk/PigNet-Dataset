# PigNet-Dataset

This repository provides the dataset, source code, and trained model weights for the manuscript:

PigNet: Multi-step forecasting of pig body weight from multi-source data using an LSTM with time-aware attention and depthwise separable convolutional smoothing

## Overview

This repository is organized to support the reproducibility of the study on short-term pig body-weight forecasting from multi-source data.

The repository currently includes:

- main-cohort data used for model development and evaluation
- second-batch data used for independent-cohort evaluation
- source code for PigNet and the compared baseline models
- trained model weights for representative models

The study focuses on forecasting the next 7 days of individual pig body weight from a 14-day look-back window using daily records of feeding behavior, age, initial body weight, and environmental variables.

## Repository structure

```text
PigNet-Dataset/
├─ code/
├─ data/
│  ├─ main_cohort/
│  └─ second_batch/
├─ weights/
└─ README.md
```

## Data

### 1. Main cohort

The folder `data/main_cohort/` contains the main experimental Excel files used in the manuscript.

These files are used for model training, ablation study, input-availability analysis, and ear-tag-grouped 10-fold cross-validation.

### 2. Second batch

The folder `data/second_batch/` contains the independently collected second-batch Excel files used for the independent-cohort evaluation with limited target-domain fine-tuning.

This part is used to assess transferability under cross-cohort conditions.

## Code

The folder `code/` contains the scripts used in the study, including PigNet and the baseline models.

Current scripts include:

- `PigNet.py`
- `LSTM-10_fold_CV_win14.py`
- `LightTS.py`
- `TimesNet.py`
- `RF-10_fold_CV_win14.py`
- `XGBoost-10_fold_CV_win14.py`
- `Light-10_fold_CV_win14.py`
- `Cat-10_fold_CV_win14.py`
- `simple_baselines.py`

These scripts correspond to the experiments reported in the manuscript, including the proposed PigNet model, deep-learning baselines, machine-learning baselines, and simple limited-input baselines.

## Trained weights

The folder `weights/` contains trained model weights for representative models:

- `PigNet.pt`
- `LSTM.pt`
- `TimesNet.pt`

These weights are provided to facilitate result verification and reproducibility.

## Experimental setting

The main experiments in the manuscript are based on:

- daily-granularity forecasting
- look-back window length of 14 days
- forecast horizon of 7 days
- ear-tag-grouped outer 10-fold cross-validation

Additional experiments include:

- ablation study
- input-availability analysis
- fold-to-fold variability analysis
- independent-cohort evaluation with limited target-domain fine-tuning

## Data format

The raw Excel files keep the original Chinese column names used during data collection and analysis. The provided code reads these original field names directly.

For clarity, an English reference for the column names is provided in `column_mapping.md`.

## Environment

The scripts are implemented in Python and mainly rely on common scientific computing and deep-learning packages, including:

- numpy
- pandas
- scikit-learn
- torch
- openpyxl
- xlsxwriter

A typical installation command is:

```bash
pip install numpy pandas scikit-learn torch openpyxl xlsxwriter
```

## Notes

- The repository is intended to improve the transparency and reproducibility of the study.
- The current data were collected under the measuring-station-based experimental setting described in the manuscript.
- The present repository supports the forecasting setting studied in the paper and should not be interpreted as direct validation under truly low-cost sensing conditions.

## Citation

If you use this repository, please cite the corresponding manuscript.

## Contact

For questions regarding the dataset or code, please contact:

Zhenlong Wu  
China Agricultural University  
E-mail: wuzhenlong@cau.edu.cn
