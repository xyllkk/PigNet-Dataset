# PigNet: Multi-step forecasting of pig body weight from multi-source data using an LSTM with time-aware attention and depthwise separable convolutional smoothing

## Overview
PigNet formulates week-ahead body‑weight forecasting as a direct multi‑step problem: a 14‑day input window predicts 7 future horizons (h1–h7). The model couples a time‑aware attention module (one learnable query per horizon) with a depthwise‑separable smoothing module along the horizon axis to improve cross‑horizon consistency while remaining lightweight. The pipeline consumes routinely collected farm data such as RFID feeding logs, age, initial body weight, and pig‑house climate sensors (temperature, relative humidity, NH3, CO2).

## Highlights
- Seven‑day multi‑step forecasts aligned with weekly operations.  
- Horizon‑specific attention plus horizon‑axis smoothing for stability.  
- Cost‑effective deployment relying on standard farm sensors and logs.  
- Reproducible data schema, windowing, cross‑validation, and metrics.  

## Repository layout
```
scripts/          data preparation and windowing examples
metrics/          evaluation utilities (RMSE, MAE, R²; per‑horizon)
notebooks/        quick demos and visualization (optional)
models/           model code (if you decide to factor it out later)
README.md         this file
```
Your current script already includes end‑to‑end training and evaluation; the above layout is a suggested organization.

## Data schema
Daily table, one row per pig per day. Map your actual column names via DATE_COL, PIG_COL, and WEIGHT_COL in the script.

Required
- date (one day per pig)  
- pig identifier  
- body weight

Recommended
- age in days  
- feeding intake, counts, duration (hh:mm:ss or numeric)  
- climate sensors: temperature, relative humidity, NH3, CO2

Notes
- duplicates on [date, pig] are removed; rows are sorted by pig → date  
- duration strings are converted to seconds with a new “_s” suffix  
- non‑numeric columns are ignored when building features

Example (CSV header)
```
date, pig_id, weight, age_d, intake_kg, feed_time, temp_c, rh, nh3_ppm, co2_ppm
2024-01-01, Duroc-1-1, 35.2, 80, 2.31, 01:23:40, 22.5, 63, 3, 950
```

## Windowing and targets
- input window W = 14 days, horizons H = 7 (h1..h7)  
- per pig, sliding windows are created in time order  
- the first available weight is used as an initial‑weight feature through FiLM (if enabled)  
- outputs are daily weights for the next 7 days

## Quick start

### Requirements
Python 3.8+, PyTorch 1.10+ (match your CUDA), numpy, pandas, scikit‑learn, openpyxl, xlsxwriter

Install example
```bash
# choose the official torch command that matches your CUDA; example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn openpyxl xlsxwriter
```

### Run
Cross‑validation (10‑fold GroupKFold by pig id; default entry in main)
```bash
python your_script_name.py
```
Fixed split (Global901)
- uncomment `run_global901_only()` in `__main__`

Minimal runner (edit the constants at the top of the script)
```python
DATA_PATH = "/path/to/daily_table_or_folder"
OUT_EXCEL = "/path/to/output/PigNet"
WINDOW, HORIZON = 14, 7
run_cv_kfold_by_pig(n_splits=10)
```

## Configuration knobs
Model switches
- USE_FILM_INIT: FiLM conditioning from initial body weight  
- USE_HQ_ATTENTION: 7 queries for time‑aware attention (one per horizon)  
- USE_H_CONV: depthwise‑separable 1D conv along the horizon axis  
- USE_SMOOTH_LOSS: adjacent‑horizon smoothness penalty (SMOOTH_LAMBDA)

Training
- BEST_PARAMS: hidden_size, num_layers, dropout, lr, epochs, batch_size  
- optimizer: Adam (weight_decay=3e‑4); loss: Huber (or MSE)  
- scheduler: ReduceLROnPlateau (if a validation split exists)  
- early stopping with patience EARLY_PATIENCE

Reproducibility
- seeds are fixed to 42 for NumPy and PyTorch; cudnn.benchmark is enabled

## Metrics and outputs
Per‑horizon and averaged RMSE/MAE/R² are reported. The script writes raw per‑sample predictions to Excel under OUT_EXCEL.

Cross‑validation file
- `{prefix}_CV10_W14_H7.xlsx`  
  - per‑fold sheets `CV10_F1 ... CV10_F10` with long‑form predictions  
  - `CV10_ALL_RAW` concatenated long‑form predictions  
  - `splits_info` listing train/val pigs and selected hyperparameters

Fixed split file
- `{prefix}_W14_H7.xlsx` with `Global901` and `splits_info` sheets

Raw prediction columns
- set (Train/Val/Test), pig id, breed, station, window index, anchor date, horizon h1–h7, target date, y_true, y_pred, fold tag, W, H

## Tips
- column names: map your actual names via DATE_COL, PIG_COL, WEIGHT_COL  
- Excel I/O: ensure both openpyxl and xlsxwriter are installed  
- OOM: reduce batch_size or hidden_size; disable H‑Conv if necessary  
- no numeric features found: adjust `pick_features` or EXCLUDE_FEATURE_KEYWORDS

## License
To be added. Choose a permissive license for code (e.g., MIT) and a suitable non‑commercial license for data if needed.

## Acknowledgments
We thank collaborating farms and partners for data acquisition and on‑farm deployment support.

## Citation
The related resources will be released after the manuscript is accepted.
