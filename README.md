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
