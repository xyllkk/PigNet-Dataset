"""Run LSTM adaptation and evaluation on the independent cohort."""

from independent_cohort_common import main


if __name__ == "__main__":
    main("lstm", "LSTM", "CV10_F7")
