"""Run PigNet adaptation and evaluation on the independent cohort."""

from independent_cohort_common import main


if __name__ == "__main__":
    main("pignet", "PigNet", "CV10_F4")
