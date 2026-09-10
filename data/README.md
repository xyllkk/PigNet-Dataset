# Data

## Organisation

- `main_cohort/` contains the source cohort used for model development and
  nested pig-level cross-validation.
- `second_batch/` contains the independently collected target cohort used for
  the repeated adaptation experiment (33 retained Duroc pigs: 3 fine-tuning
  and 30 test pigs per split).

The repository retains the supplied workbook organisation. Scripts accept
either a directory of `.xlsx`, `.xls`, and `.csv` files or a single compatible
table. Unreadable files are skipped only when a directory is supplied; execution
fails if no readable tables remain.

## Schema

The original column names are retained because they are part of the released
data schema. The required columns are:

- `日期`: calendar date;
- `耳缺号`: pig identifier;
- `体重`: body weight in kilograms.

Candidate predictors include daily feeding amount and frequency, feeding
duration, age, and environmental variables. Feeding-duration fields are
converted to seconds when needed. See [`../column_mapping.md`](../column_mapping.md)
for the full English mapping.

## Preprocessing and window construction

Records are parsed, de-duplicated by pig and date, and sorted within pig. The
first non-missing body weight in each pig's chronologically sorted series is the
single initial body-weight anchor. No body-weight measurements from the 14-day
input interval are used as time-varying predictors.

Every sample requires 21 consecutive calendar days for one pig: 14 input days
immediately followed by 7 target days. A candidate crossing a missing calendar
day is excluded. Missing predictor values are mean-imputed and standardized
using statistics estimated only from the applicable training partition. For
independent-cohort adaptation, the stored source-cohort means and standard
deviations are retained.

Pig identifiers are the grouping unit throughout; windows from one pig never
cross training, validation, or test partitions.
