# Results

Executable scripts write generated workbooks, checkpoints, and fold manifests
under this directory. These run-specific outputs are ignored by Git to avoid
publishing large checkpoints and temporary spreadsheets.

`reference_metrics/` contains small, reviewable tables that record the final
manuscript-aligned benchmark values, the supplementary QC sensitivity summary,
and the recovered independent-cohort split assignments. They are reference
records, not substitutes for rerunning the executable code.

The three source checkpoints required for independent-cohort adaptation are
distributed in `../weights/`.
