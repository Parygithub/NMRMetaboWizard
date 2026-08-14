# Changelog

All notable changes to NMRMetaboWizard are documented here.

## [Unreleased]

- Replaced PSA-specific EDA and ML controls with optional generic numeric clinical-variable controls.
- Added median or manual cutoffs for selected clinical variables.
- Added disease-neutral variable-by-class, cutoff-group, and variable-to-NMR correlation plots.
- Replaced the PSA-only predictor mode with selected-clinical-variable-only modelling.
- Added explicit clinical-predictor selection for clinical and combined models.
- Replaced automatic disease-name positive-class inference with an optional user-defined threshold-rule benchmark.
- Added a neutral default for region removal and retained the urine water/urea interval as an optional preset.
- Added an optional custom exclusion interval for baseline estimation.
- Disabled internal referencing by default.
- Replaced the prostate-specific synthetic demonstration with generic `Class_A` and `Class_B` data.
- Added disease-neutral clinical-variable validation tests.
- Removed legacy `.xls` upload support while retaining `.xlsx` support.

## [0.1.0] - 2026-07-16

- Initial public research-software release.
