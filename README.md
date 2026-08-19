# NMRMetaboWizard

NMRMetaboWizard is an interactive, no-code Shiny for Python application for stepwise preprocessing, clinical-metadata integration, exploratory analysis, outlier assessment, and machine-learning analysis of one-dimensional Bruker ¹H NMR metabolomics data.

[![Tests](https://github.com/Parygithub/NMRMetaboWizard/actions/workflows/tests.yml/badge.svg)](https://github.com/Parygithub/NMRMetaboWizard/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/nmrmetabowizard/badge/?version=latest)](https://nmrmetabowizard.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- **Source code:** https://github.com/Parygithub/NMRMetaboWizard
- **Documentation:** https://nmrmetabowizard.readthedocs.io/
- **Public web application:** https://parygithub-nmrmetabowizard.share.connect.posit.cloud/

## Why this software?

Raw NMR free induction decays require a sequence of processing and quality-control decisions before they become a statistical feature matrix. NMRMetaboWizard exposes those decisions in a stepwise interface rather than hiding them in a black-box workflow. Users can inspect, apply, adjust, or skip optional preprocessing steps, align spectra with clinical metadata, explore PCA and PLS-DA results, screen possible outliers by cohort, and compare several machine-learning models.

The application is disease-neutral. Clinical-variable stratification is optional and can use any numeric variable supplied by the user; no PSA, cancer type, or positive outcome is assumed by the software.

## Main capabilities

- Bruker ZIP import from experiment folders containing `fid` and `acqus`
- Raw complex FID reconstruction and time-domain visualization
- Group-delay handling using `GRPDLY` or a manual override
- Optional FID-domain solvent-residual suppression
- Exponential or Gaussian apodization
- Optional zero filling
- Fourier transformation and ppm-axis construction
- Automatic zero-order or manual phase correction
- Optional internal chemical-shift referencing
- ALS, arPLS, or airPLS baseline correction
- Optional custom exclusion interval for baseline estimation
- Optional cross-correlation-based integer peak alignment
- Negative-value zeroing and spectral-window selection
- Optional region-removal presets or a custom interval
- Binning by width or total number of bins using trapezoidal or rectangular integration
- PQN, total-area, SNV, or no normalization
- Clinical metadata import from CSV, TSV, TXT, or XLSX
- Sample alignment by `study_id` and outcome labels from `Class`
- PCA scores, PCA loading bar plots, PLS-DA scores, univariate testing, and clinical correlations
- Score coloring by aligned numeric or categorical clinical variables
- Optional selected-variable cutoff views using the median or a manual cutoff
- Hotelling's T² or robust PCA-distance outlier screening, globally or by cohort
- Logistic regression, random forest, linear SVM, and ANN/MLP models
- NMR-only, clinical-only, NMR-plus-clinical, or selected-variable-only predictors
- Explicit selection of clinical predictors to reduce unintended information leakage
- Optional PCA within the machine-learning pipeline and optional cross-validation
- Optional user-defined threshold-rule benchmark for binary outcomes
- Downloadable spectra, tables, predictions, feature importance, and model-run history
- Downloadable fully synthetic demonstration NMR and clinical datasets

## Scientific status

NMRMetaboWizard is a research analysis platform. It is not a validated medical device or clinical diagnostic system. Results require appropriate quality control, independent validation, and interpretation by suitably qualified researchers.

## Requirements

- Python 3.10 or newer
- A modern web browser
- Dependencies listed in `requirements.txt`

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Parygithub/NMRMetaboWizard.git
cd NMRMetaboWizard
```

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

```bash
python -m shiny run --reload --launch-browser app.py
```

For a network-accessible local deployment:

```bash
python -m shiny run --host 0.0.0.0 --port 8000 app.py
```

A public hosted instance is available at:

https://parygithub-nmrmetabowizard.share.connect.posit.cloud/

## Synthetic demonstration data

Step 1 provides two downloads:

- a ZIP archive containing 16 fully synthetic Bruker-like FIDs;
- a matching clinical CSV containing 8 `Class_A` and 8 `Class_B` samples.

The clinical file includes generic numeric variables such as `biomarker` and `clinical_measure`. The files contain no patient data and are intended only for testing the workflow. They must not be used for biological or clinical interpretation.

## Input data

### Bruker ZIP archive

The archive should contain one or more Bruker experiments with both `fid` and `acqus`:

```text
study_id_001/
└── 1/
    ├── fid
    └── acqus
```

When the experiment directory is numeric, the parent folder is used as the biological sample ID. Folder and sample names become `study_id` values and must match the clinical metadata.

### Clinical metadata

The clinical file should contain, at minimum:

```text
study_id
Class
```

Example:

```csv
study_id,Class,age,biomarker,height,weight
study_id_001,Class_A,54,3.2,168,67
study_id_002,Class_B,61,6.8,174,76
```

Clinical variables become predictors only when the user selects them in the machine-learning section. Exclude diagnosis-derived, post-outcome, or otherwise inappropriate predictors to avoid data leakage.

Only synthetic or appropriately de-identified data should be used with a public deployment. Do not commit confidential or identifiable clinical data to this repository.

## Workflow summary

1. Upload and inspect raw FIDs.
2. Apply or skip preprocessing steps.
3. Create and normalize the binned matrix.
4. Import clinical metadata and align samples by `study_id`.
5. Run exploratory data analysis.
6. Optionally choose a numeric clinical variable for cutoff-based plots.
7. Screen possible outliers and decide whether to remove them.
8. Repeat exploratory analysis after the outlier decision.
9. Select predictor sources and train machine-learning models.
10. Download processed data, plots, predictions, and processing logs.

## Reproducibility

- Plot downsampling is display-only; calculations use the full arrays.
- Imputation, scaling, and optional PCA are fitted inside the machine-learning pipeline.
- Machine-learning runs are recorded in a downloadable session-history table.
- Processing parameters are recorded in the application log.
- Synthetic demonstration data are generated deterministically from a fixed random seed.
- The threshold-rule benchmark requires an explicitly selected variable, cutoff, direction, and positive class.


## Large cohorts

For cohorts with 200 or more spectra, the app automatically switches to a
memory-efficient representation when moving from window selection to region
removal. This preserves all data required for region removal, binning,
normalization, EDA, and machine learning while releasing older full-resolution
intermediate arrays. Inspect or download earlier preprocessing plots before
continuing. Binning is vectorized and displays progress for large datasets.

## Testing

Run the main smoke test:

```bash
python tests/smoke_test.py
```

Run the synthetic demonstration-data test:

```bash
python tests/test_demo_data.py
```

Run the disease-neutral clinical-variable test:

```bash
python tests/test_generic_clinical_variable.py
```

GitHub Actions runs the tests on Python 3.10, 3.11, and 3.12.

## Documentation

Full user documentation is available at:

https://nmrmetabowizard.readthedocs.io/

To build it locally:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W -b html docs/source docs/build/html
```

## Citation

Citation metadata are provided in `CITATION.cff` and `CITATION.bib`. Please cite the software version used in your analysis. A formal article citation can be added after publication.

## License

NMRMetaboWizard is distributed under the [MIT License](LICENSE).

## Support and contributions

- Report reproducible software problems through [GitHub Issues](https://github.com/Parygithub/NMRMetaboWizard/issues).
- Submit feature requests through [GitHub Issues](https://github.com/Parygithub/NMRMetaboWizard/issues).
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.
- See [SECURITY.md](SECURITY.md) for security and private-data concerns.
