# NMRMetaboWizard

**NMRMetaboWizard** is an interactive, no-code Shiny for Python application for stepwise preprocessing, clinical-metadata integration, exploratory analysis, outlier assessment, and machine-learning analysis of one-dimensional Bruker \(^1\)H NMR metabolomics data.

> Repository URL to use after publication: `https://github.com/Parygithub/NMRMetaboWizard`  
> Documentation URL to use after Read the Docs setup: `https://nmrmetabowizard.readthedocs.io`

[![Tests](https://github.com/Parygithub/NMRMetaboWizard/actions/workflows/tests.yml/badge.svg)](https://github.com/Parygithub/NMRMetaboWizard/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/nmrmetabowizard/badge/?version=latest)](https://nmrmetabowizard.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why this software?

Raw NMR FIDs require a sequence of processing and quality-control decisions before they become a statistical feature matrix. NMRMetaboWizard exposes those decisions in a stepwise interface rather than hiding them in a black-box workflow. Users can inspect, apply, adjust, or skip optional preprocessing steps, align spectra with clinical metadata, explore PCA/PLS-DA results, screen outliers by cohort, and compare several machine-learning models.

## Main capabilities

- Bruker ZIP import from folders containing `fid` and `acqus`
- Raw complex FID reconstruction and time-domain visualization
- Group-delay handling using `GRPDLY` or a manual override
- Optional FID-domain solvent-residual suppression
- Exponential or Gaussian apodization
- Optional zero filling
- Fourier transformation and ppm-axis construction
- Automatic zero-order or manual phase correction
- Internal chemical-shift referencing
- ALS, arPLS, or airPLS baseline correction
- Optional cross-correlation-based integer peak alignment
- Negative-value zeroing, spectral-window selection, and region removal
- Binning by width or by total number of bins using trapezoidal or rectangular integration
- PQN, total-area, SNV, or no normalization
- Clinical metadata import from CSV, TSV, TXT, XLSX, or XLS
- ID alignment using `study_id` and class labels using `Class`
- PCA scores, PCA loadings, PLS-DA scores, univariate testing, and clinical correlations
- Score coloring by aligned clinical variables
- Hotelling's T² or robust PCA-distance outlier screening, globally or by cohort
- Logistic regression, random forest, linear SVM, and ANN/MLP
- NMR-only, clinical-only, NMR+clinical, or PSA-only predictor sets
- Optional PCA within the ML pipeline and optional cross-validation
- Downloadable spectra, tables, predictions, feature importance, and model-run history

## Important scientific status

NMRMetaboWizard is a **research analysis platform**, not a validated medical device or clinical diagnostic system. Results must be interpreted with appropriate quality control, validation, and domain expertise.

## Requirements

- Python 3.10 or newer
- A modern browser (Chrome or Edge recommended)
- Dependencies listed in `requirements.txt`

## Installation

### Windows

```bat
cd /d C:\Users\YOUR_USERNAME\Documents
python -m venv NMRMetaboWizard\.venv
NMRMetaboWizard\.venv\Scripts\activate.bat
cd NMRMetaboWizard
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

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

## Input data

### Bruker ZIP archive

The archive should contain one or more Bruker experiments with both:

```text
fid
acqus
```

Recommended structure:

```text
study_id_001/
└── 1/
    ├── fid
    └── acqus
```

When the experiment directory is numeric, the parent folder is used as the biological sample ID. **Folder/sample names become `study_id` values and must match the clinical metadata.**

Remove diagnostic or dummy acquisition folders such as `99999` before creating the ZIP.

### Clinical metadata

The clinical file should contain at minimum:

```text
study_id
Class
```

Example:

```csv
study_id,Class,age,psa,height,weight
study_id_001,PCa,68,7.4,176,81
study_id_002,BPH,64,5.9,171,77
```

Only synthetic or fully de-identified examples should be committed to a public repository.

## Workflow summary

1. Upload and inspect raw FIDs.
2. Apply or skip preprocessing steps.
3. Create and normalize the binned matrix.
4. Import clinical metadata and align by `study_id`.
5. Run EDA and inspect PCA/PLS-DA results.
6. Screen possible outliers and decide whether to remove them.
7. Rerun EDA after the outlier decision.
8. Train and compare machine-learning models.
9. Download data tables, predictions, plots, and processing logs.

## Reproducibility

- Plotting downsampling is display-only; calculations use the full arrays.
- Imputation, scaling, and optional PCA are fitted inside the ML pipeline.
- Every ML run is recorded in a downloadable session history table.
- Processing parameters are logged.
- Use a tagged release and archive it with Zenodo before manuscript submission.

## Testing

Run the synthetic smoke test:

```bash
python tests/smoke_test.py
```

GitHub Actions also compiles the source files, builds the documentation, and runs the smoke test.

## Documentation

Full documentation is in `docs/source/`.

Build locally:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html`.

## Citation

See [`CITATION.cff`](CITATION.cff) and [`CITATION.bib`](CITATION.bib).

Suggested citation before publication:

> Vahdatiyekta P, Huynh T-P. NMRMetaboWizard: a dynamic Shiny platform for \(^1\)H NMR untargeted metabolomics. Software version 0.1.0.

Replace this with the final paper citation after acceptance.

## Availability and implementation

A manuscript-ready statement is provided in [`journal/AVAILABILITY_AND_IMPLEMENTATION.md`](journal/AVAILABILITY_AND_IMPLEMENTATION.md).

## License

The repository includes a proposed MIT License. Confirm intellectual-property and licensing requirements with Åbo Akademi University before public release.

## Support and contributions

- Bug reports: GitHub Issues
- Feature requests: GitHub Issues
- Contributions: see [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security/private-data concerns: see [`SECURITY.md`](SECURITY.md)
