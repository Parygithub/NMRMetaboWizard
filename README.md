# NMRMetaboWizard

NMRMetaboWizard is an interactive, no-code Shiny for Python application for stepwise preprocessing, clinical-metadata integration, exploratory analysis, outlier assessment, and machine-learning analysis of one-dimensional Bruker \(^1\)H NMR metabolomics data.

[![Tests](https://github.com/Parygithub/NMRMetaboWizard/actions/workflows/tests.yml/badge.svg)](https://github.com/Parygithub/NMRMetaboWizard/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/nmrmetabowizard/badge/?version=latest)](https://nmrmetabowizard.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- **Source code:** https://github.com/Parygithub/NMRMetaboWizard
- **Documentation:** https://nmrmetabowizard.readthedocs.io/
- **Public web application:** https://parygithub-nmrmetabowizard.share.connect.posit.cloud/

## Why this software?

Raw NMR FIDs require a sequence of processing and quality-control decisions before they become a statistical feature matrix. NMRMetaboWizard exposes those decisions in a stepwise interface rather than hiding them in a black-box workflow. Users can inspect, apply, adjust, or skip optional preprocessing steps, align spectra with clinical metadata, explore PCA and PLS-DA results, screen possible outliers by cohort, and compare several machine-learning models.

## Main capabilities

- Bruker ZIP import from experiment folders containing `fid` and `acqus`
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
- Binning by width or total number of bins using trapezoidal or rectangular integration
- PQN, total-area, SNV, or no normalization
- Clinical metadata import and alignment by `study_id`
- PCA scores, PCA loadings, PLS-DA scores, univariate testing, and clinical correlations
- Score coloring by aligned clinical variables
- Hotelling's T² or robust PCA-distance outlier screening, globally or by cohort
- Logistic regression, random forest, linear SVM, and ANN/MLP models
- NMR-only, clinical-only, NMR-plus-clinical, or PSA-only predictor sets
- Optional PCA within the machine-learning pipeline and optional cross-validation
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

Create and activate a virtual environment.

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

Step 1 of the application provides two downloads:

- a ZIP archive containing 16 fully synthetic Bruker-like FIDs;
- a matching clinical CSV containing 8 BPH-labelled and 8 PCa-labelled samples.

The demonstration files contain no patient data and are intended only for testing the software workflow. They must not be used for biological or clinical interpretation.

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

When the experiment directory is numeric, the parent folder is used as the biological sample ID. Folder and sample names become `study_id` values and must match the clinical metadata.

Remove diagnostic or non-sample acquisition folders before creating the ZIP.

### Clinical metadata

The clinical file should contain, at minimum:

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

Only synthetic or appropriately de-identified data should be used with a public deployment. Do not commit confidential or identifiable clinical data to this repository.

## Workflow summary

1. Upload and inspect raw FIDs.
2. Apply or skip preprocessing steps.
3. Create and normalize the binned matrix.
4. Import clinical metadata and align samples by `study_id`.
5. Run exploratory data analysis.
6. Screen possible outliers and decide whether to remove them.
7. Repeat exploratory analysis after the outlier decision.
8. Train and compare machine-learning models.
9. Download processed data, plots, predictions, and processing logs.

## Reproducibility

- Plotting downsampling is display-only; calculations use the full arrays.
- Imputation, scaling, and optional PCA are fitted inside the machine-learning pipeline.
- Every machine-learning run is recorded in a downloadable session-history table.
- Processing parameters are recorded in the application log.
- Synthetic demonstration data are generated deterministically from a fixed random seed.

## Testing

Run the smoke test:

```bash
python tests/smoke_test.py
```

Run the synthetic demonstration-data validation test:

```bash
python tests/test_demo_data.py
```

The demonstration-data test verifies:

- 16 `fid` files;
- 16 `acqus` files;
- 16 unique FID SHA-256 hashes;
- exact agreement between NMR sample-folder names and clinical `study_id` values.

GitHub Actions runs the tests on Python 3.10, 3.11, and 3.12.

## Documentation

Full user documentation is available at:

https://nmrmetabowizard.readthedocs.io/

To build the documentation locally:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W -b html docs/source docs/build/html
```

Then open:

```text
docs/build/html/index.html
```

## Citation

Citation metadata are provided in:

- `CITATION.cff`
- `CITATION.bib`

Please cite the software version used in your analysis. A formal article citation can be added here after publication.

## License

NMRMetaboWizard is distributed under the MIT License. See [LICENSE](LICENSE).

## Support and contributions

- Report reproducible software problems through [GitHub Issues](https://github.com/Parygithub/NMRMetaboWizard/issues).
- Submit feature requests through [GitHub Issues](https://github.com/Parygithub/NMRMetaboWizard/issues).
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.
- See [SECURITY.md](SECURITY.md) for security and private-data concerns.
