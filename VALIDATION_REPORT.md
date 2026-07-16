# Validation report

Date: 2026-07-16  
Software version: 0.1.0 release candidate

## Static validation

- `app.py`: compiled successfully
- `clinical_analysis.py`: compiled successfully
- `nmr_pipeline.py`: compiled successfully

## Synthetic smoke test

Command:

```bash
python tests/smoke_test.py
```

Exit status: 0

Standard output:

```text
NMRMetaboWizard smoke test passed.
```

The smoke test exercises:

- synthetic complex FID creation;
- group delay;
- optional solvent step;
- apodization;
- zero filling;
- Fourier transformation;
- automatic phase correction;
- referencing bypass;
- baseline correction;
- alignment bypass;
- negative-value zeroing;
- spectral windowing;
- region removal;
- binning and PQN normalization;
- clinical alignment;
- PCA and PCA loadings;
- PLS-DA;
- groupwise outlier screening;
- logistic regression;
- random forest;
- linear SVM;
- ANN/MLP.

## Scope

This report verifies that the source modules compile and that representative
functions execute on synthetic data. It does not establish clinical validity,
equivalence to vendor software, or correctness for every Bruker acquisition
configuration. Those require study-specific quality control and independent
scientific validation.
