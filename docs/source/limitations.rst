Limitations
===========

- The importer currently expects Bruker ``fid`` and ``acqus`` files.
- Diagnostic or non-sample folders are not automatically excluded.
- Automatic phase correction is zero-order only.
- Alignment is a simple global integer shift within one window, not full
  icoshift or local warping.
- Baseline and solvent procedures are user-adjustable and require visual QC.
- Sample-type-specific region presets require user justification.
- Binning does not perform metabolite identification.
- PLS-DA output is exploratory and does not currently include permutation tests.
- Univariate p-values are not automatically corrected for multiple testing.
- Linear SVM probability calibration is not currently implemented.
- ANN feature importance is heuristic.
- Cross-validation is not nested hyperparameter optimization.
- The optional threshold-rule benchmark is a descriptive comparator, not a
  trained model or clinical decision rule.
- The app is not validated for clinical decision-making.
- Model history and intermediate state are session-based unless downloaded.
