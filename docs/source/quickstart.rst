Quick start
===========

Try the synthetic example
-------------------------

At Step 1, the application provides two downloadable demonstration files:

- a ZIP archive containing 16 fully synthetic Bruker-like FIDs;
- a matching clinical metadata CSV containing 8 ``Class_A`` and 8
  ``Class_B`` samples.

Upload the NMR ZIP at Step 1. At Step 17, upload the matching clinical CSV and
use ``study_id`` as the sample ID column and ``Class`` as the class column.
The demonstration data contain generic numeric variables such as
``biomarker`` and ``clinical_measure``. They contain no patient data and must
not be used for biological or clinical interpretation.

Standard workflow
-----------------

1. Prepare a ZIP file containing Bruker experiment folders with ``fid`` and
   ``acqus`` files.
2. Ensure that sample-folder names match the clinical ``study_id`` values.
3. Remove diagnostic or non-sample acquisition folders.
4. Start the app and upload the ZIP.
5. Inspect the raw FIDs and apply or skip each preprocessing step.
6. Create the binned table and apply normalization.
7. Upload a clinical file containing ``study_id`` and ``Class``.
8. Review the alignment summary and unmatched IDs.
9. Run EDA.
10. Optionally choose any numeric clinical variable for cutoff-based plots.
11. Review possible outliers and decide whether to retain or remove them.
12. Rerun EDA after the outlier decision.
13. Select predictor sources and configure machine learning.
14. Download processing logs, plot data, feature matrices, predictions, and
    model-performance history.

General first-pass settings
---------------------------

- group delay: use Bruker ``GRPDLY``;
- referencing: off unless an appropriate reference peak is present;
- alignment: off unless visible drift is present;
- spectral window: choose a range appropriate for the experiment;
- region removal: ``None`` unless a known solvent or artifact interval should
  be excluded;
- bin width: 0.01 ppm as an exploratory starting value;
- integration: trapezoidal;
- normalization: PQN as an exploratory starting value;
- ML PCA: off initially;
- outlier detection: groupwise Hotelling's T² followed by manual review.

The urine water/urea preset at 4.5-6.1 ppm should be selected only for urine
workflows where that exclusion is scientifically appropriate.
