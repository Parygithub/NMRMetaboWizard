Clinical metadata
=================

Alignment model
---------------

NMRMetaboWizard keeps three objects separate:

- ``X`` - NMR-derived feature matrix;
- ``y`` - class/outcome labels;
- ``clinical_aligned`` - clinical metadata aligned to the same sample order.

Clinical variables do not automatically become predictors. The user chooses
NMR-only, clinical-only, PSA-only, or combined features in the ML step.

ID matching
-----------

IDs are stripped of whitespace and normalized for matching. Numeric IDs such
as ``001`` and ``1`` may be normalized to the same match key. Duplicate
clinical IDs are reported, and the first occurrence is used.

Required review
---------------

Before EDA, inspect:

- number of spectral samples;
- number of clinical rows;
- matched samples with non-empty class labels;
- duplicated clinical IDs;
- unmatched spectral IDs;
- unmatched clinical IDs.

Avoiding leakage
----------------

Do not use variables that are consequences of the outcome as predictors when
the scientific question is diagnosis prediction. For example, Gleason score
or pathological stage may leak diagnosis-related information into a PCa/BPH
classifier.
