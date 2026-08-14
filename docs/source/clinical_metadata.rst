Clinical metadata
=================

Alignment model
---------------

NMRMetaboWizard keeps three objects separate:

- ``X`` - the NMR-derived feature matrix;
- ``y`` - class or outcome labels;
- ``clinical_aligned`` - clinical metadata in the same sample order.

Clinical variables do not automatically become predictors. The user chooses
NMR-only, clinical-only, NMR-plus-clinical, or selected-variable-only features
in the ML section. For clinical-only and combined models, the predictor list is
selected explicitly.

Optional clinical-variable stratification
-----------------------------------------

EDA and ML can optionally use any numeric clinical variable for cutoff-based
visualization or subsetting. The cutoff can be the variable median or a manual
value. Selecting ``None`` disables this behavior.

The optional threshold-rule benchmark is available only for binary outcomes.
It requires the user to select the variable, cutoff, direction, and positive
class explicitly. The app does not infer a positive class from disease names.

ID matching
-----------

IDs are stripped of whitespace and normalized for matching. Numeric IDs such
as ``001`` and ``1`` may normalize to the same match key. Duplicate clinical
IDs are reported, and the first occurrence is used.

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

Do not use variables that are consequences of the target outcome, measured
after the outcome, or direct encodings of the class when the scientific goal
is prediction. Examples may include pathology-confirmed stage, diagnosis
codes, treatment response measured after therapy, or manually assigned group
indicators.
