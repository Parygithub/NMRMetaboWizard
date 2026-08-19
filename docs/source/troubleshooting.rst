Troubleshooting
===============

No Bruker experiments found
---------------------------

Confirm that the uploaded ZIP contains Bruker experiment directories with
both ``fid`` and ``acqus`` files. Avoid placing another unsupported archive
inside the ZIP.

If fewer samples are detected than expected, check that every intended
experiment contains both files. A folder containing ``acqus`` but no ``fid``
cannot be imported as a raw NMR experiment.

Clinical data do not match
--------------------------

- Compare the detected NMR sample-folder names with ``study_id``.
- Remove leading or trailing spaces and accidental spreadsheet formatting.
- Check for duplicate sample IDs.
- Confirm that the selected class column is non-empty.
- Review unmatched IDs in the alignment summary.

The ``study_id`` values in the clinical table must correspond to the sample
names detected from the Bruker archive.

Plot does not update as expected
--------------------------------

Confirm that the selected clinical variable contains suitable values for the
requested plot and that sample IDs are unique.

For continuous coloring, select a numeric clinical variable. For categorical
coloring, select an appropriate grouping variable.

If the application state becomes inconsistent after changing an earlier
processing step, restart the workflow and process the data again using the
intended settings.

Disconnected from the server or Reload appears
-----------------------------------------------

A browser message such as ``Disconnected from the server`` or ``Reload`` can
occur if the application process stops unexpectedly. For large NMR cohorts,
one possible cause is excessive memory use during processing.

For cohorts containing 200 or more spectra, NMRMetaboWizard automatically
activates a memory-efficient mode when progressing from window selection to
region removal. Older full-resolution intermediate arrays are released while
the data required for region removal, binning, normalization, exploratory
analysis, and machine learning are retained.

Before continuing beyond window selection in a large cohort:

- inspect important preprocessing quality-control plots;
- download any intermediate plot data that should be retained;
- use only as much zero filling as required for the analysis;
- avoid unnecessary alignment or computationally expensive options;
- use a reasonable bin width rather than an extremely small value.

After large-cohort memory mode is activated, earlier full-resolution
preprocessing plots cannot be regenerated within the same session. Restart
the preprocessing workflow if those earlier stages need to be inspected or
changed.

If disconnection continues during local use, inspect the terminal for errors
such as ``MemoryError``, a killed process, or other traceback information.

For hosted deployments, available memory depends on the deployment
environment. Very large cohorts may therefore be more suitable for local
processing when the hosted memory limit is insufficient.

Processing is slow
------------------

Processing time increases with the number of samples, spectral resolution,
zero-filled points, number of bins, and computationally intensive analysis
options.

To reduce processing time:

- use the fast baseline point limit where appropriate;
- avoid unnecessarily large zero-filling values;
- avoid unnecessarily small bin widths;
- disable peak alignment unless it is required;
- reduce ANN size or the number of training iterations during exploratory
  analysis;
- reduce cross-validation folds during exploratory work.

Binning uses a vectorized implementation and displays progress while spectra
are processed.

ROC curve or predicted probabilities are unavailable
-----------------------------------------------------

A valid holdout test and a model that provides probability estimates are
required for predicted probabilities and probability-based ROC analysis.

The current linear SVM implementation does not provide predicted
probabilities.

Feature-importance plot is blank
--------------------------------

Run a machine-learning model first and confirm that usable predictor features
remain after filtering.

Direct feature importance is available for models such as logistic
regression, linear SVM, and random forest. Some models or configurations may
not provide directly interpretable feature importance.

Selected clinical-variable analysis is unavailable
---------------------------------------------------

Optional cutoff-based clinical-variable analysis requires a numeric clinical
variable.

If no suitable numeric variable is selected:

- cutoff-based subsetting is not applied;
- selected-variable-only modelling is unavailable;
- threshold-rule baseline analysis is unavailable.

This behavior is intentional. NMRMetaboWizard does not assume that a
particular biomarker, disease, or clinical variable is present.

Threshold-rule baseline cannot be calculated
--------------------------------------------

The optional threshold-rule benchmark requires:

- a numeric clinical variable;
- a valid cutoff;
- an explicitly selected positive class;
- a threshold direction.

The positive class is not inferred automatically from the disease or class
name.

Large-cohort quality-control plots are no longer available
----------------------------------------------------------

For large cohorts, memory-efficient mode releases older full-resolution
intermediate arrays after window selection.

This does not remove the downstream spectral data required for region
removal, binning, normalization, EDA, outlier analysis, or machine learning.

If an earlier preprocessing plot must be regenerated, restart the workflow
and inspect or export the required quality-control output before progressing
to the downstream large-cohort stages.
