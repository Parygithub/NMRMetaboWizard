Troubleshooting
===============

No Bruker experiments found
----------------------------

Confirm that the ZIP contains directories with both ``fid`` and ``acqus``.
Avoid placing another unsupported archive inside the ZIP.

Clinical data do not match
--------------------------

- Compare detected folder names with ``study_id``.
- Remove spaces and accidental spreadsheet formatting.
- Check duplicate IDs.
- Confirm that ``Class`` is non-empty.
- Review unmatched IDs in the alignment summary.

Plot hangs after selecting a clinical color
-------------------------------------------

Check that the selected variable is numeric when a continuous scale is
expected and that sample IDs are unique. Restart the application after
replacing source files with a newer version.

ROC or probabilities are unavailable
-------------------------------------

A valid holdout test and a model with probability output are required. The
current ``LinearSVC`` implementation does not generate probabilities.

Feature-importance plot is blank
--------------------------------

Run a model first. Check that features remain after filtering. Direct
importance is most interpretable for logistic regression, linear SVM, and
random forest.

Processing is slow
------------------

- Use the fast baseline point limit.
- Avoid unnecessarily small bin widths.
- Disable alignment unless needed.
- Reduce ANN size or iterations.
- Reduce cross-validation folds during exploratory work.
