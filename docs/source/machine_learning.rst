Machine learning
================

Machine-learning algorithms identify patterns in training data and use them to
classify samples. NMRMetaboWizard supports disease-neutral predictor selection
and does not assume a particular biomarker or positive class.

Predictor sources
-----------------

- NMR only
- Clinical only
- NMR + clinical
- Selected clinical variable only

For clinical-only and combined models, users select the numeric clinical
predictors to include. This makes the design explicit and helps avoid accidental
use of diagnosis-derived, post-outcome, or otherwise inappropriate variables.

Optional clinical-variable subsetting
-------------------------------------

Any numeric clinical variable may be selected for optional subsetting. The
cutoff can be the variable median or a manual value, and the model can use:

- all samples;
- samples below the cutoff;
- samples at or above the cutoff.

Selecting ``None`` is valid when all samples and non-selected-variable feature
modes are used.

Threshold-rule baseline
-----------------------

For binary outcomes, users may optionally compare the ML model with a simple
one-variable threshold rule. The user must explicitly select:

- the numeric variable;
- cutoff;
- positive class;
- whether values below or at/above the cutoff predict the positive class.

This benchmark is not a fitted ML model and is disabled by default. The app
never infers the positive class from words such as a disease name.

Models
------

- logistic regression;
- random forest;
- linear support vector machine using ``LinearSVC``;
- artificial neural network or multilayer perceptron using ``MLPClassifier``.

Holdout evaluation
------------------

By default, 25% of samples are reserved for testing, and the split is
stratified when class sizes permit. The app reports training and test accuracy,
balanced accuracy, classification reports, confusion matrices, and individual
predictions.

Cross-validation
----------------

Five-fold stratified cross-validation is enabled by default but can be
disabled. The number of folds is reduced automatically when the smallest class
contains fewer samples than requested.

Leakage control
---------------

Median imputation, standardization, and optional PCA are inside the
scikit-learn ``Pipeline`` and are fitted separately in each training split or
cross-validation fold. Predictor selection still requires scientific review;
pipeline encapsulation cannot make an inappropriate clinical variable valid.

Optional PCA
------------

PCA is off by default. When enabled, the metrics distinguish the input feature
count from the number of components seen by the classifier.

Probability and ROC outputs
---------------------------

Logistic regression, random forest, and ANN provide probability output and can
produce ROC plots after a valid holdout fit. The current linear SVM uses
``LinearSVC`` and does not provide calibrated probabilities.

Feature importance
------------------

For logistic regression and linear SVM, importance is calculated from mean
absolute coefficients. Random-forest importance is impurity based. ANN
importance is approximated from mean absolute first-layer weights and should
be interpreted cautiously.

When optional PCA is used, the displayed original-variable importance table is
obtained from a separate full-data model without PCA. It is exploratory and is
not a held-out performance estimate.

Model history
-------------

Each model run is appended to a session history table with settings and
metrics. Download the table before closing or restarting the app.
