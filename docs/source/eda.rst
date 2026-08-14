Exploratory data analysis
=========================

EDA helps identify patterns, trends, possible outliers, and data-quality issues
before further statistical analysis or modelling.

Principal component analysis
----------------------------

PCA is applied after median imputation and standardization of the NMR bins. The
app provides:

- selectable component count;
- interactive 2D or 3D score plots;
- explained-variance table;
- coloring by aligned clinical variables;
- PCA loading bar plots for one or more PCs.

PCA loading plots
-----------------

Loadings are plotted against ppm-bin centers using the conventional reversed
NMR orientation. Loadings indicate which bins contribute to a component; they
do not by themselves identify metabolites.

Partial least squares-discriminant analysis
-------------------------------------------

PLS-DA uses one-hot encoded class labels and ``PLSRegression``. Score plots are
exploratory. The displayed approximate X-variance percentages are not
equivalent to PCA explained variance.

Clinical-variable coloring and cutoff views
-------------------------------------------

The score-color menu is generated from aligned clinical metadata. Numeric
variables use a continuous color scale and categorical variables use discrete
traces.

Users may optionally select any numeric clinical variable and a cutoff. The
cutoff can be the variable median or a manual value. This enables:

- selected variable by Class;
- Class counts below and at or above the cutoff;
- selected-variable to NMR-bin Spearman correlations;
- score coloring by the derived variable group.

Selecting ``None`` disables the derived cutoff groups. No biomarker name or
clinical cutoff is assumed by the app.

Other analyses
--------------

- class counts;
- Welch t-test for two classes;
- one-way ANOVA for more than two classes;
- Spearman clinical-clinical correlations;
- Spearman clinical-bin correlations.

Multiple-testing caution
------------------------

The current univariate table reports raw p-values and simple effect
differences. Apply an appropriate multiple-testing procedure before making
confirmatory claims.
