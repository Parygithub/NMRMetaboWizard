Exploratory data analysis (EDA)

EDA helps identify patterns, trends, outliers, and potential data-quality issues before further statistical analysis or modelling.

=========================

PCA
---

PCA is fitted to median-imputed and standardized NMR bins. The app provides:

- selectable component count;
- interactive 2D or 3D score plots;
- explained-variance table;
- clinical-variable coloring;
- PCA loading bar plots for one or more PCs.

PCA loading plots
-----------------

Loadings are plotted against ppm-bin centers. Numeric ppm axes use conventional
reversed NMR orientation and integer tick labels. Loadings indicate which bins
contribute to a component; they do not by themselves identify metabolites.

PLS-DA
------

PLS-DA uses one-hot encoded class labels and ``PLSRegression``. Score plots are
exploratory. The displayed approximate X-variance percentages are not
equivalent to PCA explained variance.

Clinical-variable coloring
---------------------------

The color menu is generated from aligned clinical metadata. Numeric variables
use a continuous color scale; categorical variables use discrete traces.

Other analyses
--------------

- class counts;
- Welch t-test for two classes;
- one-way ANOVA for more than two classes;
- Spearman clinical-clinical correlations;
- Spearman clinical-bin correlations;
- PSA box plots and PSA-stratified class counts.

Multiple-testing caution
------------------------

The current univariate table reports raw p-values and simple effect
differences. Apply an appropriate false-discovery-rate procedure before making
confirmatory claims.
