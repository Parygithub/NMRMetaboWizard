Outputs
=======

Preprocessing
-------------

- current plotted data as CSV;
- binned NMR matrix as CSV;
- normalized NMR matrix as CSV;
- processing log as TXT.

Clinical alignment
------------------

- aligned sample table as CSV;
- alignment summary in the app.

EDA
---

- current EDA plot data as CSV;
- PCA variance and loading tables;
- PLS-DA component table;
- class counts;
- univariate tests;
- clinical correlation tables.

Outliers
--------

- outlier table as CSV;
- retained/removed sample decision recorded in the aligned object.

Machine learning
----------------

- current ML plot data as CSV;
- feature importance as CSV;
- test predictions as CSV;
- model-performance history as CSV.

Current limitations
-------------------

The app does not yet export a serialized fitted model or a single complete
analysis bundle. Archive downloaded results together with the processing log
and software version.
