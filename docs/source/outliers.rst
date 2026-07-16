Outlier handling
================

Outlier screening occurs after the initial EDA and before a second EDA.

Methods
-------

Hotelling's T²
   Calculates a score-space statistic and an F-based confidence limit.

Robust PCA distance
   Calculates Euclidean distance from a median PCA-score center and converts
   it to a median-absolute-deviation-based robust z-score.

Groupwise mode
--------------

When enabled, limits are calculated separately within each ``Class`` cohort.
This reduces the risk of flagging a biologically shifted cohort merely because
it differs from the global center.

Decision workflow
-----------------

1. Detect possible outliers.
2. Inspect cohort-specific plots and the outlier table.
3. Review spectral quality, metadata, and biological plausibility.
4. Remove flagged samples, enter selected IDs manually, or skip removal.
5. Rerun EDA and compare results.

Outlier flags are not proof that a sample is invalid.
