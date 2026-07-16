Reproducibility
===============

Minimum information to report
-----------------------------

- software version and release DOI;
- Python and dependency versions;
- Bruker acquisition parameters;
- every applied or skipped preprocessing step;
- parameter values;
- spectral window and removed regions;
- bin definition and integration method;
- normalization method;
- metadata matching rules;
- outlier method and removal decisions;
- feature source and model hyperparameters;
- holdout and cross-validation design;
- random seed;
- class counts and missing-data handling.

Recommended archive
-------------------

For each study archive:

- processing log;
- binned and normalized matrices;
- aligned sample table;
- outlier table;
- EDA tables;
- ML predictions and model history;
- exact source-code release;
- manuscript analysis script or locked parameter record.

Version control
---------------

Use tagged semantic versions, for example ``v0.1.0``. Do not change scientific
defaults silently between releases.
