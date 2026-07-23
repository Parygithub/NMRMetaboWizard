Overview
========

Purpose
-------

NMRMetaboWizard connects raw Bruker free induction decay (FID) processing, clinical metadata,
exploratory data analysis (EDA), outlier screening, and machine learning (ML) in an
interactive interface. It is designed for researchers who need an auditable
workflow but do not want to write code for every processing step.

Workflow groups
---------------

The interface contains four groups:

1. **Preprocessing** - transforms FIDs and spectra into a feature matrix.
2. **Clinical labels** - aligns metadata using sample identifiers.
3. **EDA and outliers** - explores the matrix and supports reviewed exclusions.
4. **Machine learning** - compares predictive models and feature sources.

Design principles
-----------------

- Stepwise inspection rather than hidden batch processing.
- Optional preprocessing steps can be skipped.
- Spectral features, clinical metadata, and labels remain conceptually separate.
- Resulting figures and tables can be downloaded.
- ML preprocessing is fitted inside the model pipeline.
- Outlier detection flags samples for review rather than deleting them automatically.

Target users
------------

The software is intended for NMR metabolomics researchers, analytical
chemists, clinical collaborators, students, and data analysts.
