---
title: 'NMRMetaboWizard: a dynamic Shiny platform for 1H NMR untargeted metabolomics'
tags:
  - Python
  - Shiny
  - NMR
  - metabolomics
  - machine learning
authors:
  - name: Parastoo Vahdatiyekta
    affiliation: 1
    orcid: REPLACE_WITH_ORCID
  - name: Tan-Phat Huynh
    affiliation: 1
    orcid: REPLACE_WITH_ORCID
affiliations:
  - name: Chemistry and Chemical Technology, Faculty of Science and Engineering, Åbo Akademi University, Turku, Finland
    index: 1
date: 16 July 2026
bibliography: paper.bib
---

# Summary

NMRMetaboWizard is an interactive Shiny for Python application for converting
one-dimensional Bruker free induction decays into analysis-ready NMR
metabolomics matrices and linking those matrices to clinical metadata. The
application exposes each preprocessing step, supports interactive quality
control, and integrates exploratory analysis, cohort-aware outlier screening,
and leakage-aware machine learning.

# Statement of need

NMR metabolomics workflows commonly combine vendor software, scripts,
spreadsheets, and statistical tools. This fragmentation makes processing
choices difficult to audit and can create inconsistencies between spectral
processing, metadata matching, and model evaluation. NMRMetaboWizard provides
a no-code, stepwise environment in which optional preprocessing operations can
be inspected, adjusted, or skipped, and where spectral features, class labels,
and clinical metadata remain explicitly separated.

# Functionality

The application supports Bruker FID import, group-delay handling,
solvent-residual suppression, apodization, zero filling, Fourier
transformation, phase correction, referencing, baseline correction, optional
alignment, region removal, binning, normalization, PCA, PLS-DA, clinical
correlations, Hotelling's T² and robust-distance outlier screening, and four
classification algorithms. Imputation, scaling, and optional PCA are fitted
inside scikit-learn pipelines.

# Availability

Source code: https://github.com/Parygithub/NMRMetaboWizard

Documentation: https://nmrmetabowizard.readthedocs.io

Archived release: https://doi.org/10.5281/zenodo.REPLACE_WITH_RECORD

# Acknowledgements

Replace with funding, institutional, clinical, and technical acknowledgements.

# References
