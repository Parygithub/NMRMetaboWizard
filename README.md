# Synthetic example data

This folder contains fully synthetic data for demonstrating the main functions of NMRMetaboWizard. The files are generated for software testing and contain no patient data, measured biological spectra, or personal information.

## Files

- `demo_cohort_bruker.zip`  
  Contains 16 synthetic Bruker-like one-dimensional ^1H NMR FIDs:
  - 8 BPH-labelled samples
  - 8 PCa-labelled samples

- `demo_clinical_metadata.csv`  
  Contains matching synthetic clinical metadata, including:
  - `study_id`
  - `Class`
  - age
  - PSA
  - height
  - weight
  - creatinine
  - prostate volume

The `study_id` values in the clinical file match the sample-folder names in the Bruker ZIP archive.

## How to use the example data

1. Open NMRMetaboWizard.
2. At **Step 1 — Upload ZIP**, upload `demo_cohort_bruker.zip`.
3. Continue through preprocessing, binning, and normalization.
4. At **Step 17 — Clinical labels and metadata**, upload `demo_clinical_metadata.csv`.
5. Use:

   ```text
   Sample ID column: study_id
   Class column: Class
   ```

6. Continue to PCA, PCA loadings, PLS-DA, outlier screening, and machine-learning analysis.

## Important note

These data are provided only to demonstrate the software workflow. They must not be used for biological interpretation, biomarker discovery, diagnostic evaluation, or clinical conclusions.
