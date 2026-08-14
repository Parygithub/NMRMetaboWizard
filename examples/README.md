# Synthetic example data

This folder contains fully synthetic, disease-neutral data for demonstrating NMRMetaboWizard. The files contain no patient data, measured biological spectra, or personal information.

## Files

- `demo_cohort_bruker.zip`  
  Contains 16 synthetic Bruker-like one-dimensional ^1H NMR FIDs:
  - 8 `Class_A` samples
  - 8 `Class_B` samples

- `demo_clinical_metadata.csv`  
  Contains matching synthetic metadata, including:
  - `study_id`
  - `Class`
  - age
  - `biomarker`
  - height
  - weight
  - creatinine
  - `clinical_measure`

The `study_id` values match the sample-folder names in the Bruker ZIP archive.

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

6. In EDA, select `biomarker` or `clinical_measure` to test the optional generic cutoff plots.
7. In ML, select either variable for selected-variable modelling or the optional threshold-rule benchmark.

## Important note

These data are provided only to demonstrate the software workflow. They must not be used for biological interpretation, biomarker discovery, diagnostic evaluation, or clinical conclusions.
