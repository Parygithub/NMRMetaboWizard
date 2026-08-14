Input data
==========

Bruker archive
--------------

The importer detects directories containing both ``fid`` and ``acqus``. The
binary FID is decoded using ``BYTORDA`` and ``DTYPA`` and reconstructed from
alternating real and imaginary values.

Recommended layout::

   sample_id_001/
   └── 1/
       ├── fid
       └── acqus

When the experiment directory is numeric, its parent folder is used as the
sample ID. This ID is later matched to the clinical ``study_id`` column.

Clinical table
--------------

Accepted formats:

- CSV
- TSV
- TXT
- XLSX

Required columns:

- ``study_id`` - sample identifier;
- ``Class`` - cohort or outcome label.

Additional columns may contain age, treatment group, biomarker measurements,
tumour characteristics, anthropometric measurements, laboratory values, or
other study-specific metadata.

Clinical variables are not automatically assumed to be valid predictors. In
the ML section, users explicitly select the numeric variables to include.
Variables derived from the outcome, measured after the outcome, or otherwise
capable of leaking the target label should be excluded.

Class labels that differ only by capitalization are standardized to the most
frequent spelling.

Data-quality checks
-------------------

- Remove empty rows and repeated spreadsheet header rows.
- Check duplicated ``study_id`` values.
- Verify unmatched spectral and clinical IDs.
- Confirm that class labels are correct and non-empty.
- Do not upload identifiable patient information to a public deployment.
