Input data
==========

Bruker archive
--------------

The importer detects directories containing both ``fid`` and ``acqus``.
The binary FID is decoded using ``BYTORDA`` and ``DTYPA`` and reconstructed
from alternating real and imaginary values.

Recommended layout::

   patient_or_sample_id/
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

Additional columns may contain age, PSA, creatinine, prostate volume,
Gleason score, or other study-specific variables.

Class labels that differ only by capitalization, such as ``PCa`` and ``Pca``,
are standardized to the most frequent spelling.

Data-quality checks
-------------------

- Remove empty rows and repeated spreadsheet header rows.
- Check duplicated ``study_id`` values.
- Verify unmatched spectral and clinical IDs.
- Do not upload identifiable patient information to a public deployment.
