Validation and testing
======================

Automated smoke test
--------------------

``tests/smoke_test.py`` creates synthetic FIDs, runs the preprocessing
functions, aligns synthetic clinical metadata, calculates EDA outputs,
screens outliers, and trains supported classifiers.

Run::

   python tests/smoke_test.py

Continuous integration
----------------------

GitHub Actions workflows:

- compile all Python modules;
- run the smoke test;
- build the Sphinx documentation.

Scientific validation still required
------------------------------------

Automated tests establish software functionality, not biological or clinical
validity. A publication should additionally provide:

- comparison with a trusted processing workflow;
- parameter-sensitivity analysis;
- independent or nested validation for supervised models;
- manual review of representative spectra;
- consistency checks for sample IDs and class labels.
