Validation and testing
======================

Automated smoke test
--------------------

``tests/smoke_test.py`` creates synthetic FIDs, runs the preprocessing
functions, aligns synthetic clinical metadata, calculates EDA outputs, screens
possible outliers, and trains the supported classifiers.

.. code-block:: bash

   python tests/smoke_test.py

Synthetic demonstration-data validation
---------------------------------------

``tests/test_demo_data.py`` checks that the disease-neutral demonstration data
contain:

- 16 ``fid`` files;
- 16 ``acqus`` files;
- 16 unique FID SHA-256 hashes;
- 8 ``Class_A`` and 8 ``Class_B`` rows;
- exact agreement between clinical ``study_id`` values and NMR sample folders.

.. code-block:: bash

   python tests/test_demo_data.py

Generic clinical-variable validation
------------------------------------

``tests/test_generic_clinical_variable.py`` verifies that an arbitrarily named
numeric variable can be used for selected-variable modelling, cutoff-based
subsetting, explicit clinical-predictor selection, and the optional
threshold-rule benchmark without PSA- or disease-specific assumptions.

.. code-block:: bash

   python tests/test_generic_clinical_variable.py

Continuous integration
----------------------

GitHub Actions:

- compile all Python modules;
- run the smoke test;
- run the synthetic demonstration-data validation;
- run the generic clinical-variable validation;
- test Python 3.10, 3.11, and 3.12;
- build the Sphinx documentation with warnings treated as errors.

Scientific validation still required
------------------------------------

Automated tests establish software functionality and internal consistency.
They do not establish biological, analytical, diagnostic, or clinical
validity. Applied studies should additionally consider trusted-workflow
comparison, parameter sensitivity, manual spectral review, batch effects,
sample-ID and label checks, multiple testing, data leakage, and independent or
nested model validation.
