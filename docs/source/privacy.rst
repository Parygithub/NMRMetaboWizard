Privacy and data governance
===========================

NMRMetaboWizard can process clinical data. Public software repositories and
documentation must contain only synthetic, public, or formally de-identified
data.

Before deployment
-----------------

- use institutional authentication when required;
- restrict network access;
- configure secure temporary-file storage and cleanup;
- avoid logging identifiers;
- document retention and backup procedures;
- follow ethics approvals and data-processing agreements.

Before pushing to GitHub
------------------------

Run::

   git status

Confirm that no ZIP, Excel, CSV, raw FID, clinical, or results files are
staged. The repository ``.gitignore`` blocks common data extensions, but
manual review remains essential.
