Release and archiving
=====================

GitHub release
--------------

1. Update the version in ``CITATION.cff``, documentation, and changelog.
2. Run tests and build documentation.
3. Commit all changes.
4. Create and push a version tag, for example ``v0.1.0``.
5. Create a GitHub Release from the tag.

Zenodo
------

1. Sign in to Zenodo using GitHub.
2. Enable the NMRMetaboWizard repository.
3. Create the GitHub Release.
4. Zenodo will archive the release and mint a DOI.
5. Add the DOI to ``CITATION.cff``, README, manuscript, and documentation.
6. Create a small follow-up release containing the final DOI metadata if needed.

Read the Docs
-------------

1. Sign in with GitHub.
2. Import the repository.
3. Confirm that ``.readthedocs.yaml`` is detected.
4. Build the ``main`` branch.
5. Add the final HTTPS documentation URL to the manuscript.
