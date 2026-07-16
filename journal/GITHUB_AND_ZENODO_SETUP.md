# Create the GitHub repository and permanent links

An actual GitHub link cannot be created without access to the author's GitHub
account. Use the steps below.

## 1. Create the repository

On GitHub, create a new **empty** repository named:

```text
NMRMetaboWizard
```

Do not initialize it with a README, license, or `.gitignore`, because those
files are already included.

The final link will be:

```text
https://github.com/Parygithub/NMRMetaboWizard
```

## 2. Push this package

```bash
cd NMRMetaboWizard_journal_github_repository
git init
git add .
git status
git commit -m "Initial journal-ready NMRMetaboWizard release"
git branch -M main
git remote add origin https://github.com/Parygithub/NMRMetaboWizard.git
git push -u origin main
```

Inspect `git status` before committing. No real NMR ZIP, clinical Excel file,
patient data, or result table should be staged.

## 3. Replace placeholders

Run:

```bash
python scripts/replace_repository_placeholders.py Parygithub
```

Then review and commit the changes.

## 4. Create documentation

Import the GitHub repository at Read the Docs. The repository contains
`.readthedocs.yaml`. The expected documentation URL is:

```text
https://nmrmetabowizard.readthedocs.io
```

The exact Read the Docs slug may differ.

## 5. Create the first release

```bash
git tag -a v0.1.0 -m "NMRMetaboWizard v0.1.0"
git push origin v0.1.0
```

Create a GitHub Release from `v0.1.0`.

## 6. Archive with Zenodo

Connect Zenodo to GitHub, enable the repository, and create the release.
Replace the placeholder DOI throughout the repository and manuscript.
