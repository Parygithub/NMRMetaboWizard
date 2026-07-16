# Contributing to NMRMetaboWizard

Thank you for considering a contribution.

## Before opening an issue

1. Search existing issues.
2. Confirm that the latest tagged release is being used.
3. Remove all patient identifiers and confidential data.
4. Include the operating system, Python version, and error traceback.
5. Describe the smallest reproducible example.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python tests/smoke_test.py
```

## Pull requests

- Create a focused branch.
- Add or update tests for changed scientific functionality.
- Update the relevant documentation page.
- Do not commit real clinical data or raw patient NMR files.
- Run the smoke test and documentation build before submission.

## Scientific changes

Changes to preprocessing, statistics, outlier criteria, or ML evaluation must include:

- a clear mathematical or algorithmic description;
- the reason for the change;
- a synthetic or public test case;
- expected effects on existing outputs;
- documentation of backward compatibility.

## Coding style

- Use descriptive names.
- Prefer small testable functions.
- Keep UI code separate from scientific computation where possible.
- Avoid silently changing defaults.
