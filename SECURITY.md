# Security and confidential-data policy

## Supported versions

Security and privacy fixes are applied to the latest tagged release.

## Reporting

Do not report vulnerabilities involving confidential clinical data in a public issue. Contact the maintainers privately.

## Clinical and patient data

NMRMetaboWizard can process clinical metadata and raw NMR files, but the public repository must not contain:

- names, dates of birth, addresses, or medical-record identifiers;
- real clinical spreadsheets;
- raw patient FIDs unless formally approved for public release;
- access tokens, passwords, or private server addresses.

Use coded sample IDs and follow institutional ethics and data-management requirements.

## Temporary files

Uploaded ZIP archives are extracted to a system temporary directory by the current application. Deployers should configure operating-system cleanup and access controls appropriate to their environment.
