from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/replace_repository_placeholders.py GITHUB_USERNAME")

username = sys.argv[1].strip()
if not username:
    raise SystemExit("GitHub username cannot be empty.")

root = Path(__file__).resolve().parents[1]

replacements = {
    "YOUR_GITHUB_USERNAME": username,
}

extensions = {".md", ".rst", ".cff", ".bib", ".yaml", ".yml", ".txt"}

for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix.lower() not in extensions and path.name not in {"README.md"}:
        continue

    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)

    if updated != text:
        path.write_text(updated, encoding="utf-8")
        print(f"Updated {path.relative_to(root)}")
