"""One-off: replace Unicode em dash (U+2014) with ASCII hyphen in text files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    "dist",
    "build",
}
SKIP_SUFFIXES = {
    ".pdf",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pyc",
    ".whl",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".ts",
    ".tsx",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".example",
    ".env",
    ".html",
    ".css",
    ".sh",
    ".ps1",
}


def should_process(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    if path.suffix in TEXT_SUFFIXES or path.name in (".env.example", "Dockerfile"):
        return True
    if path.name == "Dockerfile":
        return True
    return False


def main() -> int:
    changed = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or not should_process(path):
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "\u2014" not in raw:
            continue
        new = raw.replace("\u2014", " - ")
        path.write_text(new, encoding="utf-8", newline="\n")
        changed += 1
        print(path.relative_to(ROOT))
    print(f"Updated {changed} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
