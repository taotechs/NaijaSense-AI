"""Export docs/SOLUTION_PAPER.md to docs/SOLUTION_PAPER.docx (requires markdown, html2docx)."""

from __future__ import annotations

from pathlib import Path

import markdown
from html2docx import html2docx

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "SOLUTION_PAPER.md"
OUT_PATH = ROOT / "docs" / "SOLUTION_PAPER.docx"


def main() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    html = markdown.markdown(
        text,
        extensions=[
            "markdown.extensions.extra",
            "markdown.extensions.nl2br",
            "markdown.extensions.sane_lists",
        ],
    )
    buf = html2docx(html, "NaijaSense AI — Solution Paper")
    OUT_PATH.write_bytes(buf.getvalue())
    print(f"Wrote {OUT_PATH} ({len(buf.getvalue())} bytes)")


if __name__ == "__main__":
    main()
