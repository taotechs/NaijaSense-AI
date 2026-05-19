"""Export docs/SOLUTION_PAPER.md to DOCX (and optional team-branded copy).

Requires: pip install markdown html2docx
Optional PDF: pip install docx2pdf (needs Microsoft Word on Windows) or use pandoc.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from html2docx import html2docx

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "SOLUTION_PAPER.md"
DOCX_PATH = ROOT / "docs" / "SOLUTION_PAPER.docx"
TEAM_DOCX = ROOT / "docs" / "TEAM TAOTECH SOLUTIONS SOLUTION_PAPER.docx"
TEAM_PDF = ROOT / "docs" / "TEAM TAOTECH SOLUTIONS SOLUTION_PAPER.pdf"


def md_to_docx_bytes(md_text: str) -> bytes:
    html = markdown.markdown(
        md_text,
        extensions=[
            "markdown.extensions.extra",
            "markdown.extensions.nl2br",
            "markdown.extensions.sane_lists",
            "markdown.extensions.tables",
        ],
    )
    return html2docx(html, "NaijaSense AI - Solution Paper").getvalue()


def write_docx(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print(f"Wrote {path} ({len(data)} bytes)")


def try_pdf_from_docx(docx: Path, pdf: Path) -> bool:
    """Best-effort PDF export."""
    try:
        from docx2pdf import convert  # type: ignore[import-untyped]

        convert(str(docx), str(pdf))
        print(f"Wrote {pdf} via docx2pdf")
        return True
    except Exception as exc:
        print(f"docx2pdf skipped: {exc}", file=sys.stderr)

    pandoc = shutil.which("pandoc")
    if pandoc:
        for engine in ("wkhtmltopdf", "weasyprint", "pdflatex", "xelatex"):
            try:
                cmd = [pandoc, str(MD_PATH), "-o", str(pdf)]
                if engine != "pdflatex":
                    cmd.extend([f"--pdf-engine={engine}"])
                subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
                print(f"Wrote {pdf} via pandoc ({engine})")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        try:
            subprocess.run(
                [pandoc, str(docx), "-o", str(pdf)],
                check=True,
                cwd=ROOT,
                capture_output=True,
            )
            print(f"Wrote {pdf} via pandoc (docx)")
            return True
        except subprocess.CalledProcessError:
            pass

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(pdf.parent), str(docx)],
            check=True,
            cwd=ROOT,
        )
        generated = docx.with_suffix(".pdf")
        if generated.exists() and generated != pdf:
            generated.replace(pdf)
        if pdf.exists():
            print(f"Wrote {pdf} via LibreOffice")
            return True
    return False


def main() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    data = md_to_docx_bytes(text)
    write_docx(DOCX_PATH, data)
    write_docx(TEAM_DOCX, data)
    if not try_pdf_from_docx(TEAM_DOCX, TEAM_PDF):
        print(
            "PDF not generated automatically. Open the team DOCX in Word → Save as PDF.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
