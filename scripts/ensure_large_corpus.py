"""
Ensure large_corpus.jsonl and corpus_index.json exist (idempotent).

Used by Docker build, container entrypoint, and FastAPI startup so Koyeb
production always has the 10k-scale retrieval pool.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import settings


def count_jsonl_lines(path: Path, *, stop_at: int | None = None) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
            if stop_at is not None and count >= stop_at:
                break
    return count


def corpus_is_ready(
    *,
    corpus_path: Path | None = None,
    index_path: Path | None = None,
    min_rows: int | None = None,
) -> bool:
    corpus_path = corpus_path or Path(settings.large_corpus_path)
    index_path = index_path or Path(settings.corpus_index_path)
    min_rows = min_rows if min_rows is not None else int(os.getenv("CORPUS_MIN_ROWS", "9000"))

    if not corpus_path.exists() or not index_path.exists():
        return False
    if count_jsonl_lines(corpus_path, stop_at=min_rows) >= min_rows:
        return index_path.stat().st_size > 100
    # Fast reject obviously truncated artifacts without scanning all lines.
    if corpus_path.stat().st_size < 100_000 or index_path.stat().st_size < 50_000:
        return False
    return count_jsonl_lines(corpus_path, stop_at=min_rows) >= min_rows


def ensure_large_corpus(*, rows: int | None = None, force: bool = False) -> Path:
    """
    Build corpus + index when missing or below ``CORPUS_MIN_ROWS``.
    Returns path to the JSONL corpus.
    """
    from scripts.build_large_corpus import build_corpus, build_index

    target_rows = rows if rows is not None else int(os.getenv("CORPUS_BUILD_ROWS", "10000"))
    corpus_path = Path(settings.large_corpus_path)
    index_path = Path(settings.corpus_index_path)
    seed_path = Path(settings.review_corpus_path)

    if not force and corpus_is_ready(corpus_path=corpus_path, index_path=index_path):
        print(f"[corpus] OK — {corpus_path} + {index_path} already present.")
        return corpus_path

    if not seed_path.exists():
        raise FileNotFoundError(
            f"Seed corpus missing at {seed_path}. Commit data/processed/review_corpus.jsonl "
            "or set REVIEW_CORPUS_PATH."
        )

    print(f"[corpus] Building {target_rows} rows -> {corpus_path}")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    written = build_corpus(rows=target_rows, output=corpus_path, seed_path=seed_path)
    print(f"[corpus] Wrote {written} rows; building index -> {index_path}")
    build_index(corpus_path, index_path)
    print("[corpus] Done.")
    return corpus_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ensure large corpus + index exist.")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_large_corpus(rows=args.rows, force=args.force)


if __name__ == "__main__":
    main()
