"""
Ensure corpus_index.json exists and indexes the full review_corpus.jsonl.

Task A (few-shots) and Task B (stage-1 retrieval) both read from the same
5k+ row corpus via the inverted index - no separate 3k synthetic pool.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import settings


def count_corpus_rows(path: Path, *, stop_at: int | None = None) -> int:
    if not path.exists():
        return 0
    if path.suffix.lower() == ".json":
        return _count_json_records(path, stop_at=stop_at)
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
            if stop_at is not None and count >= stop_at:
                break
    return count


def _count_json_records(path: Path, *, stop_at: int | None = None) -> int:
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if isinstance(data, list):
        n = len(data)
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        n = len(data["records"])
    else:
        return 0
    if stop_at is not None:
        return min(n, stop_at) if n >= stop_at else n
    return n


def corpus_is_ready(
    *,
    corpus_path: Path | None = None,
    index_path: Path | None = None,
    min_rows: int | None = None,
) -> bool:
    corpus_path = corpus_path or Path(settings.review_corpus_path)
    index_path = index_path or Path(settings.corpus_index_path)
    min_rows = min_rows if min_rows is not None else int(os.getenv("CORPUS_MIN_ROWS", "4000"))

    if not corpus_path.exists() or not index_path.exists():
        return False
    if count_corpus_rows(corpus_path, stop_at=min_rows) < min_rows:
        return False
    min_index_bytes = max(1000, min_rows * 12)
    if index_path.stat().st_size < min_index_bytes:
        return False
    try:
        import json

        payload = json.loads(index_path.read_text(encoding="utf-8"))
        indexed = len(payload.get("rows") or [])
        if indexed < max(1, min_rows - 200):
            return False
    except Exception:
        return False
    return True


def ensure_corpus_index(*, force: bool = False) -> Path:
    """Build inverted index over the full review_corpus.jsonl."""
    from scripts.build_large_corpus import build_index

    corpus_path = Path(settings.review_corpus_path)
    index_path = Path(settings.corpus_index_path)

    if not force and corpus_is_ready(corpus_path=corpus_path, index_path=index_path):
        n = count_corpus_rows(corpus_path)
        print(f"[corpus] OK - {n} rows indexed at {index_path}")
        return corpus_path

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Review corpus missing at {corpus_path}. "
            "Commit data/processed/review_corpus.jsonl or run build_review_corpus.py."
        )

    n = count_corpus_rows(corpus_path)
    print(f"[corpus] Indexing {n} rows from {corpus_path} -> {index_path}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    build_index(corpus_path, index_path, max_rows=max(n + 500, 6000))
    print("[corpus] Done.")
    return corpus_path


def ensure_large_corpus(*, rows: int | None = None, force: bool = False) -> Path:
    """Backward-compatible entrypoint (``rows`` ignored - uses full review corpus)."""
    _ = rows
    return ensure_corpus_index(force=force)


# Legacy alias for tests
count_jsonl_lines = count_corpus_rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ensure corpus index over review_corpus.jsonl.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_corpus_index(force=args.force)


if __name__ == "__main__":
    main()
