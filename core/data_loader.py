"""
Memory-efficient corpus I/O for 10k+ row evaluation sets.

Never loads the full dataset into RAM: uses line iterators, chunked parquet
reads, and strict wall-clock timeouts so serverless handlers stay within limits.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, TypeVar

from utils.config import settings

T = TypeVar("T")

DEFAULT_QUERY_TIMEOUT_SEC = 2.5


def corpus_paths() -> List[Path]:
    """Ordered preference: committed review corpus (5k+) → legacy fallbacks."""
    candidates = [
        Path(settings.review_corpus_path),
        Path("data/processed/review_corpus.jsonl"),
        Path("data/large_corpus.jsonl"),
        Path("data/large_corpus.json"),
        Path("data/large_corpus.parquet"),
    ]
    seen: set[str] = set()
    ordered: List[Path] = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(p)
    return ordered


def resolve_corpus_path() -> Optional[Path]:
    for path in corpus_paths():
        if path.exists():
            return path
    return None


def run_with_timeout(
    fn: Callable[[], T],
    *,
    timeout_sec: float | None = None,
    default: T,
) -> T:
    """Run ``fn`` in a worker thread; return ``default`` if it exceeds the budget."""
    budget = timeout_sec if timeout_sec is not None else settings.corpus_query_timeout_sec
    if budget <= 0:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=budget)
        except FuturesTimeout:
            return default


def iter_jsonl(path: Path, *, max_rows: int | None = None) -> Generator[Dict[str, Any], None, None]:
    """Stream one JSON object per line without holding the file in memory."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row
                count += 1
                if max_rows is not None and count >= max_rows:
                    break


def iter_corpus_rows(
    path: Path | None = None,
    *,
    chunk_size: int = 512,
) -> Iterator[Dict[str, Any]]:
    """
    Unified row iterator: JSONL (preferred), JSON array (ijson), or parquet chunks.
    """
    resolved = path or resolve_corpus_path()
    if resolved is None or not resolved.exists():
        return iter(())

    suffix = resolved.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(resolved)
        return

    if suffix == ".json":
        # Prefer JSONL for large corpora; only load small array files whole-cloth.
        if resolved.stat().st_size > 50_000_000:
            return iter(())
        data = json.loads(resolved.read_text(encoding="utf-8"))
        rows: List[Any]
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("records"), list):
            rows = data["records"]
        else:
            return
        for row in rows:
            if isinstance(row, dict):
                yield row
        return

    if suffix == ".parquet":
        yield from _iter_parquet_chunks(resolved, chunk_size=chunk_size)
        return

    yield from iter_jsonl(resolved)


def _iter_parquet_chunks(path: Path, *, chunk_size: int) -> Generator[Dict[str, Any], None, None]:
    try:
        import pandas as pd
    except ImportError:
        return

    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            frame = batch.to_pandas()
            for row in frame.to_dict(orient="records"):
                yield _normalize_row(row)
        return
    except Exception:
        pass

    frame = pd.read_parquet(path)
    for row in frame.to_dict(orient="records"):
        yield _normalize_row(row)


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map parquet/CSV columns to our normalized review schema."""
    return {
        "source": str(row.get("source", row.get("dataset", "corpus"))),
        "user_id": str(row.get("user_id", row.get("user", "anon"))),
        "item_id": str(row.get("item_id", row.get("id", ""))),
        "item_name": str(row.get("item_name", row.get("title", "item"))),
        "item_domain": str(row.get("item_domain", row.get("domain", "general"))),
        "text": str(row.get("text", row.get("review", ""))),
        "rating": float(row.get("rating", row.get("stars", 3.5))),
        "price_tier": str(row.get("price_tier", "mid")),
        "tags": row.get("tags") if isinstance(row.get("tags"), list) else [],
    }


def timed_iter(
    rows: Iterator[Dict[str, Any]],
    *,
    deadline: float,
) -> Generator[Dict[str, Any], None, None]:
    """Stop yielding when monotonic clock passes ``deadline``."""
    for row in rows:
        if time.monotonic() >= deadline:
            break
        yield row


def scan_corpus(
    *,
    score_fn: Callable[[Dict[str, Any]], float],
    limit: int,
    path: Path | None = None,
    timeout_sec: float | None = None,
) -> List[tuple[float, Dict[str, Any]]]:
    """
    Single-pass streaming top-k: keeps only ``limit`` best rows in a small heap list.
    """
    budget = timeout_sec if timeout_sec is not None else settings.corpus_query_timeout_sec
    deadline = time.monotonic() + budget
    resolved = path or resolve_corpus_path()

    def _run() -> List[tuple[float, Dict[str, Any]]]:
        if resolved is None:
            return []
        top: List[tuple[float, Dict[str, Any]]] = []
        for row in timed_iter(iter_corpus_rows(resolved), deadline=deadline):
            score = score_fn(row)
            if score <= 0:
                continue
            top.append((score, row))
            top.sort(key=lambda x: x[0], reverse=True)
            if len(top) > limit:
                top = top[:limit]
        return top

    return run_with_timeout(_run, timeout_sec=budget, default=[])
