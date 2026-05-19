"""
Build a 10k+ row JSONL corpus and lightweight inverted index for fast Task A/B retrieval.

Usage:
  python scripts/build_large_corpus.py --rows 10000
  python scripts/build_large_corpus.py --rows 12000 --output data/large_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import iter_corpus_rows

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


def _price_tier_for_row(row: Dict[str, Any]) -> str:
    blob = f"{row.get('text', '')} {row.get('item_name', '')}".lower()
    if any(k in blob for k in ("premium", "luxury", "lekki", "splurge", "expensive")):
        return "premium"
    if any(k in blob for k in ("budget", "cheap", "student", "campus", "affordable", "10k")):
        return "budget"
    rating = float(row.get("rating", 3.5))
    if rating <= 2.5:
        return "mid"
    return "mid"


def _enrich_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    domain = str(row.get("item_domain", row.get("domain", "general")))
    tags = row.get("tags")
    if not isinstance(tags, list):
        tags = list(_terms(f"{row.get('item_name', '')} {row.get('text', '')} {domain}"))
    out = {
        "source": row.get("source", "corpus"),
        "user_id": row.get("user_id", f"user_{idx % 500}"),
        "item_id": row.get("item_id") or f"item_{idx}",
        "item_name": row.get("item_name", "Item"),
        "item_domain": domain,
        "text": row.get("text", ""),
        "rating": float(row.get("rating", 3.5)),
        "price_tier": row.get("price_tier") or _price_tier_for_row(row),
        "tags": tags[:12],
    }
    return out


def _seed_rows(seed_path: Path) -> Iterator[Dict[str, Any]]:
    if seed_path.exists():
        yield from iter_jsonl(seed_path)
    offline = PROJECT_ROOT / "data" / "offline_review_samples.jsonl"
    if offline.exists():
        yield from iter_jsonl(offline)


def _synthetic_variants(base: Dict[str, Any], n: int, rng: random.Random) -> Iterator[Dict[str, Any]]:
    locations = ["Yaba, Lagos", "VI, Lagos", "Abuja Wuse", "Port Harcourt", "Ikeja"]
    prefixes = [
        "Paid about ₦{price} - ",
        "For the money at ₦{price}, ",
        "Honestly at ₦{price} ",
    ]
    suffixes = [
        " Worth it if you're nearby.",
        " I'd repeat on a weekday.",
        " Queue was manageable.",
        " Portion could be bigger though.",
    ]
    text = str(base.get("text", ""))
    for i in range(n):
        price = rng.choice([1200, 1500, 2000, 2500, 3000, 4500, 8000])
        prefix = rng.choice(prefixes).format(price=price)
        suffix = rng.choice(suffixes)
        variant = dict(base)
        variant["user_id"] = f"{base.get('user_id', 'u')}_v{i}"
        variant["item_id"] = f"{base.get('item_id', 'item')}_v{i}"
        variant["text"] = prefix + text[:180] + suffix
        variant["rating"] = round(
            max(1.0, min(5.0, float(base.get("rating", 3.5)) + rng.uniform(-0.4, 0.4))),
            1,
        )
        variant["location_hint"] = rng.choice(locations)
        yield variant


def build_corpus(*, rows: int, output: Path, seed_path: Path) -> int:
    rng = random.Random(42)
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds = [_enrich_row(r, i) for i, r in enumerate(_seed_rows(seed_path))]
    if not seeds:
        raise SystemExit("No seed rows found - add data/processed/review_corpus.jsonl first.")

    written = 0
    with output.open("w", encoding="utf-8") as handle:
        idx = 0
        while written < rows:
            base = seeds[idx % len(seeds)]
            batch = list(_synthetic_variants(base, min(8, rows - written), rng))
            for row in batch:
                row = _enrich_row(row, written)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if written >= rows:
                    break
            idx += 1
    return written


def build_index(corpus_path: Path, index_path: Path, *, max_rows: int = 12_000) -> None:
    """Write compact postings index (terms → row indices) for sub-2.5s lookups."""
    rows: List[Dict[str, Any]] = []
    postings: Dict[str, List[int]] = {}

    count = 0
    for idx, row in enumerate(iter_corpus_rows(corpus_path)):
        if count >= max_rows:
            break
        count += 1
        compact = {
            "item_id": row.get("item_id"),
            "item_name": row.get("item_name"),
            "item_domain": row.get("item_domain"),
            "text": str(row.get("text", ""))[:280],
            "rating": row.get("rating"),
            "price_tier": row.get("price_tier", "mid"),
            "tags": row.get("tags", []),
        }
        rows.append(compact)
        for term in _terms(
            f"{compact['item_name']} {compact['text']} {compact['item_domain']} "
            f"{' '.join(compact.get('tags') or [])}"
        ):
            postings.setdefault(term, []).append(idx)
            if len(postings[term]) > 400:
                postings[term] = postings[term][-400:]

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps({"rows": rows, "postings": postings}, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build large_corpus.jsonl + corpus_index.json")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--output", default="data/large_corpus.jsonl")
    parser.add_argument("--index", default="data/processed/corpus_index.json")
    parser.add_argument("--seed", default="data/processed/review_corpus.jsonl")
    args = parser.parse_args()

    out = PROJECT_ROOT / args.output
    seed = PROJECT_ROOT / args.seed
    count = build_corpus(rows=args.rows, output=out, seed_path=seed)
    print(f"Wrote {count} rows -> {out}")
    build_index(out, PROJECT_ROOT / args.index)
    print(f"Index -> {args.index}")


if __name__ == "__main__":
    main()
