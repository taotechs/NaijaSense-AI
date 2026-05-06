"""Build a normalized review corpus from available public datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.normalize import normalize_amazon, normalize_goodreads, normalize_yelp
from data_pipeline.schema import NormalizedReviewRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized review corpus JSONL.")
    parser.add_argument("--output", default="data/processed/review_corpus.jsonl")
    parser.add_argument("--limit", type=int, default=500, help="Rows per source (best effort).")
    parser.add_argument("--use_hf", action="store_true", help="Pull datasets from HuggingFace hub.")
    args = parser.parse_args()

    records: List[NormalizedReviewRecord] = []
    if args.use_hf:
        records.extend(_fetch_hf_records(limit=args.limit))

    # Always append tiny local seed corpus to ensure non-empty behavior.
    records.extend(_local_seed_records())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} normalized records to {output_path}")


def _fetch_hf_records(limit: int) -> List[NormalizedReviewRecord]:
    try:
        from datasets import load_dataset
    except Exception:
        print("datasets package unavailable; skipping HuggingFace ingest.")
        return []

    out: List[NormalizedReviewRecord] = []
    specs: list[tuple[str, str, Callable[..., Iterable[dict]], Callable[[dict], NormalizedReviewRecord]]] = [
        ("yelp_review_full", "train", load_dataset, normalize_yelp),
        ("amazon_polarity", "train", load_dataset, normalize_amazon),
    ]
    for dataset_name, split, loader, normalizer in specs:
        try:
            ds = loader(dataset_name, split=split)
            for row in ds.select(range(min(limit, len(ds)))):
                try:
                    rec = normalizer(dict(row))
                    if rec.text:
                        out.append(rec)
                except Exception:
                    continue
            print(f"Ingested {dataset_name}: {len(out)} cumulative records.")
        except Exception:
            print(f"Could not ingest {dataset_name}; continuing.")
            continue

    # Goodreads often needs custom source, so provide graceful fallback seed rows.
    out.extend(
        [
            normalize_goodreads(
                {
                    "user_id": "gr_u_1",
                    "book_id": "gr_b_1",
                    "title": "Half of a Yellow Sun",
                    "review_text": "Emotional story with powerful character development.",
                    "rating": 5,
                }
            )
        ]
    )
    return out


def _local_seed_records() -> List[NormalizedReviewRecord]:
    return [
        normalize_yelp(
            {
                "user_id": "y_seed_1",
                "business_id": "b_amala_1",
                "name": "Iya Aladura Amala Spot",
                "text": "Amala and gbegiri were rich and tasty, portion was fair.",
                "stars": 4.0,
            }
        ),
        normalize_yelp(
            {
                "user_id": "y_seed_2",
                "business_id": "b_buka_2",
                "name": "Budget Buka",
                "text": "Affordable meal but service was slow and soup arrived cold.",
                "stars": 2.0,
            }
        ),
        normalize_amazon(
            {
                "user_id": "a_seed_1",
                "asin": "earbud_1",
                "title": "Budget Earbuds",
                "text": "Battery life impressed me and bass was clean for the price.",
                "rating": 4.5,
                "category": "tech",
            }
        ),
    ]


if __name__ == "__main__":
    main()

