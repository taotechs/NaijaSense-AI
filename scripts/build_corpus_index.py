"""Build corpus_index.json from the full review_corpus.jsonl (5k+ rows)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index data/processed/review_corpus.jsonl for Task A/B retrieval."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from scripts.ensure_large_corpus import ensure_corpus_index

    path = ensure_corpus_index(force=args.force)
    print(f"Corpus index ready (source: {path})")


if __name__ == "__main__":
    main()
