"""Tests for production corpus ensure script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ensure_large_corpus import corpus_is_ready, count_jsonl_lines, ensure_corpus_index


def test_count_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert count_jsonl_lines(path) == 2


def _review_line(i: int) -> str:
    return json.dumps(
        {
            "source": "yelp",
            "user_id": f"u{i}",
            "item_id": f"item_{i}",
            "item_name": f"Test Spot {i}",
            "item_domain": "food",
            "text": f"Budget jollof spot number {i} in Lagos.",
            "rating": 4.0,
        }
    )


def test_ensure_indexes_review_corpus(tmp_path: Path, monkeypatch) -> None:
    seed = tmp_path / "seed.jsonl"
    seed.write_text("\n".join(_review_line(i) for i in range(25)) + "\n", encoding="utf-8")
    index = tmp_path / "index.json"
    monkeypatch.setattr("utils.config.settings.review_corpus_path", str(seed))
    monkeypatch.setattr("utils.config.settings.large_corpus_path", str(seed))
    monkeypatch.setattr("utils.config.settings.corpus_index_path", str(index))

    ensure_corpus_index(force=True)
    assert index.exists()
    assert count_jsonl_lines(seed) == 25
    assert corpus_is_ready(corpus_path=seed, index_path=index, min_rows=20)
