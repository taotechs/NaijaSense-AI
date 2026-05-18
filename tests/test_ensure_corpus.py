"""Tests for production corpus ensure script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ensure_large_corpus import corpus_is_ready, count_jsonl_lines, ensure_large_corpus


def test_count_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert count_jsonl_lines(path) == 2


def test_ensure_builds_from_seed(tmp_path: Path, monkeypatch) -> None:
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        json.dumps(
            {
                "item_id": "x1",
                "item_name": "Test Spot",
                "item_domain": "food",
                "text": "Budget jollof for students.",
                "rating": 4.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "large.jsonl"
    index = tmp_path / "index.json"
    monkeypatch.setenv("CORPUS_BUILD_ROWS", "20")
    monkeypatch.setenv("CORPUS_MIN_ROWS", "15")
    monkeypatch.setattr("utils.config.settings.review_corpus_path", str(seed))
    monkeypatch.setattr("utils.config.settings.large_corpus_path", str(corpus))
    monkeypatch.setattr("utils.config.settings.corpus_index_path", str(index))

    ensure_large_corpus(rows=20, force=True)
    assert corpus.exists()
    assert index.exists()
    assert count_jsonl_lines(corpus) >= 15
    assert corpus_is_ready(corpus_path=corpus, index_path=index, min_rows=15)
