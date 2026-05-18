"""Tests for streaming corpus loader, index, and timeout fallbacks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from core.corpus_index import (
    LargeCorpusIndex,
    infer_price_tier_constraint,
    row_violates_tier,
)
from core.data_loader import iter_jsonl, resolve_corpus_path, run_with_timeout, scan_corpus


@pytest.fixture()
def mini_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "mini.jsonl"
    rows = [
        {
            "item_id": "food_1",
            "item_name": "Campus Jollof Kitchen",
            "item_domain": "food",
            "text": "Cheap jollof for students near Yaba campus. ₦1500 plate.",
            "rating": 4.2,
            "price_tier": "budget",
            "tags": ["student", "jollof", "budget"],
        },
        {
            "item_id": "lux_1",
            "item_name": "Premium VI Seafood Lounge",
            "item_domain": "food",
            "text": "Luxury dining experience on the Island. Splurge night.",
            "rating": 4.8,
            "price_tier": "premium",
            "tags": ["premium", "seafood", "vi"],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_iter_jsonl_streaming(mini_corpus: Path) -> None:
    loaded = list(iter_jsonl(mini_corpus))
    assert len(loaded) == 2
    assert loaded[0]["item_id"] == "food_1"


def test_run_with_timeout_returns_default() -> None:
    def slow() -> str:
        time.sleep(3)
        return "late"

    out = run_with_timeout(slow, timeout_sec=0.1, default="fallback")
    assert out == "fallback"


def test_scan_corpus_top_k(mini_corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.data_loader.resolve_corpus_path",
        lambda: mini_corpus,
    )
    hits = scan_corpus(
        score_fn=lambda r: 1.0 if "student" in r.get("text", "").lower() else 0.1,
        limit=1,
        path=mini_corpus,
        timeout_sec=2.0,
    )
    assert hits
    assert "student" in hits[0][1]["text"].lower()


def test_budget_tier_filters_premium() -> None:
    assert infer_price_tier_constraint(
        location="Yaba",
        interests=["food"],
        context="cheap campus lunch",
    ) == "budget"
    row = {"price_tier": "premium"}
    assert row_violates_tier(row, "budget") is True


def test_corpus_index_few_shots(mini_corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.corpus_index.resolve_corpus_path",
        lambda: mini_corpus,
    )
    idx = LargeCorpusIndex(corpus_path=str(mini_corpus), index_path=str(mini_corpus.parent / "noidx.json"))
    shots = idx.search_few_shots(
        profile_terms={"student", "food", "yaba"},
        product_name="Jollof",
        product_context="campus lunch budget",
        k=2,
    )
    assert len(shots) >= 1
    assert any("student" in s.get("text", "").lower() or "jollof" in s.get("item_name", "").lower() for s in shots)


def test_task_b_student_excludes_premium(mini_corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.corpus_index.resolve_corpus_path",
        lambda: mini_corpus,
    )
    idx = LargeCorpusIndex(corpus_path=str(mini_corpus), index_path=str(mini_corpus.parent / "noidx.json"))
    pool = idx.retrieve_candidates(
        interests=["food"],
        context="cheap student campus",
        location="Yaba",
        limit=5,
    )
    titles = [p[0].title.lower() for p in pool]
    assert not any("premium" in t for t in titles)
