"""Tests for scripts/generate_corpus.py output."""

import json
from pathlib import Path

from scripts.generate_corpus import build_corpus


def test_build_corpus_shape() -> None:
    payload = build_corpus(per_domain=10, seed=1)
    assert payload["total_rows"] == 30
    assert payload["domains"] == {"yelp": 10, "amazon": 10, "goodreads": 10}
    records = payload["records"]
    yelp = [r for r in records if r["source"] == "yelp"]
    assert yelp[0]["domain_record"]["business_name"]
    assert yelp[0]["domain_record"]["category"] in ("Food", "Drinks")
    amazon = [r for r in records if r["source"] == "amazon"]
    assert "durability_rating" in amazon[0]["domain_record"]
    assert "price" in amazon[0]["domain_record"]
    gr = [r for r in records if r["source"] == "goodreads"]
    assert gr[0]["domain_record"]["narrative_style"]
    assert gr[0]["domain_record"]["thematic_tags"]


def test_large_corpus_json_loads_via_iter(tmp_path: Path) -> None:
    from core.data_loader import iter_corpus_rows

    payload = build_corpus(per_domain=5, seed=2)
    path = tmp_path / "large_corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = list(iter_corpus_rows(path))
    assert len(rows) == 15
    assert all(r.get("item_name") and r.get("text") for r in rows)
