"""Tests for Task B display item resolution."""

from core.recommendation_items import (
    canonical_item_title,
    is_placeholder_item_name,
    merge_unique_items,
    resolve_display_item,
    curated_catalog_items,
)
from core.candidate_catalog import CatalogItem
from core.corpus_index import LargeCorpusIndex


def test_canonical_title_collapses_variants() -> None:
    assert canonical_item_title("Local Jollof Kitchen - Surulere #19") == "Local Jollof Kitchen - Surulere"
    a = CatalogItem("a", "Local Jollof Kitchen - Surulere #1", "food", ("jollof",))
    b = CatalogItem("b", "Local Jollof Kitchen - Surulere #2", "food", ("jollof",))
    merged = merge_unique_items([(a, 0.9), (b, 0.8)], limit=5)
    assert len(merged) == 1


def test_rejects_yelp_placeholder() -> None:
    row = {
        "item_id": "hf_yelp_156_v3",
        "item_name": "yelp_review",
        "item_domain": "restaurant",
        "text": "Best fish sandwich in town, crispy and affordable.",
    }
    item = resolve_display_item(row)
    assert item is not None
    assert item.title == "Crispy Fish Sandwich"
    assert item.domain == "Food"
    assert "yelp" not in item.title.lower()


def test_curated_catalog_has_real_titles() -> None:
    items = curated_catalog_items()
    assert len(items) >= 20
    assert all(not is_placeholder_item_name(i.title) for i in items)


def test_retrieve_candidates_no_placeholders(monkeypatch) -> None:
    idx = LargeCorpusIndex(
        corpus_path="data/processed/review_corpus.jsonl",
        index_path="data/processed/corpus_index.json",
    )
    pool = idx.retrieve_candidates(
        interests=["food", "movies"],
        context="Student in Yaba on a budget, likes jollof and Nollywood.",
        limit=10,
    )
    for item, _ in pool:
        assert not is_placeholder_item_name(item.title)
        assert item.title != "yelp_review"
