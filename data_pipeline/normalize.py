"""Normalization helpers across Yelp/Amazon/Goodreads style rows."""

from __future__ import annotations

from typing import Any, Dict

from data_pipeline.schema import NormalizedReviewRecord


def normalize_yelp(row: Dict[str, Any]) -> NormalizedReviewRecord:
    stars = float(row.get("stars", 3.0))
    return NormalizedReviewRecord(
        source="yelp",
        user_id=str(row.get("user_id", "unknown_user")),
        item_id=str(row.get("business_id", "unknown_item")),
        item_name=str(row.get("name", row.get("business_id", "yelp_item"))),
        item_domain="restaurant",
        text=str(row.get("text", "")).strip(),
        rating=max(1.0, min(5.0, stars)),
    )


def normalize_amazon(row: Dict[str, Any]) -> NormalizedReviewRecord:
    rating = float(row.get("rating", row.get("overall", 3.0)))
    title = str(row.get("title", row.get("product_title", "amazon_item")))
    return NormalizedReviewRecord(
        source="amazon",
        user_id=str(row.get("user_id", row.get("reviewerID", "unknown_user"))),
        item_id=str(row.get("parent_asin", row.get("asin", "unknown_item"))),
        item_name=title,
        item_domain=str(row.get("category", "general")).lower(),
        text=str(row.get("text", row.get("reviewText", ""))).strip(),
        rating=max(1.0, min(5.0, rating)),
    )


def normalize_goodreads(row: Dict[str, Any]) -> NormalizedReviewRecord:
    rating = float(row.get("rating", 3.0))
    return NormalizedReviewRecord(
        source="goodreads",
        user_id=str(row.get("user_id", "unknown_user")),
        item_id=str(row.get("book_id", "unknown_item")),
        item_name=str(row.get("title", "goodreads_book")),
        item_domain="books",
        text=str(row.get("review_text", row.get("text", ""))).strip(),
        rating=max(1.0, min(5.0, rating)),
    )

