"""Shared schema for normalized review corpus rows."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NormalizedReviewRecord(BaseModel):
    source: str = Field(..., description="Dataset source e.g. yelp, amazon, goodreads")
    user_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    item_name: str = Field(..., min_length=1)
    item_domain: str = Field(default="general")
    text: str = Field(..., min_length=1)
    rating: float = Field(..., ge=1.0, le=5.0)

