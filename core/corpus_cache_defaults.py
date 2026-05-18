"""Pre-cached defaults used when corpus scans exceed the query timeout budget."""

from __future__ import annotations

from typing import Any, Dict, List

from core.candidate_catalog import CATALOG, CatalogItem

# Task A — Nigerian few-shot style references (compact; no full corpus load).
DEFAULT_FEW_SHOT_REVIEWS: List[Dict[str, Any]] = [
    {
        "item_name": "Iya Eba Amala Spot",
        "item_domain": "food",
        "rating": 4.5,
        "text": (
            "Paid about ₦2,500 for the plate — fair for the portion. Amala was soft, "
            "egusi rich without being oily. Wait was 20 minutes on a Saturday; still worth it after class."
        ),
        "price_tier": "budget",
        "tags": ["amala", "student", "lagos"],
    },
    {
        "item_name": "Budget USB-C Power Bank",
        "item_domain": "tech",
        "rating": 4.0,
        "text": (
            "Build feels solid for the price band. Battery holds a full workday on campus; "
            "only gripe is the cable in the box is short for desk setup."
        ),
        "price_tier": "budget",
        "tags": ["power", "student", "gadget"],
    },
    {
        "item_name": "Suya & Chill Stand",
        "item_domain": "food",
        "rating": 3.0,
        "text": (
            "Suya was tasty but skewers ran small for ₦2k. Pepper level was accurate; "
            "I'd return only if the queue stays under 15 minutes."
        ),
        "price_tier": "mid",
        "tags": ["suya", "street", "critical"],
    },
    {
        "item_name": "Late-night Shawarma Alley",
        "item_domain": "food",
        "rating": 2.5,
        "text": (
            "Wrap had soggy bread and chicken was overcooked. For ₦3k I expected better "
            "value — might try again only if I'm stuck on the Island late."
        ),
        "price_tier": "mid",
        "tags": ["shawarma", "vi", "late-night"],
    },
]


def default_catalog_items() -> List[CatalogItem]:
    return list(CATALOG)


def default_few_shots(k: int = 2) -> List[Dict[str, Any]]:
    return DEFAULT_FEW_SHOT_REVIEWS[:k]
