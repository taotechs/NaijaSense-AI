"""Parse a single comprehensive Task B persona narrative into retrieval signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from core.nigerian_defaults import apply_cold_start_interests

_DOMAIN_HINTS = {
    "food": ("food", "restaurant", "jollof", "suya", "buka", "eat", "dining", "amala"),
    "movies": ("movie", "nollywood", "film", "cinema", "watch", "series", "entertainment"),
    "drinks": ("drink", "smoothie", "bar", "tea", "coffee", "juice", "cocktail"),
    "tech": ("tech", "gadget", "phone", "laptop", "power bank", "earbuds"),
    "wellness": ("wellness", "spa", "yoga", "fitness", "relax"),
    "fashion": ("fashion", "ankara", "thrift", "style", "shopping"),
    "books": ("book", "read", "literature", "novel"),
    "experiences": ("weekend", "experience", "outing", "market", "social"),
}

_LOCATION_RE = re.compile(
    r"\b(lagos|yaba|ikeja|vi|victoria island|abuja|port harcourt|ph|lekki|surulere|mainland)\b",
    re.I,
)


@dataclass
class ParsedPersona:
    narrative: str
    location: str
    interests: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    budget_sensitive: bool = False
    cold_start: bool = False


def parse_task_b_persona(persona_text: str, *, user_id: str = "") -> ParsedPersona:
    """Derive structured signals from one persona blob (no separate query field)."""
    narrative = (persona_text or "").strip()
    lower = narrative.lower()

    location = "Lagos, Nigeria"
    loc_match = _LOCATION_RE.search(narrative)
    if loc_match:
        location = loc_match.group(0).title()
        if "vi" in location.lower():
            location = "Victoria Island, Lagos"

    domains: List[str] = []
    for domain, hints in _DOMAIN_HINTS.items():
        if any(h in lower for h in hints):
            domains.append(domain)

    interests: Set[str] = set(domains)
    for token in re.findall(r"[a-z]{3,}", lower):
        if token in (
            "student",
            "budget",
            "street",
            "social",
            "spicy",
            "tech",
            "movies",
            "food",
            "drinks",
        ):
            interests.add(token)

    if "value" in lower or "money" in lower:
        interests.add("value for money")

    interest_list = list(interests)
    if not interest_list:
        interest_list, _ = apply_cold_start_interests([])
    else:
        interest_list, _ = apply_cold_start_interests(interest_list)

    budget_sensitive = any(
        k in lower
        for k in (
            "student",
            "budget",
            "cheap",
            "affordable",
            "low income",
            "tight budget",
            "10k",
            "campus",
            "cannot afford",
            "save money",
        )
    )

    cold_start = len(narrative) < 40 or (not domains and not interests)

    return ParsedPersona(
        narrative=narrative,
        location=location,
        interests=interest_list,
        domains=domains or interest_list[:4],
        budget_sensitive=budget_sensitive,
        cold_start=cold_start,
    )
