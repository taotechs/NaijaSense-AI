"""Parse a single comprehensive Task B persona narrative into retrieval signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from core.nigerian_defaults import apply_cold_start_interests
from core.task_b_persona_intent import infer_persona_intent

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
    team_culture_mode: bool = False
    retrieval_context: str = ""


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

    intent = infer_persona_intent(narrative, base_domains=[], base_interests=[])
    domains = list(intent.domains)
    interests: Set[str] = set(intent.interests)

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
        team_culture_mode=intent.team_culture_mode,
        retrieval_context=intent.retrieval_context,
    )
