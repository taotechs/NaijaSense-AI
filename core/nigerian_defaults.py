"""Nigerian-specific defaults for cold-start and cross-domain recommendation."""

from __future__ import annotations

from typing import List

# Baseline interests when a new user has no history (cold-start).
COLD_START_INTERESTS: List[str] = [
    "street food",
    "value for money",
    "local experiences",
    "mobile gadgets",
]

# Curated cross-domain pool - food, tech, entertainment, wellness.
NIGERIAN_CANDIDATE_CATALOG: List[str] = [
    "Iya Eba Amala Spot - Yaba",
    "Suya & Chill Stand - Ikeja",
    "Shawarma Alley - VI",
    "Local Jollof Kitchen - Surulere",
    "Budget USB-C Power Bank",
    "Wireless Earbuds (wallet-friendly)",
    "Nollywood weekend drama pick",
    "African lit paperback + café",
    "Cozy tea corner for de-stress",
    "Weekend buka hopping guide",
    "Mobile data bundle saver plan",
    "Street-style Ankara accessories pop-up",
]


def apply_cold_start_interests(interests: List[str]) -> tuple[List[str], bool]:
    """Merge Nigerian cold-start priors when interests are empty."""
    cleaned = [i.strip() for i in interests if i and i.strip()]
    if cleaned:
        return cleaned, False
    return list(COLD_START_INTERESTS), True


def build_persona_context(
    *,
    location: str | None,
    interests: List[str],
    history: str | None,
    tone_notes: str | None,
    context: str | None,
) -> str:
    """Single blob fed into Reason-Before-Recommend and candidate generation."""
    parts: List[str] = []
    if location:
        parts.append(f"Location: {location}")
    if interests:
        parts.append("Interests: " + ", ".join(interests))
    if history:
        parts.append(f"History: {history.strip()[:2000]}")
    if tone_notes:
        parts.append(f"Tone: {tone_notes.strip()[:800]}")
    if context:
        parts.append(f"Query: {context.strip()[:800]}")
    return " | ".join(parts) if parts else "General Nigerian consumer seeking useful picks."


def candidates_for_persona(
    interests: List[str],
    context: str | None,
    *,
    cross_domain: bool = False,
) -> List[str]:
    """Build a ranked candidate pool from persona + optional query."""
    from core.intent_router import _default_candidates_from_query

    query = context or " ".join(interests) or "weekend ideas Lagos"
    pool = _default_candidates_from_query(query, interests)
    if cross_domain or len(pool) < 6:
        seen = {p.lower() for p in pool}
        for item in NIGERIAN_CANDIDATE_CATALOG:
            if item.lower() not in seen:
                pool.append(item)
                seen.add(item.lower())
    return pool[:12]
