"""Task B persona intent - avoid false tech/hiring matches and steer retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Set

# Substring-safe domain hints (word boundaries).
_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "food": (
        r"\bfood\b",
        r"\brestaurant\b",
        r"\bjollof\b",
        r"\bsuya\b",
        r"\bbuka\b",
        r"\beat\b",
        r"\bdining\b",
        r"\bamala\b",
    ),
    "movies": (
        r"\bmovie\b",
        r"\bnollywood\b",
        r"\bfilm\b",
        r"\bcinema\b",
        r"\bwatch\b",
        r"\bseries\b",
    ),
    "drinks": (
        r"\bdrink\b",
        r"\bsmoothie\b",
        r"\bbar\b",
        r"\btea\b",
        r"\bcoffee\b",
        r"\bjuice\b",
    ),
    "tech": (
        r"\bgadget\b",
        r"\bphone\b",
        r"\blaptop\b",
        r"\bpower\s*bank\b",
        r"\bearbuds\b",
        r"\bkeyboard\b",
    ),
    "wellness": (r"\bwellness\b", r"\bspa\b", r"\byoga\b", r"\bfitness\b", r"\brelax\b"),
    "fashion": (r"\bfashion\b", r"\bankara\b", r"\bthrift\b", r"\bstyle\b", r"\bshopping\b"),
    "books": (r"\bbook\b", r"\bread\b", r"\bliterature\b", r"\bnovel\b"),
    "experiences": (
        r"\bweekend\b",
        r"\bexperience\b",
        r"\bouting\b",
        r"\bmarket\b",
        r"\bteam\b",
        r"\bhangout\b",
    ),
}

_LIFESTYLE_SIGNAL_RE = re.compile(
    r"\b(food|eat|eating|restaurant|buka|jollof|suya|amala|movie|nollywood|cinema|"
    r"weekend|drink|bar|coffee|smoothie|hangout|dinner|lunch|brunch|date\s+night|"
    r"student|budget|campus|₦|naira|yaba|ikeja|lekki|surulere)\b",
    re.I,
)

_ADVISORY_ONLY_RE = re.compile(
    r"\b(learn(?:ing)?|stud(?:y|ies)|course|career|advis(?:e|ed|able)|should\s+i|"
    r"how\s+to|what\s+is|century|skill|artificial\s+intelligence|machine\s+learning|"
    r"\bai\b|upskill|bootcamp|certification)\b",
    re.I,
)

_TEAM_CULTURE_RE = re.compile(
    r"\b(hire|hiring|recruit|recruiting|join\s+(my|our)\s+team|"
    r"software\s+engineer|data\s+scientist|developer|founder|co-?founder|"
    r"build\s+a\s+team|attract\s+talent|work\s+for\s+me)\b",
    re.I,
)

# Tokens that pollute retrieval when the user asks about hiring, not gadgets.
_RETRIEVAL_STOPWORDS = frozenset(
    {
        "hire",
        "hiring",
        "recruit",
        "recruiting",
        "founder",
        "cofounder",
        "co-founder",
        "engineer",
        "engineering",
        "software",
        "developer",
        "scientist",
        "data",
        "company",
        "solutions",
        "taotech",
        "join",
        "team",
        "talent",
        "attract",
        "great",
        "how",
        "can",
        "get",
        "my",
        "the",
        "am",
        "i",
        "to",
        "me",
    }
)

_COMPANY_NOISE_RE = re.compile(
    r"\b(taotech|solutions|ltd|limited|inc|plc)\b",
    re.I,
)


@dataclass(frozen=True)
class PersonaIntent:
    team_culture_mode: bool = False
    advisory_only_mode: bool = False
    domains: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()
    retrieval_context: str = ""


def infer_persona_intent(narrative: str, *, base_domains: Sequence[str], base_interests: Sequence[str]) -> PersonaIntent:
    """Refine domains/interests and flag founder/hiring personas."""
    text = (narrative or "").strip()
    lower = text.lower()

    domains: List[str] = []
    for domain, patterns in _DOMAIN_HINTS.items():
        if any(re.search(p, lower) for p in patterns):
            domains.append(domain)

    team_culture = bool(_TEAM_CULTURE_RE.search(lower))
    has_lifestyle = bool(_LIFESTYLE_SIGNAL_RE.search(lower))
    advisory_only = bool(_ADVISORY_ONLY_RE.search(lower)) and not has_lifestyle

    if advisory_only:
        domains = [d for d in domains if d in ("tech", "books", "movies", "services")]
        for d in ("tech", "books", "services"):
            if d not in domains:
                domains.append(d)
        interests = list(
            dict.fromkeys(
                ["tech", "books", "student", "learning", "remote", "lagos", "budget"]
            )
        )
    elif team_culture:
        # Hiring questions → team outings & local lifestyle, not Amazon HR gadgets.
        domains = [d for d in domains if d != "tech"]
        for d in ("experiences", "food", "entertainment", "drinks"):
            if d not in domains:
                domains.append(d)
        interests = list(
            dict.fromkeys(
                [
                    "team",
                    "social",
                    "food",
                    "experiences",
                    "entertainment",
                    "founder",
                    "lagos",
                ]
            )
        )
    else:
        interests = list(base_interests)
        if not domains:
            domains = list(base_domains)

    # Lifestyle tokens (skip misleading hiring/company terms).
    for token in re.findall(r"[a-z]{3,}", lower):
        if token in _RETRIEVAL_STOPWORDS:
            continue
        if token in (
            "student",
            "budget",
            "street",
            "social",
            "spicy",
            "movies",
            "food",
            "drinks",
            "weekend",
            "campus",
        ):
            interests.append(token)

    interests = list(dict.fromkeys(interests))
    if not domains:
        domains = list(base_domains) or ["food", "experiences"]

    ctx = _COMPANY_NOISE_RE.sub(" ", lower)
    ctx = " ".join(w for w in ctx.split() if w not in _RETRIEVAL_STOPWORDS)

    return PersonaIntent(
        team_culture_mode=team_culture,
        advisory_only_mode=advisory_only,
        domains=tuple(domains),
        interests=tuple(interests),
        retrieval_context=ctx[:2000],
    )
