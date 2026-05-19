"""
Curated recommendable products for Task B (not raw review rows).

Stage-1 retrieval targets real items (Food, Movie, Drink, etc.) with human-readable
titles - never review-dataset placeholders like ``yelp_review`` or ``hf_yelp_*``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from core.candidate_catalog import CATALOG, CatalogItem

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(*parts: str) -> tuple[str, ...]:
    terms: Set[str] = set()
    for part in parts:
        terms.update(t for t in _TOKEN.findall((part or "").lower()) if len(t) > 2)
    return tuple(sorted(terms)[:10])

_PLACEHOLDER_NAMES = frozenset(
    {
        "yelp_review",
        "amazon_item",
        "goodreads_book",
        "yelp_item",
        "item",
        "unknown_item",
        "product",
    }
)
_PLACEHOLDER_ID_RE = re.compile(
    r"^(hf_yelp_|hf_amz_|off_[a-z]_\d|yelp_review|amazon_|unknown)",
    re.I,
)
_INTERNAL_TITLE_RE = re.compile(r"^(yelp|amazon|goodreads|hf)_", re.I)

# Keyword → display title hints extracted from review prose.
_TEXT_TITLE_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(jollof|rice)\b", re.I), "Local Jollof Plate", "food"),
    (re.compile(r"\b(suya|skewer)\b", re.I), "Street Suya Skewers", "food"),
    (re.compile(r"\b(amala|egusi|ewedu)\b", re.I), "Amala & Soup Combo", "food"),
    (re.compile(r"\b(shawarma|wrap)\b", re.I), "Shawarma Wrap", "food"),
    (re.compile(r"\b(pepper soup|goat pepper)\b", re.I), "Goat Pepper Soup", "food"),
    (re.compile(r"\b(fish sandwich|fried fish)\b", re.I), "Crispy Fish Sandwich", "food"),
    (re.compile(r"\b(reuben)\b", re.I), "Classic Reuben Sandwich", "food"),
    (re.compile(r"\b(pizza|white pizza)\b", re.I), "Wood-Fired Pizza", "food"),
    (re.compile(r"\b(smoothie|juice bar)\b", re.I), "Fresh Fruit Smoothie", "drinks"),
    (re.compile(r"\b(coffee|latte|chai)\b", re.I), "Specialty Coffee", "drinks"),
    (re.compile(r"\b(craft beer|beer selection)\b", re.I), "Craft Beer Flight", "drinks"),
    (re.compile(r"\b(nollywood|movie|film|cinema)\b", re.I), "Nollywood Weekend Movie", "movies"),
    (re.compile(r"\b(documentary|series)\b", re.I), "African Documentary Series", "movies"),
    (re.compile(r"\b(power bank|battery)\b", re.I), "USB-C Power Bank", "tech"),
    (re.compile(r"\b(earbuds|headphones)\b", re.I), "Wireless Earbuds", "tech"),
    (re.compile(r"\b(spa|massage)\b", re.I), "Relaxation Spa Session", "wellness"),
]

# Opinion / review prose mistaken for product titles in corpus rows.
_REVIEW_SNIPPET_TITLE_RE = re.compile(
    r"^(one of the|i wanted|i loved|i hate|i thought|food porn|would not|wouldn't|"
    r"don't|do not|never again|so good|so bad|amazing|terrible|disappointed|"
    r"highly recommend|not worth|waste of|overrated|underrated)\b",
    re.I,
)
_REVIEW_SNIPPET_MID_RE = re.compile(
    r"\b(but\.\.\.|however,|although |because |though |'\s*\w|\"\s*\w)",
    re.I,
)

_DOMAIN_FALLBACK_TITLE = {
    "Food": "Premium Local Restaurant",
    "Movie": "A Thrilling Sci-Fi Film",
    "Drink": "Specialty Drinks Spot",
    "Books": "A Compelling Read",
    "Tech": "Practical Tech Pick",
    "Wellness": "Relaxing Wellness Experience",
    "Fashion": "Stylish Fashion Find",
    "Experience": "Memorable Local Experience",
    "Service": "Trusted Local Service",
    "General": "Recommended Local Pick",
}

_DOMAIN_LABELS = {
    "food": "Food",
    "restaurant": "Food",
    "drinks": "Drink",
    "drink": "Drink",
    "movies": "Movie",
    "movie": "Movie",
    "entertainment": "Movie",
    "books": "Books",
    "tech": "Tech",
    "wellness": "Wellness",
    "fashion": "Fashion",
    "experiences": "Experience",
    "services": "Service",
    "general": "General",
}


def is_placeholder_item_name(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n or len(n) < 3:
        return True
    if n in _PLACEHOLDER_NAMES:
        return True
    if _INTERNAL_TITLE_RE.match(n):
        return True
    if n.endswith("_review") or n.endswith("_item"):
        return True
    return False


def is_internal_item_id(item_id: str) -> bool:
    iid = (item_id or "").strip()
    if not iid:
        return True
    return bool(_PLACEHOLDER_ID_RE.match(iid))


def display_domain(raw: str) -> str:
    key = (raw or "general").lower().strip()
    if key in _DOMAIN_LABELS:
        return _DOMAIN_LABELS[key]
    if "restaurant" in key or "food" in key:
        return "Food"
    if "movie" in key or "entertain" in key:
        return "Movie"
    if "drink" in key or "bar" in key:
        return "Drink"
    return key.capitalize() if key else "General"


def looks_like_review_snippet(title: str) -> bool:
    """True when a string looks like review comment text, not a product/venue name."""
    t = (title or "").strip()
    if not t or len(t) < 4:
        return False
    if t.startswith(('"', "'", "“", "”", "(")):
        return True
    if "..." in t or "…" in t:
        return True
    if _REVIEW_SNIPPET_TITLE_RE.search(t):
        return True
    if _REVIEW_SNIPPET_MID_RE.search(t):
        return True
    if t.count("!") >= 2 or t.count("?") >= 2:
        return True
    words = t.split()
    if len(words) > 12:
        return True
    if len(words) >= 6 and t[0].islower():
        return True
    return False


def prompt_display_title(
    title: str,
    *,
    domain: str = "General",
    context_text: str = "",
) -> str:
    """
    Clean title for Task B prompts and generated prose.
    Never pass raw review snippets through as display names.
    """
    base = canonical_item_title(title)
    if base and not looks_like_review_snippet(base):
        return base

    blob = " ".join(filter(None, [context_text, title, " ".join(_tokenize(title, domain))]))
    inferred = infer_title_from_text(blob)
    if inferred:
        return inferred[0]

    label = display_domain(domain)
    return _DOMAIN_FALLBACK_TITLE.get(label, _DOMAIN_FALLBACK_TITLE["General"])


def infer_title_from_text(text: str) -> Optional[Tuple[str, str]]:
    blob = text or ""
    for pattern, title, domain in _TEXT_TITLE_PATTERNS:
        if pattern.search(blob):
            return title, domain
    return None


def resolve_display_item(row: Dict[str, Any], *, idx: int = 0) -> Optional[CatalogItem]:
    """
    Map a corpus row to a recommendable catalog entry, or None if not displayable.
    """
    raw_name = str(row.get("item_name", "")).strip()
    text = str(row.get("text", "")).strip()
    raw_domain = str(row.get("item_domain", row.get("domain", "general")))

    title = raw_name
    domain = raw_domain

    if is_placeholder_item_name(raw_name):
        inferred = infer_title_from_text(text)
        if not inferred:
            return None
        title, domain = inferred
    elif len(raw_name) > 80:
        inferred = infer_title_from_text(text)
        title = inferred[0] if inferred else raw_name[:60]
        domain = inferred[1] if inferred else raw_domain

    title = prompt_display_title(title, domain=domain, context_text=text)
    if is_placeholder_item_name(title) or looks_like_review_snippet(title):
        return None

    iid = str(row.get("item_id") or "").strip()
    if is_internal_item_id(iid) or not iid:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:40]
        iid = f"rec_{slug}_{idx}"

    tags = row.get("tags")
    tag_tuple = tuple(str(t).lower() for t in tags) if isinstance(tags, list) else ()
    if not tag_tuple:
        tag_tuple = _tokenize(title, domain, text)

    return CatalogItem(
        item_id=iid,
        title=title,
        domain=display_domain(domain),
        tags=tag_tuple,
    )


def curated_catalog_items() -> List[CatalogItem]:
    """Static Nigerian catalog with normalized display domains."""
    items: List[CatalogItem] = []
    for c in CATALOG:
        items.append(
            CatalogItem(
                item_id=c.item_id,
                title=c.title,
                domain=display_domain(c.domain),
                tags=c.tags,
            )
        )
    return items


_VARIANT_SUFFIX_RE = re.compile(r"\s*#\d+\s*$", re.I)
_VARIANT_PAREN_RE = re.compile(r"\s*\(variant\s+\d+\)\s*$", re.I)


def canonical_item_title(title: str) -> str:
    """Collapse corpus variants like 'Local Jollof Kitchen - Surulere #19' → base name."""
    t = (title or "").strip()
    t = _VARIANT_SUFFIX_RE.sub("", t)
    t = _VARIANT_PAREN_RE.sub("", t)
    return t.strip() or "local pick"


def merge_unique_items(
    candidates: List[Tuple[CatalogItem, float]],
    *,
    limit: int,
) -> List[Tuple[CatalogItem, float]]:
    """Deduplicate by canonical title, keep highest score."""
    by_title: Dict[str, Tuple[CatalogItem, float]] = {}
    for item, score in candidates:
        key = canonical_item_title(item.title).lower()
        if key not in by_title or score > by_title[key][1]:
            by_title[key] = (item, score)
    merged = sorted(by_title.values(), key=lambda x: x[1], reverse=True)
    return merged[:limit]
