"""Parse unified Task A persona + product text blobs into agent signals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from core.persona_parser import parse_task_b_persona

# Product-domain detection (from Product Details text only - not persona interests).
_DOMAIN_RULES: dict[str, dict[str, object]] = {
    "food": {
        "keywords": (
            "food",
            "restaurant",
            "buka",
            "jollof",
            "amala",
            "suya",
            "shawarma",
            "meal",
            "dish",
            "kitchen",
            "cafe",
            "eat",
            "dining",
            "snack",
        ),
        "trade_offs": "taste, portion size, wait time, value-for-money, spice level, freshness",
        "forbidden_unless_food": "portion, jollof, amala, suya, pepper, plate, buka, kitchen taste",
    },
    "tech": {
        "keywords": (
            "software",
            "app",
            "saas",
            "platform",
            "tech",
            "gadget",
            "phone",
            "laptop",
            "api",
            "digital",
            "device",
            "hardware",
            "startup",
            "solutions",
        ),
        "trade_offs": "build quality, speed, reliability, battery life, UX, support, value for money",
        "forbidden_unless_food": "portion, jollof, amala, suya, plate, buka, pepper level, dish",
    },
    "service": {
        "keywords": (
            "service",
            "consulting",
            "agency",
            "support",
            "delivery",
            "repair",
            "installation",
            "subscription",
        ),
        "trade_offs": "responsiveness, professionalism, turnaround time, clarity, value delivered",
        "forbidden_unless_food": "portion, jollof, taste of food, plate, buka",
    },
    "book": {
        "keywords": ("book", "novel", "author", "chapter", "read", "paperback", "literature"),
        "trade_offs": "writing style, pacing, engagement, clarity, worth the read",
        "forbidden_unless_food": "portion, jollof, plate, buka, spice",
    },
    "hospitality": {
        "keywords": ("hotel", "lodge", "spa", "resort", "accommodation", "room", "check-in"),
        "trade_offs": "cleanliness, comfort, staff attitude, location, value for the stay",
        "forbidden_unless_food": "portion, jollof, software bug, api",
    },
}

_SENTIMENT_RE = re.compile(
    r"\b(positive|critical|balanced|harsh|generous|upbeat|skeptical)\b",
    re.I,
)


@dataclass
class ParsedTaskAInputs:
    persona_text: str
    product_text: str
    item_name: str
    item_context: str
    product_domain: str
    domain_trade_offs: str
    location: str
    interests: List[str]
    sentiment_bias: str

    def to_user_model(self) -> Dict[str, Any]:
        return {
            "user_id": "task_a_user",
            "location": self.location,
            "interests": self.interests,
            "bias": self.sentiment_bias,
            "tone": self.persona_text[:500],
            "persona_narrative": self.persona_text,
            "persona_style": "nigerian_twitter",
            "product_domain": self.product_domain,
            "domain_trade_offs": self.domain_trade_offs,
        }


def infer_product_domain(product_text: str, item_name: str = "") -> str:
    """Classify Product Details domain before any LLM call."""
    blob = f"{product_text} {item_name}".lower()
    scores: dict[str, int] = {k: 0 for k in _DOMAIN_RULES}
    for domain, meta in _DOMAIN_RULES.items():
        for kw in meta["keywords"]:  # type: ignore[union-attr]
            if kw in blob:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "general"


def domain_prompt_block(product_domain: str) -> str:
    """Instructions + allowed vocabulary for the detected product domain."""
    meta = _DOMAIN_RULES.get(product_domain, {})
    trade_offs = str(
        meta.get("trade_offs", "value-for-money, quality, reliability, overall experience")
    )
    label = product_domain.replace("_", " ").title()
    lines = [
        f"PRODUCT DOMAIN (from Product Details only): {label}",
        f"Use ONLY vocabulary and trade-offs relevant to {label}: {trade_offs}.",
        "The User Persona shapes your tone and priorities - it is NOT the subject of the review.",
    ]
    if product_domain != "food":
        forbidden = meta.get("forbidden_unless_food", "portion, jollof, amala, suya, plate, buka")
        lines.append(
            f"FORBIDDEN unless the product is food: {forbidden}."
        )
    lines.append(
        "Perspective: first-person - you ARE the User Persona reviewing the product/experience "
        "described in Product Details."
    )
    return "\n".join(lines)


def _extract_item_name(product_text: str) -> str:
    text = product_text.strip()
    if not text:
        return "Product"
    first_line = text.splitlines()[0].strip()
    if len(first_line) <= 120:
        return first_line.rstrip(".")
    sentence = re.split(r"[.!?]\s+", text, maxsplit=1)[0].strip()
    return (sentence[:120] if sentence else text[:120]).strip()


def _extract_sentiment(persona_text: str) -> str:
    match = _SENTIMENT_RE.search(persona_text)
    if not match:
        return "balanced"
    word = match.group(1).lower()
    if word in ("positive", "generous", "upbeat"):
        return "positive"
    if word in ("critical", "harsh", "skeptical"):
        return "critical"
    return "balanced"


def parse_task_a_inputs(user_persona: str, product_details: str) -> ParsedTaskAInputs:
    persona_text = (user_persona or "").strip()
    product_text = (product_details or "").strip()
    profile = parse_task_b_persona(persona_text or "Nigerian consumer", user_id="task_a")

    item_name = _extract_item_name(product_text)
    product_domain = infer_product_domain(product_text, item_name)
    meta = _DOMAIN_RULES.get(product_domain, {})
    trade_offs = str(
        meta.get("trade_offs", "value-for-money, quality, reliability, overall experience")
    )

    return ParsedTaskAInputs(
        persona_text=persona_text,
        product_text=product_text,
        item_name=item_name,
        item_context=product_text,
        product_domain=product_domain,
        domain_trade_offs=trade_offs,
        location=profile.location,
        interests=profile.interests,
        sentiment_bias=_extract_sentiment(persona_text),
    )
