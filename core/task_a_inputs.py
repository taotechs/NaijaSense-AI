"""Parse unified Task A persona + product text blobs into agent signals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from core.persona_parser import parse_task_b_persona

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
        }


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

    return ParsedTaskAInputs(
        persona_text=persona_text,
        product_text=product_text,
        item_name=_extract_item_name(product_text),
        item_context=product_text,
        location=profile.location,
        interests=profile.interests,
        sentiment_bias=_extract_sentiment(persona_text),
    )
