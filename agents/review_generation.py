"""Review generation agent."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

from agents.base import BaseAgent
from models.llm_wrapper import LLMWrapper


class ReviewGenerationAgent(BaseAgent):
    """Generate realistic persona-aligned review text and rating."""

    def __init__(self, llm: LLMWrapper) -> None:
        self.llm = llm

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_model = payload["user_model"]
        item_name = payload["item_name"]
        item_context = (payload.get("item_context", "") or "").strip()
        retrieved_examples: List[Dict[str, Any]] = payload.get("retrieved_examples", [])
        domain = self._infer_domain(item_name, item_context)

        opening = self._build_opening(
            persona_style=user_model.get("persona_style", "formal"),
            bias=user_model.get("bias", "balanced"),
            domain=domain,
            seed=f"{item_name}|{item_context}|{user_model.get('user_id', '')}",
        )
        detail_sentence = self._build_detail_sentence(
            item_name=item_name,
            item_context=item_context,
            domain=domain,
            retrieved_examples=retrieved_examples,
        )
        personalization = self._build_personalization(user_model=user_model, domain=domain)
        review_text = f"{opening} {detail_sentence} {personalization}".strip()

        rating = self._score_rating(
            bias=user_model.get("bias", "balanced"),
            interests=user_model.get("interests", []),
            item_name=item_name,
            item_context=item_context,
            retrieved_examples=retrieved_examples,
        )

        return {
            "review_text": review_text,
            "rating": round(rating, 1),
            "persona_breakdown": user_model,
        }

    def _build_opening(self, persona_style: str, bias: str, domain: str, seed: str) -> str:
        food_hint = "the food" if domain == "food" else "this one"
        templates_ng = {
            "positive": [
                f"Omo {food_hint} slap well, no cap.",
                f"I go lie for you? {food_hint.capitalize()} really delivered.",
                f"{food_hint.capitalize()} dey sweet die, I liked it immediately.",
            ],
            "balanced": [
                f"{food_hint.capitalize()} good, but e still get small room to improve.",
                f"I tested {food_hint}, e dey okay sha.",
                f"{food_hint.capitalize()} try, no be perfect but e no bad.",
            ],
            "critical": [
                f"{food_hint.capitalize()} no really hit the mark for me.",
                f"Honestly, this one no try as I expect.",
                f"I expected better from {food_hint}.",
            ],
        }
        templates_formal = {
            "positive": [
                f"My first impression of {food_hint} was strongly positive.",
                f"I had a very satisfying experience with {food_hint}.",
            ],
            "balanced": [
                f"My experience with {food_hint} was moderate overall.",
                f"{food_hint.capitalize()} performs decently with a few compromises.",
            ],
            "critical": [
                f"My experience with {food_hint} was below expectation.",
                f"{food_hint.capitalize()} has notable quality issues.",
            ],
        }

        bank = templates_ng if persona_style == "nigerian_twitter" else templates_formal
        options = bank.get(bias, bank["balanced"])
        index = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(options)
        return options[index]

    @staticmethod
    def _build_detail_sentence(
        item_name: str,
        item_context: str,
        domain: str,
        retrieved_examples: List[Dict[str, Any]],
    ) -> str:
        if item_context:
            cleaned = re.sub(r"\s+", " ", item_context).strip().rstrip(".")
            return f"For {item_name}, this stood out: {cleaned}."
        if retrieved_examples:
            example = retrieved_examples[0]
            exemplar_text = str(example.get("text", "")).strip().rstrip(".")
            if exemplar_text:
                return f"Similar reviews mention: {exemplar_text}."
        if domain == "food":
            return f"For {item_name}, taste and consistency were the main deciding factors."
        if domain == "tech":
            return f"For {item_name}, performance and reliability influenced the experience most."
        return f"For {item_name}, value for money influenced my final impression."

    @staticmethod
    def _build_personalization(user_model: Dict[str, Any], domain: str) -> str:
        interests = [str(i).lower() for i in user_model.get("interests", [])]
        tone = user_model.get("tone", "casual")
        if interests:
            return (
                f"It aligns with my interest in {interests[0]}, "
                f"and the overall tone feels {tone} for day-to-day use."
            )
        if domain == "food":
            return "If you enjoy local dishes, you may still find this worth trying once."
        return "It may suit users looking for a practical and straightforward option."

    @staticmethod
    def _infer_domain(item_name: str, item_context: str) -> str:
        text = f"{item_name} {item_context}".lower()
        if any(k in text for k in ["amala", "gbegiri", "food", "restaurant", "buka", "ewedu", "jollof"]):
            return "food"
        if any(k in text for k in ["earbud", "phone", "laptop", "smartwatch", "tech", "gadget"]):
            return "tech"
        if any(k in text for k in ["bag", "shoe", "fashion", "cloth", "sneaker"]):
            return "fashion"
        return "general"

    @staticmethod
    def _score_rating(
        bias: str,
        interests: list[Any],
        item_name: str,
        item_context: str,
        retrieved_examples: List[Dict[str, Any]],
    ) -> float:
        base = {"positive": 4.2, "balanced": 3.5, "critical": 2.7}.get(bias, 3.5)
        text = f"{item_name} {item_context}".lower()
        positive_words = ["great", "sweet", "fresh", "clean", "fast", "excellent", "love"]
        negative_words = ["bad", "slow", "cold", "overpriced", "poor", "terrible", "no try"]
        base += 0.15 * sum(1 for w in positive_words if w in text)
        base -= 0.2 * sum(1 for w in negative_words if w in text)
        base += 0.15 if any(str(i).lower() in text for i in interests) else 0.0
        if retrieved_examples:
            avg = sum(float(x.get("rating", 3.5)) for x in retrieved_examples) / len(retrieved_examples)
            base = (base * 0.7) + (avg * 0.3)
        return max(1.0, min(5.0, base))

