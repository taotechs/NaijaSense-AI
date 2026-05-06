"""Review generation agent."""

from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent
from models.llm_wrapper import LLMWrapper


class ReviewGenerationAgent(BaseAgent):
    """Generate realistic persona-aligned review text and rating."""

    def __init__(self, llm: LLMWrapper) -> None:
        self.llm = llm

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_model = payload["user_model"]
        item_name = payload["item_name"]
        item_context = payload.get("item_context", "")

        rating_map = {"positive": 4.5, "balanced": 3.8, "critical": 2.8}
        rating = rating_map.get(user_model["bias"], 3.8)

        if user_model["persona_style"] == "nigerian_twitter":
            sentiment_line = self._nigerian_sentiment_line(user_model.get("bias", "balanced"))
            seed_text = f"{sentiment_line} {item_context or 'E get strong points, but no be perfect.'}"
        else:
            seed_text = (
                f"My experience with {item_name} was mostly positive. "
                f"{item_context or 'It performs as expected with minor tradeoffs.'}"
            )

        prompt = (
            f"Persona: {user_model['persona_style']}\n"
            f"Tone: {user_model['tone']}\n"
            f"Generate one concise review: {seed_text}"
        )
        review_text = self.llm.generate(prompt).text.split("Generate one concise review:")[-1].strip()

        return {
            "review_text": review_text,
            "rating": round(rating, 1),
            "persona_breakdown": user_model,
        }

    @staticmethod
    def _nigerian_sentiment_line(bias: str) -> str:
        """
        Return localized opening lines with controlled slang intensity.

        This improves cultural realism for demo quality while keeping output readable.
        """
        if bias == "critical":
            return "This place no try at all for this kind thing."
        if bias == "positive":
            return "Omo this one slap! I just tried am now now."
        return "Omo, I don test am and e dey okay sha."

