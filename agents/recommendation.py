"""Recommendation agent."""

from __future__ import annotations

from typing import Any, Dict, List

from agents.base import BaseAgent


class RecommendationAgent(BaseAgent):
    """Rank items from user profile and memory evidence."""

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_model = payload["user_model"]
        candidate_items: List[str] = payload["candidate_items"]
        memory_hits: List[str] = payload.get("memory_hits", [])
        top_k = payload["top_k"]

        interests = [i.lower() for i in user_model.get("interests", [])]
        bias_bonus = 0.1 if user_model.get("bias") == "positive" else 0.0

        scored = []
        for item in candidate_items:
            item_l = item.lower()
            interest_overlap = sum(1 for tag in interests if tag in item_l)
            memory_overlap = sum(1 for hit in memory_hits if item_l in hit.lower())
            score = round(interest_overlap * 0.6 + memory_overlap * 0.3 + 0.5 + bias_bonus, 3)
            scored.append(
                {
                    "item_name": item,
                    "score": score,
                    "explanation": (
                        f"Matches {interest_overlap} interests and appears in {memory_overlap} memory signals."
                    ),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"recommendations": scored[:top_k]}

