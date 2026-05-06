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
        recommender_personality = payload.get("recommender_personality", "analyst")
        conversational_mode = bool(payload.get("conversational_mode", True))

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
        top_recommendations = scored[:top_k]
        explainability = {
            "personality_selected": recommender_personality,
            "scoring_formula": "score = 0.6*interest_overlap + 0.3*memory_overlap + 0.5 + bias_bonus",
            "memory_hit_count": len(memory_hits),
            "reasoning_summary": "Items are ranked by preference overlap and prior behavior evidence.",
        }
        conversational_response = (
            self._build_conversational_response(top_recommendations, recommender_personality)
            if conversational_mode
            else None
        )
        return {
            "recommendations": top_recommendations,
            "conversational_response": conversational_response,
            "explainability": explainability,
        }

    def _build_conversational_response(
        self, recommendations: List[Dict[str, Any]], personality: str
    ) -> str:
        if not recommendations:
            return "I could not find a strong match yet. Share more preferences and I can refine it."
        names = ", ".join(item["item_name"] for item in recommendations)
        if personality == "nigerian_twitter":
            return f"Omo, based on your vibe, you should check these first: {names}."
        if personality == "friend":
            return f"I'd personally suggest starting with {names}. These feel like your kind of picks."
        if personality == "coach":
            return f"Solid move: prioritize {names}. They align with your current goals and usage pattern."
        return f"Top recommendations: {names}. Ranked by profile-interest and memory relevance."

