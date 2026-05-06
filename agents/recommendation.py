"""Recommendation agent."""

from __future__ import annotations

import re
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
        contextual_query = str(payload.get("contextual_query", payload.get("context", "")))
        conversation_history: List[str] = payload.get("conversation_history", [])

        interests = [i.lower() for i in user_model.get("interests", [])]
        bias_bonus = 0.1 if user_model.get("bias") == "positive" else 0.0
        query_terms = self._terms(f"{contextual_query} {' '.join(conversation_history)}")
        cold_start = len(memory_hits) == 0
        cross_domain = self._is_cross_domain(interests, candidate_items)

        scored = []
        for item in candidate_items:
            item_l = item.lower()
            interest_overlap = sum(1 for tag in interests if tag in item_l)
            memory_overlap = sum(1 for hit in memory_hits if item_l in hit.lower())
            query_overlap = len(query_terms.intersection(self._terms(item_l)))
            cold_start_bonus = 0.2 if cold_start and query_overlap > 0 else 0.0
            cross_domain_bonus = 0.1 if cross_domain and query_overlap > 0 else 0.0
            score = round(
                interest_overlap * 0.5
                + memory_overlap * 0.25
                + query_overlap * 0.2
                + 0.4
                + bias_bonus
                + cold_start_bonus
                + cross_domain_bonus,
                3,
            )
            scored.append(
                {
                    "item_name": item,
                    "score": score,
                    "explanation": (
                        f"Interest overlap={interest_overlap}, memory overlap={memory_overlap}, "
                        f"context overlap={query_overlap}, cold-start boost={round(cold_start_bonus, 2)}."
                    ),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_recommendations = scored[:top_k]
        explainability = {
            "personality_selected": recommender_personality,
            "scoring_formula": (
                "score = 0.5*interest_overlap + 0.25*memory_overlap + 0.2*context_overlap "
                "+ base + bias + conditional boosts"
            ),
            "memory_hit_count": len(memory_hits),
            "cold_start": cold_start,
            "cross_domain": cross_domain,
            "multiturn_turns_used": len(conversation_history),
            "reasoning_summary": "Items are ranked by preference, retrieved behavior, and conversational context.",
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

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _is_cross_domain(interests: List[str], candidate_items: List[str]) -> bool:
        if not interests:
            return True
        blob = " ".join(candidate_items).lower()
        return not any(i in blob for i in interests)

