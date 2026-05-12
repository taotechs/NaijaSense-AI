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
        query_blob = f"{contextual_query} {' '.join(conversation_history)}".strip()
        query_terms = self._terms(query_blob)
        cold_start = len(memory_hits) == 0
        cross_domain = self._is_cross_domain(interests, candidate_items)
        low_budget = bool(re.search(r"\b(under|below|less than|budget|cheap|affordable|[0-9]+k)\b", query_blob.lower()))
        wants_spicy = bool(re.search(r"\b(spicy|pepper|hot)\b", query_blob.lower()))
        wants_relax = bool(re.search(r"\b(relax|calm|chill|unwind|stress)\b", query_blob.lower()))
        query_domain_hints = self._query_domain_hints(query_blob)

        scored = []
        for item in candidate_items:
            item_l = item.lower()
            interest_overlap = sum(1 for tag in interests if tag in item_l)
            memory_overlap = sum(1 for hit in memory_hits if item_l in hit.lower())
            query_overlap = len(query_terms.intersection(self._terms(item_l)))
            domain_alignment = 1 if any(h in item_l for h in query_domain_hints) else 0
            spicy_bonus = 0.1 if wants_spicy and any(k in item_l for k in ("pepper", "suya", "jollof", "spicy")) else 0.0
            budget_bonus = 0.1 if low_budget and any(k in item_l for k in ("budget", "under", "wallet", "affordable", "street")) else 0.0
            relax_bonus = 0.1 if wants_relax and any(k in item_l for k in ("tea", "calm", "cozy", "relax", "chill", "cafe")) else 0.0
            cold_start_bonus = 0.15 if cold_start and query_overlap > 0 else 0.0
            cross_domain_bonus = 0.15 if cross_domain and query_overlap > 0 else 0.0
            score = round(
                interest_overlap * 0.5
                + memory_overlap * 0.25
                + query_overlap * 0.2
                + domain_alignment * 0.2
                + 0.4
                + bias_bonus
                + spicy_bonus
                + budget_bonus
                + relax_bonus
                + cold_start_bonus
                + cross_domain_bonus,
                3,
            )
            # Penalize placeholder-y items that look like templates.
            if any(k in item_l for k in ["starter pack", "bundle", "choice", "pick"]) and query_overlap == 0:
                score = round(max(0.0, score - 0.25), 3)
            scored.append(
                {
                    "item_name": item,
                    "score": score,
                    "explanation": (
                        f"Interest overlap={interest_overlap}, memory overlap={memory_overlap}, "
                        f"context overlap={query_overlap}, domain alignment={domain_alignment}, "
                        f"bonus(spicy/budget/relax)={round(spicy_bonus + budget_bonus + relax_bonus, 2)}."
                    ),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_recommendations = scored[:top_k]
        # Chain-of-thought trace: name the signals that drove the ranking so
        # the reasoning is auditable even though the scorer is deterministic.
        cot_trace: List[str] = []
        if cold_start:
            cot_trace.append("Cold-start path: no prior memory hits, leaning on context overlap.")
        else:
            cot_trace.append(
                f"Warm-start path: {len(memory_hits)} memory snippet(s) shape interest priors."
            )
        if cross_domain:
            cot_trace.append("Cross-domain detected: candidates disjoint from known interests.")
        if conversation_history:
            cot_trace.append(
                f"Multi-turn aware: folded {len(conversation_history)} prior turn(s) into the query."
            )
        if wants_spicy or low_budget or wants_relax:
            intents = ", ".join(
                kind
                for kind, present in (
                    ("spicy", wants_spicy),
                    ("budget", low_budget),
                    ("relax", wants_relax),
                )
                if present
            )
            cot_trace.append(f"Intent boosts active: {intents}.")
        if top_recommendations:
            top = top_recommendations[0]
            cot_trace.append(
                f"Final pick: {top['item_name']} (score={top['score']}) wins on the "
                "weighted sum above."
            )
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
            "chain_of_thought": cot_trace,
            "reasoning_summary": (
                "Items are ranked by preference, retrieved behavior, and "
                "conversational context. See chain_of_thought for the step-by-step trace."
            ),
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
            return f"Based on your vibe, start with {names}. These look strongest right now."
        if personality == "friend":
            return f"I'd personally suggest starting with {names}. These feel like your kind of picks."
        if personality == "coach":
            return f"Solid move: prioritize {names}. They align with your current goals and usage pattern."
        return (
            f"Top recommendations: {names}. "
            "Ranking combines profile fit, memory evidence, and context overlap."
        )

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _query_domain_hints(query: str) -> set[str]:
        q = query.lower()
        hints: set[str] = set()
        if any(k in q for k in ("eat", "food", "restaurant", "hungry", "dinner", "lunch", "breakfast")):
            hints.update({"amala", "jollof", "suya", "shawarma", "buka", "kitchen", "food", "restaurant"})
        if any(k in q for k in ("watch", "movie", "netflix", "series", "film")):
            hints.update({"movie", "series", "drama", "comedy", "docu", "feature"})
        if any(k in q for k in ("buy", "gadget", "tech", "device", "phone", "laptop")):
            hints.update({"earbud", "charger", "hub", "watch", "keyboard", "tech"})
        if any(k in q for k in ("relax", "calm", "stress", "unwind")):
            hints.update({"tea", "cozy", "calm", "chill", "cafe"})
        return hints

    @staticmethod
    def _is_cross_domain(interests: List[str], candidate_items: List[str]) -> bool:
        if not interests:
            return True
        blob = " ".join(candidate_items).lower()
        return not any(i in blob for i in interests)

