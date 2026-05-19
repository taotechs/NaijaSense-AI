"""User modeling agent.

Implements the *default vs. override* contract of the stateful agentic
workflow:

1. The orchestrator silently retrieves a ``historical_persona`` summary
   (avg rating, rating tendency, tone signal, dominant domains, inferred
   interests) for the incoming ``user_id``.
2. This agent treats that summary as the **baseline persona**.
3. The UI-supplied :class:`utils.schemas.UserProfile` fields then act as
   **explicit overrides** - but only where the user actually set them.
   This is the bit the brief calls out: persona settings on the screen
   are an override, not a replacement.
4. The returned ``merge_meta`` lists exactly which fields were
   overridden so the reasoning trace can surface it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.base import BaseAgent
from models.llm_wrapper import LLMWrapper
from utils.config import settings


class UserModelingAgent(BaseAgent):
    """Infer a reusable structured user persona from profile + history."""

    def __init__(self, llm: LLMWrapper) -> None:
        self.llm = llm

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_profile = payload.get("user_profile")
        user_history: List[str] = payload.get("user_history", []) or []
        historical_persona: Dict[str, Any] = payload.get("historical_persona") or {}
        preferences: Dict[str, Any] = payload.get("preferences", {}) or {}
        metadata: Dict[str, Any] = payload.get("metadata", {}) or {}
        persona_style = payload.get("persona_style") or settings.default_persona_style

        prompt = self._build_reasoning_prompt(
            user_profile=user_profile,
            user_history=user_history,
            historical_persona=historical_persona,
            preferences=preferences,
            metadata=metadata,
            persona_style=persona_style,
        )
        llm_text = self.llm.generate(prompt).text
        inferred = self._parse_llm_json(llm_text)
        if not inferred:
            inferred = self._fallback_persona(
                user_profile=user_profile,
                user_history=user_history,
                historical_persona=historical_persona,
                preferences=preferences,
                metadata=metadata,
                persona_style=persona_style,
            )

        merged, merge_meta = self._merge_history_with_overrides(
            inferred=inferred,
            user_profile=user_profile,
            historical_persona=historical_persona,
            persona_style=persona_style,
        )

        return {
            "user_id": user_profile.user_id,
            "tone": merged["tone"],
            "rating_tendency": merged["rating_tendency"],
            "interests": merged["interests"],
            "cultural_context": merged["cultural_context"],
            "behavior_patterns": merged["behavior_patterns"],
            "reasoning_summary": merged["reasoning_summary"],
            "persona_style": persona_style,
            "bias": merged["bias"],
            "location": merged["location"],
            "merge_meta": merge_meta,
            "historical_persona": historical_persona,
        }

    # ---- Merge semantics --------------------------------------------

    @staticmethod
    def _merge_history_with_overrides(
        inferred: Dict[str, Any],
        user_profile: Any,
        historical_persona: Dict[str, Any],
        persona_style: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Combine inferred + historical baseline with explicit UI fields.

        Override rule: the UI value wins **only when the user actively
        set it** (non-empty for free-text, non-default for enums).
        Otherwise the historical baseline survives.
        """
        overridden: List[str] = []
        baseline_source: Dict[str, str] = {}

        # Tone -----------------------------------------------------------------
        ui_tone = (getattr(user_profile, "tone_preference", None) or "").strip().lower()
        history_tone = (historical_persona.get("tone_signal") or "").strip().lower()
        inferred_tone = (inferred.get("tone") or "").strip().lower()
        if ui_tone in {"formal", "casual", "slang-heavy"}:
            tone = ui_tone
            overridden.append("tone")
            baseline_source["tone"] = "ui"
        elif history_tone:
            tone = history_tone
            baseline_source["tone"] = "history"
        else:
            tone = inferred_tone or "casual"
            baseline_source["tone"] = "inferred"

        # Sentiment bias -------------------------------------------------------
        ui_bias = (getattr(user_profile, "sentiment_bias", None) or "").strip().lower()
        history_bias = (historical_persona.get("sentiment_bias") or "").strip().lower()
        if ui_bias and ui_bias != "balanced":
            bias = ui_bias
            overridden.append("sentiment_bias")
            baseline_source["sentiment_bias"] = "ui"
        elif history_bias:
            bias = history_bias
            baseline_source["sentiment_bias"] = "history"
        else:
            bias = ui_bias or "balanced"
            baseline_source["sentiment_bias"] = "ui_default"

        # Rating tendency ------------------------------------------------------
        # No UI field for this; history wins, otherwise derive from bias.
        history_tendency = (historical_persona.get("rating_tendency") or "").strip().lower()
        if history_tendency:
            rating_tendency = history_tendency
            baseline_source["rating_tendency"] = "history"
        else:
            rating_tendency = (
                "strict" if bias in {"critical", "negative"} else "generous"
            )
            baseline_source["rating_tendency"] = "derived"

        # Interests ------------------------------------------------------------
        ui_interests = [
            str(i).strip()
            for i in getattr(user_profile, "interests", []) or []
            if str(i).strip()
        ]
        history_interests = [
            str(i).strip()
            for i in historical_persona.get("inferred_interests", []) or []
            if str(i).strip()
        ]
        inferred_interests = [
            str(i).strip()
            for i in inferred.get("interests", []) or []
            if str(i).strip()
        ]
        if ui_interests:
            merged_interests: List[str] = []
            for item in ui_interests + history_interests + inferred_interests:
                if item.lower() not in {x.lower() for x in merged_interests}:
                    merged_interests.append(item)
            interests = merged_interests
            overridden.append("interests")
            baseline_source["interests"] = "ui+history"
        elif history_interests:
            interests = history_interests + [
                i for i in inferred_interests if i.lower() not in {h.lower() for h in history_interests}
            ]
            baseline_source["interests"] = "history"
        else:
            interests = inferred_interests or ["general lifestyle"]
            baseline_source["interests"] = "inferred"

        # Location -------------------------------------------------------------
        ui_location = (getattr(user_profile, "location", None) or "").strip()
        if ui_location and ui_location.lower() != "nigeria":
            location = ui_location
            overridden.append("location")
            baseline_source["location"] = "ui"
        else:
            location = ui_location or "Nigeria"
            baseline_source["location"] = "ui_default"

        cultural_context = inferred.get(
            "cultural_context",
            "General consumer behavior with regional adaptation.",
        )
        # The LLM occasionally returns ``behavior_patterns`` as a free-form
        # string instead of an object; coerce defensively so we never call
        # ``dict()`` on a string and trip a TypeError downstream.
        raw_behavior = inferred.get("behavior_patterns")
        if isinstance(raw_behavior, dict):
            behavior_patterns: Dict[str, Any] = dict(raw_behavior)
        elif isinstance(raw_behavior, str) and raw_behavior.strip():
            behavior_patterns = {"summary": raw_behavior.strip()[:300]}
        else:
            behavior_patterns = {}
        behavior_patterns.setdefault("persona_style_hint", persona_style)
        if historical_persona.get("n_reviews"):
            behavior_patterns["historical_n_reviews"] = historical_persona["n_reviews"]
        if historical_persona.get("avg_rating") is not None:
            behavior_patterns["historical_avg_rating"] = historical_persona["avg_rating"]
        if historical_persona.get("top_domains"):
            behavior_patterns["historical_top_domains"] = historical_persona["top_domains"]

        reasoning_summary_bits: List[str] = []
        if historical_persona.get("n_reviews"):
            reasoning_summary_bits.append(
                f"Baseline persona built from {historical_persona['n_reviews']} historical reviews"
            )
        if overridden:
            reasoning_summary_bits.append(
                "UI overrides applied to " + ", ".join(sorted(set(overridden)))
            )
        if not reasoning_summary_bits:
            reasoning_summary_bits.append(
                "No historical signal; persona inferred from UI profile only"
            )
        reasoning_summary = ". ".join(reasoning_summary_bits) + "."

        merged = {
            "tone": tone,
            "bias": bias,
            "rating_tendency": rating_tendency,
            "interests": interests,
            "location": location,
            "cultural_context": cultural_context,
            "behavior_patterns": behavior_patterns,
            "reasoning_summary": reasoning_summary,
        }
        merge_meta = {
            "overridden_fields": sorted(set(overridden)),
            "source_per_field": baseline_source,
            "has_history": bool(historical_persona.get("n_reviews")),
        }
        return merged, merge_meta

    # ---- LLM prompt + parsing ---------------------------------------

    def _build_reasoning_prompt(
        self,
        user_profile: Any,
        user_history: List[str],
        historical_persona: Dict[str, Any],
        preferences: Dict[str, Any],
        metadata: Dict[str, Any],
        persona_style: str,
    ) -> str:
        history_blob = "\n".join(f"- {h}" for h in user_history[:5]) or "(no past reviews available)"
        return (
            "You are a persona inference engine. Infer a user persona from the inputs.\n"
            "Return JSON only with keys: tone, rating_tendency, interests, cultural_context, "
            "behavior_patterns, reasoning_summary.\n"
            "Allowed tone values: formal, casual, slang-heavy.\n"
            "Allowed rating_tendency values: strict, generous, balanced.\n"
            f"User profile: {user_profile.model_dump()}\n"
            f"Historical persona summary: {historical_persona}\n"
            f"Recent past reviews:\n{history_blob}\n"
            f"Preferences: {preferences}\n"
            f"Metadata: {metadata}\n"
            f"Persona style hint: {persona_style}\n"
        )

    def _parse_llm_json(self, raw_text: str) -> Dict[str, Any]:
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                return {}
            data = json.loads(raw_text[start : end + 1])
            if not isinstance(data, dict):
                return {}
            required = {
                "tone",
                "rating_tendency",
                "interests",
                "cultural_context",
                "behavior_patterns",
                "reasoning_summary",
            }
            if not required.issubset(data.keys()):
                return {}
            # Type-coerce fields the downstream merge expects in specific shapes
            # so we never let an unexpected LLM payload propagate.
            if not isinstance(data.get("interests"), list):
                data["interests"] = []
            if not isinstance(data.get("tone"), str):
                data["tone"] = ""
            if not isinstance(data.get("rating_tendency"), str):
                data["rating_tendency"] = ""
            if not isinstance(data.get("cultural_context"), str):
                data["cultural_context"] = ""
            if not isinstance(data.get("reasoning_summary"), str):
                data["reasoning_summary"] = ""
            return data
        except (json.JSONDecodeError, TypeError, AttributeError):
            return {}

    def _fallback_persona(
        self,
        user_profile: Any,
        user_history: List[str],
        historical_persona: Dict[str, Any],
        preferences: Dict[str, Any],
        metadata: Dict[str, Any],
        persona_style: str,
    ) -> Dict[str, Any]:
        history_blob = " ".join(user_history).lower()
        preference_tone = str(preferences.get("tone", "")).lower()
        metadata_tone = str(metadata.get("tone", "")).lower()
        history_tone = str(historical_persona.get("tone_signal") or "").lower()

        if history_tone in {"slang-heavy", "formal", "casual"}:
            tone = history_tone
        elif any(key in history_blob for key in ["omo", "abeg", "wahala", "sha"]) or "slang" in preference_tone:
            tone = "slang-heavy"
        elif user_profile.tone_preference in {"formal", "casual", "slang-heavy"}:
            tone = user_profile.tone_preference
        elif "formal" in preference_tone or "formal" in metadata_tone:
            tone = "formal"
        else:
            tone = "casual"

        bias = (
            historical_persona.get("sentiment_bias")
            or (user_profile.sentiment_bias or "balanced")
        ).lower()
        rating_tendency = (
            historical_persona.get("rating_tendency")
            or ("strict" if bias in {"critical", "negative"} else "generous")
        )

        interests = list(
            dict.fromkeys(
                list(user_profile.interests or [])
                + list(historical_persona.get("inferred_interests", []) or [])
                + list(preferences.get("interests", []) or [])
            )
        )
        if not interests:
            interests = ["general lifestyle"]

        location = (user_profile.location or "").lower()
        if "nigeria" in location or location in {"lagos", "abuja", "port harcourt", "ibadan"}:
            cultural_context = (
                "Nigerian consumer behavior: value-conscious, social-proof driven, "
                "and expressive language in informal channels."
            )
        else:
            cultural_context = "General consumer behavior with regional adaptation."

        return {
            "tone": tone,
            "rating_tendency": rating_tendency,
            "interests": interests,
            "cultural_context": cultural_context,
            "behavior_patterns": {
                "consistency": "medium",
                "feedback_style": "short-form social expression" if tone != "formal" else "structured feedback",
                "decision_driver": "value and relevance",
                "persona_style_hint": persona_style,
            },
            "reasoning_summary": (
                "Persona inferred from sentiment bias, interest overlap, tone indicators, "
                "and region-specific behavior cues."
            ),
        }
