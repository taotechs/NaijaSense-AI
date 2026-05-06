"""User modeling agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.base import BaseAgent
from models.llm_wrapper import LLMWrapper
from utils.config import settings


class UserModelingAgent(BaseAgent):
    """Infer a reusable structured user persona from profile and behavior context."""

    def __init__(self, llm: LLMWrapper) -> None:
        self.llm = llm

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_profile = payload.get("user_profile")
        user_history: List[str] = payload.get("user_history", [])
        preferences: Dict[str, Any] = payload.get("preferences", {})
        metadata: Dict[str, Any] = payload.get("metadata", {})
        persona_style = payload.get("persona_style") or settings.default_persona_style

        prompt = self._build_reasoning_prompt(
            user_profile=user_profile,
            user_history=user_history,
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
                preferences=preferences,
                metadata=metadata,
                persona_style=persona_style,
            )

        return {
            "user_id": user_profile.user_id,
            "tone": inferred["tone"],
            "rating_tendency": inferred["rating_tendency"],
            "interests": inferred["interests"],
            "cultural_context": inferred["cultural_context"],
            "behavior_patterns": inferred["behavior_patterns"],
            "reasoning_summary": inferred["reasoning_summary"],
            # Backward-compatible fields used by other agents.
            "persona_style": persona_style,
            "bias": user_profile.sentiment_bias or "balanced",
            "location": user_profile.location or "Nigeria",
        }

    def _build_reasoning_prompt(
        self,
        user_profile: Any,
        user_history: List[str],
        preferences: Dict[str, Any],
        metadata: Dict[str, Any],
        persona_style: str,
    ) -> str:
        return (
            "You are a persona inference engine. Infer a user persona from the inputs.\n"
            "Return JSON only with keys: tone, rating_tendency, interests, cultural_context, "
            "behavior_patterns, reasoning_summary.\n"
            "Allowed tone values: formal, casual, slang-heavy.\n"
            "Allowed rating_tendency values: strict, generous.\n"
            f"User profile: {user_profile.model_dump()}\n"
            f"User history: {user_history}\n"
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
            return data
        except (json.JSONDecodeError, TypeError, AttributeError):
            return {}

    def _fallback_persona(
        self,
        user_profile: Any,
        user_history: List[str],
        preferences: Dict[str, Any],
        metadata: Dict[str, Any],
        persona_style: str,
    ) -> Dict[str, Any]:
        history_blob = " ".join(user_history).lower()
        preference_tone = str(preferences.get("tone", "")).lower()
        metadata_tone = str(metadata.get("tone", "")).lower()

        if any(key in history_blob for key in ["omo", "abeg", "wahala", "sha"]) or "slang" in preference_tone:
            tone = "slang-heavy"
        elif user_profile.tone_preference in {"formal", "casual", "slang-heavy"}:
            tone = user_profile.tone_preference
        elif "formal" in preference_tone or "formal" in metadata_tone:
            tone = "formal"
        else:
            tone = "casual"

        bias = (user_profile.sentiment_bias or "balanced").lower()
        rating_tendency = "strict" if bias in {"critical", "negative"} else "generous"

        interests = list(dict.fromkeys(user_profile.interests + preferences.get("interests", [])))
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

