"""Task A — two-pass star rating then aligned review (RMSE / fidelity)."""

from __future__ import annotations

import json
import re
import secrets
from typing import Any, Dict, List, Optional

from agents.review_generation import ReviewGenerationAgent
from models.llm_wrapper import LLMWrapper
from utils.config import settings

# Few-shot matrix: profile → authentic Nigerian review tone (trade-offs, not stereotypes).
NIGERIAN_FEW_SHOT_MATRIX = """
FEW-SHOT PROFILE → REVIEW MATRIX (style only — do not copy facts):

| Profile | Sample review tone |
|---------|-------------------|
| Lagos student, budget food | "Paid about ₦1,500 for the plate — fair for the portion. Jollof had decent smoke but rice was slightly clumped; still worth it if you're hungry after class." |
| VI professional, formal tech | "Build feels solid for the price band. Battery holds a full workday; only gripe is the cable in the box is short for desk setup." |
| Abuja family diner, balanced | "Service was polite and the egusi was rich without being oily. Wait was 25 minutes on a Sunday — manageable if you order before the rush." |
| Critical street-food explorer | "Suya was tasty but skewers ran small for ₦2k. Pepper level was accurate; I'd return only if the queue stays under 15 minutes." |

Localized trade-offs to weigh: value-for-money, wait time, portion size, durability under daily use,
power/battery on gadgets, and whether the experience matches the price band in Lagos/Abuja/PH context.
"""


class TaskATwoPassAgent:
    """
    Pass 1: derive numerical rating + short review_reasoning.
    Pass 2: generate review_text locked to that rating (sentiment alignment for RMSE).
    """

    def __init__(
        self,
        router_llm: Optional[LLMWrapper] = None,
        generator_llm: Optional[LLMWrapper] = None,
    ) -> None:
        self.router = router_llm or LLMWrapper(role="router")
        self.generator = generator_llm or LLMWrapper(role="generator")
        self._review_helper = ReviewGenerationAgent(llm=self.generator, critic_llm=self.router)

    def run(
        self,
        *,
        user_model: Dict[str, Any],
        item_name: str,
        item_context: str,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
        language: str = "english",
    ) -> Dict[str, Any]:
        retrieved_examples = retrieved_examples or []
        rating, review_reasoning = self._pass1_rating(
            user_model=user_model,
            item_name=item_name,
            item_context=item_context,
            retrieved_examples=retrieved_examples,
        )
        review_text = self._pass2_review(
            user_model=user_model,
            item_name=item_name,
            item_context=item_context,
            rating=rating,
            review_reasoning=review_reasoning,
            retrieved_examples=retrieved_examples,
            language=language,
        )
        return {
            "rating": rating,
            "review_reasoning": review_reasoning,
            "review_text": review_text,
        }

    def _pass1_rating(
        self,
        *,
        user_model: Dict[str, Any],
        item_name: str,
        item_context: str,
        retrieved_examples: List[Dict[str, Any]],
    ) -> tuple[float, str]:
        bias = str(user_model.get("bias", "balanced"))
        interests = user_model.get("interests", [])
        persona_blob = json.dumps(
            {
                "user_id": user_model.get("user_id"),
                "location": user_model.get("location"),
                "interests": interests,
                "bias": bias,
                "tone": user_model.get("tone"),
                "merge_meta": user_model.get("merge_meta"),
            },
            ensure_ascii=False,
        )

        system = (
            "You are a Nigerian consumer rating analyst. "
            "Output ONLY valid JSON with keys: rating (float 1.0-5.0), review_reasoning (string, 2-3 sentences). "
            "The rating must be justified from persona + product facts. "
            "Weigh value-for-money, local utility, and stated experience cues."
        )
        user_msg = (
            f"{NIGERIAN_FEW_SHOT_MATRIX}\n\n"
            f"PERSONA:\n{persona_blob}\n\n"
            f"PRODUCT:\n- name: {item_name}\n- context: {item_context or '(none)'}\n\n"
            "Return JSON only."
        )

        raw = self.router.generate(user_msg, system=system, temperature=0.2).text.strip()
        parsed = self._parse_json_block(raw)
        if parsed and "rating" in parsed:
            rating = max(1.0, min(5.0, float(parsed["rating"])))
            reasoning = str(parsed.get("review_reasoning", "") or "").strip()
            if reasoning:
                return round(rating, 1), reasoning

        # Heuristic fallback (deterministic) when router unavailable.
        rating = ReviewGenerationAgent._score_rating(
            bias=bias,
            interests=interests,
            item_name=item_name,
            item_context=item_context,
            retrieved_examples=retrieved_examples,
        )
        reasoning = (
            f"Pass-1 heuristic: mapped sentiment_bias={bias} and product cues to "
            f"rating={rating:.1f}; value-for-money and experience keywords in context were weighted."
        )
        return round(rating, 1), reasoning

    def _pass2_review(
        self,
        *,
        user_model: Dict[str, Any],
        item_name: str,
        item_context: str,
        rating: float,
        review_reasoning: str,
        retrieved_examples: List[Dict[str, Any]],
        language: str,
    ) -> str:
        user_model = dict(user_model)
        user_model["language"] = language
        few_shot = ReviewGenerationAgent._build_few_shot_block(retrieved_examples)

        bias = str(user_model.get("bias", "balanced"))
        persona_style = str(user_model.get("persona_style", "nigerian_twitter"))

        system_msg = (
            "You are a Nigerian consumer-review writer for NaijaSense AI. "
            f"{NIGERIAN_FEW_SHOT_MATRIX}\n"
            "CRITICAL: The star rating is ALREADY DECIDED. Your review prose MUST match that rating. "
            "Do not contradict the score. Mention value-for-money or durability where relevant. "
            "Return ONLY the review text (2-4 sentences), no JSON, no rating line."
        )

        user_msg = (
            f"LOCKED RATING: {rating}/5.0\n"
            f"RATIONALE (Pass 1): {review_reasoning}\n\n"
            f"ITEM: {item_name}\n"
            f"FACTS: {item_context or '(none)'}\n"
            f"PERSONA STYLE: {persona_style} | bias: {bias}\n"
            f"LANGUAGE: {language}\n\n"
            f"{few_shot}"
            "Write the review now — tone must align with the locked rating:"
        )

        seed = secrets.randbelow(2**31 - 1)
        out = self.generator.generate(user_msg, system=system_msg, seed=seed).text.strip()
        if out and len(out) > 20:
            return out.strip().strip('"')

        # Template fallback aligned to rating band.
        if rating >= 4.0:
            tone = "Solid experience overall — worth the spend for what you get."
        elif rating >= 3.0:
            tone = "Decent but not perfect; price and wait time are the main trade-offs."
        else:
            tone = "Below expectations for the money; I'd think twice before a repeat visit."
        return f"For {item_name}, {tone} {item_context[:120] if item_context else ''}".strip()

    @staticmethod
    def _parse_json_block(raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip()
        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None
