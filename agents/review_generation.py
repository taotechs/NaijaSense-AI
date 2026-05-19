"""Review generation agent."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from models.llm_wrapper import LLMWrapper
from utils.config import settings


class ReviewGenerationAgent(BaseAgent):
    """Generate realistic persona-aligned review text and rating."""

    def __init__(self, llm: LLMWrapper, critic_llm: Optional[LLMWrapper] = None) -> None:
        self.llm = llm
        # Critic is the cheap/fast model used in the critique→regenerate loop.
        # Falls back to the generator if no dedicated critic was wired.
        self.critic_llm = critic_llm or llm

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
            seed=f"{item_name}|{item_context}|{user_model.get('user_id', '')}|{secrets.token_hex(4)}",
        )
        detail_sentence = self._build_detail_sentence(
            item_name=item_name,
            item_context=item_context,
            domain=domain,
            retrieved_examples=retrieved_examples,
        )
        personalization = self._build_personalization(user_model=user_model, domain=domain)
        draft_text = f"{opening} {detail_sentence} {personalization}".strip()
        review_text = self._rewrite_with_llm_if_available(
            draft_text=draft_text,
            user_model=user_model,
            item_name=item_name,
            item_context=item_context,
            domain=domain,
            retrieved_examples=retrieved_examples,
        )

        critique_meta: Dict[str, Any] = {"applied": False}
        # Only run critique on real LLM output, not the deterministic fallback.
        if (
            settings.review_critique_enabled
            and review_text
            and review_text != draft_text
        ):
            review_text, critique_meta = self._critique_and_maybe_rewrite(
                review_text=review_text,
                user_model=user_model,
                item_name=item_name,
                item_context=item_context,
                domain=domain,
                retrieved_examples=retrieved_examples,
                draft_text=draft_text,
            )

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
            "critique_meta": critique_meta,
        }

    def _rewrite_with_llm_if_available(
        self,
        draft_text: str,
        user_model: Dict[str, Any],
        item_name: str,
        item_context: str,
        domain: str,
        retrieved_examples: List[Dict[str, Any]],
    ) -> str:
        """Generate a fresh review with the LLM using facts + few-shot retrieval.

        We pass structured facts (NOT the deterministic draft) so the model
        writes from scratch each call. A per-call variation token + random seed
        guarantees byte-different output for identical inputs. Falls back to
        ``draft_text`` only when the provider is unavailable or returns nothing.
        """
        persona_style = str(user_model.get("persona_style", "formal"))
        bias = str(user_model.get("bias", "balanced"))
        tone = str(user_model.get("tone", "casual"))
        interests = [str(i).strip() for i in user_model.get("interests", []) if str(i).strip()]
        interests_blob = ", ".join(interests) if interests else "general lifestyle"

        if persona_style == "nigerian_twitter":
            style_rule = (
                "Speak with light Naija-twitter / pidgin colour where it feels natural. "
                "Use AT MOST one slang phrase per sentence and never force it."
            )
        else:
            style_rule = (
                "Use clear, neutral global English. Do NOT use Nigerian slang or pidgin."
            )

        # Hard language rule, threaded from the agent gateway language toggle.
        # ``english`` keeps current behaviour. ``pidgin`` and ``yoruba_mix``
        # request specific local registers regardless of persona_style above.
        language = str(user_model.get("language", "english") or "english").lower().strip()
        if language == "pidgin":
            language_rule = (
                "WRITE PRIMARILY IN NIGERIAN PIDGIN ENGLISH. Use natural pidgin "
                "constructions (e.g. 'I no go lie', 'e dey sweet die', 'na vibes'). "
                "Mix in standard English only where pidgin would obscure meaning. "
                "This rule OVERRIDES the persona style rule above."
            )
        elif language == "yoruba_mix":
            language_rule = (
                "WRITE IN ENGLISH WITH NATURAL YORUBA WORDS / PHRASES sprinkled in "
                "(e.g. 'omo', 'jare', 'gan-an', 'gbono feli feli'). Keep the sentence "
                "structure English so non-Yoruba speakers can still follow. "
                "This rule OVERRIDES the persona style rule above."
            )
        else:
            language_rule = "Write in standard global English."

        bias_rule = {
            "positive": (
                "Overall sentiment is positive but specific - no gushing, no hyperbole."
            ),
            "balanced": (
                "Overall sentiment is balanced: name one concrete strength AND one honest caveat."
            ),
            "critical": (
                "Overall sentiment is critical but fair: name a real shortcoming and what would fix it."
            ),
        }.get(bias, "Overall sentiment is balanced.")

        few_shot_block = self._build_few_shot_block(retrieved_examples)
        nonce = secrets.token_hex(3)
        seed = secrets.randbelow(2**31 - 1)

        system_msg = (
            "You are a Nigerian consumer-review writer for the NaijaSense AI platform. "
            "Write like a real buyer in Lagos, Abuja, or Port Harcourt: value-for-money "
            "matters, service speed and 'worth it' matter, and tone is direct but fair. "
            "Use natural Nigerian English (light pidgin or local adjectives only when "
            "persona style allows - e.g. 'e dey sweet', 'no cap', 'worth the hype', "
            "'cash-wise'). You produce short, specific, human-sounding reviews grounded "
            "ONLY in the facts provided. Never invent prices, brands, or experiences not "
            "in the facts. Return ONLY the review text - no preamble, headings, markdown, "
            "or quotation marks."
        )

        user_msg = (
            "Write a fresh 2-4 sentence review.\n\n"
            "FACTS\n"
            f"- Item: {item_name}\n"
            f"- Domain: {domain}\n"
            f"- Reviewer interests: {interests_blob}\n"
            f"- Reviewer tone bucket: {tone}\n"
            f"- Persona style: {persona_style}\n"
            f"- What the user actually said about it: {item_context or '(none provided)'}\n\n"
            "STYLE RULES\n"
            f"- {language_rule}\n"
            f"- {style_rule}\n"
            f"- {bias_rule}\n"
            "- Be concrete: mention at least one specific detail "
            "(texture, speed, taste, price band / 'for the money', queue time, "
            "location vibe, power/battery) instead of generic praise.\n"
            "- Nigerian fidelity: if facts mention cost, comment on whether it "
            "felt worth it; mention delivery/wait or 'stress' when relevant.\n"
            "- Do NOT start with any of these openings: "
            "\"My first impression\", \"My experience with\", \"Honestly\", "
            "\"Overall\", \"Omo\". Vary sentence structure every time.\n"
            "- Return plain prose only, no bullets, no markdown.\n\n"
            f"{few_shot_block}"
            f"VARIATION TOKEN (use to ensure this output differs from prior calls): {nonce}\n\n"
            "Now write the review:"
        )

        out = self.llm.generate(user_msg, system=system_msg, seed=seed).text.strip()
        if not out or "Nigerian consumer-review writer" in out:
            return draft_text

        out = out.strip().strip('"').strip("'").strip()
        for prefix in ("review:", "Review:", "REVIEW:"):
            if out.startswith(prefix):
                out = out[len(prefix):].strip()
                break
        return out

    def _critique_and_maybe_rewrite(
        self,
        review_text: str,
        user_model: Dict[str, Any],
        item_name: str,
        item_context: str,
        domain: str,
        retrieved_examples: List[Dict[str, Any]],
        draft_text: str,
    ) -> tuple[str, Dict[str, Any]]:
        """Score the review for specificity; rewrite once if below threshold.

        Uses the cheap critic model so this stays fast and low-cost. Returns
        the (possibly rewritten) review and a meta dict for explainability.
        """
        critique = self._critique_review(
            review_text=review_text,
            persona_style=str(user_model.get("persona_style", "formal")),
            bias=str(user_model.get("bias", "balanced")),
            item_name=item_name,
        )
        meta: Dict[str, Any] = {
            "applied": True,
            "specificity_score": critique.get("specificity_score"),
            "issues": critique.get("issues", []),
            "rewritten": False,
        }
        # Gate on score only - the small critic's needs_rewrite boolean is
        # noisy and biased toward "yes". Score is more reliable.
        score = critique.get("specificity_score")
        needs_rewrite = isinstance(score, int) and score < settings.review_critique_threshold
        if not needs_rewrite:
            return review_text, meta

        rewritten = self._regenerate_with_feedback(
            previous_review=review_text,
            critic_issues=critique.get("issues", []),
            user_model=user_model,
            item_name=item_name,
            item_context=item_context,
            domain=domain,
            retrieved_examples=retrieved_examples,
        )
        if rewritten and rewritten != review_text:
            meta["rewritten"] = True
            return rewritten, meta
        # Critic said rewrite but generator returned empty / identical text.
        return review_text, meta

    def _critique_review(
        self,
        review_text: str,
        persona_style: str,
        bias: str,
        item_name: str,
    ) -> Dict[str, Any]:
        """Ask the critic model for a JSON verdict on the review."""
        system_msg = (
            "You are a calibrated review-quality critic. You score consumer reviews "
            "for specificity. You are NOT a stylist or a perfectionist - your job is "
            "to flag genuinely vague or generic reviews, not to nitpick decent ones. "
            "You output ONLY a single JSON object - no prose, no markdown, no code fences."
        )
        user_msg = (
            "Score the following review on specificity using this rubric:\n"
            "  5 = excellent: multiple concrete details (e.g. specific texture, "
            "named feature, exact price/time/location, sensory detail). Could only "
            "be about this exact experience.\n"
            "  4 = good: at least one strong concrete detail beyond generic praise. "
            "Reads like a real person's review. ← APPROVE.\n"
            "  3 = mediocre: mostly generic with one weak detail.\n"
            "  2 = vague: generic adjectives only, could apply to any similar item.\n"
            "  1 = empty calories: zero substance.\n\n"
            "Be honest but not harsh. A 4 is the typical score for a normal good "
            "review - do NOT reserve 4-5 for poetry. Only score 1-3 if the review "
            "is genuinely lacking concrete detail.\n\n"
            f"Item being reviewed: {item_name}\n"
            f"Required persona style: {persona_style} "
            "(nigerian_twitter = light pidgin colour allowed; formal = neutral English, no slang)\n"
            f"Required sentiment bias: {bias}\n\n"
            "REVIEW:\n"
            f"{review_text}\n\n"
            "Return JSON with EXACTLY these keys:\n"
            "  - specificity_score: integer 1-5 per the rubric above\n"
            "  - issues: array of short strings describing concrete problems "
            "(generic phrasing, no specific detail, wrong tone, forbidden opening, "
            "invented facts). Empty array if none.\n"
            "  - needs_rewrite: true if score is 1-3, else false."
        )
        raw = self.critic_llm.generate(
            user_msg,
            system=system_msg,
            temperature=0.0,
            max_tokens=220,
        ).text.strip()
        return self._parse_critique_json(raw)

    @staticmethod
    def _parse_critique_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return {}
        try:
            data = json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        score = data.get("specificity_score")
        if isinstance(score, float):
            score = int(round(score))
        if not isinstance(score, int) or not 1 <= score <= 5:
            score = None
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = []
        issues = [str(item).strip() for item in issues if str(item).strip()][:6]
        return {
            "specificity_score": score,
            "issues": issues,
            "needs_rewrite": bool(data.get("needs_rewrite")),
        }

    def _regenerate_with_feedback(
        self,
        previous_review: str,
        critic_issues: List[str],
        user_model: Dict[str, Any],
        item_name: str,
        item_context: str,
        domain: str,
        retrieved_examples: List[Dict[str, Any]],
    ) -> str:
        """Re-call the generator with the critic's issues injected as rules."""
        persona_style = str(user_model.get("persona_style", "formal"))
        bias = str(user_model.get("bias", "balanced"))
        tone = str(user_model.get("tone", "casual"))
        interests = [str(i).strip() for i in user_model.get("interests", []) if str(i).strip()]
        interests_blob = ", ".join(interests) if interests else "general lifestyle"
        few_shot_block = self._build_few_shot_block(retrieved_examples)
        nonce = secrets.token_hex(3)
        seed = secrets.randbelow(2**31 - 1)

        issues_block = (
            "\n".join(f"- {issue}" for issue in critic_issues)
            if critic_issues
            else "- review was too generic and not specific enough"
        )

        system_msg = (
            "You are a Nigerian consumer-review writer for the NaijaSense AI platform. "
            "A previous draft was rejected by a quality critic. Write a DIFFERENT, "
            "more concrete review that fixes the listed issues. You return ONLY the "
            "review text - no preamble, no headings, no markdown, no quotation marks."
        )
        user_msg = (
            "Rewrite this review so it is more specific and human, addressing the "
            "critic's issues. Do NOT repeat the previous review's phrasing.\n\n"
            "FACTS\n"
            f"- Item: {item_name}\n"
            f"- Domain: {domain}\n"
            f"- Reviewer interests: {interests_blob}\n"
            f"- Reviewer tone bucket: {tone}\n"
            f"- Persona style: {persona_style}\n"
            f"- Sentiment bias: {bias}\n"
            f"- What the user actually said about it: {item_context or '(none provided)'}\n\n"
            "ISSUES TO FIX (from the critic)\n"
            f"{issues_block}\n\n"
            "RULES\n"
            "- 2-4 sentences, plain prose only.\n"
            "- Add at least one new concrete detail not in the previous review "
            "(specific texture, price band, time, location, mood, feature).\n"
            "- Vary sentence structure; do NOT reuse the opening of the previous review.\n\n"
            f"{few_shot_block}"
            "PREVIOUS REVIEW (to avoid copying):\n"
            f"{previous_review}\n\n"
            f"VARIATION TOKEN: {nonce}\n\n"
            "Now write the improved review:"
        )

        out = self.llm.generate(
            user_msg,
            system=system_msg,
            seed=seed,
            # Slightly higher temperature on the rewrite to escape the previous mode.
            temperature=min(0.95, (settings.gen_temperature or 0.85) + 0.05),
        ).text.strip()
        if not out:
            return ""
        out = out.strip().strip('"').strip("'").strip()
        for prefix in ("review:", "Review:", "REVIEW:"):
            if out.startswith(prefix):
                out = out[len(prefix):].strip()
                break
        return out

    @staticmethod
    def _build_few_shot_block(retrieved_examples: List[Dict[str, Any]]) -> str:
        """Format up to three retrieved corpus rows as a style/concreteness reference.

        Examples are shown for STYLE & concreteness only - the system prompt forbids
        copying their facts. Snippets are truncated and stripped of newlines so the
        block stays compact.
        """
        if not retrieved_examples:
            return ""
        lines: List[str] = [
            "FEW-SHOT EXAMPLES (style and concreteness reference - do NOT copy their facts):",
        ]
        shown = 0
        for ex in retrieved_examples[:3]:
            text = str(ex.get("text", "") or "").strip()
            if len(text) < 40:
                continue
            item = str(ex.get("item_name", "") or "").strip() or "similar item"
            snippet = " ".join(text.split())[:280]
            lines.append(f"- ({item}) {snippet}")
            shown += 1
        if shown == 0:
            return ""
        lines.append("")
        return "\n".join(lines) + "\n"

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
                f"I tested {food_hint}, it was okay overall.",
                f"{food_hint.capitalize()} tried, not perfect but still decent.",
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
            cleaned = re.sub(r"^\s*(review|rate)\s+[^:]+:\s*", "", cleaned, flags=re.I)
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
        """Avoid classifying as food from a lone 'food' word in context (e.g. boilerplate reviews)."""
        name = (item_name or "").lower()
        ctx = (item_context or "").lower()

        if any(k in name for k in ["earbud", "phone", "laptop", "smartwatch", "tech", "gadget", "charger", "keyboard", "hub"]):
            return "tech"
        if any(k in name for k in ["bag", "shoe", "fashion", "cloth", "sneaker", "dress"]):
            return "fashion"

        strong_food = (
            "amala",
            "gbegiri",
            "buka",
            "jollof",
            "ewedu",
            "restaurant",
            "suya",
            "kitchen",
            "chef",
            "menu",
        )
        if any(k in name for k in strong_food) or any(k in ctx for k in strong_food):
            return "food"
        if re.search(r"\bfood\b", ctx) and re.search(
            r"\b(meal|dish|plate|menu|taste|flavor|dining|portion|cooked|served)\b", ctx
        ):
            return "food"
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
        # Small per-request spread so repeat calls are not identical in logs/UI.
        jitter = (secrets.randbelow(201) - 100) / 1000.0
        return max(1.0, min(5.0, base + jitter))

