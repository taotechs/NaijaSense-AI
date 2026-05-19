"""Task A - two-pass star rating then aligned review (RMSE / fidelity)."""

from __future__ import annotations

import json
import re
import secrets
from typing import Any, Dict, List, Optional

from agents.review_generation import ReviewGenerationAgent
from core.corpus_index import build_few_shot_matrix_block, get_corpus_index
from core.task_a_inputs import domain_prompt_block
from models.llm_wrapper import LLMWrapper

_PERSPECTIVE_RULE = (
    "You are the User Persona. Write a first-person review expressing your personal "
    "experience using or visiting the item described in Product Details. "
    "Do NOT review, summarize, or describe the persona profile itself."
)


class TaskATwoPassAgent:
    """
    Pass 1: derive numerical rating + short review_reasoning (domain-aware).
    Pass 2: generate review_text locked to that rating (domain vocabulary only).
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
        product_domain = str(user_model.get("product_domain", "general"))
        domain_block = domain_prompt_block(product_domain)

        few_shot_block = self._dynamic_few_shot_block(
            user_model=user_model,
            item_name=item_name,
            item_context=item_context,
            product_domain=product_domain,
            retrieved_examples=retrieved_examples,
        )
        rating, review_reasoning = self._pass1_rating(
            user_model=user_model,
            item_name=item_name,
            item_context=item_context,
            retrieved_examples=retrieved_examples,
            few_shot_block=few_shot_block,
            domain_block=domain_block,
        )
        review_text = self._pass2_review(
            user_model=user_model,
            item_name=item_name,
            item_context=item_context,
            rating=rating,
            review_reasoning=review_reasoning,
            retrieved_examples=retrieved_examples,
            language=language,
            few_shot_block=few_shot_block,
            domain_block=domain_block,
        )
        return {
            "rating": rating,
            "review_reasoning": review_reasoning,
            "review_text": review_text,
        }

    def _dynamic_few_shot_block(
        self,
        *,
        user_model: Dict[str, Any],
        item_name: str,
        item_context: str,
        product_domain: str,
        retrieved_examples: List[Dict[str, Any]],
    ) -> str:
        """Domain-filtered few-shots (style only) from corpus."""
        from core.corpus_index import _terms

        profile_terms = _terms(
            str(user_model.get("persona_narrative", "")),
            str(user_model.get("bias", "")),
        )
        index = get_corpus_index()
        examples = index.search_few_shots(
            profile_terms=profile_terms,
            product_name=item_name,
            product_context=item_context,
            sentiment_bias=str(user_model.get("bias", "balanced")),
            k=4,
        )
        if retrieved_examples:
            for ex in retrieved_examples:
                examples.append(ex)

        filtered = self._filter_examples_by_domain(examples, product_domain)
        return build_few_shot_matrix_block(filtered[:2])

    @staticmethod
    def _filter_examples_by_domain(
        examples: List[Dict[str, Any]],
        product_domain: str,
    ) -> List[Dict[str, Any]]:
        """Prefer few-shots from the same product domain; drop food style for tech/service."""
        domain_map = {
            "food": ("food", "restaurant"),
            "tech": ("tech", "general", "service"),
            "service": ("service", "tech", "general"),
            "book": ("books", "book", "general"),
            "hospitality": ("hospitality", "wellness", "general"),
            "general": ("general", "food", "tech", "service"),
        }
        allowed = domain_map.get(product_domain, ("general",))

        matched: List[Dict[str, Any]] = []
        rest: List[Dict[str, Any]] = []
        for ex in examples:
            ex_domain = str(ex.get("item_domain", "general")).lower()
            if any(a in ex_domain for a in allowed):
                matched.append(ex)
            elif product_domain != "food" and ex_domain in ("restaurant", "food"):
                continue
            else:
                rest.append(ex)
        return matched + rest

    def _pass1_rating(
        self,
        *,
        user_model: Dict[str, Any],
        item_name: str,
        item_context: str,
        retrieved_examples: List[Dict[str, Any]],
        few_shot_block: str,
        domain_block: str,
    ) -> tuple[float, str]:
        bias = str(user_model.get("bias", "balanced"))
        interests = user_model.get("interests", [])
        persona_text = str(user_model.get("persona_narrative", ""))[:3000]

        system = (
            "You are a Nigerian consumer rating analyst.\n"
            f"{_PERSPECTIVE_RULE}\n"
            "STEP 1: Read Product Details and identify the product domain "
            "(Food, Tech/Software, Service, Book, Hospitality, etc.).\n"
            "STEP 2: Use the User Persona only to decide tone and how strictly you score - "
            "not to change what product is being rated.\n"
            "Output ONLY valid JSON: "
            '{"rating": float 1.0-5.0, "review_reasoning": "2-3 sentences"}.\n'
            "review_reasoning must justify the score using domain-appropriate trade-offs only."
        )
        user_msg = (
            f"{domain_block}\n\n"
            f"{few_shot_block}\n\n"
            f"USER PERSONA (voice + priorities - NOT the product being rated):\n{persona_text}\n\n"
            f"PRODUCT DETAILS (subject of the review):\n"
            f"- name: {item_name}\n"
            f"- details: {item_context or '(none)'}\n\n"
            "Return JSON only."
        )

        raw = self.router.generate(user_msg, system=system, temperature=0.2).text.strip()
        parsed = self._parse_json_block(raw)
        if parsed and "rating" in parsed:
            rating = max(1.0, min(5.0, float(parsed["rating"])))
            reasoning = str(parsed.get("review_reasoning", "") or "").strip()
            if reasoning:
                return round(rating, 1), reasoning

        rating = ReviewGenerationAgent._score_rating(
            bias=bias,
            interests=interests,
            item_name=item_name,
            item_context=item_context,
            retrieved_examples=retrieved_examples,
        )
        trade_offs = user_model.get("domain_trade_offs", "overall experience")
        reasoning = (
            f"Pass-1 heuristic: persona bias={bias}; product domain cues mapped to "
            f"rating={rating:.1f} using {trade_offs}."
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
        few_shot_block: str,
        domain_block: str,
    ) -> str:
        persona_text = str(user_model.get("persona_narrative", ""))[:3000]
        bias = str(user_model.get("bias", "balanced"))
        product_domain = str(user_model.get("product_domain", "general"))
        domain_examples = self._filter_examples_by_domain(retrieved_examples, product_domain)
        few_shot = ReviewGenerationAgent._build_few_shot_block(domain_examples)

        system_msg = (
            "You are a Nigerian consumer-review writer for NaijaSense AI.\n"
            f"{_PERSPECTIVE_RULE}\n"
            f"{domain_block}\n"
            f"{few_shot_block}\n"
            "CRITICAL - RATING LOCK:\n"
            "- Pass 1 already set the star rating. Your prose MUST match that exact score.\n"
            "- Explicitly justify the locked rating using ONLY domain-appropriate trade-offs.\n"
            "- Do not contradict the score or mention a different rating.\n"
            "Return ONLY the review text (2-4 sentences, first person). No JSON, no 'X/5' line."
        )

        user_msg = (
            f"LOCKED RATING: {rating}/5.0 (do not change)\n"
            f"PASS-1 RATIONALE: {review_reasoning}\n\n"
            f"USER PERSONA (you are this person - first person 'I'):\n{persona_text}\n\n"
            f"PRODUCT DETAILS (what you used/visited - write about THIS):\n"
            f"- {item_name}\n"
            f"- {item_context or '(none)'}\n\n"
            f"Bias tendency: {bias} | Language: {language}\n"
            f"{few_shot}"
            f"Write the review now. Match {rating}/5 and use {product_domain}-appropriate words only:"
        )

        seed = secrets.randbelow(2**31 - 1)
        out = self.generator.generate(user_msg, system=system_msg, seed=seed).text.strip()
        if out and len(out) > 20:
            cleaned = out.strip().strip('"')
            if not self._looks_like_persona_review(cleaned, persona_text):
                return cleaned
            # Regenerate once if model reviewed the persona instead of the product.
            retry_msg = user_msg + "\n\nREMINDER: Review the PRODUCT only, not the persona description."
            out2 = self.generator.generate(retry_msg, system=system_msg, seed=seed + 1).text.strip()
            if out2 and len(out2) > 20:
                return out2.strip().strip('"')

        return self._fallback_review(
            item_name=item_name,
            item_context=item_context,
            rating=rating,
            product_domain=product_domain,
            trade_offs=str(user_model.get("domain_trade_offs", "")),
        )

    @staticmethod
    def _looks_like_persona_review(review: str, persona: str) -> bool:
        """Heuristic: reject reviews that describe the persona instead of the product."""
        r = review.lower()
        if "persona" in r or "profile" in r or "as a user who" in r:
            return True
        persona_snip = persona.lower()[:80]
        if persona_snip and persona_snip in r and "product" not in r:
            return True
        return False

    @staticmethod
    def _fallback_review(
        *,
        item_name: str,
        item_context: str,
        rating: float,
        product_domain: str,
        trade_offs: str,
    ) -> str:
        templates = {
            "food": {
                "high": "I enjoyed {item} - taste and portion were fair for the price, and I'd go again.",
                "mid": "Mixed feelings on {item}; decent taste but wait/price could be better.",
                "low": "Wouldn't rush back to {item} - portion or value didn't match what I paid.",
            },
            "tech": {
                "high": "I've been using {item} and it's solid - fast, reliable, and worth the spend for me.",
                "mid": "{item} works, but a few rough edges on speed/support for the price.",
                "low": "Disappointed with {item} - reliability and value aren't there yet for me.",
            },
            "service": {
                "high": "{item} delivered well - responsive team and clear value for what I paid.",
                "mid": "Okay experience with {item}; turnaround and communication were average.",
                "low": "{item} fell short - slow response and not enough value for the cost.",
            },
            "book": {
                "high": "{item} kept me engaged - pacing and writing style really worked for me.",
                "mid": "{item} was readable but not gripping throughout.",
                "low": "Struggled to finish {item} - pacing didn't hold my attention.",
            },
        }
        band = "high" if rating >= 4.0 else "mid" if rating >= 3.0 else "low"
        domain_templates = templates.get(product_domain, templates["tech"])
        line = domain_templates[band].format(item=item_name)
        snippet = (item_context or "")[:100].strip()
        if snippet:
            return f"{line} {snippet}"
        return line

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
