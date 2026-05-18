"""Task B — 2-stage retrieval (top-30) + LLM agentic reranker (persona-only input)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from core.candidate_catalog import CatalogItem, retrieve_top_k
from core.persona_parser import ParsedPersona, parse_task_b_persona
from models.llm_wrapper import LLMWrapper

ARCHETYPE_BRIDGES = {
    "social": ("entertainment", "experiences", "food", "movies"),
    "spicy": ("food", "entertainment"),
    "budget": ("food", "services", "tech", "fashion", "drinks"),
    "student": ("food", "tech", "services", "experiences", "movies"),
    "relax": ("wellness", "entertainment", "books", "movies"),
    "tech": ("tech", "services"),
    "movie": ("movies", "entertainment", "food"),
    "drink": ("drinks", "food", "entertainment"),
}


class TaskBPipelineAgent:
    """Stage-1 corpus retrieval → Stage-2 Reason-Before-Recommend LLM rerank."""

    DEFAULT_TOP_K = 10

    def __init__(self, router_llm: Optional[LLMWrapper] = None) -> None:
        self.llm = router_llm or LLMWrapper(role="router")

    def run(
        self,
        *,
        user_id: str,
        persona_narrative: str,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        parsed = parse_task_b_persona(persona_narrative, user_id=user_id)
        k = top_k or self.DEFAULT_TOP_K
        interests = list(parsed.interests)
        cross_domain = parsed.cold_start or len(set(parsed.domains)) >= 2

        if cross_domain:
            interests = self._expand_cross_domain_interests(interests, parsed.narrative)

        user_model: Dict[str, Any] = {
            "user_id": user_id,
            "location": parsed.location,
            "bias": "balanced",
            "persona_narrative": parsed.narrative,
            "budget_sensitive": parsed.budget_sensitive,
            "domains": parsed.domains,
        }

        stage1_pool = self._stage1_retrieve(
            parsed=parsed,
            interests=interests,
            cold_start=parsed.cold_start,
            cross_domain=cross_domain,
        )

        recommendations, agent_reasoning = self._stage2_rerank(
            user_model=user_model,
            parsed=parsed,
            interests=interests,
            pool=stage1_pool,
            top_k=k,
            cold_start=parsed.cold_start,
            cross_domain=cross_domain,
        )

        return {
            "recommendations": recommendations,
            "agent_reasoning": agent_reasoning,
        }

    def _stage1_retrieve(
        self,
        *,
        parsed: ParsedPersona,
        interests: List[str],
        cold_start: bool,
        cross_domain: bool,
    ) -> List[Tuple[CatalogItem, float]]:
        """Top-30 candidates from corpus metadata (titles/domains unchanged)."""
        return retrieve_top_k(
            interests=interests,
            context=parsed.narrative,
            limit=30,
            cold_start=cold_start,
            cross_domain=cross_domain,
            location=parsed.location,
            tone_notes=None,
        )

    def _stage2_rerank(
        self,
        *,
        user_model: Dict[str, Any],
        parsed: ParsedPersona,
        interests: List[str],
        pool: List[Tuple[CatalogItem, float]],
        top_k: int,
        cold_start: bool,
        cross_domain: bool,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """LLM assigns confidence scores; titles/domains come from stage-1 corpus rows."""
        persona_block = parsed.narrative.strip()[:4000]

        lines = [
            f"- id={item.item_id} | domain={item.domain} | title={item.title} | stage1={score:.3f}"
            for item, score in pool
        ]
        catalog_blob = "\n".join(lines)

        monologue_seed = self._build_persona_monologue(parsed, interests, cold_start, cross_domain)

        system = (
            "You are a Nigerian cross-category recommendation agent (Food, Movies, Drinks, Tech, etc.). "
            "Read ONLY the user persona narrative — there is no separate search query. "
            "Reason about lifestyle, financial limits, location, and cross-domain tastes, then rank candidates. "
            "Use exact item_id values from the list; do not invent or rename titles. "
            "Output ONLY valid JSON: "
            '{"agent_reasoning": "...", "rankings": [{"item_id": "...", "confidence_score": 0.0-1.0}]} '
            f"Return exactly {top_k} items, best-first. confidence_score must reflect persona fit."
        )
        user_msg = (
            f"{monologue_seed}\n\n"
            f"USER PERSONA (sole input):\n{persona_block}\n\n"
            f"CANDIDATES — metadata from corpus (stage-1 top {len(pool)}):\n{catalog_blob}\n\n"
            f"Rank top {top_k}. agent_reasoning must explain persona-driven weights "
            "(budget, lifestyle, location) — do not mention a manual query string."
        )

        raw = self.llm.generate(user_msg, system=system, temperature=0.25).text.strip()
        parsed_json = self._parse_json(raw)
        id_to_item = {item.item_id: (item, s1) for item, s1 in pool}

        if parsed_json and parsed_json.get("rankings"):
            agent_reasoning = str(parsed_json.get("agent_reasoning", "")).strip() or monologue_seed
            recs: List[Dict[str, Any]] = []
            for entry in parsed_json["rankings"][:top_k]:
                iid = str(entry.get("item_id", ""))
                conf = float(entry.get("confidence_score", 0.5))
                if iid in id_to_item:
                    item, _s1 = id_to_item[iid]
                    recs.append(
                        {
                            "item_id": item.item_id,
                            "title": item.title,
                            "domain": item.domain,
                            "confidence_score": round(min(1.0, max(0.0, conf)), 4),
                        }
                    )
            if recs:
                return recs, agent_reasoning

        agent_reasoning = (
            f"{monologue_seed} Stage-2 LLM rerank unavailable; confidence derived from "
            "stage-1 persona match scores."
        )
        max_s = max((s for _, s in pool), default=1.0) or 1.0
        recs = []
        for item, s1 in pool[:top_k]:
            recs.append(
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "domain": item.domain,
                    "confidence_score": round(min(1.0, s1 / max_s), 4),
                }
            )
        return recs, agent_reasoning

    @staticmethod
    def _build_persona_monologue(
        parsed: ParsedPersona,
        interests: List[str],
        cold_start: bool,
        cross_domain: bool,
    ) -> str:
        lines = [
            "Reason-Before-Recommend (persona-only evaluation):",
            f"- Inferred location: {parsed.location}",
            f"- Lifestyle / category signals: {', '.join(parsed.domains[:6]) or 'general'}",
            f"- Interest weights: {', '.join(interests[:8])}",
        ]
        if parsed.budget_sensitive:
            lines.append(
                "- Financial constraint: budget-sensitive persona — down-rank premium-tier items."
            )
        else:
            lines.append("- Financial constraint: no strict budget cap detected from persona.")
        if cold_start:
            lines.append(
                "- Cold-start: sparse persona → demographic priors for popular localized picks."
            )
        if cross_domain:
            lines.append(
                "- Cross-domain: bridging tastes (e.g. social food energy → movies/experiences)."
            )
        return "\n".join(lines)

    @staticmethod
    def _expand_cross_domain_interests(interests: List[str], persona_narrative: str) -> List[str]:
        expanded = list(interests)
        blob = f"{' '.join(interests)} {persona_narrative}".lower()
        for archetype, domains in ARCHETYPE_BRIDGES.items():
            if archetype in blob:
                expanded.extend(domains)
        return list(dict.fromkeys(expanded))

    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip()
        try:
            if raw.startswith("{"):
                return json.loads(raw)
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return None
