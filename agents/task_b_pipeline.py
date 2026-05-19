"""Task B — stage-1 retrieval (top-30) + Gemini structured rerank (persona-only)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agents.task_b_gemini import TaskBRerankError, rerank_with_gemini
from core.candidate_catalog import CatalogItem, retrieve_top_k
from core.persona_parser import ParsedPersona, parse_task_b_persona
from core.recommendation_items import display_domain
from utils.task_schemas import TaskBResponse

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
    """Stage-1 corpus retrieval → Stage-2 Gemini Reason-Before-Recommend rerank."""

    DEFAULT_TOP_K = 10

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

        stage1_pool = self._stage1_retrieve(
            parsed=parsed,
            interests=interests,
            cold_start=parsed.cold_start,
            cross_domain=cross_domain,
        )

        if not stage1_pool:
            raise TaskBRerankError("Stage-1 retrieval returned no candidates.")

        want = min(k, len(stage1_pool))
        recommendations, agent_reasoning = self._stage2_rerank_gemini(
            parsed=parsed,
            interests=interests,
            pool=stage1_pool,
            top_k=want,
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
        return retrieve_top_k(
            interests=interests,
            context=parsed.narrative,
            limit=30,
            cold_start=cold_start,
            cross_domain=cross_domain,
            location=parsed.location,
            tone_notes=None,
        )

    def _stage2_rerank_gemini(
        self,
        *,
        parsed: ParsedPersona,
        interests: List[str],
        pool: List[Tuple[CatalogItem, float]],
        top_k: int,
        cold_start: bool,
        cross_domain: bool,
    ) -> Tuple[List[Dict[str, Any]], str]:
        persona_block = parsed.narrative.strip()[:4000]
        monologue_seed = self._build_persona_monologue(parsed, interests, cold_start, cross_domain)

        candidate_items_list = "\n".join(
            f"- item_id={item.item_id} | domain={item.domain} | title={item.title} | stage1_score={score:.3f}"
            for item, score in pool
        )

        user_persona = f"{monologue_seed}\n\nUSER PERSONA:\n{persona_block}"

        result: TaskBResponse = rerank_with_gemini(
            user_persona=user_persona,
            candidate_items_list=candidate_items_list,
            top_k=top_k,
        )

        allowed_ids = {item.item_id for item, _ in pool}
        catalog_by_id = {item.item_id: item for item, _ in pool}

        recs: List[Dict[str, Any]] = []
        for entry in result.recommendations:
            if entry.item_id not in allowed_ids:
                raise TaskBRerankError(
                    f"Gemini returned item_id not in stage-1 pool: {entry.item_id!r}"
                )
            catalog_item = catalog_by_id[entry.item_id]
            title = (entry.title or "").strip() or catalog_item.title
            domain = (entry.domain or "").strip() or display_domain(catalog_item.domain)
            recs.append(
                {
                    "item_id": entry.item_id,
                    "title": title,
                    "domain": domain,
                    "confidence_score": round(min(1.0, max(0.0, float(entry.confidence_score))), 4),
                }
            )
            if len(recs) >= top_k:
                break

        if len(recs) < top_k:
            raise TaskBRerankError(
                f"Gemini returned {len(recs)} valid recommendations; expected {top_k}."
            )

        reasoning = (result.agent_reasoning or "").strip()
        if not reasoning:
            raise TaskBRerankError("Gemini returned empty agent_reasoning.")

        return recs, reasoning

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
