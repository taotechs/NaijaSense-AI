"""Task B — diversified stage-1 retrieval + two-step LLM (rank → paragraph)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from agents.task_b_errors import TaskBRerankError
from agents.task_b_rerank import rerank_task_b
from core.candidate_catalog import CatalogItem, retrieve_top_k
from core.persona_parser import ParsedPersona, parse_task_b_persona
from core.task_b_diversify import diversify_stage1_pool
from utils.config import settings
from utils.task_schemas import TaskBResponse

_NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d+[\.\)]\s+")
_NAIRA_RE = re.compile(r"₦\s*[\d,]+|(\d+)\s*k\b", re.I)

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
    """Stage-1 retrieval → router rank → generator paragraph (Groq by default)."""

    DEFAULT_TOP_K = 6
    STAGE1_LIMIT = 30

    def run(
        self,
        *,
        user_id: str,
        persona_narrative: str,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        parsed = parse_task_b_persona(persona_narrative, user_id=user_id)
        k = top_k or getattr(settings, "task_b_top_k", None) or self.DEFAULT_TOP_K
        interests = list(parsed.interests)
        cross_domain = parsed.cold_start or len(set(parsed.domains)) >= 2

        if cross_domain:
            interests = self._expand_cross_domain_interests(interests, parsed.narrative)

        retrieval_context = parsed.retrieval_context or parsed.narrative
        if parsed.team_culture_mode:
            cross_domain = True

        stage1_pool = self._stage1_retrieve(
            parsed=parsed,
            interests=interests,
            cold_start=parsed.cold_start,
            cross_domain=cross_domain,
            context=retrieval_context,
        )

        if not stage1_pool:
            raise TaskBRerankError("Stage-1 retrieval returned no candidates.")

        diversified = diversify_stage1_pool(
            stage1_pool,
            limit=self.STAGE1_LIMIT,
            persona_domains=parsed.domains,
            min_unique_domains=3 if cross_domain or len(parsed.domains) >= 2 else 2,
        )

        want = min(k, len(diversified))
        recommendations_text, agent_reasoning = self._stage2_rerank(
            parsed=parsed,
            interests=interests,
            pool=diversified,
            top_k=want,
            cold_start=parsed.cold_start,
            cross_domain=cross_domain,
            team_culture_mode=parsed.team_culture_mode,
        )

        return {
            "recommendations": recommendations_text,
            "agent_reasoning": agent_reasoning,
        }

    def _stage1_retrieve(
        self,
        *,
        parsed: ParsedPersona,
        interests: List[str],
        cold_start: bool,
        cross_domain: bool,
        context: str | None = None,
    ) -> List[Tuple[CatalogItem, float]]:
        return retrieve_top_k(
            interests=interests,
            context=context or parsed.narrative,
            limit=self.STAGE1_LIMIT,
            cold_start=cold_start,
            cross_domain=cross_domain,
            location=parsed.location,
            tone_notes="budget" if parsed.budget_sensitive else None,
            team_culture_mode=parsed.team_culture_mode,
        )

    def _stage2_rerank(
        self,
        *,
        parsed: ParsedPersona,
        interests: List[str],
        pool: List[Tuple[CatalogItem, float]],
        top_k: int,
        cold_start: bool,
        cross_domain: bool,
        team_culture_mode: bool = False,
    ) -> Tuple[str, str]:
        persona_block = parsed.narrative.strip()[:4000]
        monologue_seed = self._build_persona_monologue(parsed, interests, cold_start, cross_domain)

        candidate_items_list = "\n".join(
            f"- item_id={item.item_id} | domain={item.domain} | title={item.title} | stage1_score={score:.3f}"
            for item, score in pool
        )

        user_persona = f"{monologue_seed}\n\nUSER PERSONA:\n{persona_block}"

        result: TaskBResponse = rerank_task_b(
            user_persona=user_persona,
            candidate_items_list=candidate_items_list,
            top_k=top_k,
            pool=pool,
            team_culture_mode=team_culture_mode,
        )

        return _finalize_paragraph_response(result)

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
        naira = _NAIRA_RE.search(parsed.narrative)
        if naira:
            lines.append(f"- Budget signal detected in persona: {naira.group(0)}")
        if parsed.budget_sensitive:
            lines.append(
                "- Financial constraint: budget-sensitive — avoid premium-tier picks."
            )
        else:
            lines.append("- Financial constraint: no strict budget cap detected from persona.")
        if cold_start:
            lines.append(
                "- Cold-start: sparse persona → demographic priors for popular localized picks."
            )
        if cross_domain:
            lines.append(
                "- Cross-domain: spread recommendations across food, entertainment, and lifestyle."
            )
        if parsed.team_culture_mode:
            lines.append(
                "- Team-culture mode: persona asks about hiring — recommend Lagos lifestyle perks "
                "(team meals, outings, experiences), not HR advice or a shopping list of gadgets."
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


def _finalize_paragraph_response(result: TaskBResponse) -> Tuple[str, str]:
    paragraph = (result.recommendations or "").strip()
    if len(paragraph) < 80:
        raise TaskBRerankError("Recommendations paragraph too short.")
    if _NUMBERED_LIST_RE.search(paragraph):
        raise TaskBRerankError(
            "Recommendations must be one fluid paragraph, not a numbered list."
        )
    reasoning = (result.agent_reasoning or "").strip()
    if not reasoning:
        raise TaskBRerankError("agent_reasoning is empty.")
    return paragraph, reasoning
