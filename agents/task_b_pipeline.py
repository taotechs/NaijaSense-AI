"""Task B — 2-stage retrieval (top-30) + LLM agentic reranker."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.candidate_catalog import CatalogItem, retrieve_top_k
from core.nigerian_defaults import apply_cold_start_interests, build_persona_context
from models.llm_wrapper import LLMWrapper
from utils.config import settings

# Cross-domain archetype bridges (food history → entertainment, etc.).
ARCHETYPE_BRIDGES = {
    "social": ("entertainment", "experiences", "food"),
    "spicy": ("food", "entertainment"),
    "budget": ("food", "services", "tech", "fashion"),
    "student": ("food", "tech", "services", "experiences"),
    "relax": ("wellness", "entertainment", "books"),
    "tech": ("tech", "services"),
}


class TaskBPipelineAgent:
    """Stage-1 retrieval → Stage-2 Reason-Before-Recommend LLM rerank."""

    def __init__(self, router_llm: Optional[LLMWrapper] = None) -> None:
        self.llm = router_llm or LLMWrapper(role="router")

    def run(
        self,
        *,
        user_model: Dict[str, Any],
        interests: List[str],
        context: str | None,
        top_k: int = 5,
        cold_start: bool = False,
        cross_domain: bool = False,
        explicit_titles: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        # Cold-start intercept: empty history indices → demographic priors.
        if cold_start or not interests:
            cold_start = True
            interests, _ = apply_cold_start_interests(interests)

        if cross_domain:
            interests = self._expand_cross_domain_interests(interests, context)

        stage1_pool = self._stage1_retrieve(
            interests=interests,
            context=context,
            cold_start=cold_start,
            cross_domain=cross_domain,
            explicit_titles=explicit_titles,
        )

        recommendations, agent_reasoning = self._stage2_rerank(
            user_model=user_model,
            interests=interests,
            context=context,
            pool=stage1_pool,
            top_k=top_k,
            cold_start=cold_start,
            cross_domain=cross_domain,
        )

        return {
            "recommendations": recommendations,
            "agent_reasoning": agent_reasoning,
        }

    def _stage1_retrieve(
        self,
        *,
        interests: List[str],
        context: str | None,
        cold_start: bool,
        cross_domain: bool,
        explicit_titles: Optional[Sequence[str]],
    ) -> List[Tuple[CatalogItem, float]]:
        """Fetch top-30 candidates by semantic persona match."""
        if explicit_titles:
            from core.candidate_catalog import CATALOG

            by_title = {c.title.lower(): c for c in CATALOG}
            pool: List[Tuple[CatalogItem, float]] = []
            for idx, title in enumerate(explicit_titles[:30]):
                key = title.strip().lower()
                if key in by_title:
                    pool.append((by_title[key], 1.0 - idx * 0.01))
                else:
                    slug = re.sub(r"[^a-z0-9]+", "_", key)[:40] or f"custom_{idx}"
                    pool.append(
                        (
                            CatalogItem(f"custom_{slug}", title.strip(), "general", ()),
                            0.5 - idx * 0.01,
                        )
                    )
            if len(pool) >= 3:
                return pool[:30]

        return retrieve_top_k(
            interests=interests,
            context=context,
            limit=30,
            cold_start=cold_start,
            cross_domain=cross_domain,
        )

    def _stage2_rerank(
        self,
        *,
        user_model: Dict[str, Any],
        interests: List[str],
        context: str | None,
        pool: List[Tuple[CatalogItem, float]],
        top_k: int,
        cold_start: bool,
        cross_domain: bool,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Reason-Before-Recommend: LLM reranks stage-1 pool with dynamic weights."""
        persona_context = build_persona_context(
            location=user_model.get("location"),
            interests=interests,
            history=None,
            tone_notes=user_model.get("tone_notes") if isinstance(user_model.get("tone_notes"), str) else None,
            context=context,
        )

        lines = [
            f"- id={item.item_id} | domain={item.domain} | title={item.title} | s1_score={score}"
            for item, score in pool
        ]
        catalog_blob = "\n".join(lines)

        monologue_prefix = [
            "Reason-Before-Recommend internal monologue:",
            f"Persona location={user_model.get('location', 'Nigeria')}; "
            f"interests={', '.join(interests[:8])}; bias={user_model.get('bias', 'balanced')}.",
        ]
        if cold_start:
            monologue_prefix.append(
                "Cold-start intercept: applied demographic default weights "
                "(popular localized food, budget tech, weekend experiences)."
            )
        if cross_domain:
            monologue_prefix.append(
                "Cross-domain: mapped behavioural archetypes (e.g. social/street-food energy) "
                "to entertainment and experience candidates."
            )
        if context and re.search(r"\b(student|campus|budget|cheap|10k)\b", context.lower()):
            monologue_prefix.append(
                "Context boost: budget/student intent increases weight on affordable picks."
            )

        system = (
            "You are a Nigerian recommendation agent. "
            "First reason about persona-context fit, then rank items. "
            "Output ONLY valid JSON: "
            '{"agent_reasoning": "...", "rankings": [{"item_id": "...", "confidence_score": 0.0-1.0}]} '
            "Include exactly the requested number of top picks in rankings, ordered best-first."
        )
        user_msg = (
            f"{chr(10).join(monologue_prefix)}\n\n"
            f"PROFILE:\n{persona_context}\n\n"
            f"CANDIDATES (stage-1 top {len(pool)}):\n{catalog_blob}\n\n"
            f"Return top {top_k} in rankings with confidence_score in [0,1]."
        )

        raw = self.llm.generate(user_msg, system=system, temperature=0.25).text.strip()
        parsed = self._parse_json(raw)
        id_to_item = {item.item_id: (item, score) for item, score in pool}

        if parsed and parsed.get("rankings"):
            agent_reasoning = str(parsed.get("agent_reasoning", "")).strip()
            if not agent_reasoning:
                agent_reasoning = " ".join(monologue_prefix)
            recs: List[Dict[str, Any]] = []
            for rank in parsed["rankings"][:top_k]:
                iid = str(rank.get("item_id", ""))
                conf = float(rank.get("confidence_score", 0.5))
                if iid in id_to_item:
                    item, s1 = id_to_item[iid]
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

        # Deterministic fallback: stage-1 order normalized to confidence.
        agent_reasoning = " ".join(monologue_prefix) + " LLM rerank unavailable; using stage-1 retrieval order."
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
    def _expand_cross_domain_interests(interests: List[str], context: str | None) -> List[str]:
        expanded = list(interests)
        blob = f"{' '.join(interests)} {context or ''}".lower()
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
