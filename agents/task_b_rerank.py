"""Task B stage-2: persona-conditioned recommendation paragraph (Groq / Gemini / fallback)."""

from __future__ import annotations

import json
import os
import re
from typing import List, Tuple

from agents.task_b_errors import TaskBRerankError
from core.candidate_catalog import CatalogItem
from core.recommendation_items import display_domain
from models.llm_wrapper import LLMWrapper
from utils.config import settings
from utils.task_schemas import TaskBResponse

_PARAGRAPH_RULES = (
    "Write agent_reasoning first (Reason-Before-Recommend trace). "
    "Then set recommendations to ONE fluid paragraph with exactly {top_k} woven sentences "
    "(no numbered list, no bullets, no markdown). Each sentence names a pick from the candidate "
    "list using human-readable titles — never raw item_id values. Tie picks to the persona's "
    "budget, location, and lifestyle. Output ONLY valid JSON matching this schema: "
    '{{"agent_reasoning": "...", "recommendations": "..."}}'
)


def rerank_task_b(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
    pool: List[Tuple[CatalogItem, float]],
) -> TaskBResponse:
    """
    Produce TaskBResponse using configured provider.

    Default (``groq``): Groq free tier — same key as Task A.
    ``gemini``: Google GenAI when credits are available.
    ``auto``: Groq → Gemini → deterministic paragraph from stage-1.
    """
    mode = (settings.task_b_rerank_provider or "groq").lower().strip()
    errors: list[str] = []

    if mode in ("groq", "auto"):
        try:
            return _rerank_with_groq(
                user_persona=user_persona,
                candidate_items_list=candidate_items_list,
                top_k=top_k,
            )
        except Exception as exc:
            errors.append(f"Groq: {exc}")

    if mode in ("gemini", "auto"):
        try:
            from agents.task_b_gemini import rerank_with_gemini

            return rerank_with_gemini(
                user_persona=user_persona,
                candidate_items_list=candidate_items_list,
                top_k=top_k,
            )
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

    if pool:
        return _paragraph_from_stage1(pool, top_k=top_k, errors=errors)

    raise TaskBRerankError("; ".join(errors) or "Task B rerank failed.")


def _rerank_with_groq(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
) -> TaskBResponse:
    if not settings.groq_api_key:
        raise TaskBRerankError("GROQ_API_KEY is not set.")

    llm = LLMWrapper(role="router")
    system = (
        "You are a Nigerian cross-category recommendation agent. "
        + _PARAGRAPH_RULES.format(top_k=top_k)
    )
    user_msg = (
        f"{user_persona}\n\n"
        f"Filtered Database Candidates:\n{candidate_items_list}\n\n"
        "Return JSON only."
    )
    raw = llm.generate(user_msg, system=system, temperature=0.25, max_tokens=900).text.strip()
    if not raw:
        raise TaskBRerankError("Groq returned empty output.")

    parsed = _parse_task_b_json(raw)
    return _validate_task_b_response(parsed, top_k=top_k)


def _paragraph_from_stage1(
    pool: List[Tuple[CatalogItem, float]],
    *,
    top_k: int,
    errors: list[str],
) -> TaskBResponse:
    """Deterministic prose from stage-1 scores when LLMs are unavailable."""
    picks = pool[:top_k]
    sentences: list[str] = []
    for idx, (item, _score) in enumerate(picks):
        domain = display_domain(item.domain)
        title = (item.title or "a local pick").strip()
        if idx == 0:
            sentences.append(
                f"Given your profile, {title} ({domain}) is the strongest fit for value and lifestyle."
            )
        elif idx == len(picks) - 1:
            sentences.append(
                f"You could also try {title} in the {domain} space — it lines up with your interests without overspending."
            )
        else:
            sentences.append(
                f"{title} is another solid {domain} option that matches your budget and routine."
            )

    note = " | ".join(errors[:2]) if errors else "LLM unavailable"
    reasoning = (
        "Reason-Before-Recommend: ranked stage-1 corpus matches by persona overlap and budget signals. "
        f"Prose assembled locally ({note})."
    )
    return TaskBResponse(agent_reasoning=reasoning, recommendations=" ".join(sentences))


def _parse_task_b_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group(0))
    raise TaskBRerankError("Could not parse JSON from LLM output.")


def _validate_task_b_response(data: dict, *, top_k: int) -> TaskBResponse:
    result = TaskBResponse.model_validate(data)
    paragraph = (result.recommendations or "").strip()
    if len(paragraph) < 80:
        raise TaskBRerankError("Recommendations paragraph too short.")
    if re.search(r"(?m)^\s*\d+[\.\)]\s+", paragraph):
        raise TaskBRerankError("Recommendations must be prose, not a numbered list.")
    if not (result.agent_reasoning or "").strip():
        raise TaskBRerankError("agent_reasoning is empty.")
    return result
