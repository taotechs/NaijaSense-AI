"""Task B stage-2: router rank → generator paragraph (Groq default, optional Gemini)."""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from agents.task_b_errors import TaskBRerankError
from core.candidate_catalog import CatalogItem
from core.recommendation_items import display_domain
from models.llm_wrapper import LLMWrapper
from utils.config import settings
from utils.task_schemas import TaskBRankResponse, TaskBResponse

_NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d+[\.\)]\s+")

_GENERATOR_FEW_SHOT = (
    "EXAMPLE — student in Yaba on ₦10k/week, locked picks only:\n"
    "Picks: Late-night Akara & Pap — Yaba | Nollywood weekend drama pick | "
    "Local Jollof Kitchen — Surulere\n"
    "Paragraph: If you are stretching a tight weekly budget around Yaba, grab Late-night "
    "Akara & Pap for a cheap, filling bite before a Nollywood weekend drama pick with friends. "
    "When you want proper rice without Island prices, Local Jollof Kitchen in Surulere keeps "
    "portions honest and the vibe low-key."
)

_ROUTER_SYSTEM = (
    "You are a Nigerian cross-category recommendation ranker. Read ONLY the persona — "
    "no separate search query. Reason about budget, location, and lifestyle, then rank "
    "exactly {top_k} item_id values from the candidate list (best-first). "
    "Output ONLY valid JSON: "
    '{{"agent_reasoning": "2-4 sentences on strategy", '
    '"rankings": [{{"item_id": "...", "brief_why": "short"}}]}} '
    "Use item_id values exactly as given; do not invent ids."
)

_GENERATOR_SYSTEM = (
    "You are a Nigerian lifestyle recommendation writer. Write ONE fluid paragraph only "
    "({sentence_count} sentences woven together). Rules: no numbered list, no bullets, no "
    "markdown; vary sentence openings; mention each locked pick by its human-readable title; "
    "tie choices to budget/location from the persona; light Nigerian English is fine. "
    "Return ONLY the paragraph text — no JSON, no headings.\n\n"
    + _GENERATOR_FEW_SHOT
)


def rerank_task_b(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
    pool: List[Tuple[CatalogItem, float]],
) -> TaskBResponse:
    """Rank with router, write paragraph with generator; Gemini/fallback if Groq fails."""
    mode = (settings.task_b_rerank_provider or "groq").lower().strip()
    errors: list[str] = []

    if mode in ("groq", "auto") and settings.groq_api_key:
        try:
            return _two_step_groq(
                user_persona=user_persona,
                candidate_items_list=candidate_items_list,
                top_k=top_k,
                pool=pool,
            )
        except Exception as exc:
            errors.append(f"Groq two-step: {exc}")

    if mode in ("gemini", "auto"):
        try:
            from agents.task_b_gemini import rerank_with_gemini

            result = rerank_with_gemini(
                user_persona=user_persona,
                candidate_items_list=candidate_items_list,
                top_k=top_k,
            )
            return _finalize_task_b_response(result)
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

    if pool:
        return _paragraph_from_stage1(pool, top_k=top_k, errors=errors)

    raise TaskBRerankError("; ".join(errors) or "Task B rerank failed.")


def _two_step_groq(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
    pool: List[Tuple[CatalogItem, float]],
) -> TaskBResponse:
    rank = _router_rank(
        user_persona=user_persona,
        candidate_items_list=candidate_items_list,
        top_k=top_k,
        pool=pool,
    )
    locked = _resolve_locked_picks(rank, pool, top_k=top_k)
    paragraph = _generator_paragraph(
        user_persona=user_persona,
        locked_picks=locked,
        top_k=top_k,
    )
    return TaskBResponse(agent_reasoning=rank.agent_reasoning.strip(), recommendations=paragraph)


def _router_rank(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
    pool: List[Tuple[CatalogItem, float]],
) -> TaskBRankResponse:
    llm = LLMWrapper(role="router")
    user_msg = (
        f"{user_persona}\n\n"
        f"Filtered Database Candidates:\n{candidate_items_list}\n\n"
        f"Rank exactly {top_k} item_id values. Return JSON only."
    )
    raw = llm.generate(
        user_msg,
        system=_ROUTER_SYSTEM.format(top_k=top_k),
        temperature=0.2,
        max_tokens=700,
    ).text.strip()
    if not raw:
        raise TaskBRerankError("Router rank step returned empty output.")

    data = _parse_json(raw)
    rank = TaskBRankResponse.model_validate(data)
    allowed = {item.item_id for item, _ in pool}
    valid_rankings = []
    for entry in rank.rankings:
        iid = entry.item_id.strip()
        if iid in allowed:
            valid_rankings.append(entry)
    if len(valid_rankings) < min(top_k, len(pool)):
        valid_rankings = _heuristic_rank(pool, top_k=top_k)
        reasoning = (rank.agent_reasoning or "").strip() or "Heuristic rank fallback applied."
        return TaskBRankResponse(agent_reasoning=reasoning, rankings=valid_rankings)

    return TaskBRankResponse(
        agent_reasoning=rank.agent_reasoning,
        rankings=valid_rankings[:top_k],
    )


def _heuristic_rank(
    pool: List[Tuple[CatalogItem, float]],
    *,
    top_k: int,
) -> List:
    from utils.task_schemas import TaskBRankedPick

    return [
        TaskBRankedPick(item_id=item.item_id, brief_why="stage-1 score")
        for item, _ in pool[:top_k]
    ]


def _resolve_locked_picks(
    rank: TaskBRankResponse,
    pool: List[Tuple[CatalogItem, float]],
    *,
    top_k: int,
) -> List[CatalogItem]:
    by_id = {item.item_id: item for item, _ in pool}
    locked: List[CatalogItem] = []
    for entry in rank.rankings[:top_k]:
        item = by_id.get(entry.item_id.strip())
        if item:
            locked.append(item)
    if len(locked) < min(top_k, len(pool)):
        for item, _ in pool:
            if item.item_id not in {p.item_id for p in locked}:
                locked.append(item)
            if len(locked) >= top_k:
                break
    return locked


def _generator_paragraph(
    *,
    user_persona: str,
    locked_picks: List[CatalogItem],
    top_k: int,
) -> str:
    if not locked_picks:
        raise TaskBRerankError("No locked picks for paragraph generation.")

    pick_lines = "\n".join(
        f"- {item.title.strip()} ({display_domain(item.domain)})" for item in locked_picks
    )
    sentence_count = min(top_k, len(locked_picks))
    llm = LLMWrapper(role="generator")
    user_msg = (
        f"USER PERSONA:\n{user_persona}\n\n"
        f"LOCKED PICKS (mention each by title — do not add other venues):\n{pick_lines}\n\n"
        "Write the recommendation paragraph now."
    )
    system = _GENERATOR_SYSTEM.format(sentence_count=sentence_count)

    for attempt in range(2):
        raw = llm.generate(
            user_msg if attempt == 0 else user_msg + "\n\nREMINDER: Use ONLY the locked pick titles above.",
            system=system,
            temperature=0.55 if attempt == 0 else 0.45,
            max_tokens=520,
        ).text.strip()
        paragraph = _clean_paragraph(raw)
        if len(paragraph) >= 80 and not _NUMBERED_LIST_RE.search(paragraph):
            if _paragraph_mentions_picks(paragraph, locked_picks, min_hits=max(1, len(locked_picks) - 1)):
                return paragraph

    raise TaskBRerankError("Generator failed to produce a grounded recommendation paragraph.")


def _paragraph_mentions_picks(
    paragraph: str,
    picks: List[CatalogItem],
    *,
    min_hits: int,
) -> bool:
    lower = paragraph.lower()
    hits = 0
    for item in picks:
        title = (item.title or "").lower()
        tokens = [t for t in re.findall(r"[a-z0-9]+", title) if len(t) > 3]
        if not tokens:
            continue
        if any(tok in lower for tok in tokens[:4]):
            hits += 1
            continue
        if title[:12] in lower:
            hits += 1
    return hits >= min_hits


def _clean_paragraph(raw: str) -> str:
    text = raw.strip().strip('"').strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def _paragraph_from_stage1(
    pool: List[Tuple[CatalogItem, float]],
    *,
    top_k: int,
    errors: list[str],
) -> TaskBResponse:
    picks = pool[:top_k]
    templates = {
        "food": "For something tasty without overspending, {title} is a reliable {domain} pick near your routine.",
        "movie": "When you want a low-cost hangout, {title} fits a {domain} night that will not drain your wallet.",
        "entertainment": "{title} is a fun {domain} option that matches a social, budget-conscious weekend.",
        "tech": "If you need practical value, {title} is a sensible {domain} choice for everyday student life.",
        "drink": "For a treat that still respects your budget, {title} is a solid {domain} stop.",
        "default": "{title} lines up with your lifestyle as a worthwhile {domain} recommendation.",
    }
    sentences: list[str] = []
    for item, _ in picks:
        domain = display_domain(item.domain).lower()
        title = (item.title or "a local pick").strip()
        key = domain if domain in templates else "default"
        sentences.append(templates[key].format(title=title, domain=domain))

    note = " | ".join(errors[:2]) if errors else "LLM unavailable"
    reasoning = (
        "Reason-Before-Recommend: stage-1 corpus scores ranked candidates by persona overlap. "
        f"Prose assembled locally ({note})."
    )
    return TaskBResponse(agent_reasoning=reasoning, recommendations=" ".join(sentences))


def _finalize_task_b_response(result: TaskBResponse) -> TaskBResponse:
    paragraph = (result.recommendations or "").strip()
    if len(paragraph) < 80:
        raise TaskBRerankError("Recommendations paragraph too short.")
    if _NUMBERED_LIST_RE.search(paragraph):
        raise TaskBRerankError("Recommendations must be prose, not a numbered list.")
    if not (result.agent_reasoning or "").strip():
        raise TaskBRerankError("agent_reasoning is empty.")
    return result


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group(0))
    raise TaskBRerankError("Could not parse JSON from LLM output.")
