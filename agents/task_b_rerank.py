"""Task B stage-2: Groq router rank → Groq generator paragraph."""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from agents.task_b_errors import TaskBRerankError
from core.candidate_catalog import CatalogItem
from core.recommendation_items import canonical_item_title, display_domain
from models.llm_wrapper import LLMWrapper
from utils.config import settings
from utils.task_schemas import TaskBRankResponse, TaskBResponse

_NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d+[\.\)]\s+")

_GENERATOR_FEW_SHOT = (
    "EXAMPLE - student in Yaba on ₦10k/week, locked picks only:\n"
    "Picks: Late-night Akara & Pap - Yaba | Nollywood weekend drama pick | "
    "Local Jollof Kitchen - Surulere\n"
    "Paragraph: If you are stretching a tight weekly budget around Yaba, grab Late-night "
    "Akara & Pap for a cheap, filling bite before a Nollywood weekend drama pick with friends. "
    "When you want proper rice without Island prices, Local Jollof Kitchen in Surulere keeps "
    "portions honest and the vibe low-key."
)

_ROUTER_SYSTEM = (
    "You are a Nigerian cross-category recommendation ranker. Read ONLY the persona - "
    "no separate search query. Reason about budget, location, and lifestyle, then rank "
    "exactly {top_k} item_id values from the candidate list (best-first). "
    "Output ONLY valid JSON: "
    '{{"agent_reasoning": "2-4 sentences on strategy", '
    '"rankings": [{{"item_id": "...", "brief_why": "short"}}]}} '
    "Use item_id values exactly as given; do not invent ids."
)

_GENERATOR_SYSTEM = (
    "You are a Nigerian lifestyle recommendation writer. Write ONE tight paragraph "
    "({sentence_count} sentences max). Rules: no numbered list, no bullets, no markdown; "
    "mention each locked pick exactly ONCE using its display title; no repeated venue names; "
    "no general life/career/education advice (do not say whether someone should learn AI, "
    "change careers, etc.); only describe why each pick fits as a lifestyle/product choice. "
    "Stay on-topic to the locked picks only - no digressions. "
    "Return ONLY the paragraph text.\n\n"
    + _GENERATOR_FEW_SHOT
)

_TEAM_CULTURE_ROUTER_EXTRA = (
    " The persona is about hiring or building a team. Rank team-friendly lifestyle picks "
    "(food outings, experiences, social venues). At most ONE tech accessory if any; avoid "
    "random Amazon gadget SKUs as a hiring strategy."
)

_TEAM_CULTURE_GENERATOR_EXTRA = (
    " The user mentioned hiring or founding a company. You are NOT an HR consultant. "
    "Frame each pick as a welcome perk or team hangout in Lagos (meals, outings, culture) - "
    "never as 'buy this to attract engineers'. Mention the company name at most once."
)

_ADVISORY_ROUTER_EXTRA = (
    " The persona is general advice (learning/career), not a food outing request. "
    "Rank study-friendly tech, books, or tools - avoid random restaurants unless clearly "
    "study-relevant (e.g. cafe with wifi). Max one food pick."
)

_ADVISORY_GENERATOR_EXTRA = (
    " CRITICAL: Do NOT answer whether the user should learn AI or any career question. "
    "Do NOT write 'it is advisable' or similar. Start directly with lifestyle picks that "
    "support someone upskilling (gear, books, quiet cafe). One sentence per pick; "
    "each title appears once only."
)


def rerank_task_b(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
    pool: List[Tuple[CatalogItem, float]],
    team_culture_mode: bool = False,
    advisory_only_mode: bool = False,
) -> TaskBResponse:
    """Rank with Groq router, write paragraph with Groq generator; stage-1 fallback if Groq fails."""
    errors: list[str] = []

    if settings.groq_api_key:
        try:
            return _two_step_groq(
                user_persona=user_persona,
                candidate_items_list=candidate_items_list,
                top_k=top_k,
                pool=pool,
                team_culture_mode=team_culture_mode,
                advisory_only_mode=advisory_only_mode,
            )
        except Exception as exc:
            errors.append(f"Groq two-step: {exc}")
    else:
        errors.append("GROQ_API_KEY not set")

    if pool:
        return _paragraph_from_stage1(pool, top_k=top_k, errors=errors)

    raise TaskBRerankError("; ".join(errors) or "Task B rerank failed.")


def _two_step_groq(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
    pool: List[Tuple[CatalogItem, float]],
    team_culture_mode: bool = False,
    advisory_only_mode: bool = False,
) -> TaskBResponse:
    rank = _router_rank(
        user_persona=user_persona,
        candidate_items_list=candidate_items_list,
        top_k=top_k,
        pool=pool,
        team_culture_mode=team_culture_mode,
        advisory_only_mode=advisory_only_mode,
    )
    locked = _resolve_locked_picks(rank, pool, top_k=top_k)
    paragraph = _generator_paragraph(
        user_persona=user_persona,
        locked_picks=locked,
        top_k=top_k,
        team_culture_mode=team_culture_mode,
        advisory_only_mode=advisory_only_mode,
    )
    paragraph = _polish_paragraph(paragraph)
    paragraph = _collapse_duplicate_title_mentions(paragraph, locked)
    return TaskBResponse(agent_reasoning=rank.agent_reasoning.strip(), recommendations=paragraph)


def _router_rank(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
    pool: List[Tuple[CatalogItem, float]],
    team_culture_mode: bool = False,
    advisory_only_mode: bool = False,
) -> TaskBRankResponse:
    llm = LLMWrapper(role="router")
    user_msg = (
        f"{user_persona}\n\n"
        f"Filtered Database Candidates:\n{candidate_items_list}\n\n"
        f"Rank exactly {top_k} item_id values. Return JSON only."
    )
    system = _ROUTER_SYSTEM.format(top_k=top_k)
    if team_culture_mode:
        system += _TEAM_CULTURE_ROUTER_EXTRA
    if advisory_only_mode:
        system += _ADVISORY_ROUTER_EXTRA
    raw = llm.generate(
        user_msg,
        system=system,
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

    picks: List[TaskBRankedPick] = []
    seen: set[str] = set()
    for item, _ in pool:
        key = canonical_item_title(item.title).lower()
        if key in seen:
            continue
        seen.add(key)
        picks.append(TaskBRankedPick(item_id=item.item_id, brief_why="stage-1 score"))
        if len(picks) >= top_k:
            break
    return picks


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
    return _dedupe_locked_by_title(locked)


def _dedupe_locked_by_title(items: List[CatalogItem]) -> List[CatalogItem]:
    seen: set[str] = set()
    out: List[CatalogItem] = []
    for item in items:
        key = canonical_item_title(item.title).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _display_title(item: CatalogItem) -> str:
    return canonical_item_title(item.title)


def _generator_paragraph(
    *,
    user_persona: str,
    locked_picks: List[CatalogItem],
    top_k: int,
    team_culture_mode: bool = False,
    advisory_only_mode: bool = False,
) -> str:
    if not locked_picks:
        raise TaskBRerankError("No locked picks for paragraph generation.")

    pick_lines = "\n".join(
        f"- {_display_title(item)} ({display_domain(item.domain)})" for item in locked_picks
    )
    sentence_count = min(top_k, len(locked_picks))
    llm = LLMWrapper(role="generator")
    user_msg = (
        f"USER PERSONA:\n{user_persona}\n\n"
        f"LOCKED PICKS (mention each by title - do not add other venues):\n{pick_lines}\n\n"
        "Write the recommendation paragraph now."
    )
    system = _GENERATOR_SYSTEM.format(sentence_count=sentence_count)
    if team_culture_mode:
        system += _TEAM_CULTURE_GENERATOR_EXTRA
    if advisory_only_mode:
        system += _ADVISORY_GENERATOR_EXTRA

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
                if not _paragraph_has_title_repetition(paragraph, locked_picks):
                    if not _paragraph_has_advisory_digression(paragraph, advisory_only_mode):
                        return paragraph

    raise TaskBRerankError("Generator failed to produce a grounded recommendation paragraph.")


_ADVISORY_DIGRESSION_RE = re.compile(
    r"\b(it is advisable|is indeed advisable|you should learn|in this century|"
    r"learning ai|artificial intelligence is|career advice)\b",
    re.I,
)


def _paragraph_has_advisory_digression(paragraph: str, advisory_only_mode: bool) -> bool:
    if not advisory_only_mode:
        return False
    return bool(_ADVISORY_DIGRESSION_RE.search(paragraph))


def _paragraph_has_title_repetition(paragraph: str, picks: List[CatalogItem]) -> bool:
    lower = paragraph.lower()
    for item in picks:
        base = canonical_item_title(item.title).lower()
        if len(base) < 6:
            continue
        if lower.count(base) > 1:
            return True
        # Short brand token repeated too often (e.g. "local jollof kitchen" x5)
        token = base.split(" - ")[0].strip()[:24]
        if len(token) > 8 and lower.count(token) > 1:
            return True
    return False


def _collapse_duplicate_title_mentions(paragraph: str, picks: List[CatalogItem]) -> str:
    """Drop repeated sentences that re-use the same venue name."""
    if not _paragraph_has_title_repetition(paragraph, picks):
        return paragraph
    seen: set[str] = set()
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        s = sentence.strip()
        if not s:
            continue
        lower = s.lower()
        duplicate = False
        for item in picks:
            base = canonical_item_title(item.title).lower()
            if len(base) >= 8 and base in lower:
                if base in seen:
                    duplicate = True
                    break
                seen.add(base)
        if not duplicate:
            kept.append(s)
    return " ".join(kept) if kept else paragraph


def _paragraph_mentions_picks(
    paragraph: str,
    picks: List[CatalogItem],
    *,
    min_hits: int,
) -> bool:
    lower = paragraph.lower()
    hits = 0
    for item in picks:
        title = canonical_item_title(item.title or "").lower()
        tokens = [t for t in re.findall(r"[a-z0-9]+", title) if len(t) > 3]
        if not tokens:
            continue
        if any(tok in lower for tok in tokens[:4]):
            hits += 1
            continue
        if title[:12] in lower:
            hits += 1
    return hits >= min_hits


def _polish_paragraph(text: str) -> str:
    """Trim repetitive boilerplate common in founder/hiring personas."""
    out = text
    for phrase in (
        "learning ai in this century is indeed advisable",
        "learning ai in this century is advisable",
        "it is advisable",
        "is indeed advisable",
        "in this century",
        "to attract top tech talent",
        "to attract top data scientists",
        "to attract and retain top talent",
        "attract top data scientists",
        "attract top tech talent",
        "in the competitive field of data science",
    ):
        while phrase.lower() in out.lower():
            idx = out.lower().find(phrase.lower())
            out = (out[:idx] + out[idx + len(phrase) :]).strip()
            out = re.sub(r"\s{2,}", " ", out)
            out = re.sub(r"\s+([,.])", r"\1", out)
    out = re.sub(r"\b(which is why|meanwhile,|by offering)\b", "", out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


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


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group(0))
    raise TaskBRerankError("Could not parse JSON from LLM output.")
