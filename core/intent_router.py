"""LangChain-backed routing: Task A (review) vs Task B (recommendation)."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from utils.config import settings
from utils.logger import get_logger

_log = get_logger("naijasense.intent_router")


class RouterDecision(BaseModel):
    task: Literal["review", "recommend"] = Field(
        description="review = simulate a review for one item; recommend = rank suggestions"
    )
    rationale: str = Field(description="Brief justification for routing")
    item_name: Optional[str] = None
    item_context: Optional[str] = None
    candidate_items: list[str] = Field(default_factory=list)
    context: Optional[str] = None
    persona_style: Optional[str] = Field(
        default=None, description="nigerian_twitter or formal when generating reviews"
    )


ROUTER_SYSTEM = """You route user messages for NaijaSense AI (Nigerian-flavored reviews & recommendations).

- task=review: User wants a written review, rating, or feedback about ONE specific thing they tried or bought.
  Set item_name and item_context from the message. Keep item_name short (the product/place/experience).

- task=recommend: User asks what to pick — food, watch, buy, gift ideas, "suggest", "what should I", planning, cold-start discovery.
  Fill candidate_items with 4–8 concrete, diverse short names. Include cross-domain ideas when it helps
  (e.g. after discussing books, a calming drink or café still fits).

Prefer review if they name one item and describe an experience. Prefer recommend for open choice.
persona_style defaults to formal/neutral. Use nigerian_twitter only when the user explicitly asks for Nigerian slang/local tone.
"""


def _looks_like_explicit_review_request(q: str) -> bool:
    if re.search(r"\b(review|rate|rating)\b.*\b(for|about|on)\b", q, re.I):
        return True
    if "out of 5" in q or "/5" in q:
        return True
    return False


def _extract_item_from_query(query: str) -> tuple[str, str]:
    m = re.search(r"\b(?:review|rate)\s+(?:for|about|on)?\s*[:\-]?\s*(.+?)(?:[.?!]|$)", query, re.I)
    if m:
        name = m.group(1).split(",")[0].strip()[:120]
        return name, query[:800]
    quoted = re.search(r"['\"]([^'\"]{2,80})['\"]", query)
    if quoted:
        return quoted.group(1).strip(), query[:800]
    words = re.findall(r"[A-Za-z]{3,}", query)
    if words:
        return words[0][:80], query[:800]
    return "this experience", query[:800]


def _default_candidates_from_query(query: str, interests: list[str]) -> list[str]:
    q = query.lower()
    pool: list[str] = []
    if any(k in q for k in ("relax", "calm", "chill", "unwind", "stress")):
        pool.extend(
            [
                "Cozy cafe corner",
                "Chamomile tea set",
                "Ambient playlist session",
                "Soft lamp reading nook",
            ]
        )
    if any(k in q for k in ("eat", "food", "buka", "amala", "restaurant", "hungry")):
        pool.extend(
            [
                "Iya Eba Amala Spot",
                "Suya & Chill Stand",
                "Shawarma Alley",
                "Local Jollof Kitchen",
            ]
        )
    if any(k in q for k in ("watch", "movie", "series", "show", "netflix")):
        pool.extend(["Nollywood drama pick", "Light comedy night", "Docu mini-series", "Family-friendly feature"])
    if any(k in q for k in ("buy", "gadget", "tech", "phone")):
        pool.extend(["Budget earbuds", "Power bank slim", "USB-C hub", "Fitness band"])
    if any(k in q for k in ("read", "book", "novel")):
        pool.extend(["African lit bestseller", "Cozy café + new paperback", "Hibiscus tea pairing"])
    pool.extend(["Weekend outing idea", "Budget-friendly comfort pick", "Quick practical choice"])
    dedup: list[str] = []
    seen: set[str] = set()
    for p in pool:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(p)
    return dedup[:8]


def _infer_persona_style(query: str, user_persona: dict[str, Any]) -> str:
    text = f"{query} {user_persona.get('tone_notes', '')}".lower()
    if any(k in text for k in ("nigerian", "naija", "pidgin", "street slang", "twitter tone")):
        return "nigerian_twitter"
    if any(k in text for k in ("formal", "neutral", "professional", "plain english")):
        return "formal"
    return settings.default_persona_style


def heuristic_route(query: str, user_persona: dict[str, Any], persona_interests: list[str]) -> RouterDecision:
    q = query.lower().strip()
    persona_style = _infer_persona_style(query, user_persona)
    rec_signals = (
        "recommend",
        "suggest",
        "what should i",
        "what to ",
        "give me ideas",
        "ideas for",
        "options for",
        "help me pick",
        "where can i",
        "best place",
        "gift for",
    )
    if any(s in q for s in rec_signals) and not _looks_like_explicit_review_request(q):
        return RouterDecision(
            task="recommend",
            rationale="Heuristic: open-ended suggestion / discovery request.",
            candidate_items=_default_candidates_from_query(query, persona_interests),
            context=query,
            persona_style=persona_style,
        )
    name, ctx = _extract_item_from_query(query)
    return RouterDecision(
        task="review",
        rationale="Heuristic: specific item or experience to review.",
        item_name=name,
        item_context=ctx,
        persona_style=persona_style,
    )


def _api_key_ok(provider: str) -> bool:
    if provider == "openai":
        return bool(settings.openai_api_key)
    if provider == "groq":
        return bool(settings.groq_api_key)
    return False


def invoke_llm_router(user_persona: dict[str, Any], query: str, provider: str) -> RouterDecision:
    from langchain_core.prompts import ChatPromptTemplate

    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.2)
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=settings.orchestrator_model, api_key=settings.openai_api_key, temperature=0.2)

    structured = llm.with_structured_output(RouterDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM),
            ("human", "User persona (JSON):\n{persona}\n\nQuery:\n{query}"),
        ]
    )
    chain = prompt | structured
    return chain.invoke(
        {
            "persona": json.dumps(user_persona, ensure_ascii=False, indent=2),
            "query": query,
        }
    )


def route_query(user_persona: dict[str, Any], query: str) -> tuple[RouterDecision, Literal["llm", "heuristic"]]:
    provider = (settings.orchestrator_provider or "none").lower().strip()
    interests = list(user_persona.get("interests") or [])
    if provider in ("openai", "groq") and _api_key_ok(provider):
        try:
            decision = invoke_llm_router(user_persona, query, provider)
            if not (decision.persona_style or "").strip():
                decision.persona_style = _infer_persona_style(query, user_persona)
            if decision.task == "recommend" and len(decision.candidate_items) < 2:
                decision.candidate_items = _default_candidates_from_query(query, interests)
            if decision.task == "review" and not (decision.item_name or "").strip():
                n, c = _extract_item_from_query(query)
                decision.item_name = n
                decision.item_context = decision.item_context or c
            return decision, "llm"
        except Exception as exc:
            _log.warning("LLM router failed; using heuristic: %s", exc)
    return heuristic_route(query, user_persona, interests), "heuristic"
