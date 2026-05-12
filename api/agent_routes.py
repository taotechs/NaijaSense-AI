"""Unified agent gateway: one endpoint, LLM/heuristic routing to Task A or B."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, status

from api.deps import (
    api_logger,
    get_recent_turns,
    orchestrator,
    record_user_turn,
)
from core.intent_router import route_query
from utils.config import settings
from utils.schemas import (
    AgentGatewayRequest,
    AgentGatewayResponse,
    AgentRecommendationResult,
    AgentReviewResult,
    ItemData,
    RecommendationRequest,
    ReviewSimulationRequest,
    UserProfile,
)

router = APIRouter(tags=["agent"])

_AMBIGUOUS_QUERY_RE = re.compile(r"^[a-z]{3,10}$", re.I)


def _enriched_query(payload: AgentGatewayRequest) -> str:
    parts: list[str] = []
    if payload.user_persona.history:
        parts.append(f"[Background]\n{payload.user_persona.history.strip()[:6000]}")
    parts.append(payload.query.strip())
    if payload.user_persona.tone_notes:
        parts.append(f"[Tone / style notes]\n{payload.user_persona.tone_notes.strip()[:1500]}")
    return "\n\n".join(parts)

def _tone_overrides(tone_notes: str | None) -> tuple[str | None, str | None]:
    """
    Returns (tone_preference, persona_style) overrides when tone notes are explicit.

    Important: phrases like "avoid slang" or "slang minimal" mean formal/neutral.
    """
    if not tone_notes:
        return None, None
    notes = tone_notes.lower()

    wants_neutral = any(
        k in notes
        for k in (
            "neutral",
            "professional",
            "plain english",
            "clear, natural english",
            "clear natural english",
            "avoid slang",
            "no slang",
            "minimal slang",
            "slang minimal",
            "keep slang minimal",
        )
    )
    wants_naija = any(k in notes for k in ("nigerian", "naija", "pidgin", "twitter tone"))
    wants_slang = any(k in notes for k in ("use slang", "more slang", "slang-heavy", "slang heavy"))

    if wants_neutral and not wants_naija and not wants_slang:
        return "formal", "formal"
    if wants_naija or wants_slang:
        return "slang-heavy", "nigerian_twitter"
    return None, None


def _to_user_profile(payload: AgentGatewayRequest) -> UserProfile:
    up = payload.user_persona
    interests = list(up.interests) if up.interests else ["general lifestyle"]
    tone_pref, _ = _tone_overrides(up.tone_notes)
    return UserProfile(
        user_id=up.user_id,
        location=up.location,
        interests=interests,
        sentiment_bias=up.sentiment_bias or "balanced",
        tone_preference=tone_pref,
    )


def _sanitize_review_item_name(raw: str, query: str) -> str:
    cleaned = (raw or "").strip()
    if cleaned.lower() in {"review", "rate", "rating", "this experience"}:
        m = query.strip()
        m = re.sub(r"^\s*(review|rate)\s*[:\-]?\s*", "", m, flags=re.I)
        m = m.split(",")[0].split(".")[0].strip()
        if m:
            cleaned = m[:120]
    return cleaned or "this experience"


@router.post("/v1", response_model=AgentGatewayResponse)
def agent_gateway(payload: AgentGatewayRequest) -> AgentGatewayResponse:
    raw_query = payload.query.strip()
    if len(raw_query) < 8 and _AMBIGUOUS_QUERY_RE.match(raw_query) and " " not in raw_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ambiguous_query",
                "detail": (
                    "Input is too short/unclear. Provide item/context for review or ask a recommendation question."
                ),
            },
        )

    # Stateful agentic workflow: record this turn into the rolling
    # conversation buffer keyed by user_id BEFORE routing. Multi-turn
    # context is threaded into Task B via ``conversation_history`` on
    # the recommendation request below.
    record_user_turn(payload.user_persona.user_id, raw_query)

    enriched = _enriched_query(payload)
    persona_dict = payload.user_persona.model_dump()
    try:
        decision, routing_source = route_query(persona_dict, raw_query)
    except Exception as exc:
        api_logger.exception("agent route_query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "routing_failed", "detail": str(exc)},
        ) from exc

    profile = _to_user_profile(payload)
    persona_style = (decision.persona_style or settings.default_persona_style).strip()
    _, style_override = _tone_overrides(payload.user_persona.tone_notes)
    if style_override:
        persona_style = style_override

    if decision.task == "review":
        item_name = _sanitize_review_item_name(decision.item_name or "this experience", payload.query)
        item_context = (decision.item_context or enriched).strip()[:1000]
        req = ReviewSimulationRequest(
            user_profile=profile,
            item_data=ItemData(item_name=item_name, item_context=item_context),
            persona_style=persona_style,
        )
        try:
            res = orchestrator.simulate_review(req)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "bad_request", "detail": str(exc)},
            ) from exc
        steps = [f"Routed to Task A (review). {decision.rationale}"] + res.reasoning_steps
        return AgentGatewayResponse(
            task="review",
            orchestrator_rationale=decision.rationale,
            routing_source=routing_source,
            review=AgentReviewResult(
                review_text=res.review_text,
                rating=res.rating,
                persona_breakdown=res.persona_breakdown,
            ),
            reasoning_steps=steps,
        )

    candidates = [c.strip() for c in decision.candidate_items if c and str(c).strip()]
    if len(candidates) < 1:
        candidates = ["Local experience A", "Local experience B", "Local experience C"]
    ctx = (decision.context or enriched).strip()[:1000]
    prior_turns = get_recent_turns(payload.user_persona.user_id, exclude_latest=True)
    req = RecommendationRequest(
        user_profile=profile,
        candidate_items=candidates,
        context=ctx,
        top_k=min(payload.top_k, len(candidates)),
        recommender_personality=("nigerian_twitter" if persona_style == "nigerian_twitter" else "analyst"),
        conversational_mode=True,
        conversation_history=prior_turns,
    )
    try:
        res = orchestrator.recommend(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "detail": str(exc)},
        ) from exc
    steps = [f"Routed to Task B (recommend). {decision.rationale}"] + res.reasoning_steps
    return AgentGatewayResponse(
        task="recommend",
        orchestrator_rationale=decision.rationale,
        routing_source=routing_source,
        recommendation=AgentRecommendationResult(
            recommendations=res.recommendations,
            conversational_response=res.conversational_response,
            explainability=res.explainability or {},
            memory_retrieved=res.memory_retrieved,
        ),
        reasoning_steps=steps,
    )
