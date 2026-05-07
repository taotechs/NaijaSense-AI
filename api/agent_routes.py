"""Unified agent gateway: one endpoint, LLM/heuristic routing to Task A or B."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.deps import api_logger, orchestrator
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


def _enriched_query(payload: AgentGatewayRequest) -> str:
    parts: list[str] = []
    if payload.user_persona.history:
        parts.append(f"[Background]\n{payload.user_persona.history.strip()[:6000]}")
    parts.append(payload.query.strip())
    if payload.user_persona.tone_notes:
        parts.append(f"[Tone / style notes]\n{payload.user_persona.tone_notes.strip()[:1500]}")
    return "\n\n".join(parts)


def _to_user_profile(payload: AgentGatewayRequest) -> UserProfile:
    up = payload.user_persona
    interests = list(up.interests) if up.interests else ["general lifestyle"]
    return UserProfile(
        user_id=up.user_id,
        location=up.location,
        interests=interests,
        sentiment_bias=up.sentiment_bias or "balanced",
    )


@router.post("/v1", response_model=AgentGatewayResponse)
def agent_gateway(payload: AgentGatewayRequest) -> AgentGatewayResponse:
    enriched = _enriched_query(payload)
    persona_dict = payload.user_persona.model_dump()
    try:
        decision, routing_source = route_query(persona_dict, enriched)
    except Exception as exc:
        api_logger.exception("agent route_query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "routing_failed", "detail": str(exc)},
        ) from exc

    profile = _to_user_profile(payload)
    persona_style = (decision.persona_style or settings.default_persona_style).strip()

    if decision.task == "review":
        item_name = (decision.item_name or "this experience").strip()
        item_context = (decision.item_context or payload.query).strip()[:1000]
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
    ctx = (decision.context or payload.query).strip()[:1000]
    req = RecommendationRequest(
        user_profile=profile,
        candidate_items=candidates,
        context=ctx,
        top_k=min(payload.top_k, len(candidates)),
        recommender_personality="nigerian_twitter",
        conversational_mode=True,
        conversation_history=[],
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
