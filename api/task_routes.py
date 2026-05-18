"""Hackathon Task A / Task B endpoints (dual-link submission)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from api.deps import api_logger, orchestrator
from api.task_page import task_endpoint_html
from core.nigerian_defaults import (
    apply_cold_start_interests,
    build_persona_context,
    candidates_for_persona,
)
from utils.schemas import ItemData, RecommendationRequest, ReviewSimulationRequest, UserProfile
from utils.task_schemas import (
    RankedRecommendation,
    TaskARequest,
    TaskAResponse,
    TaskBRequest,
    TaskBResponse,
)

router = APIRouter(tags=["hackathon-tasks"])


def _persona_to_profile(persona) -> UserProfile:  # noqa: ANN001 — UserPersona
    interests, _ = apply_cold_start_interests(list(persona.interests or []))
    tone_pref = None
    notes = (persona.tone_notes or "").lower()
    if any(k in notes for k in ("pidgin", "naija", "nigerian", "twitter")):
        tone_pref = "slang-heavy"
    elif any(k in notes for k in ("formal", "professional", "neutral")):
        tone_pref = "formal"
    return UserProfile(
        user_id=persona.user_id,
        location=persona.location or "Lagos, Nigeria",
        interests=interests,
        tone_preference=tone_pref,
        sentiment_bias=persona.sentiment_bias or "balanced",
    )


def _normalise_language(raw: str | None) -> str:
    val = (raw or "english").lower().strip()
    return val if val in ("english", "pidgin", "yoruba_mix") else "english"


_TASK_A_EXAMPLE = {
    "user_persona": {
        "user_id": "judge_demo",
        "location": "Lagos",
        "interests": ["street food"],
        "sentiment_bias": "balanced",
    },
    "product_details": {
        "item_name": "Iya Eba Amala Spot",
        "item_context": "Saturday lunch, amala soft, about 2k each.",
    },
}

_TASK_B_EXAMPLE = {
    "user_persona": {
        "user_id": "judge_demo",
        "location": "Yaba, Lagos",
        "interests": ["food"],
    },
    "context": "Cheap weekend food spots nearby.",
    "top_k": 5,
}


@router.get(
    "/task-a/user-modeling",
    response_class=HTMLResponse,
    include_in_schema=False,
    summary="Task A — browser info page (use POST to run)",
)
def task_a_user_modeling_get() -> HTMLResponse:
    return HTMLResponse(
        task_endpoint_html(
            task_name="Task A — User modeling",
            path="/task-a/user-modeling",
            description="Input: user persona + product details. Output: rating (1–5) + review text.",
            example_body=_TASK_A_EXAMPLE,
        )
    )


@router.get(
    "/task-b/recommendation",
    response_class=HTMLResponse,
    include_in_schema=False,
    summary="Task B — browser info page (use POST to run)",
)
def task_b_recommendation_get() -> HTMLResponse:
    return HTMLResponse(
        task_endpoint_html(
            task_name="Task B — Recommendation",
            path="/task-b/recommendation",
            description="Input: user persona (+ optional context). Output: ranked recommendations.",
            example_body=_TASK_B_EXAMPLE,
        )
    )


@router.post(
    "/task-a/user-modeling",
    response_model=TaskAResponse,
    summary="Task A — User modeling (review + rating)",
)
def task_a_user_modeling(payload: TaskARequest) -> TaskAResponse:
    """
    **Input:** user persona + product details.

    **Output:** simulated star rating (1–5) and a written review with Nigerian consumer fidelity.
    """
    profile = _persona_to_profile(payload.user_persona)
    language = _normalise_language(payload.user_persona.language)
    persona_style = (payload.persona_style or "nigerian_twitter").strip()

    # Thread optional pasted history into item context when product context is thin.
    item_context = (payload.product_details.item_context or "").strip()
    if not item_context and payload.user_persona.history:
        item_context = payload.user_persona.history.strip()[:1000]

    req = ReviewSimulationRequest(
        user_profile=profile,
        item_data=ItemData(
            item_name=payload.product_details.item_name,
            item_context=item_context or None,
            category=payload.product_details.category,
        ),
        persona_style=persona_style,
    )
    try:
        res = orchestrator.simulate_review(req, language=language)
    except ValueError as exc:
        api_logger.warning("task-a validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "detail": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover
        api_logger.exception("task-a failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "detail": "Task A failed."},
        ) from exc

    return TaskAResponse(
        rating=res.rating,
        review=res.review_text,
        reasoning_steps=res.reasoning_steps,
        persona_breakdown=res.persona_breakdown,
    )


@router.post(
    "/task-b/recommendation",
    response_model=TaskBResponse,
    summary="Task B — Personalized recommendation (ranked list)",
)
def task_b_recommendation(payload: TaskBRequest) -> TaskBResponse:
    """
    **Input:** user persona (optional context query).

    **Output:** ranked recommendations with Reason-Before-Recommend chain-of-thought.
    """
    persona = payload.user_persona
    interests, cold_start = apply_cold_start_interests(list(persona.interests or []))
    profile = _persona_to_profile(persona)
    profile.interests = interests
    language = _normalise_language(persona.language)

    context_blob = build_persona_context(
        location=persona.location,
        interests=interests,
        history=persona.history,
        tone_notes=persona.tone_notes,
        context=payload.context,
    )

    candidates = [c.strip() for c in (payload.candidate_items or []) if c and c.strip()]
    cross_domain = False
    if not candidates:
        cross_domain = not interests or len(set(i.lower() for i in interests)) >= 3
        candidates = candidates_for_persona(interests, payload.context, cross_domain=cross_domain)
    if len(candidates) < 3:
        candidates = candidates_for_persona(interests, payload.context, cross_domain=True)

    req = RecommendationRequest(
        user_profile=profile,
        candidate_items=candidates,
        context=context_blob,
        top_k=min(payload.top_k, len(candidates)),
        recommender_personality="nigerian_twitter",
        conversational_mode=False,
    )
    try:
        res = orchestrator.recommend(req, language=language)
    except ValueError as exc:
        api_logger.warning("task-b validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "detail": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover
        api_logger.exception("task-b failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "detail": "Task B failed."},
        ) from exc

    explain = res.explainability or {}
    cot = list(explain.get("chain_of_thought") or [])
    # Prepend explicit Reason-Before-Recommend lines from orchestrator step 2.
    reason_before = [
        f"Persona scan: location={persona.location or 'Nigeria'}, "
        f"interests={', '.join(interests[:5])}, bias={persona.sentiment_bias or 'balanced'}.",
    ]
    if cold_start:
        reason_before.append(
            "Cold-start: no prior interests — applied Nigerian default priors "
            "(street food, value-for-money, local experiences)."
        )
    if cross_domain:
        reason_before.append(
            "Cross-domain: expanded candidate pool across food, tech, and lifestyle."
        )
    if persona.history:
        reason_before.append("Incorporated pasted user history into context scoring.")
    chain_of_thought = reason_before + cot

    ranked = [
        RankedRecommendation(
            rank=idx,
            item_name=item.item_name,
            score=item.score,
            explanation=item.explanation,
        )
        for idx, item in enumerate(res.recommendations, start=1)
    ]

    return TaskBResponse(
        recommendations=ranked,
        chain_of_thought=chain_of_thought,
        reasoning_steps=res.reasoning_steps,
        scenario_flags={
            "cold_start": cold_start or bool(explain.get("cold_start")),
            "cross_domain": cross_domain or bool(explain.get("cross_domain")),
        },
    )
