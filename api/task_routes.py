"""Hackathon Task A / Task B endpoints (dual-link submission)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from agents.task_a_two_pass import TaskATwoPassAgent
from agents.task_b_pipeline import TaskBPipelineAgent
from api.deps import api_logger, orchestrator
from api.task_page import task_endpoint_html
from core.nigerian_defaults import apply_cold_start_interests
from utils.schemas import UserProfile
from utils.task_schemas import (
    RecommendationItem,
    TaskARequest,
    TaskAResponse,
    TaskBRequest,
    TaskBResponse,
)

router = APIRouter(tags=["hackathon-tasks"])

_task_a_agent = TaskATwoPassAgent()
_task_b_agent = TaskBPipelineAgent()


def _persona_to_profile(persona) -> UserProfile:  # noqa: ANN001
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


def _is_cold_start(persona) -> bool:  # noqa: ANN001
    """Cold-start when interests and pasted history are both empty."""
    no_interests = not [i for i in (persona.interests or []) if str(i).strip()]
    no_history = not (persona.history or "").strip()
    return no_interests or (no_interests and no_history)


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
        "persona": (
            "I'm a 22-year-old UNILAG student living in Yaba on a tight ₦10k weekly budget. "
            "I love affordable street food and jollof spots, weekend Nollywood movies with friends, "
            "and occasional smoothies — value-for-money matters more than luxury."
        ),
    },
}


@router.get("/task-a/user-modeling", response_class=HTMLResponse, include_in_schema=False)
def task_a_user_modeling_get() -> HTMLResponse:
    return HTMLResponse(
        task_endpoint_html(
            task_name="Task A — User modeling",
            path="/task-a/user-modeling",
            description="Output: rating + review_reasoning + review_text (two-pass aligned).",
            example_body=_TASK_A_EXAMPLE,
        )
    )


@router.get("/task-b/recommendation", response_class=HTMLResponse, include_in_schema=False)
def task_b_recommendation_get() -> HTMLResponse:
    return HTMLResponse(
        task_endpoint_html(
            task_name="Task B — Recommendation",
            path="/task-b/recommendation",
            description="Input: user_persona only. Output: recommendations[] + agent_reasoning.",
            example_body=_TASK_B_EXAMPLE,
        )
    )


@router.post("/task-a/user-modeling", response_model=TaskAResponse)
def task_a_user_modeling(payload: TaskARequest) -> TaskAResponse:
    """
    Two-pass Task A: Pass 1 locks star rating; Pass 2 writes review aligned to it.
    """
    profile = _persona_to_profile(payload.user_persona)
    language = _normalise_language(payload.user_persona.language)
    persona_style = (payload.persona_style or "nigerian_twitter").strip()

    item_context = (payload.product_details.item_context or "").strip()
    if not item_context and payload.user_persona.history:
        item_context = payload.user_persona.history.strip()[:1000]

    try:
        user_model = orchestrator.prepare_user_model(
            profile,
            persona_style=persona_style,
            tone_notes=payload.user_persona.tone_notes,
        )
        user_model["persona_style"] = persona_style
        user_model["bias"] = profile.sentiment_bias or "balanced"

        query = f"{payload.product_details.item_name} {item_context}".strip()
        retrieved = (
            orchestrator.corpus_store.search(query, top_k=3)
            if orchestrator.corpus_store
            else []
        )

        result = _task_a_agent.run(
            user_model=user_model,
            item_name=payload.product_details.item_name,
            item_context=item_context,
            retrieved_examples=retrieved,
            language=language,
        )
    except ValueError as exc:
        api_logger.warning("task-a validation error: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        api_logger.exception("task-a failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task A failed.",
        ) from exc

    return TaskAResponse(
        rating=result["rating"],
        review_reasoning=result["review_reasoning"],
        review_text=result["review_text"],
    )


@router.post("/task-b/recommendation", response_model=TaskBResponse)
def task_b_recommendation(payload: TaskBRequest) -> TaskBResponse:
    """
    Task B: stage-1 top-30 retrieval → stage-2 LLM Reason-Before-Recommend rerank.
    """
    persona = payload.user_persona

    try:
        result = _task_b_agent.run(
            user_id=persona.user_id,
            persona_narrative=persona.persona,
        )
    except ValueError as exc:
        api_logger.warning("task-b validation error: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        api_logger.exception("task-b failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task B failed.",
        ) from exc

    recs = [
        RecommendationItem(
            item_id=item["item_id"],
            title=item["title"],
            domain=item["domain"],
            confidence_score=item["confidence_score"],
        )
        for item in result["recommendations"]
    ]

    return TaskBResponse(
        recommendations=recs,
        agent_reasoning=result["agent_reasoning"],
    )
