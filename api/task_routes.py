"""Hackathon Task A / Task B endpoints (dual-link submission)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from agents.task_a_two_pass import TaskATwoPassAgent
from agents.task_b_pipeline import TaskBPipelineAgent
from api.deps import api_logger, orchestrator
from api.task_page import task_endpoint_html
from core.nigerian_defaults import apply_cold_start_interests
from core.task_a_inputs import parse_task_a_inputs
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
    "user_persona": (
        "Lagos-based foodie in Yaba, mid-20s, balanced but honest reviewer. "
        "Cares about value-for-money, wait times, and authentic local taste — "
        "writes in relatable Nigerian English, not overly formal."
    ),
    "product_details": (
        "Iya Eba Amala Spot — Saturday lunch with a friend. Amala was soft, "
        "egusi rich without too much oil, about ₦2,000 each, waited roughly 20 minutes."
    ),
}

_TASK_B_EXAMPLE = {
    "user_persona": {
        "user_id": "demo_user",
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
            description="Input: user_persona + product_details strings. Output: rating + review_text.",
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
    Task A: unified persona + product text → star rating + aligned review_text.
    """
    parsed = parse_task_a_inputs(payload.user_persona, payload.product_details)
    user_model = parsed.to_user_model()

    try:
        query = f"{parsed.item_name} {parsed.item_context}".strip()
        retrieved = (
            orchestrator.corpus_store.search(query, top_k=3)
            if orchestrator.corpus_store
            else []
        )

        result = _task_a_agent.run(
            user_model=user_model,
            item_name=parsed.item_name,
            item_context=parsed.item_context,
            retrieved_examples=retrieved,
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
