"""Unified agent gateway: one endpoint, LLM/heuristic routing to Task A or B."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from api.deps import (
    api_logger,
    get_recent_turns,
    orchestrator,
    record_user_turn,
)
from core.intent_router import route_query
from core.safety import check_input, check_output
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

# Valid output-language tokens. We normalise unknown values to ``english``
# rather than failing — the agent should always respond, even if the UI
# sends an unsupported tag from a future build.
_VALID_LANGUAGES = ("english", "pidgin", "yoruba_mix")

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


def _normalise_language(raw: Optional[str]) -> str:
    val = (raw or "english").lower().strip()
    return val if val in _VALID_LANGUAGES else "english"


def _run_pipeline(
    payload: AgentGatewayRequest,
    *,
    skip_history: bool,
    on_step: Optional[Callable[[Dict[str, object]], None]] = None,
) -> AgentGatewayResponse:
    """Single pass through the agent. Shared by /v1 and /v1/stream.

    Pulled out of the route handler so the streaming endpoint can call the
    exact same code path with an ``on_step`` callback wired to its queue.
    The ``skip_history`` flag short-circuits the silent retrieval step and
    powers the ``compare_with_no_history`` A/B view.
    """
    started_at = time.perf_counter()
    raw_query = payload.query.strip()
    language = _normalise_language(payload.user_persona.language)

    # --- Safety: input checks (advisory, never blocking) -------------
    input_flags = check_input(
        raw_query,
        payload.user_persona.tone_notes or "",
        payload.user_persona.history or "",
    )

    # Stateful agentic workflow: record this turn into the rolling
    # conversation buffer keyed by user_id BEFORE routing. We skip this in
    # the no-history variant so the A/B side does not pollute future runs.
    if not skip_history:
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

    if on_step:
        on_step({
            "type": "route",
            "task": decision.task,
            "source": routing_source,
            "rationale": decision.rationale,
        })

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
            res = orchestrator.simulate_review(
                req,
                skip_history=skip_history,
                language=language,
                on_step=on_step,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "bad_request", "detail": str(exc)},
            ) from exc
        steps = [f"Routed to Task A (review). {decision.rationale}"] + res.reasoning_steps

        # Output safety: grounded against the user's own query + item context.
        output_flags = check_output(
            res.review_text,
            grounding_sources=(raw_query, item_context),
        )
        timing_ms = int((time.perf_counter() - started_at) * 1000)

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
            safety_flags=sorted(set(input_flags + output_flags)),
            timing_ms=timing_ms,
            language=language,
        )

    candidates = [c.strip() for c in decision.candidate_items if c and str(c).strip()]
    if len(candidates) < 1:
        candidates = ["Local experience A", "Local experience B", "Local experience C"]
    ctx = (decision.context or enriched).strip()[:1000]
    prior_turns = (
        []
        if skip_history
        else get_recent_turns(payload.user_persona.user_id, exclude_latest=True)
    )
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
        res = orchestrator.recommend(
            req,
            skip_history=skip_history,
            language=language,
            on_step=on_step,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "detail": str(exc)},
        ) from exc
    steps = [f"Routed to Task B (recommend). {decision.rationale}"] + res.reasoning_steps

    # Output safety against the user's query + the chosen recommendation
    # explanations (which is where free-text hallucinations would live).
    rec_text_blob = " ".join(item.explanation for item in res.recommendations)
    if res.conversational_response:
        rec_text_blob = f"{res.conversational_response} {rec_text_blob}"
    output_flags = check_output(rec_text_blob, grounding_sources=(raw_query, ctx))
    timing_ms = int((time.perf_counter() - started_at) * 1000)

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
        safety_flags=sorted(set(input_flags + output_flags)),
        timing_ms=timing_ms,
        language=language,
    )


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

    main = _run_pipeline(payload, skip_history=not payload.include_history)

    # Optional second pass for the A/B comparison view. We always disable
    # history on this pass regardless of include_history, so callers can
    # opt in to seeing what the same query produces without behavioural
    # priors. We swallow exceptions on the variant so the main pass still
    # returns even if the comparison fails.
    if payload.compare_with_no_history and payload.include_history:
        try:
            variant = _run_pipeline(payload, skip_history=True)
            main.no_history_variant = variant
        except Exception as exc:  # pragma: no cover - defensive
            api_logger.warning("no-history variant failed: %s", exc)
            main.safety_flags = sorted(set(main.safety_flags + ["compare_variant_failed"]))

    return main


# --- Streaming endpoint ----------------------------------------------
#
# Returns an NDJSON stream (``application/x-ndjson``): one JSON object per
# line. The first events describe the agent's reasoning steps as they
# fire (``{"type": "step_start", ...}`` / ``{"type": "step_end", ...}``);
# the final event is ``{"type": "final", "result": AgentGatewayResponse}``.
# The client should parse lines incrementally to drive a live UI.
#
# We use NDJSON over POST (not SSE/EventSource) because EventSource is
# GET-only and our payload is non-trivial JSON. ``fetch`` + ReadableStream
# on the browser side handles this cleanly.


@router.post("/v1/stream")
async def agent_gateway_stream(payload: AgentGatewayRequest) -> StreamingResponse:
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

    async def event_stream() -> AsyncIterator[bytes]:
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def push_step(event: Dict[str, Any]) -> None:
            # Called from the worker thread; hop back onto the loop thread
            # to enqueue safely.
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def run_main() -> None:
            try:
                result = await asyncio.to_thread(
                    _run_pipeline,
                    payload,
                    skip_history=not payload.include_history,
                    on_step=push_step,
                )
                if payload.compare_with_no_history and payload.include_history:
                    try:
                        variant = await asyncio.to_thread(
                            _run_pipeline,
                            payload,
                            skip_history=True,
                            on_step=push_step,
                        )
                        result.no_history_variant = variant
                    except Exception as exc:  # pragma: no cover - defensive
                        api_logger.warning("stream no-history variant failed: %s", exc)
                        result.safety_flags = sorted(
                            set(result.safety_flags + ["compare_variant_failed"])
                        )
                await queue.put({"type": "final", "result": result.model_dump()})
            except HTTPException as exc:
                await queue.put({
                    "type": "error",
                    "status": exc.status_code,
                    "detail": exc.detail,
                })
            except Exception as exc:  # pragma: no cover - defensive
                api_logger.exception("stream pipeline failed: %s", exc)
                await queue.put({"type": "error", "status": 500, "detail": str(exc)})
            finally:
                await queue.put(None)  # sentinel: end of stream

        worker = asyncio.create_task(run_main())
        # Initial heartbeat so the browser sees bytes immediately (kills
        # any "Failed to fetch" timeouts from intermediaries on cold start).
        yield (json.dumps({"type": "start", "ts": int(time.time() * 1000)}) + "\n").encode("utf-8")
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield (json.dumps(event, default=str) + "\n").encode("utf-8")
        finally:
            await worker

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            # Defeat buffering at common intermediaries so steps arrive
            # in real time instead of being held until the body completes.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
