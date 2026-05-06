"""API routes for NaijaSense AI."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from core.orchestrator import NaijaSenseOrchestrator
from memory.review_corpus_store import ReviewCorpusStore
from memory.user_memory import UserMemory
from memory.vector_store import InMemoryVectorStore
from utils.config import settings
from utils.logger import get_logger
from utils.schemas import (
    ErrorResponse,
    RecommendationRequest,
    RecommendationResponse,
    ReviewSimulationRequest,
    ReviewSimulationResponse,
)

router = APIRouter()

_vector_store = InMemoryVectorStore()
_user_memory = UserMemory(vector_store=_vector_store)
_corpus_store = ReviewCorpusStore(corpus_path=settings.review_corpus_path)
_orchestrator = NaijaSenseOrchestrator(user_memory=_user_memory, corpus_store=_corpus_store)
_logger = get_logger("naijasense.api")


@router.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "naijasense-ai"}


@router.post(
    "/simulate-review",
    response_model=ReviewSimulationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request payload."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
)
def simulate_review(payload: ReviewSimulationRequest) -> ReviewSimulationResponse:
    try:
        return _orchestrator.simulate_review(payload)
    except ValueError as exc:
        _logger.warning("simulate-review validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "detail": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover
        _logger.exception("simulate-review unexpected failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "detail": "Failed to simulate review."},
        ) from exc


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request payload."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
)
def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    try:
        return _orchestrator.recommend(payload)
    except ValueError as exc:
        _logger.warning("recommend validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "detail": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover
        _logger.exception("recommend unexpected failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "detail": "Failed to generate recommendations."},
        ) from exc

