"""API routes for NaijaSense AI."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.deps import api_logger, orchestrator
from utils.config import settings
from utils.schemas import (
    ErrorResponse,
    RecommendationRequest,
    RecommendationResponse,
    ReviewSimulationRequest,
    ReviewSimulationResponse,
)

router = APIRouter()


@router.get("/health")
def healthcheck() -> dict:
    from scripts.ensure_large_corpus import corpus_is_ready

    return {
        "status": "ok",
        "service": "naijasense-ai",
        "corpus": {
            "large_corpus_ready": corpus_is_ready(),
            "large_corpus_path": settings.large_corpus_path,
            "corpus_index_path": settings.corpus_index_path,
        },
    }


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
        return orchestrator.simulate_review(payload)
    except ValueError as exc:
        api_logger.warning("simulate-review validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "detail": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover
        api_logger.exception("simulate-review unexpected failure: %s", exc)
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
        return orchestrator.recommend(payload)
    except ValueError as exc:
        api_logger.warning("recommend validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "detail": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover
        api_logger.exception("recommend unexpected failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "detail": "Failed to generate recommendations."},
        ) from exc

