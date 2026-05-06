"""API routes for NaijaSense AI."""

from __future__ import annotations

from fastapi import APIRouter

from core.orchestrator import NaijaSenseOrchestrator
from memory.user_memory import UserMemory
from memory.vector_store import InMemoryVectorStore
from utils.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ReviewSimulationRequest,
    ReviewSimulationResponse,
)

router = APIRouter()

_vector_store = InMemoryVectorStore()
_user_memory = UserMemory(vector_store=_vector_store)
_orchestrator = NaijaSenseOrchestrator(user_memory=_user_memory)


@router.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "naijasense-ai"}


@router.post("/simulate-review", response_model=ReviewSimulationResponse)
def simulate_review(payload: ReviewSimulationRequest) -> ReviewSimulationResponse:
    return _orchestrator.simulate_review(payload)


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    return _orchestrator.recommend(payload)

