"""Shared API and domain schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    user_id: str = Field(..., description="Unique user identifier.")
    age_range: Optional[str] = None
    location: Optional[str] = "Nigeria"
    interests: List[str] = Field(default_factory=list)
    tone_preference: Optional[str] = None
    sentiment_bias: Optional[str] = "balanced"


class ReviewSimulationRequest(BaseModel):
    user_profile: UserProfile
    item_name: str
    item_context: Optional[str] = None
    persona_style: Optional[str] = None


class ReviewSimulationResponse(BaseModel):
    review_text: str
    rating: float
    persona_breakdown: Dict[str, Any]
    reasoning_steps: List[str]


class RecommendationRequest(BaseModel):
    user_profile: UserProfile
    candidate_items: List[str]
    context: Optional[str] = None
    top_k: int = 3


class RecommendationItem(BaseModel):
    item_name: str
    score: float
    explanation: str


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    memory_retrieved: List[str]
    reasoning_steps: List[str]

