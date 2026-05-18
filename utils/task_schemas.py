"""Hackathon submission schemas — Task A / Task B strict I/O."""

from typing import List, Optional

from pydantic import BaseModel, Field


class UserPersona(BaseModel):
    """User persona for both tasks (minimal, judge-friendly)."""

    user_id: str = Field(..., min_length=1, description="Unique user identifier.")
    location: Optional[str] = Field(default="Lagos, Nigeria")
    interests: List[str] = Field(default_factory=list)
    sentiment_bias: Optional[str] = Field(
        default="balanced",
        description="positive | balanced | critical",
    )
    tone_notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Free-form style hints (e.g. value-for-money focus, pidgin).",
    )
    history: Optional[str] = Field(
        default=None,
        max_length=8000,
        description="Optional pasted profile or past behaviour narrative.",
    )
    language: Optional[str] = Field(
        default="english",
        description="english | pidgin | yoruba_mix",
    )


class ProductDetails(BaseModel):
    """Legacy structured product block (unified agent / older clients)."""

    item_name: str = Field(..., min_length=1)
    item_context: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="What the user experienced (price, wait, taste, etc.).",
    )
    category: Optional[str] = Field(default=None, max_length=120)


class TaskARequest(BaseModel):
    """Task A hackathon body: two unified text fields only."""

    user_persona: str = Field(
        ...,
        min_length=20,
        max_length=8000,
        description="Comprehensive user profile narrative (location, tone, budget, preferences).",
    )
    product_details: str = Field(
        ...,
        min_length=10,
        max_length=4000,
        description="Product or experience being reviewed (name, price, wait, taste, etc.).",
    )


class TaskAResponse(BaseModel):
    rating: float = Field(..., ge=1.0, le=5.0)
    review_reasoning: str = Field(
        ...,
        description="Pass-1 rationale linking persona and product facts to the star rating.",
    )
    review_text: str = Field(
        ...,
        description="Pass-2 review prose aligned to the locked rating.",
    )


class TaskBUserPersona(BaseModel):
    """Task B accepts only a comprehensive user persona (hackathon brief)."""

    user_id: str = Field(..., min_length=1, description="Unique user identifier.")
    persona: str = Field(
        ...,
        min_length=20,
        max_length=8000,
        description=(
            "Full profile narrative: lifestyle, budget/financial limits, location, and "
            "tastes across categories (e.g. Movies, Food, Drinks). No separate query field."
        ),
    )


class TaskBRequest(BaseModel):
    user_persona: TaskBUserPersona


class RecommendationItem(BaseModel):
    item_id: str
    title: str
    domain: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class TaskBResponse(BaseModel):
    recommendations: List[RecommendationItem]
    agent_reasoning: str = Field(
        ...,
        description="Mandatory Reason-Before-Recommend internal monologue for judges.",
    )
