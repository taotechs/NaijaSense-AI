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
    """Product / experience being reviewed (Task A)."""

    item_name: str = Field(..., min_length=1)
    item_context: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="What the user experienced (price, wait, taste, etc.).",
    )
    category: Optional[str] = Field(default=None, max_length=120)


class TaskARequest(BaseModel):
    user_persona: UserPersona
    product_details: ProductDetails
    persona_style: Optional[str] = Field(
        default="nigerian_twitter",
        description="formal | nigerian_twitter",
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


class TaskBRequest(BaseModel):
    user_persona: UserPersona
    top_k: int = Field(default=5, ge=1, le=20)
    context: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional free-text query (e.g. cheap food in Yaba).",
    )
    candidate_items: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit candidate pool; auto-built when omitted.",
    )


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
