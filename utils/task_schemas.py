"""Hackathon submission schemas — straightforward Task A / Task B I/O."""

from typing import Any, Dict, List, Optional

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
    rating: float = Field(..., ge=1.0, le=5.0, description="Simulated star rating (1–5).")
    review: str = Field(..., description="Written review text.")
    reasoning_steps: List[str] = Field(default_factory=list)
    persona_breakdown: Dict[str, Any] = Field(default_factory=dict)


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
        description="Optional explicit candidate pool; auto-built from persona when omitted.",
    )


class RankedRecommendation(BaseModel):
    rank: int = Field(..., ge=1)
    item_name: str
    score: float = Field(..., ge=0.0)
    explanation: str


class TaskBResponse(BaseModel):
    recommendations: List[RankedRecommendation]
    chain_of_thought: List[str] = Field(
        default_factory=list,
        description="Reason-Before-Recommend trace (persona analysis before ranking).",
    )
    reasoning_steps: List[str] = Field(default_factory=list)
    scenario_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="cold_start, cross_domain, etc.",
    )
