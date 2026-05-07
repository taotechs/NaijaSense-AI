"""Shared API and domain schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class UserProfile(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique user identifier.")
    age_range: Optional[str] = None
    location: Optional[str] = "Nigeria"
    interests: List[str] = Field(default_factory=list)
    tone_preference: Optional[str] = None
    sentiment_bias: Optional[str] = "balanced"


class ItemData(BaseModel):
    item_name: str = Field(..., min_length=1, description="Item/product name.")
    item_context: Optional[str] = Field(
        default=None, max_length=1000, description="Short contextual description for the item."
    )
    category: Optional[str] = Field(default=None, max_length=120)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReviewSimulationRequest(BaseModel):
    user_profile: UserProfile
    item_data: ItemData
    persona_style: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def support_legacy_item_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if "item_data" in values:
            return values
        item_name = values.get("item_name")
        if item_name:
            values["item_data"] = {
                "item_name": item_name,
                "item_context": values.get("item_context"),
            }
        return values


class ReviewSimulationResponse(BaseModel):
    review_text: str
    rating: float = Field(..., ge=1.0, le=5.0)
    persona_breakdown: Dict[str, Any]
    reasoning_steps: List[str]


class RecommendationRequest(BaseModel):
    user_profile: UserProfile
    candidate_items: List[str] = Field(
        ..., min_length=1, description="Candidate item names to rank."
    )
    context: Optional[str] = Field(default=None, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=20)
    recommender_personality: str = Field(
        default="analyst",
        description="How recommendation response should sound: analyst, coach, friend, nigerian_twitter.",
    )
    conversational_mode: bool = Field(
        default=True,
        description="If true, API returns a chat-style recommendation summary.",
    )
    conversation_history: List[str] = Field(
        default_factory=list,
        description="Previous user turns for multi-turn recommendation context.",
    )


class RecommendationItem(BaseModel):
    item_name: str
    score: float = Field(..., ge=0.0)
    explanation: str


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    memory_retrieved: List[str]
    reasoning_steps: List[str]
    conversational_response: Optional[str] = None
    explainability: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str


# --- Unified agent gateway (/api/agent/v1) ---


class UserPersonaInput(BaseModel):
    """Flexible persona for the unified chat agent."""

    user_id: str = Field(default="agent_user", min_length=1)
    location: Optional[str] = "Lagos"
    interests: List[str] = Field(default_factory=list)
    sentiment_bias: Optional[str] = Field(default="balanced", description="positive | balanced | critical")
    tone_notes: Optional[str] = Field(default=None, max_length=2000, description="Free-form style hints")
    history: Optional[str] = Field(
        default=None,
        max_length=8000,
        description="Optional pasted history or profile narrative (fed into context).",
    )


class AgentGatewayRequest(BaseModel):
    user_persona: UserPersonaInput
    query: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(default=4, ge=1, le=20)


class AgentReviewResult(BaseModel):
    review_text: str
    rating: float = Field(..., ge=1.0, le=5.0)
    persona_breakdown: Dict[str, Any] = Field(default_factory=dict)


class AgentRecommendationResult(BaseModel):
    recommendations: List[RecommendationItem]
    conversational_response: Optional[str] = None
    explainability: Dict[str, Any] = Field(default_factory=dict)
    memory_retrieved: List[str] = Field(default_factory=list)


class AgentGatewayResponse(BaseModel):
    task: str = Field(description="review | recommend")
    orchestrator_rationale: str
    routing_source: str = Field(description="llm | heuristic")
    review: Optional[AgentReviewResult] = None
    recommendation: Optional[AgentRecommendationResult] = None
    reasoning_steps: List[str] = Field(default_factory=list)

