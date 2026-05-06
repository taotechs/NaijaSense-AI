"""Main orchestrator coordinating all agents."""

from __future__ import annotations

from agents.recommendation import RecommendationAgent
from agents.review_generation import ReviewGenerationAgent
from agents.user_modeling import UserModelingAgent
from memory.user_memory import UserMemory
from models.llm_wrapper import LLMWrapper
from utils.config import settings
from utils.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ReviewSimulationRequest,
    ReviewSimulationResponse,
)


class NaijaSenseOrchestrator:
    """Coordinates agent workflows and memory interactions."""

    def __init__(self, user_memory: UserMemory) -> None:
        llm = LLMWrapper(model_name=settings.model_name)
        self.user_modeling_agent = UserModelingAgent(llm=llm)
        self.review_generation_agent = ReviewGenerationAgent(llm=llm)
        self.recommendation_agent = RecommendationAgent()
        self.user_memory = user_memory

    def simulate_review(self, request: ReviewSimulationRequest) -> ReviewSimulationResponse:
        reasoning_steps = []
        reasoning_steps.append("Modeled user persona from profile.")
        user_model = self.user_modeling_agent.run(
            {
                "user_profile": request.user_profile,
                "persona_style": request.persona_style,
            }
        )

        reasoning_steps.append("Generated review text aligned with persona and tone.")
        review_output = self.review_generation_agent.run(
            {
                "user_model": user_model,
                "item_name": request.item_name,
                "item_context": request.item_context or "",
            }
        )

        memory_note = f"Reviewed {request.item_name}: {review_output['review_text']}"
        self.user_memory.save_interaction(user_id=request.user_profile.user_id, content=memory_note)
        reasoning_steps.append("Stored review interaction in long-term memory.")

        return ReviewSimulationResponse(
            review_text=review_output["review_text"],
            rating=review_output["rating"],
            persona_breakdown=review_output["persona_breakdown"],
            reasoning_steps=reasoning_steps,
        )

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        reasoning_steps = []
        reasoning_steps.append("Modeled user persona for recommendation strategy.")
        user_model = self.user_modeling_agent.run({"user_profile": request.user_profile})

        query = f"{request.context or ''} {' '.join(request.candidate_items)}".strip()
        memory_hits = self.user_memory.get_relevant_context(
            user_id=request.user_profile.user_id,
            query=query,
            top_k=request.top_k,
        )
        reasoning_steps.append("Retrieved relevant memory context for ranking.")

        rec_output = self.recommendation_agent.run(
            {
                "user_model": user_model,
                "candidate_items": request.candidate_items,
                "memory_hits": memory_hits,
                "top_k": request.top_k,
            }
        )
        reasoning_steps.append("Scored and ranked candidate items.")

        return RecommendationResponse(
            recommendations=rec_output["recommendations"],
            memory_retrieved=memory_hits,
            reasoning_steps=reasoning_steps,
        )

