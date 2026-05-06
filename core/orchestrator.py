"""Central orchestrator for all NaijaSense agent workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from agents.recommendation import RecommendationAgent
from agents.review_generation import ReviewGenerationAgent
from agents.user_modeling import UserModelingAgent
from memory.user_memory import UserMemory
from models.llm_wrapper import LLMWrapper
from utils.config import settings
from utils.logger import get_logger
from utils.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ReviewSimulationRequest,
    ReviewSimulationResponse,
)


@dataclass
class WorkflowPlan:
    """Executable plan describing chosen workflow and rationale."""

    flow_name: str
    steps: List[str]
    rationale: str


class NaijaSenseOrchestrator:
    """Coordinates agents with dynamic planning and decision logging."""

    def __init__(self, user_memory: UserMemory) -> None:
        llm = LLMWrapper(model_name=settings.model_name)
        self.user_modeling_agent = UserModelingAgent(llm=llm)
        self.review_generation_agent = ReviewGenerationAgent(llm=llm)
        self.recommendation_agent = RecommendationAgent()
        self.user_memory = user_memory
        self.logger = get_logger("naijasense.orchestrator")
        self._hooks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {
            "before_step": [],
            "after_step": [],
        }

    def register_hook(self, event: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register extension hooks for orchestration observability."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for callback in self._hooks.get(event, []):
            callback(payload)

    def _log_decision(self, action: str, details: Dict[str, Any]) -> None:
        self.logger.info("decision=%s details=%s", action, details)

    def _plan_task_a(self, request: ReviewSimulationRequest) -> WorkflowPlan:
        return WorkflowPlan(
            flow_name="task_a_review_simulation",
            steps=[
                "reason_about_persona_strategy",
                "build_persona_from_profile",
                "generate_review_with_persona_tone",
                "persist_review_to_memory",
            ],
            rationale=(
                "Persona-first sequence selected to maximize realism before text generation."
            ),
        )

    def _plan_task_b(self, request: RecommendationRequest) -> WorkflowPlan:
        has_context = bool((request.context or "").strip())
        return WorkflowPlan(
            flow_name="task_b_memory_recommendation",
            steps=[
                "reason_about_retrieval_strategy",
                "retrieve_relevant_user_memory",
                "run_recommendation_ranking",
                "return_ranked_output",
            ],
            rationale=(
                "Memory-first sequence selected for ranking personalization."
                + (" Context-aware mode enabled." if has_context else "")
            ),
        )

    def simulate_review(self, request: ReviewSimulationRequest) -> ReviewSimulationResponse:
        reasoning_steps: List[str] = []
        plan = self._plan_task_a(request)
        reasoning_steps.append(f"Planned {plan.flow_name}: {plan.rationale}")
        self._log_decision("plan_workflow", {"flow": plan.flow_name, "steps": plan.steps})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[0]})
        reasoning_steps.append("Reasoned that persona construction should happen before generation.")
        self._log_decision(
            "reason_about_persona_strategy",
            {"persona_style": request.persona_style or settings.default_persona_style},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[0]})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[1]})
        user_model = self.user_modeling_agent.run(
            {
                "user_profile": request.user_profile,
                "persona_style": request.persona_style,
            }
        )
        reasoning_steps.append("Built persona model from user profile and style hints.")
        self._log_decision(
            "build_persona_from_profile",
            {"user_id": request.user_profile.user_id, "tone": user_model.get("tone")},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[1]})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[2]})
        review_output = self.review_generation_agent.run(
            {
                "user_model": user_model,
                "item_name": request.item_data.item_name,
                "item_context": request.item_data.item_context or "",
            }
        )
        reasoning_steps.append("Generated review output with persona-conditioned tone.")
        self._log_decision(
            "generate_review_with_persona_tone",
            {"item_name": request.item_data.item_name, "rating": review_output["rating"]},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[2]})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[3]})
        memory_note = f"Reviewed {request.item_data.item_name}: {review_output['review_text']}"
        self.user_memory.save_interaction(user_id=request.user_profile.user_id, content=memory_note)
        reasoning_steps.append("Stored generated review in memory for downstream tasks.")
        self._log_decision(
            "persist_review_to_memory",
            {"user_id": request.user_profile.user_id, "memory_preview": memory_note[:120]},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[3]})

        return ReviewSimulationResponse(
            review_text=review_output["review_text"],
            rating=review_output["rating"],
            persona_breakdown=review_output["persona_breakdown"],
            reasoning_steps=reasoning_steps,
        )

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        reasoning_steps: List[str] = []
        plan = self._plan_task_b(request)
        reasoning_steps.append(f"Planned {plan.flow_name}: {plan.rationale}")
        self._log_decision("plan_workflow", {"flow": plan.flow_name, "steps": plan.steps})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[0]})
        query = f"{request.context or ''} {' '.join(request.candidate_items)}".strip()
        reasoning_steps.append("Reasoned about retrieval query before ranking action.")
        self._log_decision(
            "reason_about_retrieval_strategy",
            {"query": query, "context_present": bool(request.context)},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[0]})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[1]})
        memory_hits = self.user_memory.get_relevant_context(
            user_id=request.user_profile.user_id,
            query=query,
            top_k=request.top_k,
        )
        reasoning_steps.append("Retrieved relevant memory snippets for recommendation context.")
        self._log_decision(
            "retrieve_relevant_user_memory",
            {"user_id": request.user_profile.user_id, "memory_hit_count": len(memory_hits)},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[1]})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[2]})
        user_model = self.user_modeling_agent.run({"user_profile": request.user_profile})
        rec_output = self.recommendation_agent.run(
            {
                "user_model": user_model,
                "candidate_items": request.candidate_items,
                "memory_hits": memory_hits,
                "top_k": request.top_k,
                "recommender_personality": request.recommender_personality,
                "conversational_mode": request.conversational_mode,
            }
        )
        reasoning_steps.append("Ranked candidate items with memory-informed scoring.")
        self._log_decision(
            "run_recommendation_ranking",
            {
                "candidate_count": len(request.candidate_items),
                "returned_count": len(rec_output["recommendations"]),
            },
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[2]})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[3]})
        self._log_decision(
            "return_ranked_output",
            {"top_k": request.top_k, "flow": plan.flow_name},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[3]})

        return RecommendationResponse(
            recommendations=rec_output["recommendations"],
            memory_retrieved=memory_hits,
            reasoning_steps=reasoning_steps,
            conversational_response=rec_output.get("conversational_response"),
            explainability=rec_output.get("explainability", {}),
        )

