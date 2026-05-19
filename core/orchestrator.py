"""Central orchestrator for all NaijaSense agent workflows.

Stateful agentic workflow
-------------------------
For every Task A and Task B request, the orchestrator runs a *silent*
context-retrieval step before any LLM call. It pulls the user's past
behaviour from :class:`memory.historical_user_store.HistoricalUserStore`
(indexed by ``user_id``), seeds the in-memory vector store with those
snippets, and derives a behavioural baseline persona that the
:class:`agents.user_modeling.UserModelingAgent` merges with the
UI-supplied override fields. This is what makes the behavioural-fidelity
scoring criterion meaningful: even before the user types anything, the
agent already "knows" how this user has historically rated and written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agents.recommendation import RecommendationAgent
from agents.review_generation import ReviewGenerationAgent
from agents.user_modeling import UserModelingAgent
from memory.historical_user_store import (
    HistoricalEntry,
    HistoricalPersona,
    HistoricalUserStore,
)
from memory.review_corpus_store import ReviewCorpusStore
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


@dataclass
class SilentContext:
    """Output of the silent context-retrieval step."""

    user_id: str
    history_entries: List[HistoricalEntry]
    history_snippets: List[str]
    historical_persona: HistoricalPersona

    @property
    def has_history(self) -> bool:
        return bool(self.history_entries) or not self.historical_persona.is_empty()

    def reasoning_line(self) -> str:
        if not self.has_history:
            return (
                f"Silent context: no historical records for user_id={self.user_id}; "
                "treating UI persona as the primary signal."
            )
        hp = self.historical_persona
        bits = [
            f"Silent context: pulled {hp.n_reviews} past review(s) for "
            f"user_id={self.user_id}"
        ]
        if hp.avg_rating is not None:
            bits.append(
                f"avg_rating={hp.avg_rating} (tendency={hp.rating_tendency})"
            )
        if hp.tone_signal:
            bits.append(f"tone_signal={hp.tone_signal}")
        if hp.top_domains:
            bits.append("domains=" + "/".join(hp.top_domains))
        return "; ".join(bits) + "."


class NaijaSenseOrchestrator:
    """Coordinates agents with dynamic planning and decision logging."""

    SILENT_RETRIEVAL_STEP = "silent_context_retrieval"
    HISTORY_LIMIT = 5

    def __init__(
        self,
        user_memory: UserMemory,
        corpus_store: ReviewCorpusStore | None = None,
        historical_store: Optional[HistoricalUserStore] = None,
    ) -> None:
        # Cheap/fast model for persona inference and any classification-style calls.
        router_llm = LLMWrapper(role="router")
        # Strong model for review writing - higher temperature and diversity controls
        # are applied per-call inside the agent.
        generator_llm = LLMWrapper(role="generator")
        self.user_modeling_agent = UserModelingAgent(llm=router_llm)
        # Generator does the writing; router (small/fast) acts as the critic in
        # the optional critique→regenerate loop. See ReviewGenerationAgent.
        self.review_generation_agent = ReviewGenerationAgent(
            llm=generator_llm, critic_llm=router_llm
        )
        self.recommendation_agent = RecommendationAgent()
        self.user_memory = user_memory
        self.corpus_store = corpus_store
        self.historical_store = historical_store
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

    # ---- Silent context retrieval ------------------------------------

    def _silent_context_retrieval(self, user_id: str) -> SilentContext:
        """Pull historical context for ``user_id`` and refresh user memory.

        Runs on every request, regardless of task, before any LLM call.
        Falls back to an empty :class:`SilentContext` for unknown users.
        """
        if not user_id or not self.historical_store:
            return SilentContext(
                user_id=user_id,
                history_entries=[],
                history_snippets=[],
                historical_persona=HistoricalPersona(user_id=user_id),
            )
        entries = self.historical_store.get_history(user_id, limit=self.HISTORY_LIMIT)
        snippets = [entry.as_memory_snippet() for entry in entries]
        # Top up the per-user vector memory so downstream retrieval also
        # sees the historical corpus and not just in-session interactions.
        for snippet in snippets:
            self.user_memory.save_interaction(user_id=user_id, content=snippet)
        persona = self.historical_store.summarize_persona(user_id)
        return SilentContext(
            user_id=user_id,
            history_entries=entries,
            history_snippets=snippets,
            historical_persona=persona,
        )

    # ---- Plans -------------------------------------------------------

    def _plan_task_a(self, request: ReviewSimulationRequest) -> WorkflowPlan:
        return WorkflowPlan(
            flow_name="task_a_review_simulation",
            steps=[
                self.SILENT_RETRIEVAL_STEP,
                "reason_about_persona_strategy",
                "build_persona_from_profile_and_history",
                "generate_review_with_persona_tone",
                "persist_review_to_memory",
            ],
            rationale=(
                "Silent-history first, then persona, then generation - so the "
                "review is conditioned on the user's actual past behaviour."
            ),
        )

    def _plan_task_b(self, request: RecommendationRequest) -> WorkflowPlan:
        has_context = bool((request.context or "").strip())
        has_multiturn = bool(request.conversation_history)
        return WorkflowPlan(
            flow_name="task_b_memory_recommendation",
            steps=[
                self.SILENT_RETRIEVAL_STEP,
                "reason_about_retrieval_strategy",
                "retrieve_relevant_user_memory",
                "run_recommendation_ranking",
                "return_ranked_output",
            ],
            rationale=(
                "Silent-history first, then retrieval, then ranking. "
                + ("Context-aware mode. " if has_context else "")
                + ("Multi-turn turns threaded. " if has_multiturn else "")
            ).strip(),
        )

    # ---- Task A ------------------------------------------------------

    def simulate_review(
        self,
        request: ReviewSimulationRequest,
        *,
        skip_history: bool = False,
        language: str = "english",
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ReviewSimulationResponse:
        reasoning_steps: List[str] = []
        plan = self._plan_task_a(request)
        reasoning_steps.append(f"Planned {plan.flow_name}: {plan.rationale}")
        self._log_decision("plan_workflow", {"flow": plan.flow_name, "steps": plan.steps})
        if on_step:
            on_step({"type": "plan", "flow": plan.flow_name, "steps": plan.steps})

        # Step 1 - silent context retrieval (skipped on demand for A/B compare).
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[0]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[0]})
        if skip_history:
            ctx = SilentContext(
                user_id=request.user_profile.user_id,
                history_entries=[],
                history_snippets=[],
                historical_persona=HistoricalPersona(user_id=request.user_profile.user_id),
            )
            reasoning_steps.append(
                "Silent context: SKIPPED (compare-mode / include_history=false). "
                "Persona is derived purely from UI overrides."
            )
        else:
            ctx = self._silent_context_retrieval(request.user_profile.user_id)
            reasoning_steps.append(ctx.reasoning_line())
        self._log_decision(
            self.SILENT_RETRIEVAL_STEP,
            {
                "user_id": ctx.user_id,
                "n_history": len(ctx.history_entries),
                "historical_persona": ctx.historical_persona.to_dict(),
                "skipped": skip_history,
            },
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[0]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[0]})

        # Step 2 - reasoning about persona strategy.
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[1]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[1]})
        reasoning_steps.append(
            "Reasoned about persona strategy: history baseline + UI overrides."
        )
        self._log_decision(
            "reason_about_persona_strategy",
            {"persona_style": request.persona_style or settings.default_persona_style},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[1]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[1]})

        # Step 3 - build persona (history + UI override).
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[2]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[2]})
        user_model = self.user_modeling_agent.run(
            {
                "user_profile": request.user_profile,
                "persona_style": request.persona_style,
                "user_history": ctx.history_snippets,
                "historical_persona": ctx.historical_persona.to_dict(),
            }
        )
        # Thread the output-language preference into the user_model so the
        # generator prompt can apply a hard rule (english | pidgin | yoruba_mix).
        user_model["language"] = language
        merge_meta = user_model.get("merge_meta") or {}
        reasoning_steps.append(
            "Built persona by merging historical baseline with UI overrides "
            f"(fields_overridden={merge_meta.get('overridden_fields', [])})."
        )
        self._log_decision(
            "build_persona_from_profile_and_history",
            {
                "user_id": request.user_profile.user_id,
                "tone": user_model.get("tone"),
                "rating_tendency": user_model.get("rating_tendency"),
                "merge_meta": merge_meta,
            },
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[2]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[2]})

        # Step 4 - generate review.
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[3]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[3]})
        review_query = f"{request.item_data.item_name} {request.item_data.item_context or ''}".strip()
        retrieved_examples = (
            self.corpus_store.search(review_query, top_k=3) if self.corpus_store else []
        )
        if retrieved_examples:
            reasoning_steps.append(
                f"Retrieved {len(retrieved_examples)} similar reviews from corpus for style grounding."
            )
        review_output = self.review_generation_agent.run(
            {
                "user_model": user_model,
                "item_name": request.item_data.item_name,
                "item_context": request.item_data.item_context or "",
                "retrieved_examples": retrieved_examples,
            }
        )
        reasoning_steps.append("Generated review output with persona-conditioned tone.")
        critique_meta = review_output.get("critique_meta") or {}
        if critique_meta.get("applied"):
            score = critique_meta.get("specificity_score")
            if critique_meta.get("rewritten"):
                reasoning_steps.append(
                    f"Critique pass rewrote the review (specificity_score={score})."
                )
            else:
                reasoning_steps.append(
                    f"Critique pass approved the review (specificity_score={score})."
                )
        self._log_decision(
            "generate_review_with_persona_tone",
            {
                "item_name": request.item_data.item_name,
                "rating": review_output["rating"],
                "critique": critique_meta,
            },
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[3]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[3]})

        # Step 5 - persist to memory.
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[4]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[4]})
        memory_note = f"Reviewed {request.item_data.item_name}: {review_output['review_text']}"
        # In skip_history A/B mode we do NOT want to pollute the user's memory
        # with the no-history variant - that would corrupt the next request's
        # baseline. The main pass still writes to memory.
        if not skip_history:
            self.user_memory.save_interaction(user_id=request.user_profile.user_id, content=memory_note)
            reasoning_steps.append("Stored generated review in memory for downstream tasks.")
        else:
            reasoning_steps.append("Memory write SKIPPED (no-history compare variant).")
        self._log_decision(
            "persist_review_to_memory",
            {
                "user_id": request.user_profile.user_id,
                "memory_preview": memory_note[:120],
                "skipped": skip_history,
            },
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[4]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[4]})

        # Surface the silent context on the persona breakdown so callers
        # and the UI can show what history was used.
        persona_breakdown = dict(review_output["persona_breakdown"])
        persona_breakdown["historical_signal"] = ctx.historical_persona.to_dict()
        persona_breakdown["history_used"] = ctx.history_snippets[:3]

        return ReviewSimulationResponse(
            review_text=review_output["review_text"],
            rating=review_output["rating"],
            persona_breakdown=persona_breakdown,
            reasoning_steps=reasoning_steps,
        )

    # ---- Task B ------------------------------------------------------

    def recommend(
        self,
        request: RecommendationRequest,
        *,
        skip_history: bool = False,
        language: str = "english",
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> RecommendationResponse:
        reasoning_steps: List[str] = []
        plan = self._plan_task_b(request)
        reasoning_steps.append(f"Planned {plan.flow_name}: {plan.rationale}")
        self._log_decision("plan_workflow", {"flow": plan.flow_name, "steps": plan.steps})
        if on_step:
            on_step({"type": "plan", "flow": plan.flow_name, "steps": plan.steps})

        # Step 1 - silent context retrieval (skipped on demand for A/B compare).
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[0]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[0]})
        if skip_history:
            ctx = SilentContext(
                user_id=request.user_profile.user_id,
                history_entries=[],
                history_snippets=[],
                historical_persona=HistoricalPersona(user_id=request.user_profile.user_id),
            )
            reasoning_steps.append(
                "Silent context: SKIPPED (compare-mode / include_history=false)."
            )
        else:
            ctx = self._silent_context_retrieval(request.user_profile.user_id)
            reasoning_steps.append(ctx.reasoning_line())
        self._log_decision(
            self.SILENT_RETRIEVAL_STEP,
            {
                "user_id": ctx.user_id,
                "n_history": len(ctx.history_entries),
                "historical_persona": ctx.historical_persona.to_dict(),
                "skipped": skip_history,
            },
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[0]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[0]})

        # Step 2 - reason about retrieval strategy (chain-of-thought line).
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[1]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[1]})
        query = f"{request.context or ''} {' '.join(request.candidate_items)}".strip()
        cot_parts = ["Reasoned about retrieval strategy"]
        if ctx.has_history:
            cot_parts.append(
                f"prior {len(ctx.history_entries)} interactions inform interest priors"
            )
        if request.conversation_history:
            cot_parts.append(
                f"{len(request.conversation_history)} prior conversation turn(s) considered"
            )
        if request.context:
            cot_parts.append("free-text context will be tokenised for overlap scoring")
        reasoning_steps.append("; ".join(cot_parts) + ".")
        self._log_decision(
            "reason_about_retrieval_strategy",
            {
                "query": query,
                "context_present": bool(request.context),
                "multiturn_turns": len(request.conversation_history),
                "historical_turns": len(ctx.history_entries),
            },
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[1]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[1]})

        # Step 3 - retrieve relevant memory (which now includes history).
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[2]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[2]})
        memory_hits = (
            []
            if skip_history
            else self.user_memory.get_relevant_context(
                user_id=request.user_profile.user_id,
                query=query,
                top_k=max(request.top_k, 3),
            )
        )
        reasoning_steps.append(
            f"Retrieved {len(memory_hits)} memory snippet(s) "
            "(in-session interactions + silent historical context)."
        )
        self._log_decision(
            "retrieve_relevant_user_memory",
            {"user_id": request.user_profile.user_id, "memory_hit_count": len(memory_hits)},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[2]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[2]})

        # Step 4 - rank.
        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[3]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[3]})
        user_model = self.user_modeling_agent.run(
            {
                "user_profile": request.user_profile,
                "user_history": ctx.history_snippets,
                "historical_persona": ctx.historical_persona.to_dict(),
            }
        )
        user_model["language"] = language
        rec_output = self.recommendation_agent.run(
            {
                "user_model": user_model,
                "candidate_items": request.candidate_items,
                "memory_hits": memory_hits,
                "contextual_query": request.context or "",
                "conversation_history": request.conversation_history,
                "top_k": request.top_k,
                "recommender_personality": request.recommender_personality,
                "conversational_mode": request.conversational_mode,
            }
        )
        reasoning_steps.append("Ranked candidate items with memory-informed scoring.")
        if rec_output.get("recommendations"):
            top = rec_output["recommendations"][0]
            reasoning_steps.append(
                f"Top pick: {top['item_name']} (score={top['score']}) - {top['explanation']}"
            )
        self._log_decision(
            "run_recommendation_ranking",
            {
                "candidate_count": len(request.candidate_items),
                "returned_count": len(rec_output["recommendations"]),
            },
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[3]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[3]})

        self._emit("before_step", {"flow": plan.flow_name, "step": plan.steps[4]})
        if on_step:
            on_step({"type": "step_start", "flow": plan.flow_name, "step": plan.steps[4]})
        self._log_decision(
            "return_ranked_output",
            {"top_k": request.top_k, "flow": plan.flow_name},
        )
        self._emit("after_step", {"flow": plan.flow_name, "step": plan.steps[4]})
        if on_step:
            on_step({"type": "step_end", "flow": plan.flow_name, "step": plan.steps[4]})

        explainability = dict(rec_output.get("explainability", {}))
        explainability["historical_signal"] = ctx.historical_persona.to_dict()
        explainability["history_turns_used"] = len(ctx.history_entries)

        return RecommendationResponse(
            recommendations=rec_output["recommendations"],
            memory_retrieved=memory_hits,
            reasoning_steps=reasoning_steps,
            conversational_response=rec_output.get("conversational_response"),
            explainability=explainability,
        )

    # ---- Hackathon submission paths (Task A / Task B) ----------------

    def prepare_user_model(
        self,
        profile: UserProfile,
        *,
        persona_style: Optional[str] = None,
        tone_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Silent history + UI persona merge for hackathon task endpoints."""
        ctx = self._silent_context_retrieval(profile.user_id)
        user_model = self.user_modeling_agent.run(
            {
                "user_profile": profile,
                "persona_style": persona_style,
                "user_history": ctx.history_snippets,
                "historical_persona": ctx.historical_persona.to_dict(),
            }
        )
        if tone_notes:
            user_model["tone_notes"] = tone_notes
        user_model["persona_style"] = persona_style or settings.default_persona_style
        user_model["historical_signal"] = ctx.historical_persona.to_dict()
        return user_model
