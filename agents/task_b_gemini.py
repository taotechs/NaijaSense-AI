"""Task B stage-2 reranking via Google Gemini structured output."""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from agents.task_b_errors import TaskBRerankError
from utils.config import settings
from utils.task_schemas import TaskBResponse

_SYSTEM_INSTRUCTION = (
    "You are an advanced agentic recommendation engine for Nigerian lifestyle contexts. "
    "Execute your reasoning trace first, storing it in 'agent_reasoning'. "
    "Then write the 'recommendations' field as ONE fluid paragraph only: exactly {top_k} "
    "woven recommendation sentences in natural prose (no numbered list, no bullet points, "
    "no markdown). Each sentence should name a specific pick from the Filtered Database "
    "Candidates using compelling human-readable titles and categories — never raw item_id "
    "strings or database indices. Tie every sentence to the persona's budget, location, "
    "and lifestyle. Select only items from the candidate list; do not invent venues. "
    "There is no separate search query — infer needs only from the User Profile Persona."
)


def rerank_with_gemini(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
) -> TaskBResponse:
    """Call Gemini with strict TaskBResponse schema; no stage-1 fallback."""
    if not (settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip():
        raise TaskBRerankError(
            "GEMINI_API_KEY is not set. Task B stage-2 requires the Google GenAI client."
        )
    if settings.gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key

    client = genai.Client()
    response = client.models.generate_content(
        model=settings.task_b_gemini_model,
        contents=[
            f"User Profile Persona: {user_persona}",
            f"Filtered Database Candidates: {candidate_items_list}",
        ],
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION.format(top_k=top_k),
            response_mime_type="application/json",
            response_schema=TaskBResponse,
            temperature=0.2,
        ),
    )

    parsed = response.parsed
    if parsed is None:
        raise TaskBRerankError("Gemini returned no parsed TaskBResponse payload.")
    if isinstance(parsed, TaskBResponse):
        return parsed
    return TaskBResponse.model_validate(parsed)
