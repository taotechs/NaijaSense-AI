"""Unit tests for Task A two-pass and Task B catalog pipeline."""

import re
from unittest.mock import patch

import pytest

from agents.task_a_two_pass import TaskATwoPassAgent
from agents.task_b_pipeline import TaskBPipelineAgent
from core.candidate_catalog import retrieve_top_k
from evals import run_mock_validation
from utils.task_schemas import RecommendationItem, TaskBResponse


def test_retrieve_top_30() -> None:
    pool = retrieve_top_k(interests=["food", "student"], context="cheap lunch yaba", limit=30)
    assert len(pool) <= 30
    assert len(pool) >= 5


def test_task_a_two_pass_heuristic() -> None:
    agent = TaskATwoPassAgent()
    out = agent.run(
        user_model={
            "user_id": "t",
            "bias": "balanced",
            "interests": ["food"],
            "persona_style": "nigerian_twitter",
            "persona_narrative": "Lagos foodie, balanced tone.",
            "product_domain": "food",
            "domain_trade_offs": "taste, portion, wait time",
        },
        item_name="Suya Spot",
        item_context="Spicy, fair price, quick service.",
    )
    assert 1.0 <= out["rating"] <= 5.0
    assert out["review_reasoning"]
    assert len(out["review_text"]) > 15


def mock_rerank_from_candidate_blob(
    *,
    user_persona: str,
    candidate_items_list: str,
    top_k: int,
) -> TaskBResponse:
    """Build a TaskBResponse from the same candidate lines the pipeline sends to Gemini."""
    items: list[RecommendationItem] = []
    for line in candidate_items_list.strip().splitlines():
        id_match = re.search(r"item_id=([^\s|]+)", line)
        if not id_match:
            continue
        domain_match = re.search(r"domain=([^\s|]+)", line)
        title_match = re.search(r"title=([^|]+)", line)
        items.append(
            RecommendationItem(
                item_id=id_match.group(1),
                title=f"Great pick: {(title_match.group(1) if title_match else 'pick').strip()}",
                domain=(domain_match.group(1) if domain_match else "general").strip(),
                confidence_score=round(0.9 - 0.08 * len(items), 2),
            )
        )
        if len(items) >= top_k:
            break
    return TaskBResponse(
        agent_reasoning="Persona-driven rerank: budget student prefers affordable food and movies.",
        recommendations=items,
    )


def test_task_b_pipeline_persona_only() -> None:
    with patch(
        "agents.task_b_pipeline.rerank_with_gemini",
        side_effect=mock_rerank_from_candidate_blob,
    ):
        agent = TaskBPipelineAgent()
        out = agent.run(
            user_id="new_user",
            persona_narrative=(
                "Lagos student with low budget, enjoys street food, movies on weekends, and cheap drinks."
            ),
            top_k=3,
        )

    assert len(out["recommendations"]) == 3
    assert out["agent_reasoning"]
    assert out["recommendations"][0]["confidence_score"] >= 0.0
    assert out["recommendations"][0]["title"].startswith("Great pick:")


def test_task_b_pipeline_rejects_invalid_gemini_item_id() -> None:
    def _bad_rerank(**_kwargs: object) -> TaskBResponse:
        return TaskBResponse(
            agent_reasoning="Test reasoning trace.",
            recommendations=[
                RecommendationItem(
                    item_id="not_in_pool",
                    title="Fake",
                    domain="food",
                    confidence_score=0.5,
                )
            ],
        )

    with patch("agents.task_b_pipeline.rerank_with_gemini", side_effect=_bad_rerank):
        agent = TaskBPipelineAgent()
        with pytest.raises(Exception) as exc_info:
            agent.run(
                user_id="u1",
                persona_narrative="Lagos foodie who loves jollof and suya on a budget in Yaba.",
                top_k=1,
            )
        assert "not in stage-1 pool" in str(exc_info.value).lower()


def test_evals_mock_validation() -> None:
    report = run_mock_validation()
    assert report["status"] == "ok"
    assert "rmse" in report["task_a"] or "rmse_explicit" in report["task_a"]
