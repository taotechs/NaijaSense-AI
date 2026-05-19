"""Unit tests for Task A two-pass and Task B catalog pipeline."""

import re
from unittest.mock import patch

import pytest

from agents.task_a_two_pass import TaskATwoPassAgent
from agents.task_b_pipeline import TaskBPipelineAgent
from core.candidate_catalog import retrieve_top_k
from core.task_b_diversify import diversify_stage1_pool
from evals import run_mock_validation
from utils.task_schemas import TaskBResponse


def test_diversify_stage1_pool_spreads_domains() -> None:
    pool = retrieve_top_k(interests=["food", "movies", "student"], context="yaba budget", limit=30)
    diversified = diversify_stage1_pool(
        pool,
        limit=12,
        persona_domains=["food", "movies", "entertainment"],
        min_unique_domains=3,
    )
    domains = {(item.domain or "").lower() for item, _ in diversified}
    assert len(diversified) <= 12
    assert len(domains) >= 2


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
    pool: object = None,
) -> TaskBResponse:
    """Build a paragraph-style TaskBResponse from stage-1 candidate lines."""
    titles: list[str] = []
    for line in candidate_items_list.strip().splitlines():
        title_match = re.search(r"title=([^|]+)", line)
        if title_match:
            titles.append(title_match.group(1).strip())
        if len(titles) >= top_k:
            break

    sentences = [
        f"If you are watching your spend, {titles[0]} is a solid weekend pick for local flavour."
        if titles
        else "Start with a budget-friendly jollof spot near campus for reliable value.",
    ]
    for title in titles[1:top_k]:
        sentences.append(
            f"You might also enjoy {title}, which fits a student budget and social weekends."
        )

    paragraph = " ".join(sentences)
    return TaskBResponse(
        agent_reasoning="Persona-driven rerank: budget student prefers affordable food and movies.",
        recommendations=paragraph,
    )


def test_task_b_pipeline_persona_only() -> None:
    with patch(
        "agents.task_b_pipeline.rerank_task_b",
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

    assert isinstance(out["recommendations"], str)
    assert len(out["recommendations"]) >= 80
    assert out["agent_reasoning"]
    assert "1." not in out["recommendations"][:20]


def test_task_b_pipeline_rejects_numbered_list() -> None:
    def _numbered_rerank(**_kwargs: object) -> TaskBResponse:
        return TaskBResponse(
            agent_reasoning="Test reasoning trace with enough detail for validation.",
            recommendations=(
                "1. First pick for your budget.\n"
                "2. Second pick for weekend movies.\n"
                "3. Third pick with extra padding text so the paragraph meets minimum length requirements."
            ),
        )

    with patch("agents.task_b_pipeline.rerank_task_b", side_effect=_numbered_rerank):
        agent = TaskBPipelineAgent()
        with pytest.raises(Exception) as exc_info:
            agent.run(
                user_id="u1",
                persona_narrative="Lagos foodie who loves jollof and suya on a budget in Yaba.",
                top_k=3,
            )
        assert "numbered list" in str(exc_info.value).lower()


def test_evals_mock_validation() -> None:
    report = run_mock_validation()
    assert report["status"] == "ok"
    assert "rmse" in report["task_a"] or "rmse_explicit" in report["task_a"]
