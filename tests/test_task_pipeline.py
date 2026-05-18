"""Unit tests for Task A two-pass and Task B catalog pipeline."""

from agents.task_a_two_pass import TaskATwoPassAgent
from agents.task_b_pipeline import TaskBPipelineAgent
from core.candidate_catalog import retrieve_top_k
from evals import run_mock_validation


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
        },
        item_name="Suya Spot",
        item_context="Spicy, fair price, quick service.",
    )
    assert 1.0 <= out["rating"] <= 5.0
    assert out["review_reasoning"]
    assert len(out["review_text"]) > 15


def test_task_b_pipeline_cold_start() -> None:
    agent = TaskBPipelineAgent()
    out = agent.run(
        user_model={"user_id": "new", "location": "Lagos", "bias": "balanced"},
        interests=[],
        context="weekend food",
        top_k=3,
        cold_start=True,
        cross_domain=True,
    )
    assert len(out["recommendations"]) == 3
    assert out["agent_reasoning"]
    assert out["recommendations"][0]["confidence_score"] >= 0.0


def test_evals_mock_validation() -> None:
    report = run_mock_validation()
    assert report["status"] == "ok"
    assert "rmse" in report["task_a"] or "rmse_explicit" in report["task_a"]
