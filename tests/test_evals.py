"""Tests for hackathon evals.py module."""

from evals import (
    cold_start_profile,
    compute_rmse,
    cross_domain_candidates,
    score_task_a_batch,
    score_task_b_batch,
)


def test_cold_start_profile() -> None:
    profile = cold_start_profile([])
    assert profile["cold_start_applied"] is True
    assert "street food" in profile["interests"]


def test_cross_domain_candidates() -> None:
    items = cross_domain_candidates(["tech"], "weekend in Lagos")
    assert len(items) >= 6


def test_score_task_a_batch() -> None:
    metrics = score_task_a_batch(
        ["good food worth the money"],
        ["nice meal good value"],
        [4.0],
        [4.5],
    )
    assert "rmse" in metrics
    assert metrics["rmse"] >= 0.0


def test_score_task_b_batch() -> None:
    metrics = score_task_b_batch(
        [["A", "B", "C"]],
        [["B"]],
        k=3,
    )
    assert "ndcg@3" in metrics
    assert "hit_rate@3" in metrics


def test_rmse() -> None:
    assert compute_rmse([3.0, 4.0], [3.0, 5.0]) == 0.707107
