"""
Hackathon evaluation utilities (Task A & Task B KPIs).

Task A: RMSE (rating), ROUGE, BERTScore (or token-F1 fallback).
Task B: NDCG@10, Hit Rate@10.

Also exposes cold-start / cross-domain scenario helpers with Nigerian default priors.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from core.nigerian_defaults import apply_cold_start_interests, candidates_for_persona
from evaluation.metrics import (
    compute_bertscore,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_rmse,
    compute_rouge_scores,
    evaluate_task_a,
    evaluate_task_b,
)

__all__ = [
    "evaluate_task_a",
    "evaluate_task_b",
    "compute_rmse",
    "compute_rouge_scores",
    "compute_bertscore",
    "compute_ndcg_at_k",
    "compute_hit_rate_at_k",
    "score_task_a_batch",
    "score_task_b_batch",
    "cold_start_profile",
    "cross_domain_candidates",
]


def score_task_a_batch(
    predicted_reviews: Sequence[str],
    reference_reviews: Sequence[str],
    predicted_ratings: Sequence[float],
    reference_ratings: Sequence[float],
) -> Dict[str, float | str]:
    """Aggregate Task A metrics for a batch of samples."""
    return evaluate_task_a(
        predicted_reviews,
        reference_reviews,
        predicted_ratings,
        reference_ratings,
    )


def score_task_b_batch(
    ranked_item_lists: Sequence[Sequence[str]],
    relevant_items: Sequence[Sequence[str]],
    k: int = 10,
) -> Dict[str, float]:
    """Aggregate Task B metrics (NDCG@k, Hit Rate@k)."""
    return evaluate_task_b(ranked_item_lists, relevant_items, k=k)


def cold_start_profile(interests: List[str] | None = None) -> Dict[str, object]:
    """
    Return a judge-ready cold-start persona with Nigerian default preferences.

    Use when ``user_id`` is new and no behavioural history exists.
    """
    merged, applied = apply_cold_start_interests(list(interests or []))
    return {
        "interests": merged,
        "cold_start_applied": applied,
        "location": "Lagos, Nigeria",
        "sentiment_bias": "balanced",
        "tone_notes": "Value-for-money focus; honest, relatable Nigerian tone.",
    }


def cross_domain_candidates(
    interests: List[str] | None = None,
    context: str | None = None,
) -> List[str]:
    """Candidate pool spanning multiple domains (food, tech, entertainment)."""
    merged, _ = apply_cold_start_interests(list(interests or []))
    return candidates_for_persona(merged, context, cross_domain=True)
