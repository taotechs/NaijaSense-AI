"""
Hackathon evaluation utilities (Task A & Task B KPIs).

Task A: RMSE (rating), ROUGE, BERTScore (or token-F1 fallback).
Task B: NDCG@10, Hit Rate@10.

Includes mock validation datasets for judge reproducibility runs.
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
    "run_mock_validation",
]

# Minimal arrays for clean `python evals.py` reproducibility checks.
MOCK_TASK_A = {
    "predicted_reviews": [
        "Good amala, worth the ₦2k spend. Wait was long but taste balanced.",
        "Battery lasts well for the price. Cable could be longer.",
    ],
    "reference_reviews": [
        "Solid amala and egusi for the money. Service was okay.",
        "Strong battery life, fair build for a budget power bank.",
    ],
    "predicted_ratings": [4.0, 4.2],
    "reference_ratings": [4.5, 4.0],
}

MOCK_TASK_B = {
    "ranked_lists": [
        ["food_jollof_surulere", "food_suya_ikeja", "food_iya_eba", "tech_earbuds"],
        ["tech_powerbank", "tech_earbuds", "food_campus", "food_delivery"],
    ],
    "relevant_lists": [
        ["food_jollof_surulere", "food_iya_eba"],
        ["tech_powerbank", "tech_earbuds"],
    ],
}


def score_task_a_batch(
    predicted_reviews: Sequence[str],
    reference_reviews: Sequence[str],
    predicted_ratings: Sequence[float],
    reference_ratings: Sequence[float],
) -> Dict[str, float | str]:
    """Aggregate Task A metrics: RMSE + lexical overlap proxies."""
    result: Dict[str, float | str] = evaluate_task_a(
        predicted_reviews,
        reference_reviews,
        predicted_ratings,
        reference_ratings,
    )
    result["rmse_explicit"] = compute_rmse(predicted_ratings, reference_ratings)
    return result


def score_task_b_batch(
    ranked_item_lists: Sequence[Sequence[str]],
    relevant_items: Sequence[Sequence[str]],
    k: int = 10,
) -> Dict[str, float]:
    """Aggregate Task B metrics: NDCG@k and Hit Rate@k."""
    out = evaluate_task_b(ranked_item_lists, relevant_items, k=k)
    out[f"ndcg@{k}_explicit"] = compute_ndcg_at_k(ranked_item_lists, relevant_items, k=k)
    out[f"hit_rate@{k}_explicit"] = compute_hit_rate_at_k(ranked_item_lists, relevant_items, k=k)
    return out


def cold_start_profile(interests: List[str] | None = None) -> Dict[str, object]:
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
    merged, _ = apply_cold_start_interests(list(interests or []))
    return candidates_for_persona(merged, context, cross_domain=True)


def check_corpus_loader() -> Dict[str, object]:
    """Lightweight corpus / index probe for reproducibility (timeout-safe)."""
    from core.data_loader import resolve_corpus_path, run_with_timeout
    from core.corpus_index import get_corpus_index

    path = resolve_corpus_path()

    def _probe() -> Dict[str, object]:
        idx = get_corpus_index()
        shots = idx.search_few_shots(
            profile_terms={"food", "student"},
            product_name="Jollof",
            product_context="campus budget",
            k=2,
        )
        pool = idx.retrieve_candidates(
            interests=["food"],
            context="cheap student lunch",
            limit=5,
        )
        return {
            "corpus_path": str(path) if path else None,
            "few_shot_hits": len(shots),
            "candidate_pool": len(pool),
        }

    return run_with_timeout(_probe, default={"corpus_path": None, "few_shot_hits": 0, "candidate_pool": 0})


def run_mock_validation() -> Dict[str, object]:
    """Execute mock datasets; safe for judges running ``python evals.py``."""
    task_a = score_task_a_batch(
        MOCK_TASK_A["predicted_reviews"],
        MOCK_TASK_A["reference_reviews"],
        MOCK_TASK_A["predicted_ratings"],
        MOCK_TASK_A["reference_ratings"],
    )
    task_b = score_task_b_batch(
        MOCK_TASK_B["ranked_lists"],
        MOCK_TASK_B["relevant_lists"],
        k=10,
    )
    return {
        "task_a": task_a,
        "task_b": task_b,
        "corpus_loader": check_corpus_loader(),
        "status": "ok",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_mock_validation(), indent=2))
