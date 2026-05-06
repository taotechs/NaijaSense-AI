"""Metric implementations for Task A and Task B evaluation."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rouge_score import rouge_scorer


def compute_rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """Compute average ROUGE-1/2/L F1 scores."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = 0.0, 0.0, 0.0
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        r1 += float(score["rouge1"].fmeasure)
        r2 += float(score["rouge2"].fmeasure)
        rl += float(score["rougeL"].fmeasure)

    count = float(len(predictions))
    return {
        "rouge1_f1": round(r1 / count, 6),
        "rouge2_f1": round(r2 / count, 6),
        "rougeL_f1": round(rl / count, 6),
    }


def compute_bertscore(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """
    Compute BERTScore if dependency is available.

    Falls back to token-overlap F1 approximation when bert-score is unavailable.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        return {"bertscore_f1": 0.0, "bertscore_mode": "empty"}

    try:
        from bert_score import score as bert_score_fn  # type: ignore

        _, _, f1 = bert_score_fn(
            cands=list(predictions),
            refs=list(references),
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
        )
        return {
            "bertscore_f1": round(float(f1.mean().item()), 6),
            "bertscore_mode": "bert-score",
        }
    except Exception:
        return {
            "bertscore_f1": round(_token_f1_average(predictions, references), 6),
            "bertscore_mode": "token-f1-fallback",
        }


def compute_rmse(predicted_ratings: Sequence[float], true_ratings: Sequence[float]) -> float:
    """Compute RMSE for review rating quality."""
    if len(predicted_ratings) != len(true_ratings):
        raise ValueError("Predicted and true ratings must have the same length.")
    if not predicted_ratings:
        return 0.0
    pred = np.array(predicted_ratings, dtype=float)
    true = np.array(true_ratings, dtype=float)
    return round(float(np.sqrt(np.mean((pred - true) ** 2))), 6)


def compute_ndcg_at_k(
    ranked_item_lists: Sequence[Sequence[str]],
    relevant_items: Sequence[Sequence[str]],
    k: int = 10,
) -> float:
    """Compute average NDCG@k for recommendation ranking quality."""
    if len(ranked_item_lists) != len(relevant_items):
        raise ValueError("Ranked lists and relevant lists must have the same length.")
    if not ranked_item_lists:
        return 0.0

    ndcgs: List[float] = []
    for ranked, relevant in zip(ranked_item_lists, relevant_items):
        relevant_set = set(relevant)
        dcg = 0.0
        for idx, item in enumerate(list(ranked)[:k], start=1):
            rel = 1.0 if item in relevant_set else 0.0
            dcg += rel / math.log2(idx + 1.0)

        ideal_hits = min(len(relevant_set), k)
        if ideal_hits == 0:
            ndcgs.append(0.0)
            continue
        idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return round(float(np.mean(ndcgs)), 6)


def compute_hit_rate_at_k(
    ranked_item_lists: Sequence[Sequence[str]],
    relevant_items: Sequence[Sequence[str]],
    k: int = 10,
) -> float:
    """Compute Hit Rate@k: fraction of users with at least one relevant hit."""
    if len(ranked_item_lists) != len(relevant_items):
        raise ValueError("Ranked lists and relevant lists must have the same length.")
    if not ranked_item_lists:
        return 0.0

    hits = 0
    for ranked, relevant in zip(ranked_item_lists, relevant_items):
        top_k = set(list(ranked)[:k])
        if top_k.intersection(set(relevant)):
            hits += 1
    return round(hits / len(ranked_item_lists), 6)


def evaluate_task_a(
    predicted_reviews: Sequence[str],
    reference_reviews: Sequence[str],
    predicted_ratings: Sequence[float],
    reference_ratings: Sequence[float],
) -> Dict[str, float | str]:
    """Evaluate review simulation quality (Task A)."""
    result: Dict[str, float | str] = {}
    result.update(compute_rouge_scores(predicted_reviews, reference_reviews))
    result.update(compute_bertscore(predicted_reviews, reference_reviews))
    result["rmse"] = compute_rmse(predicted_ratings, reference_ratings)
    return result


def evaluate_task_b(
    ranked_item_lists: Sequence[Sequence[str]],
    relevant_items: Sequence[Sequence[str]],
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate recommendation quality (Task B)."""
    return {
        f"ndcg@{k}": compute_ndcg_at_k(ranked_item_lists, relevant_items, k=k),
        f"hit_rate@{k}": compute_hit_rate_at_k(ranked_item_lists, relevant_items, k=k),
    }


def _token_f1_average(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Fallback lexical approximation used when BERTScore package is missing."""
    scores: List[float] = []
    for pred, ref in zip(predictions, references):
        p_tokens = pred.lower().split()
        r_tokens = ref.lower().split()
        if not p_tokens and not r_tokens:
            scores.append(1.0)
            continue
        if not p_tokens or not r_tokens:
            scores.append(0.0)
            continue
        common = len(set(p_tokens).intersection(set(r_tokens)))
        precision = common / max(len(set(p_tokens)), 1)
        recall = common / max(len(set(r_tokens)), 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0

