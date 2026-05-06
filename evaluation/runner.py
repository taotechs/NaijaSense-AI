"""Evaluation runner for dataset-driven benchmark execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from evaluation.metrics import evaluate_task_a, evaluate_task_b


def run_evaluation(dataset_path: str, k: int = 10) -> Dict[str, Any]:
    """Run Task A and Task B metrics against a JSON dataset."""
    dataset = _load_dataset(dataset_path)

    task_a_samples = dataset.get("task_a", [])
    task_b_samples = dataset.get("task_b", [])

    predicted_reviews = [row["pred_review"] for row in task_a_samples]
    reference_reviews = [row["gold_review"] for row in task_a_samples]
    predicted_ratings = [float(row["pred_rating"]) for row in task_a_samples]
    reference_ratings = [float(row["gold_rating"]) for row in task_a_samples]

    ranked_item_lists: List[List[str]] = [row["ranked_items"] for row in task_b_samples]
    relevant_items: List[List[str]] = [row["relevant_items"] for row in task_b_samples]

    return {
        "task_a": evaluate_task_a(
            predicted_reviews=predicted_reviews,
            reference_reviews=reference_reviews,
            predicted_ratings=predicted_ratings,
            reference_ratings=reference_ratings,
        ),
        "task_b": evaluate_task_b(
            ranked_item_lists=ranked_item_lists,
            relevant_items=relevant_items,
            k=k,
        ),
        "meta": {
            "dataset_path": str(Path(dataset_path).resolve()),
            "task_a_samples": len(task_a_samples),
            "task_b_samples": len(task_b_samples),
            "k": k,
        },
    }


def _load_dataset(dataset_path: str) -> Dict[str, Any]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "task_a" not in data or "task_b" not in data:
        raise ValueError("Dataset must contain 'task_a' and 'task_b' keys.")
    return data

