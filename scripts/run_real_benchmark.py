"""Real-data benchmark + ablation runner for Task A and Task B.

Holds out a slice of ``data/processed/review_corpus.jsonl`` and scores the
agent end-to-end with the metrics from the brief.

Task A (review simulation):
    - Inputs:   item_name + persona inferred from gold rating
    - Outputs:  generated review + predicted rating
    - Metrics:  ROUGE-1/2/L, BERTScore (or token-F1 fallback), RMSE

Task B (recommendation):
    - Inputs:   target item + 9 distractors from the same source
    - Outputs:  ranked candidate list
    - Metrics:  NDCG@10, HitRate@10

Variants (ablations):
    - full           - production pipeline
    - no_rag         - corpus_store returns no examples
    - no_critique    - critique → regenerate disabled
    - no_llm         - provider="none", deterministic fallback only

Usage::

    python scripts/run_real_benchmark.py --sample_size 20 --task both
    python scripts/run_real_benchmark.py --sample_size 20 --all_variants
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import NaijaSenseOrchestrator
from evaluation.metrics import evaluate_task_a, evaluate_task_b
from memory.review_corpus_store import ReviewCorpusStore
from memory.user_memory import UserMemory
from memory.vector_store import InMemoryVectorStore
from utils.config import settings
from utils.schemas import ItemData, RecommendationRequest, ReviewSimulationRequest, UserProfile

VARIANTS = ("full", "no_rag", "no_critique", "no_llm")


@dataclass
class CorpusRow:
    source: str
    item_name: str
    text: str
    rating: float
    item_domain: str
    user_id: str


class _EmptyCorpusStore:
    """Stand-in corpus store used by the no_rag ablation."""

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:  # noqa: ARG002
        return []


def _load_corpus(path: Path) -> List[CorpusRow]:
    rows: List[CorpusRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(raw.get("text") or "").strip()
            item_name = str(raw.get("item_name") or "").strip()
            if not text or not item_name:
                continue
            try:
                rating = float(raw.get("rating", 3.0))
            except (TypeError, ValueError):
                rating = 3.0
            rows.append(
                CorpusRow(
                    source=str(raw.get("source", "unknown")),
                    item_name=item_name[:120],
                    text=text,
                    rating=rating,
                    item_domain=str(raw.get("item_domain", "general")),
                    user_id=str(raw.get("user_id") or "anon"),
                )
            )
    return rows


def _bias_from_rating(rating: float) -> str:
    if rating >= 4.0:
        return "positive"
    if rating <= 2.0:
        return "critical"
    return "balanced"


def _interests_for_domain(domain: str) -> List[str]:
    return {
        "restaurant": ["food", "local cuisine"],
        "tech": ["gadgets", "everyday tech"],
        "books": ["reading", "fiction"],
        "general": ["general lifestyle"],
    }.get(domain, ["general lifestyle"])


def _build_orchestrator(variant: str) -> NaijaSenseOrchestrator:
    """Construct an orchestrator wired for the requested ablation variant."""
    user_memory = UserMemory(vector_store=InMemoryVectorStore())
    if variant == "no_rag":
        corpus_store: Any = _EmptyCorpusStore()
    else:
        corpus_store = ReviewCorpusStore(corpus_path=settings.review_corpus_path)
    return NaijaSenseOrchestrator(user_memory=user_memory, corpus_store=corpus_store)


class _SettingsOverride:
    """Reversible settings tweak used to scope an ablation variant."""

    def __init__(self, **overrides: Any) -> None:
        self._overrides = overrides
        self._previous: Dict[str, Any] = {}

    def __enter__(self) -> "_SettingsOverride":
        for key, value in self._overrides.items():
            self._previous[key] = getattr(settings, key)
            setattr(settings, key, value)
        return self

    def __exit__(self, *_exc: Any) -> None:
        for key, value in self._previous.items():
            setattr(settings, key, value)


def _settings_for_variant(variant: str) -> _SettingsOverride:
    if variant == "no_critique":
        return _SettingsOverride(review_critique_enabled=False)
    if variant == "no_llm":
        return _SettingsOverride(orchestrator_provider="none")
    return _SettingsOverride()  # no-op


def _evaluate_task_a(rows: Sequence[CorpusRow], orchestrator: NaijaSenseOrchestrator) -> Dict[str, Any]:
    pred_reviews: List[str] = []
    pred_ratings: List[float] = []
    gold_reviews: List[str] = []
    gold_ratings: List[float] = []
    for row in rows:
        bias = _bias_from_rating(row.rating)
        profile = UserProfile(
            user_id=row.user_id,
            location="Lagos",
            interests=_interests_for_domain(row.item_domain),
            sentiment_bias=bias,
            tone_preference="casual",
        )
        # Important: deliberately do NOT pass the gold review text as item_context
        # - that would leak the target. The agent must infer from item_name alone.
        req = ReviewSimulationRequest(
            user_profile=profile,
            item_data=ItemData(item_name=row.item_name, item_context=""),
            persona_style="formal",
        )
        result = orchestrator.simulate_review(req)
        pred_reviews.append(result.review_text)
        pred_ratings.append(float(result.rating))
        gold_reviews.append(row.text)
        gold_ratings.append(float(row.rating))
    metrics = evaluate_task_a(
        predicted_reviews=pred_reviews,
        reference_reviews=gold_reviews,
        predicted_ratings=pred_ratings,
        reference_ratings=gold_ratings,
    )
    metrics["samples"] = len(rows)
    return metrics


def _evaluate_task_b(
    rows: Sequence[CorpusRow],
    pool: Sequence[CorpusRow],
    orchestrator: NaijaSenseOrchestrator,
    rng: random.Random,
) -> Dict[str, Any]:
    ranked_lists: List[List[str]] = []
    relevant_lists: List[List[str]] = []
    # 20-item candidate set (1 target + 19 distractors) with top_k=10.
    # Distractors are drawn from the SAME domain as the target so the
    # domain-alignment heuristic doesn't trivialise the task. This is the
    # standard recsys evaluation setup: rank within a hard candidate set
    # instead of a globally random one.
    num_distractors = 19
    top_k = 10
    by_domain: Dict[str, List[str]] = {}
    for r in pool:
        by_domain.setdefault(r.item_domain, []).append(r.item_name)
    fallback_names = list({r.item_name for r in pool if r.item_name})

    for row in rows:
        same_domain_pool = [name for name in by_domain.get(row.item_domain, []) if name != row.item_name]
        if len(same_domain_pool) >= num_distractors:
            distractors = rng.sample(same_domain_pool, num_distractors)
        else:
            # Fall back to global pool only when there are not enough
            # same-domain items (rare for our long-tail Amazon slice).
            distractor_candidates = [name for name in fallback_names if name != row.item_name]
            if len(distractor_candidates) < num_distractors:
                continue
            distractors = rng.sample(distractor_candidates, num_distractors)
        candidates = [row.item_name, *distractors]
        rng.shuffle(candidates)

        profile = UserProfile(
            user_id=row.user_id,
            location="Lagos",
            interests=_interests_for_domain(row.item_domain),
            sentiment_bias=_bias_from_rating(row.rating),
            tone_preference="casual",
        )
        req = RecommendationRequest(
            user_profile=profile,
            candidate_items=candidates,
            context=f"I'm exploring {row.item_domain} options; what should I pick first?",
            top_k=top_k,
            recommender_personality="analyst",
            conversational_mode=False,
            conversation_history=[],
        )
        res = orchestrator.recommend(req)
        ranked_lists.append([item.item_name for item in res.recommendations])
        relevant_lists.append([row.item_name])

    metrics = evaluate_task_b(ranked_lists, relevant_lists, k=top_k)
    metrics["samples"] = len(ranked_lists)
    metrics["candidate_pool_size"] = num_distractors + 1
    return metrics


def _stratified_sample(
    rows: Sequence[CorpusRow],
    sample_size: int,
    rng: random.Random,
) -> List[CorpusRow]:
    """Try to keep rating distribution roughly balanced (positive / critical)."""
    positive = [r for r in rows if r.rating >= 4.0]
    critical = [r for r in rows if r.rating <= 2.0]
    middle = [r for r in rows if 2.0 < r.rating < 4.0]
    rng.shuffle(positive)
    rng.shuffle(critical)
    rng.shuffle(middle)

    half = sample_size // 2
    chunk_pos = positive[:half]
    chunk_crit = critical[: sample_size - half]
    selected = chunk_pos + chunk_crit
    if len(selected) < sample_size:
        selected.extend(middle[: sample_size - len(selected)])
    if len(selected) < sample_size:
        selected.extend((positive + critical + middle)[: sample_size - len(selected)])
    return selected[:sample_size]


def _run_single_variant(
    variant: str,
    rows: List[CorpusRow],
    sample_size: int,
    task: str,
    rng: random.Random,
) -> Dict[str, Any]:
    sample = _stratified_sample(rows, sample_size, rng)
    with _settings_for_variant(variant):
        orchestrator = _build_orchestrator(variant)
        result: Dict[str, Any] = {"variant": variant, "sample_size": len(sample)}
        if task in ("a", "both"):
            print(f"  [{variant}] Task A - generating {len(sample)} reviews...")
            result["task_a"] = _evaluate_task_a(sample, orchestrator)
            print(f"  [{variant}] Task A done: {result['task_a']}")
        if task in ("b", "both"):
            print(f"  [{variant}] Task B - ranking with distractors...")
            result["task_b"] = _evaluate_task_b(sample, rows, orchestrator, rng)
            print(f"  [{variant}] Task B done: {result['task_b']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="data/processed/review_corpus.jsonl")
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--task", choices=("a", "b", "both"), default="both")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--all_variants", action="store_true", help="Run every ablation in sequence.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None, help="Optional path to write the JSON report.")
    args = parser.parse_args()

    rows = _load_corpus(Path(args.corpus))
    if not rows:
        raise SystemExit(f"No usable rows in corpus at {args.corpus}")
    print(f"Loaded {len(rows)} corpus rows.")

    variants_to_run = list(VARIANTS) if args.all_variants else [args.variant]
    report: Dict[str, Any] = {
        "corpus": str(Path(args.corpus).resolve()),
        "total_corpus_rows": len(rows),
        "sample_size": args.sample_size,
        "task": args.task,
        "results": [],
    }
    for variant in variants_to_run:
        print(f"\n=== Variant: {variant} ===")
        rng = random.Random(args.seed)
        report["results"].append(
            _run_single_variant(
                variant=variant,
                rows=rows,
                sample_size=args.sample_size,
                task=args.task,
                rng=rng,
            )
        )

    print("\n=== FINAL REPORT ===")
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
