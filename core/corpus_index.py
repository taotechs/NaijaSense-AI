"""
Fast keyword / profile / category index over the large evaluation corpus.

Uses streaming scans with a 2.5s budget; falls back to pre-cached defaults on timeout.
Optional pre-built JSON index (``corpus_index.json``) avoids full-file scans when present.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from core.candidate_catalog import CatalogItem
from core.corpus_cache_defaults import default_catalog_items, default_few_shots
from core.data_loader import (
    iter_corpus_rows,
    resolve_corpus_path,
    run_with_timeout,
    timed_iter,
)
from utils.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9₦]+", re.IGNORECASE)


def _terms(*blobs: str) -> Set[str]:
    out: Set[str] = set()
    for blob in blobs:
        if blob:
            out.update(t.lower() for t in _TOKEN_RE.findall(blob) if len(t) > 1)
    return out


def infer_price_tier_constraint(
    *,
    location: str | None,
    interests: Sequence[str],
    context: str | None,
    tone_notes: str | None = None,
) -> Optional[str]:
    """
    Return ``budget`` when persona/query signals low spend; ``premium`` for luxury signals.
    """
    blob = " ".join(
        [
            location or "",
            " ".join(interests),
            context or "",
            tone_notes or "",
        ]
    ).lower()
    if any(k in blob for k in ("student", "campus", "budget", "cheap", "affordable", "10k", "low cost")):
        return "budget"
    if any(k in blob for k in ("premium", "luxury", "high-end", "lekki fine", "splurge")):
        return "premium"
    return None


def row_violates_tier(row: Dict[str, Any], tier: Optional[str]) -> bool:
    if not tier:
        return False
    row_tier = str(row.get("price_tier", "mid")).lower()
    if tier == "budget" and row_tier == "premium":
        return True
    if tier == "premium" and row_tier == "budget" and "luxury" in " ".join(_row_tags(row)):
        return False
    return False


def _row_tags(row: Dict[str, Any]) -> List[str]:
    tags = row.get("tags")
    if isinstance(tags, list):
        return [str(t).lower() for t in tags]
    return []


def _score_row(
    row: Dict[str, Any],
    query_terms: Set[str],
    *,
    domain_terms: Set[str],
    rating_hint: float | None = None,
) -> float:
    name = str(row.get("item_name", ""))
    text = str(row.get("text", ""))
    domain = str(row.get("item_domain", row.get("domain", "general")))
    tag_terms = _terms(name, text, domain, " ".join(_row_tags(row)))
    overlap = len(query_terms & tag_terms)
    domain_hit = 1.5 if domain_terms and domain.lower() in domain_terms else 0.0
    if any(t in domain.lower() for t in domain_terms):
        domain_hit = max(domain_hit, 1.0)
    score = overlap * 0.4 + domain_hit
    if rating_hint is not None:
        try:
            rating = float(row.get("rating", 3.5))
            score += max(0.0, 1.0 - abs(rating - rating_hint) / 4.0) * 0.25
        except (TypeError, ValueError):
            pass
    return score


def _row_to_catalog_item(row: Dict[str, Any], idx: int) -> CatalogItem:
    iid = str(row.get("item_id") or "").strip()
    if not iid:
        slug = re.sub(r"[^a-z0-9]+", "_", str(row.get("item_name", f"item_{idx}")).lower())[:48]
        iid = f"corpus_{slug}_{idx}"
    tags = tuple(_row_tags(row))
    return CatalogItem(
        item_id=iid,
        title=str(row.get("item_name", "Unknown item")),
        domain=str(row.get("item_domain", row.get("domain", "general"))),
        tags=tags,
    )


class LargeCorpusIndex:
    """Singleton-friendly index: optional pre-built postings + streaming fallback."""

    def __init__(
        self,
        corpus_path: str | None = None,
        index_path: str | None = None,
    ) -> None:
        self.corpus_path = Path(corpus_path or settings.large_corpus_path)
        self.index_path = Path(index_path or settings.corpus_index_path)
        self._postings: Dict[str, List[int]] | None = None
        self._indexed_rows: List[Dict[str, Any]] | None = None
        self._load_light_index()

    def _load_light_index(self) -> None:
        if not self.index_path.exists():
            return

        def _read() -> Dict[str, Any]:
            return json.loads(self.index_path.read_text(encoding="utf-8"))

        payload = run_with_timeout(_read, default=None)
        if not isinstance(payload, dict):
            return
        self._indexed_rows = payload.get("rows") or []
        self._postings = payload.get("postings") or {}

    def search_few_shots(
        self,
        *,
        profile_terms: Set[str],
        product_name: str,
        product_context: str,
        sentiment_bias: str = "balanced",
        k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Top-k historical reviews for Task A few-shot injection."""
        rating_hint = {"positive": 4.3, "critical": 2.8, "balanced": 3.5}.get(
            sentiment_bias.lower(), 3.5
        )
        query = profile_terms | _terms(product_name, product_context)

        def _run() -> List[Dict[str, Any]]:
            hits = self._retrieve_rows(
                query_terms=query,
                domain_terms=profile_terms,
                limit=max(k, 2),
                rating_hint=rating_hint,
                unique_items=True,
            )
            return [row for _, row in hits[:k]]

        result = run_with_timeout(_run, default=default_few_shots(k))
        return result if result else default_few_shots(k)

    def retrieve_candidates(
        self,
        *,
        interests: Sequence[str],
        context: str | None,
        location: str | None = None,
        tone_notes: str | None = None,
        limit: int = 30,
        cold_start: bool = False,
        cross_domain: bool = False,
    ) -> List[Tuple[CatalogItem, float]]:
        """Stage-1 pool for Task B with persona constraint filtering."""
        tier = infer_price_tier_constraint(
            location=location,
            interests=interests,
            context=context,
            tone_notes=tone_notes,
        )
        interest_terms = _terms(" ".join(interests))
        query_terms = interest_terms | _terms(context or "", location or "")

        def _run() -> List[Tuple[CatalogItem, float]]:
            hits = self._retrieve_rows(
                query_terms=query_terms,
                domain_terms=interest_terms,
                limit=limit * 3,
                rating_hint=None,
                unique_items=True,
                tier_filter=tier,
            )
            scored: List[Tuple[CatalogItem, float]] = []
            for score, row in hits:
                item = _row_to_catalog_item(row, len(scored))
                adj = score
                if cold_start and item.domain in ("food", "experiences", "services", "restaurant"):
                    adj += 0.35
                if cross_domain and item.domain in ("entertainment", "experiences", "books"):
                    adj += 0.2
                if "budget" in item.tags or "student" in item.tags:
                    adj += 0.1
                scored.append((item, round(adj, 4)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]

        fallback = [(c, 0.5 - i * 0.01) for i, c in enumerate(default_catalog_items()[:limit])]
        result = run_with_timeout(_run, default=fallback)
        return result if result else fallback

    def _retrieve_rows(
        self,
        *,
        query_terms: Set[str],
        domain_terms: Set[str],
        limit: int,
        rating_hint: float | None,
        unique_items: bool,
        tier_filter: str | None = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if self._indexed_rows and self._postings:
            return self._search_prebuilt(
                query_terms=query_terms,
                domain_terms=domain_terms,
                limit=limit,
                rating_hint=rating_hint,
                unique_items=unique_items,
                tier_filter=tier_filter,
            )
        return self._search_streaming(
            query_terms=query_terms,
            domain_terms=domain_terms,
            limit=limit,
            rating_hint=rating_hint,
            unique_items=unique_items,
            tier_filter=tier_filter,
        )

    def _search_prebuilt(
        self,
        *,
        query_terms: Set[str],
        domain_terms: Set[str],
        limit: int,
        rating_hint: float | None,
        unique_items: bool,
        tier_filter: str | None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        assert self._indexed_rows is not None
        candidate_idxs: Set[int] = set()
        for term in query_terms:
            for idx in (self._postings or {}).get(term, []):
                candidate_idxs.add(idx)
        if not candidate_idxs:
            candidate_idxs = set(range(min(len(self._indexed_rows), limit * 20)))

        top: List[Tuple[float, Dict[str, Any]]] = []
        seen: Set[str] = set()
        rows = self._indexed_rows
        for idx in candidate_idxs:
            if idx >= len(rows):
                continue
            row = rows[idx]
            if tier_filter and row_violates_tier(row, tier_filter):
                continue
            key = str(row.get("item_id") or row.get("item_name", "")).lower()
            if unique_items and key in seen:
                continue
            score = _score_row(row, query_terms, domain_terms=domain_terms, rating_hint=rating_hint)
            if score <= 0:
                continue
            seen.add(key)
            top.append((score, row))
            top.sort(key=lambda x: x[0], reverse=True)
            if len(top) > limit:
                top = top[:limit]
        return top

    def _search_streaming(
        self,
        *,
        query_terms: Set[str],
        domain_terms: Set[str],
        limit: int,
        rating_hint: float | None,
        unique_items: bool,
        tier_filter: str | None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        path = resolve_corpus_path()
        if path is None:
            return []

        deadline = time.monotonic() + settings.corpus_query_timeout_sec
        top: List[Tuple[float, Dict[str, Any]]] = []
        seen: Set[str] = set()

        for row in timed_iter(iter_corpus_rows(path), deadline=deadline):
            if tier_filter and row_violates_tier(row, tier_filter):
                continue
            key = str(row.get("item_id") or row.get("item_name", "")).lower()
            if unique_items and key in seen:
                continue
            score = _score_row(row, query_terms, domain_terms=domain_terms, rating_hint=rating_hint)
            if score <= 0:
                continue
            seen.add(key)
            top.append((score, row))
            top.sort(key=lambda x: x[0], reverse=True)
            if len(top) > limit:
                top = top[:limit]
        return top


_index: LargeCorpusIndex | None = None


def get_corpus_index() -> LargeCorpusIndex:
    global _index
    if _index is None:
        _index = LargeCorpusIndex()
    return _index


def build_few_shot_matrix_block(examples: List[Dict[str, Any]]) -> str:
    """Format dynamic few-shots for Task A system prompts."""
    if not examples:
        return (
            "FEW-SHOT PROFILE → REVIEW MATRIX (localized trade-offs: value-for-money, "
            "wait time, durability — natural Nigerian syntax, not stereotypes):\n"
            "(No corpus hits — using default priors.)\n"
        )
    lines = [
        "FEW-SHOT PROFILE → REVIEW MATRIX (retrieved from corpus — style only, do NOT copy facts):",
        "",
        "| Profile cue | Sample review tone |",
        "|-------------|-------------------|",
    ]
    for ex in examples[:2]:
        domain = ex.get("item_domain", "general")
        rating = ex.get("rating", "")
        snippet = " ".join(str(ex.get("text", "")).split())[:220]
        lines.append(f"| {domain}, rating≈{rating} | \"{snippet}\" |")
    lines.append("")
    lines.append(
        "Weigh: value-for-money, wait time, portion size, durability under daily use, "
        "and whether the experience matches the price band in Lagos/Abuja/PH context."
    )
    return "\n".join(lines)
