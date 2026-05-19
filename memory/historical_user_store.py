"""Historical user-history store backed by the normalized review corpus.

This is the *silent* memory layer of the agentic workflow: on startup we
index ``data/processed/review_corpus.jsonl`` by ``user_id`` so that any
incoming request whose persona carries a known ``user_id`` automatically
gets:

* the user's past review texts (used as ``user_history`` for the
  ``UserModelingAgent``)
* a derived behavioural persona summary (avg rating, rating tendency,
  dominant domains/interests, simple tone signal)

When the ``user_id`` is unknown we degrade gracefully to empty results so
the rest of the pipeline keeps working - the UI-supplied persona then
acts as the primary signal instead of an override. This module has no
runtime dependencies beyond the stdlib, so the container build stays
unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_NAIJA_TONE_MARKERS = (
    "omo",
    "abeg",
    "wahala",
    "sha",
    "no cap",
    "as e dey",
    "wetin",
    "jare",
    "biko",
    "naija",
    "naija twitter",
    "vibes",
)
_FORMAL_TONE_MARKERS = (
    "however",
    "moreover",
    "overall",
    "consequently",
    "in summary",
    "in conclusion",
    "i would recommend",
)


@dataclass
class HistoricalEntry:
    """One past interaction by a known user."""

    source: str
    user_id: str
    item_id: str
    item_name: str
    item_domain: str
    text: str
    rating: float

    def as_memory_snippet(self) -> str:
        """Short text suitable for vector-store ingestion."""
        rating_str = f"{self.rating:.1f}"
        item = self.item_name or "an item"
        body = " ".join(self.text.split())
        body = body[:240] + ("\u2026" if len(body) > 240 else "")
        return f"[{self.source}] Rated {item} {rating_str}/5: {body}"


@dataclass
class HistoricalPersona:
    """Behaviour summary derived from a user's past reviews."""

    user_id: str
    n_reviews: int = 0
    avg_rating: Optional[float] = None
    rating_tendency: Optional[str] = None  # "generous" | "strict" | "balanced"
    sentiment_bias: Optional[str] = None  # "positive" | "balanced" | "critical"
    tone_signal: Optional[str] = None  # "slang-heavy" | "formal" | "casual"
    top_domains: List[str] = field(default_factory=list)
    inferred_interests: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.n_reviews == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "n_reviews": self.n_reviews,
            "avg_rating": self.avg_rating,
            "rating_tendency": self.rating_tendency,
            "sentiment_bias": self.sentiment_bias,
            "tone_signal": self.tone_signal,
            "top_domains": self.top_domains,
            "inferred_interests": self.inferred_interests,
            "sources": self.sources,
        }


class HistoricalUserStore:
    """Read-only index of historical reviews keyed by ``user_id``."""

    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = Path(corpus_path)
        self._by_user: Dict[str, List[HistoricalEntry]] = {}
        self._load()

    def _load(self) -> None:
        if not self.corpus_path.exists():
            return
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user_id = str(row.get("user_id") or "").strip()
                if not user_id:
                    continue
                try:
                    rating = float(row.get("rating", 0.0))
                except (TypeError, ValueError):
                    rating = 0.0
                entry = HistoricalEntry(
                    source=str(row.get("source") or "unknown"),
                    user_id=user_id,
                    item_id=str(row.get("item_id") or ""),
                    item_name=str(row.get("item_name") or ""),
                    item_domain=str(row.get("item_domain") or "general"),
                    text=str(row.get("text") or ""),
                    rating=rating,
                )
                self._by_user.setdefault(user_id, []).append(entry)

    # ---- Public API ---------------------------------------------------

    def has_user(self, user_id: str) -> bool:
        return bool(user_id) and user_id in self._by_user

    def known_user_ids(self) -> List[str]:
        return list(self._by_user.keys())

    def total_entries(self) -> int:
        return sum(len(rows) for rows in self._by_user.values())

    def get_history(self, user_id: str, limit: int = 5) -> List[HistoricalEntry]:
        """Most recent (here: list order) entries for ``user_id``.

        We don't have timestamps in the normalized schema, so ``limit``
        is applied to the natural ingestion order. Callers typically
        request a handful (3-5) for prompt budget reasons.
        """
        if not user_id:
            return []
        rows = self._by_user.get(user_id, [])
        if limit <= 0:
            return list(rows)
        return list(rows)[-limit:]

    def summarize_persona(self, user_id: str) -> HistoricalPersona:
        """Derive a behavioural baseline from past reviews.

        Returns an empty ``HistoricalPersona`` (``is_empty() is True``)
        for unknown users - callers must handle this explicitly so the
        UI-supplied persona can take over.
        """
        rows = self._by_user.get(user_id, [])
        if not rows:
            return HistoricalPersona(user_id=user_id)
        ratings = [r.rating for r in rows if r.rating > 0]
        avg = sum(ratings) / len(ratings) if ratings else None
        tendency = self._classify_rating_tendency(avg)
        bias = self._classify_sentiment_bias(avg)
        tone = self._classify_tone(rows)
        domains = self._top_n([r.item_domain for r in rows if r.item_domain], n=3)
        interests = self._infer_interests(rows)
        sources = self._top_n([r.source for r in rows if r.source], n=3)
        return HistoricalPersona(
            user_id=user_id,
            n_reviews=len(rows),
            avg_rating=round(avg, 3) if avg is not None else None,
            rating_tendency=tendency,
            sentiment_bias=bias,
            tone_signal=tone,
            top_domains=domains,
            inferred_interests=interests,
            sources=sources,
        )

    # ---- Helpers ------------------------------------------------------

    @staticmethod
    def _classify_rating_tendency(avg: Optional[float]) -> Optional[str]:
        if avg is None:
            return None
        if avg >= 4.0:
            return "generous"
        if avg <= 2.5:
            return "strict"
        return "balanced"

    @staticmethod
    def _classify_sentiment_bias(avg: Optional[float]) -> Optional[str]:
        if avg is None:
            return None
        if avg >= 4.0:
            return "positive"
        if avg <= 2.5:
            return "critical"
        return "balanced"

    @staticmethod
    def _classify_tone(rows: List[HistoricalEntry]) -> Optional[str]:
        if not rows:
            return None
        blob = " ".join(r.text.lower() for r in rows)
        naija_hits = sum(1 for marker in _NAIJA_TONE_MARKERS if marker in blob)
        formal_hits = sum(1 for marker in _FORMAL_TONE_MARKERS if marker in blob)
        if naija_hits >= 2 and naija_hits > formal_hits:
            return "slang-heavy"
        if formal_hits >= 2 and formal_hits > naija_hits:
            return "formal"
        return "casual"

    @staticmethod
    def _top_n(values: List[str], n: int) -> List[str]:
        counts = Counter(v for v in values if v)
        return [value for value, _ in counts.most_common(n)]

    @staticmethod
    def _infer_interests(rows: List[HistoricalEntry]) -> List[str]:
        """Pull a few content tokens that look like interest seeds.

        Heuristic: combine item names with the top tokens that appear
        across the user's reviews, drop low-information words.
        """
        stop = {
            "the",
            "and",
            "but",
            "with",
            "this",
            "that",
            "for",
            "was",
            "had",
            "from",
            "are",
            "you",
            "very",
            "still",
            "after",
            "have",
            "than",
            "they",
            "their",
            "there",
            "more",
            "some",
            "into",
            "much",
            "really",
            "about",
            "would",
            "could",
        }
        candidates: List[str] = []
        for row in rows:
            for token in _TOKEN_RE.findall((row.item_name + " " + row.text).lower()):
                if token in stop or len(token) < 4:
                    continue
                candidates.append(token)
        counts = Counter(candidates)
        return [tok for tok, _ in counts.most_common(5)]
