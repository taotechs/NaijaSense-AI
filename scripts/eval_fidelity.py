"""Behavioral-fidelity evaluation harness for NaijaSense AI.

Headline question: *does the silent historical-context step actually move
the needle?* This script gives a numeric answer by running the agent in
both modes on a held-out slice of the review corpus and reporting

* Rating error (MAE)     - how close the predicted rating is to truth
* Text similarity (cos)  - TF-IDF cosine between generated & real review
* Tone-match score        - does generated tone match the ground-truth tone?
* Composite fidelity      - weighted blend, higher is better

For every user_id with >=2 entries we hold out the LAST review and treat
the rest as the user's "past behaviour" already in the historical store.
We then ask the agent to review the held-out item under two modes:

  (a) ``include_history=True``  - full pipeline (default production path)
  (b) ``include_history=False`` - silent retrieval skipped (control)

Outputs
~~~~~~~
* ``data/eval/fidelity_results.jsonl`` - per-item raw scores
* ``data/eval/fidelity_summary.json``  - aggregate means + delta

Usage
~~~~~
::

    python scripts/eval_fidelity.py --limit 30 --base-url http://127.0.0.1:8000

Defaults to the local backend; pass ``--base-url`` to point at Koyeb /
any deployed instance.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make the repo importable so we can reuse the corpus loader without a
# circular dependency on the FastAPI app (we don't want to spin it up
# just to read JSONL).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.historical_user_store import HistoricalUserStore  # noqa: E402


@dataclass
class EvalSample:
    user_id: str
    item_name: str
    item_domain: str
    true_text: str
    true_rating: float
    query: str
    history_count: int


@dataclass
class ModeScores:
    rating_error: float
    text_cosine: float
    tone_match: float
    fidelity: float
    latency_ms: Optional[int] = None
    safety_flags: List[str] = field(default_factory=list)
    generated: str = ""
    rating: float = 0.0


def build_samples(store: HistoricalUserStore, limit: int) -> List[EvalSample]:
    samples: List[EvalSample] = []
    for user_id in store.known_user_ids():
        rows = store.get_history(user_id, limit=0)
        if len(rows) < 2:
            continue
        held_out = rows[-1]
        past = rows[:-1]
        if not held_out.text or len(held_out.text) < 30:
            continue
        history_blob = "\n".join(
            f"- Rated {r.item_name} {r.rating:.1f}/5: {r.text[:240]}" for r in past[-3:]
        )
        query = (
            f"Review {held_out.item_name}. Context: "
            f"{held_out.text[:140].rsplit(' ', 1)[0]}\u2026"
        )
        samples.append(
            EvalSample(
                user_id=user_id,
                item_name=held_out.item_name,
                item_domain=held_out.item_domain,
                true_text=held_out.text,
                true_rating=held_out.rating,
                query=query,
                history_count=len(past),
            )
        )
        if len(samples) >= limit:
            break
    return samples


def call_agent(
    base_url: str,
    sample: EvalSample,
    *,
    include_history: bool,
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "user_persona": {
            "user_id": sample.user_id,
            "location": "Lagos",
            "interests": [sample.item_domain or "general"],
            "sentiment_bias": "balanced",
            "tone_notes": None,
            "history": None,
            "language": "english",
        },
        "query": sample.query,
        "top_k": 3,
        "include_history": include_history,
        "compare_with_no_history": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/agent/v1",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    parsed = json.loads(body)
    parsed.setdefault("_elapsed_ms", elapsed_ms)
    return parsed


# --- Scoring metrics --------------------------------------------------


_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _tfidf_cosine(a: str, b: str) -> float:
    """Tiny TF-IDF cosine (stdlib only).

    Builds a per-pair vocabulary; IDF degenerates to 1 because we only
    have two documents, so this is really an L2-normalised TF cosine.
    Sufficient for relative ranking inside the eval.
    """
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    ca, cb = collections.Counter(ta), collections.Counter(tb)
    vocab = set(ca) | set(cb)
    dot = sum(ca[w] * cb[w] for w in vocab)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_NAIJA_MARKERS = {
    "omo", "abeg", "wahala", "sha", "no cap", "as e dey", "wetin", "jare",
    "biko", "naija", "vibes", "no go lie", "e dey", "no try",
}
_FORMAL_MARKERS = {
    "however", "moreover", "overall", "consequently", "in summary",
    "in conclusion", "i would recommend", "notably", "furthermore",
}


def _tone_bucket(text: str) -> str:
    blob = (text or "").lower()
    naija = sum(1 for m in _NAIJA_MARKERS if m in blob)
    formal = sum(1 for m in _FORMAL_MARKERS if m in blob)
    if naija >= 2 and naija > formal:
        return "slang"
    if formal >= 1 and formal >= naija:
        return "formal"
    return "casual"


def _tone_match(a: str, b: str) -> float:
    return 1.0 if _tone_bucket(a) == _tone_bucket(b) else 0.0


def _composite(rating_err: float, cos: float, tone: float) -> float:
    """Higher = better. Scaled to [0, 1]."""
    rating_term = max(0.0, 1.0 - (rating_err / 4.0))  # MAE of 4 → 0
    return round(0.4 * rating_term + 0.4 * cos + 0.2 * tone, 4)


def score_sample(sample: EvalSample, response: Dict[str, Any]) -> ModeScores:
    review = (response.get("review") or {}) if isinstance(response, dict) else {}
    generated = (review.get("review_text") or "").strip()
    rating = float(review.get("rating") or 0.0)
    rating_err = abs(rating - sample.true_rating)
    cos = _tfidf_cosine(generated, sample.true_text)
    tone = _tone_match(generated, sample.true_text)
    return ModeScores(
        rating_error=round(rating_err, 4),
        text_cosine=round(cos, 4),
        tone_match=round(tone, 4),
        fidelity=_composite(rating_err, cos, tone),
        latency_ms=response.get("timing_ms") or response.get("_elapsed_ms"),
        safety_flags=list(response.get("safety_flags") or []),
        generated=generated,
        rating=round(rating, 2),
    )


def aggregate(rows: Iterable[ModeScores]) -> Dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"n": 0}
    mean = lambda key: round(statistics.fmean(getattr(r, key) for r in rows), 4)
    return {
        "n": len(rows),
        "rating_error_mae": mean("rating_error"),
        "text_cosine_mean": mean("text_cosine"),
        "tone_match_pct": round(mean("tone_match") * 100, 1),
        "fidelity_mean": mean("fidelity"),
        "latency_ms_mean": int(statistics.fmean(r.latency_ms or 0 for r in rows)),
    }


def run(
    base_url: str,
    limit: int,
    timeout: float,
    out_dir: Path,
) -> Dict[str, Any]:
    corpus_path = ROOT / "data" / "processed" / "review_corpus.jsonl"
    store = HistoricalUserStore(corpus_path=str(corpus_path))
    samples = build_samples(store, limit=limit)
    if not samples:
        raise SystemExit("No eligible eval samples found (need users with >=2 entries).")

    print(f"[eval] {len(samples)} samples against {base_url}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "fidelity_results.jsonl"
    rows_path.unlink(missing_ok=True)

    with_history: List[ModeScores] = []
    without_history: List[ModeScores] = []
    for i, sample in enumerate(samples, start=1):
        print(f"  [{i:>3}/{len(samples)}] user={sample.user_id} item={sample.item_name[:40]!r}")
        try:
            r_with = call_agent(base_url, sample, include_history=True, timeout=timeout)
            r_without = call_agent(base_url, sample, include_history=False, timeout=timeout)
        except urllib.error.URLError as exc:
            print(f"    skip (network): {exc}")
            continue
        except Exception as exc:  # pragma: no cover - defensive
            print(f"    skip (error): {exc}")
            continue

        scored_with = score_sample(sample, r_with)
        scored_without = score_sample(sample, r_without)
        with_history.append(scored_with)
        without_history.append(scored_without)

        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "sample": asdict(sample),
                "with_history": asdict(scored_with),
                "without_history": asdict(scored_without),
            }, ensure_ascii=False) + "\n")

    summary = {
        "base_url": base_url,
        "with_history": aggregate(with_history),
        "without_history": aggregate(without_history),
    }
    a = summary["with_history"]
    b = summary["without_history"]
    if a.get("n") and b.get("n"):
        summary["delta"] = {
            "fidelity": round(a["fidelity_mean"] - b["fidelity_mean"], 4),
            "rating_error": round(b["rating_error_mae"] - a["rating_error_mae"], 4),
            "text_cosine": round(a["text_cosine_mean"] - b["text_cosine_mean"], 4),
            "tone_match_pct": round(a["tone_match_pct"] - b["tone_match_pct"], 1),
        }

    summary_path = out_dir / "fidelity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[eval] Done.")
    print(f"  results: {rows_path}")
    print(f"  summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="NaijaSense AI fidelity eval harness")
    parser.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--limit", type=int, default=20, help="max eval samples")
    parser.add_argument("--timeout", type=float, default=90.0, help="per-request HTTP timeout (s)")
    parser.add_argument("--out", default="data/eval", help="output directory")
    args = parser.parse_args()
    run(args.base_url, args.limit, args.timeout, ROOT / args.out)


if __name__ == "__main__":
    main()
