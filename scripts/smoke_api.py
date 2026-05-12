"""End-to-end smoke against the running FastAPI agent gateway.

Hits POST /api/agent/v1 with three scenarios:
  1. Specific review request, same payload twice  -> outputs should differ.
  2. Recommendation request                       -> should route to task=recommend.
  3. Vague review request                         -> may trigger critique rewrite.

Usage:
    python scripts/smoke_api.py [base_url]
Defaults to http://127.0.0.1:8001
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx


def _post(base_url: str, body: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{base_url}/api/agent/v1", json=body)
        r.raise_for_status()
        return r.json()


def _print_review(label: str, resp: dict[str, Any]) -> None:
    review = resp.get("review") or {}
    print(f"--- {label} ---")
    print(f"  task           : {resp.get('task')}")
    print(f"  routing_source : {resp.get('routing_source')}")
    print(f"  rating         : {review.get('rating')}")
    print(f"  review_text    :\n    {review.get('review_text')}")
    crit = [s for s in resp.get("reasoning_steps", []) if "Critique" in s]
    if crit:
        print(f"  critique       : {crit[0]}")
    print()


def _print_recommend(label: str, resp: dict[str, Any]) -> None:
    rec = resp.get("recommendation") or {}
    print(f"--- {label} ---")
    print(f"  task               : {resp.get('task')}")
    print(f"  routing_source     : {resp.get('routing_source')}")
    print(f"  conversational     : {rec.get('conversational_response')}")
    print("  recommendations    :")
    for item in rec.get("recommendations", []):
        print(f"    - {item['item_name']}  (score={item['score']})")
    print()


def main(base_url: str) -> None:
    persona = {
        "user_id": "u_smoke_1",
        "location": "Lagos",
        "interests": ["street food", "amala"],
        "sentiment_bias": "balanced",
        "tone_notes": "Use clear, natural English. Keep slang minimal.",
    }

    print("=" * 70)
    print("SCENARIO 1 — Specific review request fired 2x (sameness check)")
    print("=" * 70)
    review_body = {
        "user_persona": persona,
        "query": (
            "Review for Iya Eba Amala Spot. Went Saturday lunch with a friend; "
            "amala was soft, egusi rich, 20 min wait, paid about 2k each."
        ),
        "top_k": 4,
    }
    for i in range(2):
        _print_review(f"run {i + 1}", _post(base_url, review_body))

    print("=" * 70)
    print("SCENARIO 2 — Recommendation request")
    print("=" * 70)
    rec_body = {
        "user_persona": persona,
        "query": "What should I eat tonight in Lagos? I'm hungry and want something spicy.",
        "top_k": 4,
    }
    _print_recommend("recommend run", _post(base_url, rec_body))

    print("=" * 70)
    print("SCENARIO 3 — Vague review (possible critique rewrite)")
    print("=" * 70)
    vague_body = {
        "user_persona": {
            **persona,
            "tone_notes": "Use Nigerian twitter tone.",
        },
        "query": "Review for jollof rice at the campus buka. It was okay I guess.",
        "top_k": 4,
    }
    _print_review("vague run", _post(base_url, vague_body))


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
    try:
        main(url)
    except httpx.HTTPStatusError as exc:
        print("HTTP ERROR:", exc.response.status_code)
        try:
            print(json.dumps(exc.response.json(), indent=2))
        except Exception:
            print(exc.response.text)
        sys.exit(1)
