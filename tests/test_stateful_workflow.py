"""Tests for the stateful agentic workflow.

Covers the four hackathon-brief axes:

1. Silent context retrieval by user_id pulls historical reviews from the
   normalized corpus before any LLM call.
2. Task A's UserModelingAgent receives + leverages that history.
3. Task B threads multi-turn conversation history across requests.
4. UI persona fields act as overrides on top of the history-derived
   baseline, not as a replacement.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app
from api.deps import historical_store, record_user_turn, reset_user_turns
from memory.historical_user_store import HistoricalUserStore
from utils.config import settings

client = TestClient(app)


def _known_user_id() -> str:
    ids = historical_store.known_user_ids()
    assert ids, "historical store should have ingested at least one user"
    return ids[0]


def test_historical_store_loads_corpus() -> None:
    store = HistoricalUserStore(corpus_path=settings.review_corpus_path)
    assert store.total_entries() > 0
    sample_uid = store.known_user_ids()[0]
    history = store.get_history(sample_uid, limit=3)
    assert history, "expected at least one historical entry for a known user"
    persona = store.summarize_persona(sample_uid)
    assert not persona.is_empty()
    assert persona.n_reviews >= 1


def test_historical_store_unknown_user_returns_empty_persona() -> None:
    persona = historical_store.summarize_persona("not_a_real_user_xyz")
    assert persona.is_empty()
    assert historical_store.get_history("not_a_real_user_xyz") == []


def test_simulate_review_uses_silent_history_for_known_user() -> None:
    known = _known_user_id()
    payload = {
        "user_profile": {
            "user_id": known,
            "interests": [],
            "sentiment_bias": "balanced",
        },
        "item_data": {
            "item_name": "Some New Spot",
            "item_context": "First visit, casual dinner.",
        },
        "persona_style": "formal",
    }
    response = client.post("/api/v1/simulate-review", json=payload)
    assert response.status_code == 200
    body = response.json()
    persona = body["persona_breakdown"]
    assert "historical_signal" in persona, "review response must expose historical_signal"
    assert persona["historical_signal"].get("n_reviews", 0) >= 1
    # Reasoning steps should mention the silent context step explicitly.
    joined = "\n".join(body["reasoning_steps"]).lower()
    assert "silent context" in joined
    assert known.lower() in joined


def test_simulate_review_falls_back_for_unknown_user() -> None:
    payload = {
        "user_profile": {
            "user_id": "u_unknown_fallback_1",
            "interests": ["tech"],
            "sentiment_bias": "balanced",
        },
        "item_data": {"item_name": "Mystery Gadget", "item_context": "Tried it briefly."},
        "persona_style": "formal",
    }
    response = client.post("/api/v1/simulate-review", json=payload)
    assert response.status_code == 200
    body = response.json()
    persona = body["persona_breakdown"]
    assert "historical_signal" in persona
    assert persona["historical_signal"].get("n_reviews", 0) == 0
    joined = "\n".join(body["reasoning_steps"]).lower()
    assert "no historical records" in joined


def test_ui_persona_overrides_history_baseline() -> None:
    known = _known_user_id()
    # Force an explicit UI override that conflicts with the typical history.
    payload = {
        "user_profile": {
            "user_id": known,
            "interests": ["unique_ui_interest_marker"],
            "sentiment_bias": "critical",
            "tone_preference": "formal",
        },
        "item_data": {"item_name": "Override Test Spot", "item_context": "Came once."},
        "persona_style": "formal",
    }
    response = client.post("/api/v1/simulate-review", json=payload)
    assert response.status_code == 200
    body = response.json()
    persona = body["persona_breakdown"]
    merge_meta = persona.get("merge_meta") or {}
    assert merge_meta.get("has_history") is True
    overridden = set(merge_meta.get("overridden_fields") or [])
    # The UI explicitly set sentiment_bias, tone_preference, and interests
    # so they should appear in overridden_fields.
    assert "sentiment_bias" in overridden
    assert "tone" in overridden
    assert "interests" in overridden
    assert "unique_ui_interest_marker" in persona["interests"]
    assert persona["bias"] == "critical"


def test_agent_gateway_threads_multiturn_history() -> None:
    user_id = "u_multiturn_test_1"
    reset_user_turns(user_id)

    turn1 = {
        "user_persona": {"user_id": user_id, "interests": ["food"]},
        "query": "I want jollof rice tonight on a small budget.",
        "top_k": 3,
    }
    r1 = client.post("/api/agent/v1", json=turn1)
    assert r1.status_code == 200
    body1 = r1.json()
    # First turn has no prior conversation history.
    if body1["task"] == "recommend":
        assert body1["recommendation"]["explainability"].get("multiturn_turns_used", 0) == 0

    turn2 = {
        "user_persona": {"user_id": user_id, "interests": ["food"]},
        "query": "Make it spicy and close to Yaba.",
        "top_k": 3,
    }
    r2 = client.post("/api/agent/v1", json=turn2)
    assert r2.status_code == 200
    body2 = r2.json()
    if body2["task"] == "recommend":
        assert body2["recommendation"]["explainability"].get("multiturn_turns_used", 0) >= 1
        cot = body2["recommendation"]["explainability"].get("chain_of_thought") or []
        assert any("multi-turn" in line.lower() for line in cot), (
            "chain_of_thought should mention multi-turn context"
        )

    reset_user_turns(user_id)


def test_chain_of_thought_present_in_recommend_explainability() -> None:
    user_id = "u_cot_test_1"
    reset_user_turns(user_id)
    payload = {
        "user_persona": {"user_id": user_id, "interests": ["food"]},
        "query": "Recommend a cheap and spicy dinner spot in Lagos.",
        "top_k": 3,
    }
    response = client.post("/api/agent/v1", json=payload)
    assert response.status_code == 200
    body = response.json()
    if body["task"] == "recommend":
        cot = body["recommendation"]["explainability"].get("chain_of_thought")
        assert isinstance(cot, list) and len(cot) >= 1
    reset_user_turns(user_id)
