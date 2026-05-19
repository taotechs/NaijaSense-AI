"""Basic API integration tests."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from tests.test_task_pipeline import mock_rerank_from_candidate_blob

client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_simulate_review() -> None:
    payload = {
        "user_profile": {
            "user_id": "u1",
            "location": "Lagos",
            "interests": ["fashion", "gadgets"],
            "sentiment_bias": "positive",
        },
        "item_data": {
            "item_name": "Wireless Earbuds Pro",
            "item_context": "Battery lasts all day and bass is clean.",
        },
        "persona_style": "nigerian_twitter",
    }
    response = client.post("/api/v1/simulate-review", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert "review_text" in body
    assert body["rating"] >= 1.0
    assert "reasoning_steps" in body


def test_recommend() -> None:
    payload = {
        "user_profile": {
            "user_id": "u1",
            "location": "Lagos",
            "interests": ["fashion", "gadgets"],
            "sentiment_bias": "balanced",
        },
        "candidate_items": ["Gadget Hub Subscription", "Kitchen Mixer", "Fashion Nova Bag"],
        "context": "Looking for useful everyday products",
        "top_k": 2,
        "recommender_personality": "friend",
        "conversational_mode": True,
    }
    response = client.post("/api/v1/recommend", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert len(body["recommendations"]) == 2
    assert "reasoning_steps" in body
    assert "conversational_response" in body
    assert "explainability" in body


def test_agent_gateway_review_heuristic() -> None:
    payload = {
        "user_persona": {
            "user_id": "u_agent",
            "location": "Lagos",
            "interests": ["food"],
            "sentiment_bias": "positive",
        },
        "query": "Review Iya Amala — jollof was smoky and pepper balanced.",
        "top_k": 3,
    }
    response = client.post("/api/agent/v1", json=payload)
    body = response.json()
    assert response.status_code == 200
    assert body["task"] == "review"
    assert body["review"] is not None
    assert body["review"]["rating"] >= 1.0
    assert body["routing_source"] in ("llm", "heuristic")


def test_agent_gateway_recommend_heuristic() -> None:
    payload = {
        "user_persona": {"user_id": "u_agent2", "interests": ["food"]},
        "query": "What should I eat tonight in Lagos? Something not too expensive.",
        "top_k": 3,
    }
    response = client.post("/api/agent/v1", json=payload)
    body = response.json()
    assert response.status_code == 200
    assert body["task"] == "recommend"
    assert body["recommendation"] is not None
    assert len(body["recommendation"]["recommendations"]) >= 1


def test_task_a_user_modeling() -> None:
    payload = {
        "user_persona": (
            "Lagos foodie in Yaba, balanced reviewer who cares about value-for-money "
            "and honest Nigerian tone."
        ),
        "product_details": (
            "Iya Eba Amala Spot — Saturday lunch, amala soft, egusi rich, about ₦2k each, 20 min wait."
        ),
    }
    response = client.post("/task-a/user-modeling", json=payload)
    body = response.json()
    assert response.status_code == 200
    assert 1.0 <= body["rating"] <= 5.0
    assert len(body["review_text"]) > 20
    assert len(body["review_reasoning"]) > 10


def test_task_b_recommendation() -> None:
    payload = {
        "user_persona": {
            "user_id": "hackathon_b",
            "persona": (
                "Yaba student on a tight budget who loves cheap jollof, weekend Nollywood "
                "with friends, and affordable smoothies — no premium spots."
            ),
        },
    }
    with patch(
        "agents.task_b_pipeline.rerank_task_b",
        side_effect=mock_rerank_from_candidate_blob,
    ):
        response = client.post("/task-b/recommendation", json=payload)
    body = response.json()
    assert response.status_code == 200
    assert isinstance(body["recommendations"], str)
    assert len(body["recommendations"]) >= 80
    assert len(body["agent_reasoning"]) > 20


def test_landing_page() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Task A" in response.text
    assert "/task-b/recommendation" in response.text


def test_task_endpoints_get_info_pages() -> None:
    for path in ("/task-a/user-modeling", "/task-b/recommendation"):
        response = client.get(path)
        assert response.status_code == 200
        assert "Endpoint is live" in response.text
        assert "POST" in response.text


def test_simulate_review_validation_error() -> None:
    payload = {
        "user_profile": {"user_id": "u1"},
        "item_data": {"item_name": ""},
    }
    response = client.post("/api/v1/simulate-review", json=payload)
    body = response.json()

    assert response.status_code == 422
    assert body["error"] == "validation_error"

