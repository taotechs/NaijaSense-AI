"""Basic API integration tests."""

from fastapi.testclient import TestClient

from api.app import app

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
    }
    response = client.post("/api/v1/recommend", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert len(body["recommendations"]) == 2
    assert "reasoning_steps" in body


def test_simulate_review_validation_error() -> None:
    payload = {
        "user_profile": {"user_id": "u1"},
        "item_data": {"item_name": ""},
    }
    response = client.post("/api/v1/simulate-review", json=payload)
    body = response.json()

    assert response.status_code == 422
    assert body["error"] == "validation_error"

