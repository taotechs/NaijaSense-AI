"""Tests for Task B persona narrative parser."""

from core.persona_parser import parse_task_b_persona


def test_parse_student_budget() -> None:
    parsed = parse_task_b_persona(
        "UNILAG student in Yaba, ₦10k weekly budget, loves jollof and Nollywood movies."
    )
    assert parsed.budget_sensitive
    assert "food" in parsed.domains or "movies" in parsed.domains


def test_parse_no_query_field_needed() -> None:
    parsed = parse_task_b_persona(
        "Victoria Island professional who enjoys seafood dining and Afrobeats lounges."
    )
    assert parsed.location
    assert parsed.narrative
