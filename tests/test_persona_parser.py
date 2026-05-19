"""Tests for Task B persona narrative parser."""

from core.persona_parser import parse_task_b_persona


def test_parse_student_budget() -> None:
    parsed = parse_task_b_persona(
        "UNILAG student in Yaba, ₦10k weekly budget, loves jollof and Nollywood movies."
    )
    assert parsed.budget_sensitive
    assert "food" in parsed.domains or "movies" in parsed.domains


def test_taotech_does_not_trigger_tech_domain() -> None:
    parsed = parse_task_b_persona(
        "I am the founder of taotech solutions, how can I hire software engineers?"
    )
    assert parsed.team_culture_mode
    assert "tech" not in parsed.domains


def test_founder_hiring_team_culture() -> None:
    parsed = parse_task_b_persona(
        "Founder in Lagos — how do I get great data scientists to join my startup?"
    )
    assert parsed.team_culture_mode
    assert "food" in parsed.domains or "experiences" in parsed.domains


def test_parse_no_query_field_needed() -> None:
    parsed = parse_task_b_persona(
        "Victoria Island professional who enjoys seafood dining and Afrobeats lounges."
    )
    assert parsed.location
    assert parsed.narrative
