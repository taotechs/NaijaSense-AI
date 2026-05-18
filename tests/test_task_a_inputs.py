"""Tests for Task A unified input parsing."""

from core.task_a_inputs import parse_task_a_inputs


def test_parse_unified_strings() -> None:
    parsed = parse_task_a_inputs(
        "Yaba student, critical reviewer on a budget, casual Nigerian English.",
        "Iya Eba Amala Spot — lunch, ₦2k, 20 min wait, soft amala.",
    )
    assert "Amala" in parsed.item_name
    assert parsed.item_context
    assert parsed.sentiment_bias in ("positive", "balanced", "critical")
