"""Tests for Task A unified input parsing."""

from core.task_a_inputs import domain_prompt_block, infer_product_domain, parse_task_a_inputs


def test_parse_unified_strings() -> None:
    parsed = parse_task_a_inputs(
        "Yaba student, critical reviewer on a budget, casual Nigerian English.",
        "Iya Eba Amala Spot — lunch, ₦2k, 20 min wait, soft amala.",
    )
    assert "Amala" in parsed.item_name
    assert parsed.item_context
    assert parsed.sentiment_bias in ("positive", "balanced", "critical")
    assert parsed.product_domain == "food"


def test_infer_tech_domain() -> None:
    domain = infer_product_domain(
        "Taotech Solutions — B2B software platform for SMEs. Onboarding took 2 days, API stable.",
        "Taotech Solutions",
    )
    assert domain == "tech"
    block = domain_prompt_block(domain)
    assert "FORBIDDEN" in block
    assert "portion" in block.lower()


def test_infer_food_allows_portion() -> None:
    domain = infer_product_domain("Mama Put jollof plate, spicy and filling.", "Mama Put")
    assert domain == "food"
    block = domain_prompt_block(domain)
    assert "FORBIDDEN" not in block
