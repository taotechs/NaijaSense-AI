"""Smoke test: confirm critique→regenerate fires only on weak reviews.

Vague context should usually score low and trigger rewrites.
Rich context should usually score high and be approved without rewrite.
"""

from __future__ import annotations

from api.deps import orchestrator
from utils.schemas import ItemData, ReviewSimulationRequest, UserProfile


def _critique_line(steps: list[str]) -> str:
    matches = [s for s in steps if "Critique" in s]
    return matches[0] if matches else "no critique step"


def _run(label: str, req: ReviewSimulationRequest, n: int = 3) -> None:
    print(f"=== {label} ===")
    for i in range(n):
        r = orchestrator.simulate_review(req)
        first_line = r.review_text[:200].replace("\n", " ")
        print(f"run {i + 1}: rating={r.rating}  |  {_critique_line(r.reasoning_steps)}")
        print(f"  > {first_line}")
    print()


def main() -> None:
    profile = UserProfile(
        user_id="u_test_critic_smoke",
        location="Lagos",
        interests=["street food", "amala"],
        sentiment_bias="balanced",
        tone_preference="casual",
    )

    vague = ReviewSimulationRequest(
        user_profile=profile,
        item_data=ItemData(item_name="Iya Eba Amala Spot", item_context="Was okay I guess."),
        persona_style="nigerian_twitter",
    )
    _run("VAGUE CONTEXT (rewrites expected)", vague)

    rich = ReviewSimulationRequest(
        user_profile=profile,
        item_data=ItemData(
            item_name="Iya Eba Amala Spot",
            item_context=(
                "Went Saturday lunch with a friend; amala was soft, egusi rich, "
                "20 min wait, paid about 2k each."
            ),
        ),
        persona_style="nigerian_twitter",
    )
    _run("RICH CONTEXT (approvals expected)", rich)


if __name__ == "__main__":
    main()
