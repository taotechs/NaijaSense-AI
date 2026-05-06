"""Behavioral fidelity utilities for Task A human evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FidelitySample:
    prompt: str
    generated_review: str
    expected_tone: str
    expected_bias: str


def heuristic_fidelity_score(sample: FidelitySample) -> float:
    """Quick proxy score (0-1) before human eval."""
    text = sample.generated_review.lower()
    tone_markers = {
        "formal": ["overall", "experience", "performance"],
        "casual": ["honestly", "really", "feels"],
        "slang-heavy": ["omo", "sha", "abeg", "no cap"],
    }
    bias_markers = {
        "positive": ["great", "love", "sweet", "excellent", "slap"],
        "balanced": ["okay", "decent", "moderate", "room to improve"],
        "critical": ["bad", "poor", "no try", "below expectation"],
    }

    tone_hit = any(m in text for m in tone_markers.get(sample.expected_tone, []))
    bias_hit = any(m in text for m in bias_markers.get(sample.expected_bias, []))
    return (0.5 if tone_hit else 0.0) + (0.5 if bias_hit else 0.0)


def average_fidelity(samples: List[FidelitySample]) -> float:
    if not samples:
        return 0.0
    return round(sum(heuristic_fidelity_score(s) for s in samples) / len(samples), 4)

