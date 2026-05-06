"""LLM abstraction for generation tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str


class LLMWrapper:
    """
    Minimal LLM interface.

    This implementation is deterministic for reliability in local testing.
    It can be replaced with OpenAI or local model calls without impacting agents.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate text from prompt.
        For production, swap internals with an API call and return parsed content.
        """
        return LLMResponse(text=prompt)

