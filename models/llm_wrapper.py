"""LLM abstraction for generation tasks.

Role-aware: ``router`` uses the small/fast model (cheap classification &
persona inference), ``generator`` uses the strong model (review writing,
recommendations rationales). Sampling parameters and per-call seeds are
honoured for both OpenAI and Groq providers so repeated calls actually
produce different text.

The fallback path returns an *empty* response (not the prompt) so callers
must explicitly decide what to do when no provider is configured — this
prevents the prompt from accidentally leaking into user-facing output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional

from utils.config import settings
from utils.logger import get_logger


@dataclass
class LLMResponse:
    text: str


_ROUTER_DEFAULTS: dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_tokens": 512,
}


class LLMWrapper:
    """Thin wrapper over provider SDKs with safe fallback semantics."""

    def __init__(self, model_name: Optional[str] = None, *, role: str = "generator") -> None:
        self.role = role if role in {"router", "generator"} else "generator"
        self._log = get_logger(f"naijasense.llm.{self.role}")
        self._provider = (settings.orchestrator_provider or "none").lower().strip()
        self.model_name = model_name or self._default_model_for_role()

    def _default_model_for_role(self) -> str:
        if self._provider == "groq":
            if self.role == "generator":
                return settings.groq_generator_model or settings.groq_model
            return settings.groq_router_model or settings.groq_model
        return settings.orchestrator_model

    def _resolve_sampling(self, **overrides: Any) -> dict[str, Any]:
        if self.role == "generator":
            base: dict[str, Any] = {
                "temperature": settings.gen_temperature,
                "top_p": settings.gen_top_p,
                "presence_penalty": settings.gen_presence_penalty,
                "frequency_penalty": settings.gen_frequency_penalty,
                "max_tokens": settings.gen_max_tokens,
            }
        else:
            base = dict(_ROUTER_DEFAULTS)
        for key, value in overrides.items():
            if value is not None:
                base[key] = value
        return base

    @staticmethod
    def _build_messages(prompt: str, system: Optional[str]) -> List[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: List[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        return messages

    @staticmethod
    def _model_kwargs(sampling: dict[str, Any], seed: Optional[int]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key in ("top_p", "presence_penalty", "frequency_penalty"):
            value = sampling.get(key)
            if value is not None:
                kwargs[key] = value
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    def _generate_with_openai(
        self,
        prompt: str,
        system: Optional[str],
        sampling: dict[str, Any],
        seed: Optional[int],
    ) -> str:
        from langchain_openai import ChatOpenAI

        model_kwargs = self._model_kwargs(sampling, seed)
        llm = ChatOpenAI(
            model=self.model_name,
            api_key=settings.openai_api_key,
            temperature=sampling.get("temperature", 0.35),
            max_tokens=sampling.get("max_tokens"),
            model_kwargs=model_kwargs or None,
        )
        msg = llm.invoke(self._build_messages(prompt, system))
        return str(getattr(msg, "content", "") or "").strip()

    def _generate_with_groq(
        self,
        prompt: str,
        system: Optional[str],
        sampling: dict[str, Any],
        seed: Optional[int],
    ) -> str:
        from langchain_groq import ChatGroq

        model_kwargs = self._model_kwargs(sampling, seed)
        llm = ChatGroq(
            model=self.model_name,
            api_key=settings.groq_api_key,
            temperature=sampling.get("temperature", 0.35),
            max_tokens=sampling.get("max_tokens"),
            model_kwargs=model_kwargs or None,
        )
        msg = llm.invoke(self._build_messages(prompt, system))
        return str(getattr(msg, "content", "") or "").strip()

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        *,
        system: Optional[str] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text from ``prompt``.

        Returns ``LLMResponse(text="")`` when no provider is configured or the
        call fails. Callers MUST handle the empty case explicitly — we never
        echo the prompt back as fake output.
        """
        sampling = self._resolve_sampling(
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
        )
        effective_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
        try:
            if self._provider == "openai" and settings.openai_api_key:
                out = self._generate_with_openai(prompt, system, sampling, effective_seed)
                if out:
                    return LLMResponse(text=out)
            if self._provider == "groq" and settings.groq_api_key:
                out = self._generate_with_groq(prompt, system, sampling, effective_seed)
                if out:
                    return LLMResponse(text=out)
        except Exception as exc:  # pragma: no cover - provider failures
            self._log.warning("LLM provider call failed; returning empty: %s", exc)
        return LLMResponse(text="")
