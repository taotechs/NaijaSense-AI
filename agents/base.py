"""Base abstractions for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Common interface for all domain agents."""

    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic and return structured output."""
        raise NotImplementedError

