"""Pytest hooks — keep integration tests fast (no 10k corpus build on import)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_corpus_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.config.settings.corpus_build_on_startup", False)
