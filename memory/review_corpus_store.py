"""Read-only retrieval store for normalized review corpus examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from memory.vector_store import InMemoryVectorStore


class ReviewCorpusStore:
    """Loads normalized reviews and serves semantically related examples."""

    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = Path(corpus_path)
        self._rows: List[Dict[str, Any]] = []
        self._vectors: List[np.ndarray] = []
        self._vectorizer = InMemoryVectorStore()
        self._load()

    def _load(self) -> None:
        if not self.corpus_path.exists():
            return
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = f"{row.get('item_name','')} {row.get('text','')}".strip()
                if not text:
                    continue
                self._rows.append(row)
                self._vectors.append(self._vectorizer._embed(text))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._rows:
            return []
        qv = self._vectorizer._embed(query)
        scored: List[tuple[float, int]] = []
        for idx, vec in enumerate(self._vectors):
            scored.append((float(np.dot(qv, vec)), idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._rows[idx] for _, idx in scored[:top_k]]

