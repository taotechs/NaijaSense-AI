"""Simple vector store for semantic memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class VectorRecord:
    user_id: str
    text: str
    vector: np.ndarray


class InMemoryVectorStore:
    """In-memory vector store using bag-of-characters embeddings."""

    def __init__(self) -> None:
        self._records: List[VectorRecord] = []

    @staticmethod
    def _embed(text: str) -> np.ndarray:
        vec = np.zeros(64, dtype=float)
        for idx, ch in enumerate(text.lower().encode("utf-8")):
            vec[idx % 64] += float(ch)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def add(self, user_id: str, text: str) -> None:
        self._records.append(
            VectorRecord(user_id=user_id, text=text, vector=self._embed(text=text))
        )

    def search(self, user_id: str, query: str, top_k: int = 3) -> List[str]:
        query_vec = self._embed(query)
        user_records = [r for r in self._records if r.user_id == user_id]
        if not user_records:
            return []

        scored = []
        for record in user_records:
            score = float(np.dot(query_vec, record.vector))
            scored.append((score, record.text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

