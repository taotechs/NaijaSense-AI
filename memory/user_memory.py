"""User memory manager built on top of vector store."""

from __future__ import annotations

from typing import List

from memory.vector_store import InMemoryVectorStore
from utils.config import settings


class UserMemory:
    """Persist and retrieve user behavior snippets."""

    def __init__(self, vector_store: InMemoryVectorStore) -> None:
        self.vector_store = vector_store

    def save_interaction(self, user_id: str, content: str) -> None:
        self.vector_store.add(user_id=user_id, text=content)

    def get_relevant_context(self, user_id: str, query: str, top_k: int = 3) -> List[str]:
        bounded_top_k = min(top_k, settings.max_history_items)
        return self.vector_store.search(user_id=user_id, query=query, top_k=bounded_top_k)

