"""Shared API dependencies (orchestrator, stores)."""

from __future__ import annotations

from core.orchestrator import NaijaSenseOrchestrator
from memory.review_corpus_store import ReviewCorpusStore
from memory.user_memory import UserMemory
from memory.vector_store import InMemoryVectorStore
from utils.config import settings
from utils.logger import get_logger

_vector_store = InMemoryVectorStore()
user_memory = UserMemory(vector_store=_vector_store)
corpus_store = ReviewCorpusStore(corpus_path=settings.review_corpus_path)
orchestrator = NaijaSenseOrchestrator(user_memory=user_memory, corpus_store=corpus_store)
api_logger = get_logger("naijasense.api")
