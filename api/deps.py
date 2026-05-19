"""Shared API dependencies (orchestrator, stores)."""

from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, List

from core.orchestrator import NaijaSenseOrchestrator
from memory.historical_user_store import HistoricalUserStore
from memory.review_corpus_store import ReviewCorpusStore
from memory.user_memory import UserMemory
from memory.vector_store import InMemoryVectorStore
from utils.config import settings
from utils.logger import get_logger

_vector_store = InMemoryVectorStore()
user_memory = UserMemory(vector_store=_vector_store)
corpus_store = ReviewCorpusStore(corpus_path=settings.review_corpus_path)
historical_store = HistoricalUserStore(corpus_path=settings.review_corpus_path)

# Seed the per-user vector memory from the historical corpus so the first
# request against any *known* user_id (e.g. the curated
# off_y_*, hf_yelp_*, off_g_*, off_amz_* records) already has past behaviour
# indexed for retrieval. Unknown user_ids accrue history only as they
# interact, which is the original behaviour.
_PRELOAD_PER_USER = 5
for _uid in historical_store.known_user_ids():
    for _entry in historical_store.get_history(_uid, limit=_PRELOAD_PER_USER):
        user_memory.save_interaction(user_id=_uid, content=_entry.as_memory_snippet())

orchestrator = NaijaSenseOrchestrator(
    user_memory=user_memory,
    corpus_store=corpus_store,
    historical_store=historical_store,
)
api_logger = get_logger("naijasense.api")
api_logger.info(
    "historical_store loaded: %d known users, %d entries",
    len(historical_store.known_user_ids()),
    historical_store.total_entries(),
)


# ---- Multi-turn conversation buffer (stateful agentic workflow) ----
#
# Per-user rolling window of recent user turns, used to thread multi-turn
# context into Task B recommendations. The buffer is process-local; this
# is sufficient for the hackathon's stateful-workflow scoring criterion
# and survives across requests within a single container instance.

_CONVO_BUFFER_MAX = 6
_convo_lock = Lock()
_convo_buffer: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=_CONVO_BUFFER_MAX))


def record_user_turn(user_id: str, turn: str) -> None:
    """Append ``turn`` to ``user_id``'s rolling conversation buffer."""
    if not user_id or not turn:
        return
    cleaned = " ".join(turn.split())
    if not cleaned:
        return
    with _convo_lock:
        _convo_buffer[user_id].append(cleaned[:600])


def get_recent_turns(user_id: str, *, exclude_latest: bool = True) -> List[str]:
    """Return the rolling conversation history for ``user_id``.

    ``exclude_latest`` drops the most recent turn from the returned list
    so the caller can treat it as the *current* query without
    double-counting it during scoring.
    """
    if not user_id:
        return []
    with _convo_lock:
        history = list(_convo_buffer.get(user_id, ()))
    if exclude_latest and history:
        return history[:-1]
    return history


def reset_user_turns(user_id: str) -> None:
    """Wipe the rolling buffer for ``user_id`` (used by tests)."""
    with _convo_lock:
        _convo_buffer.pop(user_id, None)
