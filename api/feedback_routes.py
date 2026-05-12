"""Feedback endpoint: thumbs up/down + free-form note.

Persists user feedback to a newline-delimited JSON log so we can post-hoc
audit the agent's outputs and use the signal as fuel for the
prompt-example bank / future fine-tuning. Storage is intentionally
simple (file on local disk) to keep this hackathon-friendly — swap to
Postgres / S3 later without changing the API contract.

Endpoints
~~~~~~~~~
* ``POST /api/agent/feedback`` — append one feedback record. Returns the
  generated record id so the UI can show a "saved as #abc12" toast.
* ``GET  /api/agent/feedback/stats`` — small aggregate (count, %
  positive) so a demo / pitch can show "we collected N samples in this
  session" without exposing individual records.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from api.deps import api_logger
from utils.schemas import FeedbackAck, FeedbackPayload

router = APIRouter(tags=["feedback"])

# Configurable via env (set FEEDBACK_LOG_PATH on the host) but defaults to
# a path that exists inside the Docker image without needing a writable
# volume — Koyeb's ephemeral disk is enough for hackathon-scale capture.
_DEFAULT_PATH = Path(os.environ.get("FEEDBACK_LOG_PATH", "data/feedback.jsonl"))
_DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)

_write_lock = Lock()


def _generate_id() -> str:
    return secrets.token_hex(5)


def _append_record(record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with _DEFAULT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


@router.post("/feedback", response_model=FeedbackAck)
def submit_feedback(payload: FeedbackPayload) -> FeedbackAck:
    """Append one feedback entry to the JSONL log."""
    feedback_id = _generate_id()
    record = {
        "id": feedback_id,
        "ts": int(time.time()),
        **payload.model_dump(),
    }
    try:
        _append_record(record)
    except OSError as exc:
        api_logger.exception("feedback write failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "feedback_write_failed", "detail": str(exc)},
        ) from exc
    api_logger.info(
        "feedback id=%s task=%s rating=%d user=%s",
        feedback_id,
        payload.task,
        payload.rating,
        payload.user_id,
    )
    return FeedbackAck(received=True, id=feedback_id)


@router.get("/feedback/stats")
def feedback_stats() -> Dict[str, Any]:
    """Cheap aggregate over the JSONL log.

    Walks the whole file each call; fine at hackathon scale (thousands of
    rows). If this ever gets hot, swap to a counter file or a tiny SQLite
    table.
    """
    total = 0
    positive = 0
    negative = 0
    if _DEFAULT_PATH.exists():
        with _DEFAULT_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                rating = row.get("rating")
                if rating == 1:
                    positive += 1
                elif rating == -1:
                    negative += 1
    pct = round((positive / total) * 100, 1) if total else 0.0
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_pct": pct,
        "path": str(_DEFAULT_PATH),
    }
