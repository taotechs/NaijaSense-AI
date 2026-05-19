"""Lightweight safety / validation layer for the agent gateway.

Two responsibilities, both *advisory* - we never block a request, we just
annotate the response with a ``safety_flags`` list so the UI can show a
badge and the team can audit later.

1. Input checks (run BEFORE the LLM call)
   - Prompt-injection patterns in user-controlled fields (``query``,
     ``tone_notes``, ``history``). We look for the classic "ignore previous
     instructions" family, role-impersonation, and fenced system-block
     attempts.
   - PII shapes in free-text fields (emails, phone numbers, BVNs) so we
     can warn the user *before* feeding them into a third-party LLM.

2. Output checks (run AFTER the agent returns)
   - PII leakage in generated text.
   - Hallucinated specifics: numeric facts (prices, ratings, dates) that
     appear in the output but are not grounded in the input. The check is
     intentionally conservative - we only flag when a specific numeric
     token cannot be matched against the original ``query`` /
     ``item_context``.

This module has zero runtime dependencies beyond the stdlib so it is
safe to import from anywhere (no risk of slowing cold start).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

# --- Pattern banks ----------------------------------------------------

_INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore (?:all |the )?(?:previous|prior|above) (?:instructions?|prompts?|messages?)\b",
        r"\bdisregard (?:all |the )?(?:previous|prior|above)\b",
        r"\byou are now (?:a |an )?(?:different|new) (?:assistant|model|ai)\b",
        r"\bsystem\s*[:>]\s*you (?:are|must|should)\b",
        r"```\s*system",
        r"\bjailbreak\b",
        r"\bDAN\b\s+mode",
        r"\bprint (?:your|the) (?:system )?prompt\b",
        r"\breveal (?:your|the) (?:system )?prompt\b",
    )
)

# Loose PII shapes. Intentionally conservative to avoid false positives on
# benign numbers (e.g. "10k budget"). We flag when shapes are unambiguous.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?234|0)\d{10}\b")  # Nigerian phone shapes
_BVN_RE = re.compile(r"\bBVN[\s:#-]*\d{11}\b", re.IGNORECASE)
_NUMERIC_FACT_RE = re.compile(
    r"(?<!\w)(?:₦|N|NGN|\$|USD|£|€)\s?\d{2,}(?:[,.]\d{1,3})*"  # money
    r"|(?<!\w)\d+(?:\.\d+)?\s?(?:%|naira|dollars?|usd|min(?:utes?)?|mins?|hrs?|hours?|km|kg|g|ml|l)\b",
    re.IGNORECASE,
)


# --- Public API -------------------------------------------------------


def check_input(*fragments: str) -> List[str]:
    """Return advisory flags raised by free-text fragments BEFORE the LLM call."""
    flags: List[str] = []
    blob = "\n".join(f for f in fragments if f).strip()
    if not blob:
        return flags
    if any(p.search(blob) for p in _INJECTION_PATTERNS):
        flags.append("prompt_injection_suspected")
    if _EMAIL_RE.search(blob):
        flags.append("pii_email_in_input")
    if _PHONE_RE.search(blob):
        flags.append("pii_phone_in_input")
    if _BVN_RE.search(blob):
        flags.append("pii_bvn_in_input")
    return flags


def check_output(output_text: str, grounding_sources: Iterable[str] = ()) -> List[str]:
    """Return advisory flags raised by generated output text."""
    flags: List[str] = []
    if not output_text:
        return flags
    if _EMAIL_RE.search(output_text):
        flags.append("pii_email_in_output")
    if _PHONE_RE.search(output_text):
        flags.append("pii_phone_in_output")
    if _BVN_RE.search(output_text):
        flags.append("pii_bvn_in_output")

    grounding_blob = " ".join(str(s) for s in grounding_sources if s).lower()
    output_facts = {m.group(0).lower() for m in _NUMERIC_FACT_RE.finditer(output_text)}
    ungrounded = [
        fact for fact in output_facts if _normalise_fact(fact) not in grounding_blob
    ]
    if len(ungrounded) >= 2:
        # One stray number could be tone (e.g. "5 stars" boilerplate); two or
        # more is a stronger hallucination signal worth flagging.
        flags.append("ungrounded_numeric_specifics")
    return flags


# --- Helpers ----------------------------------------------------------


def _normalise_fact(fact: str) -> str:
    """Collapse whitespace + lowercase so substring lookup is forgiving."""
    return " ".join(fact.lower().split())
