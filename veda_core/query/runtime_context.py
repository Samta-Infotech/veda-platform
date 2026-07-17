# query/runtime_context.py
# VEDA — Runtime Context Provider (Layer 0)
#
# Deterministic answers for pure system/runtime-value questions (current date,
# current time, day of week) that reference no real table or column. These
# never need retrieval, SQL generation, or an LLM call — same "skip the
# round-trip entirely" idea as chatbot/nodes.py's smalltalk fast path.
#
# Every pattern below is anchored to the WHOLE message, so a question that
# ALSO references real data ("incidents opened today") does NOT match here —
# it falls through to the normal SQL pipeline unchanged.
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Optional

_DATETIME_RE = re.compile(
    r"^\s*what(?:'s| is) (?:the )?(?:current )?date and time\s*\??\s*$"
    r"|^\s*current date and time\s*\??\s*$",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"^\s*what(?:'s| is) (?:the )?(?:current date|today'?s? date|date(?: today)?)\s*\??\s*$"
    r"|^\s*(?:today'?s? date|current date)\s*\??\s*$"
    r"|^\s*what date is it(?: today)?\s*\??\s*$",
    re.IGNORECASE,
)
_DAY_OF_WEEK_RE = re.compile(
    r"^\s*what day (?:is it|of the week is it)(?: today)?\s*\??\s*$",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"^\s*what(?:'s| is) (?:the )?current time\s*\??\s*$"
    r"|^\s*current time\s*\??\s*$"
    r"|^\s*what time is it(?: now)?\s*\??\s*$",
    re.IGNORECASE,
)


def answer_runtime_context(query: str) -> Optional[dict]:
    """None when `query` isn't a pure runtime-value question. Otherwise a
    pipeline-shaped result dict — same "answered" contract veda/pipeline.py's
    run_query() returns — with a deterministic answer and no table/SQL.
    No LLM call is ever made here, so usage is genuinely zero (not a gap);
    latency_ms is still measured so the field is never missing/null on this
    path — same contract shape as every other answered result."""
    _t0 = time.time()
    now = datetime.now(timezone.utc)
    if _DATETIME_RE.match(query):
        answer = f"The current date and time is {now.strftime('%Y-%m-%d %H:%M')} UTC."
    elif _DATE_RE.match(query):
        answer = f"Today's date is {now.strftime('%Y-%m-%d')}."
    elif _DAY_OF_WEEK_RE.match(query):
        answer = f"Today is {now.strftime('%A')}."
    elif _TIME_RE.match(query):
        answer = f"The current time is {now.strftime('%H:%M')} UTC."
    else:
        return None
    return {"ok": True, "status": "answered", "answer": answer,
            "cols": None, "rows": None, "sql": None, "table": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": round((time.time() - _t0) * 1000, 2)}
