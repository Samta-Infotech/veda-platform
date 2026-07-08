"""chatbot.state — the state object passed through every LangGraph node."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class Turn(TypedDict):
    role: str          # "user" | "assistant"
    content: str


class ChatState(TypedDict, total=False):
    # ── input ────────────────────────────────────────────────────────────────
    message: str                      # this turn's raw user message
    # Annotated + operator.add = a REDUCER: each node's returned "history" list
    # is APPENDED to (not overwritten on top of) whatever the checkpointer
    # already has for this thread_id/session_id. Terminal nodes append this
    # turn's [user, assistant] pair; classify/resolve_followup only READ it
    # (never return it), so they never trigger an append. Callers no longer
    # need to thread conversation history through manually — the checkpointer
    # + this reducer accumulate it automatically per session.
    history: Annotated[List[Turn], operator.add]
    session_id: str
    tenant: str

    # ── supervisor decision ─────────────────────────────────────────────────
    action: str                        # "smalltalk" | "answer" | "clarify" | "followup"
    resolved_query: str                # the query actually sent to the engine
                                       # (== message, or message merged with prior context)

    # ── engine result ───────────────────────────────────────────────────────
    engine_result: Dict[str, Any]      # raw run_hybrid_query() result (dict form)
    status: str                        # "answered" | "refuse" | "clarify" | ... (engine status)

    # ── output ───────────────────────────────────────────────────────────────
    reply_text: str
    needs_clarification: bool
    clarification_question: Optional[str]
    sql: Optional[str]
    rows: Optional[list]
