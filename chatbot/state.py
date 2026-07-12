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
    source_id: Optional[int]           # forwarded to InferenceClient.stream_hybrid_query
    source_ids: Optional[List[int]]    # validated multi-source scope (P5), primary first —
                                       # forwarded alongside source_id so scoped chat turns
                                       # retrieve/federate exactly like /api/v1/query
    request_id: str                    # forwarded as X-Request-Id (tracing across api->inference)

    # ── supervisor decision ─────────────────────────────────────────────────
    action: str                        # "smalltalk" | "answer" | "clarify" | "followup"
    resolved_query: str                # the query actually sent to the engine
                                       # (== message, or message merged with prior context)

    # ── engine result ───────────────────────────────────────────────────────
    engine_result: Dict[str, Any]      # raw run_hybrid_query() result (dict form)
    status: str                        # "answered" | "refuse" | "clarify" | "no_table" |
                                       # "ungrounded" | "qualifier_dropped" | "ir_mismatch" |
                                       # "error" | "unavailable" (transport/infra failure —
                                       # distinct from a reachable engine's own refusal)

    # ── output ───────────────────────────────────────────────────────────────
    reply_text: str
    needs_clarification: bool
    clarification_question: Optional[str]
    sql: Optional[str]
    rows: Optional[list]
    engine_unavailable: bool           # True only when status == "unavailable"
