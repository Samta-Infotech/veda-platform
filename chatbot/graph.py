"""chatbot.graph — the Supervisor/Planner StateGraph.

    memory_read
       |
       v
    classify
       |-- smalltalk ------------------------------------> smalltalk_node -> END
       |-- runtime_context -------------------------------------------.
       |-- (else) history non-empty -> context_resolve_node -.        |
       |-- (else) history empty --------------------------------------+--> call_engine_node
                                                                       |-- answered -> memory_write_node
                                                                       |               -> format_reply_node -> END
                                                                       `-- (else)   -> ask_clarification_node -> END

Routing after classify is based on HISTORY, not the LLM's exact action label
(followup vs. clarify_reply vs. answer) — the LLM only needs to reliably tell
smalltalk apart from a real data question; whether history is actually needed
to resolve that question is decided deterministically from the checkpointed
state instead. This avoids a real question (e.g. "what was my incident
count") silently skipping history-aware resolution just because the
classifier called it "answer" instead of "followup" — context_resolve_node
is a no-op resolution when the message is already self-contained anyway.
"runtime_context" (classify_node's deterministic current-date/time match) is
the one exception: always self-contained, so it skips context_resolve_node
regardless of history.

`memory_read`/`memory_write` add the structured analytical memory (QueryFrame
+ DrillStack, see chatbot/memory/ and docs/MEMORY_ARCHITECTURE.md) around the
unchanged core flow above: memory_read loads it before classify ever runs
(cheap, O(1) Redis reads); memory_write persists it ONLY after a successful
("answered") engine execution — never on refuse/error/clarify, since memory
must only ever record proven evidence (docs/MEMORY_ARCHITECTURE.md §6/§12).
"""
from __future__ import annotations

import logging
import threading

from langgraph.graph import StateGraph, END

from .checkpointer import get_checkpointer
from .nodes import (
    clarify_reply_node,
    no_match_node,
    reset_node,
    recall_node,
    represent_node,
    ask_clarification_node,
    call_engine_node,
    classify_with_entry_gate,
    context_resolve_node,
    format_reply_node,
    memory_read_node,
    memory_write_node,
    smalltalk_node,
)
from .state import ChatState

logger = logging.getLogger(__name__)

_ANSWERED_STATUSES = {"answered"}


def _route_after_classify(state: ChatState) -> str:
    """Edge function after classify_node: smalltalk bypasses the engine
    entirely; any real question goes through context_resolve_node whenever
    there's history to resolve against (checkpoint-based, not label-based —
    see module docstring), straight to the engine otherwise (first turn).

    "runtime_context" (deterministic current-date/time match) always goes
    straight to the engine too, regardless of history — unlike an ordinary
    question, it's never dependent on prior turns, so context_resolve_node's
    work would be pure wasted latency."""
    action = state.get("action")
    if action == "smalltalk":
        return "smalltalk"
    if action == "clarify_reply":
        # Completes a pending request, then carries on to the engine exactly as an
        # ordinary turn does — routing, agents and execution are untouched.
        return "clarify_reply"
    if action == "reset":
        # The session's memory was just wiped — acknowledge and stop. Never the engine.
        return "reset"
    if action == "no_match":
        # Nothing in the message names anything in the data — answered honestly and
        # immediately instead of spending a full engine round-trip on a refusal.
        return "no_match"
    if action == "recall":
        # A question about the conversation itself: answered from the QueryFrame,
        # which memory_read_node has already loaded. Never reaches the engine.
        return "recall"
    if action == "represent":
        # A presentation-only follow-up: the previous result is still in the
        # checkpoint and is re-rendered as-is. Never reaches the engine.
        return "represent"
    if action != "runtime_context" and state.get("history"):
        return "context_resolve"
    return "call_engine"


def _route_after_engine(state: ChatState) -> str:
    """Edge function after call_engine_node: answered -> record memory then
    format; else -> clarify (memory_write_node is deliberately NOT on this
    branch — see its own docstring: evidence-only, never on a non-"answered"
    status)."""
    if state.get("status") in _ANSWERED_STATUSES:
        return "memory_write"
    return "ask_clarification"


def build_graph():
    """Builds and compiles the Supervisor/Planner StateGraph (see module
    docstring for the shape). Called once by get_graph(); use get_graph() in
    application code so the graph + checkpointer are built a single time."""
    logger.info("build_graph: compiling chatbot supervisor graph")
    g = StateGraph(ChatState)

    g.add_node("memory_read", memory_read_node)
    # classify_with_entry_gate == classify_node + the Turn Entry Gate's reporting.
    # The gate adds no decision; see entry_decision in chatbot/nodes.py.
    g.add_node("classify", classify_with_entry_gate)
    g.add_node("smalltalk", smalltalk_node)
    g.add_node("represent", represent_node)
    g.add_node("recall", recall_node)
    g.add_node("no_match", no_match_node)
    g.add_node("reset", reset_node)
    g.add_node("clarify_reply", clarify_reply_node)
    g.add_node("context_resolve", context_resolve_node)
    g.add_node("call_engine", call_engine_node)
    g.add_node("memory_write", memory_write_node)
    g.add_node("format_reply", format_reply_node)
    g.add_node("ask_clarification", ask_clarification_node)

    g.set_entry_point("memory_read")
    g.add_edge("memory_read", "classify")
    g.add_conditional_edges("classify", _route_after_classify, {
        "smalltalk": "smalltalk",
        "represent": "represent",
        "recall": "recall",
        "no_match": "no_match",
        "reset": "reset",
        "clarify_reply": "clarify_reply",
        "context_resolve": "context_resolve",
        "call_engine": "call_engine",
    })
    g.add_edge("context_resolve", "call_engine")
    g.add_edge("clarify_reply", "call_engine")
    g.add_conditional_edges("call_engine", _route_after_engine, {
        "memory_write": "memory_write",
        "ask_clarification": "ask_clarification",
    })
    g.add_edge("memory_write", "format_reply")
    g.add_edge("smalltalk", END)
    g.add_edge("represent", END)
    g.add_edge("recall", END)
    g.add_edge("no_match", END)
    g.add_edge("reset", END)
    g.add_edge("format_reply", END)
    g.add_edge("ask_clarification", END)

    return g.compile(checkpointer=get_checkpointer())


_GRAPH = None
_LOCK = threading.Lock()


def get_graph():
    """Process-wide compiled graph singleton — build_graph() runs once."""
    global _GRAPH
    if _GRAPH is None:
        with _LOCK:
            if _GRAPH is None:
                _GRAPH = build_graph()
    return _GRAPH
