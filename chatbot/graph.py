"""chatbot.graph — the Supervisor/Planner StateGraph.

    classify
       |-- smalltalk ------------------------------------> smalltalk_node -> END
       |-- (else) history non-empty -> resolve_followup_node -.
       |-- (else) history empty ------------------------------+--> call_engine_node
                                                                       |-- answered -> format_reply_node -> END
                                                                       `-- (else)   -> ask_clarification_node -> END

Routing after classify is based on HISTORY, not the LLM's exact action label
(followup vs. clarify_reply vs. answer) — the LLM only needs to reliably tell
smalltalk apart from a real data question; whether history is actually needed
to resolve that question is decided deterministically from the checkpointed
state instead. This avoids a real question (e.g. "what was my incident
count") silently skipping history-aware resolution just because the
classifier called it "answer" instead of "followup" — resolve_followup_node
is a no-op rewrite when the message is already self-contained anyway.
"""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from .checkpointer import get_checkpointer
from .nodes import (
    ask_clarification_node,
    call_engine_node,
    classify_node,
    format_reply_node,
    resolve_followup_node,
    smalltalk_node,
)
from .state import ChatState

logger = logging.getLogger(__name__)

_ANSWERED_STATUSES = {"answered"}


def _route_after_classify(state: ChatState) -> str:
    """Edge function after classify_node: smalltalk bypasses the engine
    entirely; any real question goes through resolve_followup_node whenever
    there's history to resolve against (checkpoint-based, not label-based —
    see module docstring), straight to the engine otherwise (first turn)."""
    if state.get("action") == "smalltalk":
        return "smalltalk"
    if state.get("history"):
        return "resolve_followup"
    return "call_engine"


def _route_after_engine(state: ChatState) -> str:
    """Edge function after call_engine_node: answered -> format, else clarify."""
    if state.get("status") in _ANSWERED_STATUSES:
        return "format_reply"
    return "ask_clarification"


def build_graph():
    """Builds and compiles the Supervisor/Planner StateGraph (see module
    docstring for the shape). Called once by get_graph(); use get_graph() in
    application code so the graph + checkpointer are built a single time."""
    logger.info("build_graph: compiling chatbot supervisor graph")
    g = StateGraph(ChatState)

    g.add_node("classify", classify_node)
    g.add_node("smalltalk", smalltalk_node)
    g.add_node("resolve_followup", resolve_followup_node)
    g.add_node("call_engine", call_engine_node)
    g.add_node("format_reply", format_reply_node)
    g.add_node("ask_clarification", ask_clarification_node)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _route_after_classify, {
        "smalltalk": "smalltalk",
        "resolve_followup": "resolve_followup",
        "call_engine": "call_engine",
    })
    g.add_edge("resolve_followup", "call_engine")
    g.add_conditional_edges("call_engine", _route_after_engine, {
        "format_reply": "format_reply",
        "ask_clarification": "ask_clarification",
    })
    g.add_edge("smalltalk", END)
    g.add_edge("format_reply", END)
    g.add_edge("ask_clarification", END)

    return g.compile(checkpointer=get_checkpointer())


_GRAPH = None


def get_graph():
    """Process-wide compiled graph singleton — build_graph() runs once."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH
