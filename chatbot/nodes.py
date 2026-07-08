"""chatbot.nodes — LangGraph node functions.

Each node takes a ChatState and returns a partial dict to merge into it
(standard LangGraph node signature).
"""
from __future__ import annotations

import json
import logging
import os
import re

from .llm import call_ollama
from .prompts import (
    FALLBACK_REPLY,
    FOLLOWUP_SYSTEM_PROMPT,
    build_followup_user_prompt,
    build_smalltalk_system_prompt,
    build_supervisor_system_prompt,
    build_supervisor_user_prompt,
)
from .state import ChatState

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"smalltalk", "followup", "clarify_reply", "answer"}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Deterministic safety net, on top of the LLM classifier: generic (schema-
# agnostic — no table/column names) words that essentially never appear in
# pure smalltalk. If the LLM still says "smalltalk" despite one of these being
# present, override it — a real question wrongly treated as smalltalk means
# smalltalk_node (which has no data access) would have to invent an answer,
# which this system must never do (refuse-over-guess, same as the rest of
# the codebase's firewall).
_DATA_QUESTION_HINTS = re.compile(
    r"\b(count|how many|total|list|show me|average|sum|status|data|record|"
    r"table|report|number of|which|what was|remind me)\b",
    re.IGNORECASE,
)

# veda_core's own data/schema/client_bge paths are cwd-relative to veda_core/
# (same reason apps/ingestion/tasks.py runs the engine subprocess with
# cwd=veda_core/) — call_engine_node chdirs here for the engine call. No
# sys.path manipulation needed: package-qualified imports (from veda_core.X
# import Y, below) resolve veda_core as an ordinary package from the repo
# root, and veda_core/__init__.py's own shim handles its internal legacy
# top-level imports (from config import ..., etc.) from there — same reason
# these stay IDE-navigable (go-to-definition works) unlike a bare
# `from veda_hybrid import ...` would.
_VEDA_CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "veda_core"))


def _turn_delta(state: ChatState, assistant_reply: str) -> list:
    """[user, assistant] pair for THIS turn — appended (not replacing) to the
    checkpointed history via the Annotated[..., operator.add] reducer on
    ChatState.history (state.py). Only terminal nodes call this, once each,
    so a turn is recorded exactly once regardless of which path it took."""
    return [
        {"role": "user", "content": state["message"]},
        {"role": "assistant", "content": assistant_reply},
    ]


def classify_node(state: ChatState) -> dict:
    """Decide what kind of message this is. Defaults to 'answer' (route to the
    engine) on any failure — refuse-over-guess: never silently short-circuit
    a real data question as smalltalk just because the classifier is down."""
    message = state["message"]
    history = state.get("history", [])

    raw = call_ollama(
        build_supervisor_system_prompt(),   # built fresh each call so "today" is always current
        build_supervisor_user_prompt(message, history),
    )
    action = "answer"
    if raw:
        match = _JSON_RE.search(raw)
        if match:
            try:
                parsed = json.loads(match.group())
                candidate = parsed.get("action")
                if candidate in _VALID_ACTIONS:
                    action = candidate
            except Exception:
                logger.warning("classify_node: could not parse LLM output: %r", raw)

    if action == "smalltalk" and _DATA_QUESTION_HINTS.search(message):
        logger.warning(
            "classify_node: LLM said smalltalk but message looks data-related, "
            "overriding to 'answer': %r", message,
        )
        action = "answer"

    logger.info("classify_node: action=%s message=%r", action, message)
    # Reset per-turn output fields — the checkpointer persists the FULL state
    # across turns (that's the point, for history/context), but sql/rows/
    # status/engine_result/clarification are this-turn-only outputs. Without
    # this reset they'd leak forward from a previous turn's answer into a
    # later turn (e.g. smalltalk) that never touches these fields itself.
    return {
        "action": action,
        "sql": None,
        "rows": None,
        "status": None,
        "engine_result": {},
        "needs_clarification": False,
        "clarification_question": None,
    }


def smalltalk_node(state: ChatState) -> dict:
    """Direct reply for greetings/thanks/chit-chat — engine bypassed entirely."""
    message = state["message"]
    reply = call_ollama(
        build_smalltalk_system_prompt(),   # built fresh each call so "today" is always current
        message,
        max_tokens=60,
    )
    if not reply:
        reply = FALLBACK_REPLY
    return {
        "reply_text": reply,
        "needs_clarification": False,
        "resolved_query": "",
        "history": _turn_delta(state, reply),
    }


def resolve_followup_node(state: ChatState) -> dict:
    """Rewrite the message as a fully self-contained question using prior turns.
    Used for both 'followup' and 'clarify_reply' actions — same mechanics:
    merge new input with context, then treat like a normal question."""
    message = state["message"]
    history = state.get("history", [])

    rewritten = call_ollama(
        FOLLOWUP_SYSTEM_PROMPT,
        build_followup_user_prompt(message, history),
        max_tokens=80,
    )
    resolved = (rewritten or message).strip().strip('"')
    logger.info("resolve_followup_node: %r -> %r", message, resolved)
    return {"resolved_query": resolved}


def call_engine_node(state: ChatState) -> dict:
    """Call the existing VEDA engine VERBATIM.

    TESTING: direct import of veda_core.veda_hybrid.run_hybrid_query.
    PRODUCTION (later, when wired into apps/chat): swap this for
    apps.query.inference_client.InferenceClient.run_hybrid_query(), which goes
    over HTTP to the inference service instead — same call shape, no other
    node needs to change.

    Never lets an engine-side exception (DB down, unexpected internal error)
    crash the graph — falls through to ask_clarification_node with
    status="error" instead, same as any other non-answered status.
    """
    query = state.get("resolved_query") or state["message"]

    from veda_core.veda_hybrid import run_hybrid_query

    # veda_core's own data/schema/client_bge paths are cwd-relative to veda_core/
    # (same reason apps/ingestion/tasks.py runs the engine subprocess with
    # cwd=veda_core/) — chdir there for the call, then always restore, so this
    # node works no matter which directory the caller (chat_cli.py, apps/chat,
    # a test runner) was launched from.
    _prev_cwd = os.getcwd()
    os.chdir(_VEDA_CORE_DIR)
    try:
        result = run_hybrid_query(query, verbose=False)
    except Exception:
        logger.exception("call_engine_node: engine raised for query=%r", query)
        return {"engine_result": {}, "status": "error"}
    finally:
        os.chdir(_prev_cwd)

    # MultiResult -> first item's result dict (mirrors apps/chat/services.py's
    # _first_item_result, so both layers read the same shape).
    items = getattr(result, "items", None) or []
    item0 = items[0] if items else None
    res0 = getattr(item0, "result", {}) if item0 is not None else {}
    if not isinstance(res0, dict):
        res0 = {}

    status = res0.get("status", "error")
    logger.info("call_engine_node: status=%s query=%r", status, query)
    return {"engine_result": res0, "status": status}


def ask_clarification_node(state: ChatState) -> dict:
    """Turn a refusal into a conversational clarifying question — reuses the
    engine's own deterministic explanation (veda/feedback.py), never invents
    reasons of its own (refuse-over-guess, same as the rest of the codebase)."""
    res0 = state.get("engine_result", {})
    status = state.get("status", "refuse")

    try:
        from veda_core.veda.feedback import explain_failure
        explanation = explain_failure(status, res0.get("sm"), msg=res0.get("refusal"))
        question = explanation.get("text") or "Could you clarify what you're asking about?"
    except Exception:
        logger.exception("ask_clarification_node: explain_failure failed for status=%r", status)
        question = "Could you clarify what you're asking about?"

    return {
        "reply_text": question,
        "needs_clarification": True,
        "clarification_question": question,
        "history": _turn_delta(state, question),
    }


def format_reply_node(state: ChatState) -> dict:
    """Final assembly for the 'answered' path."""
    res0 = state.get("engine_result", {})
    answer = res0.get("answer") or "Here's what I found."
    return {
        "reply_text": answer,
        "needs_clarification": False,
        "sql": res0.get("sql"),
        "rows": res0.get("rows"),
        "history": _turn_delta(state, answer),
    }
