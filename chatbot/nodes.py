"""chatbot.nodes — LangGraph node functions.

Each node takes a ChatState and returns a partial dict to merge into it
(standard LangGraph node signature). Nodes that need to report mid-turn
progress (classify_node, resolve_followup_node, call_engine_node) also
declare a `config: RunnableConfig` parameter — LangGraph injects it
automatically for any node function whose signature names a parameter
`config` (see langgraph/_internal/_runnable.py's KWARGS_CONFIG_KEYS); nodes
that don't need it are unaffected, mixed signatures in one graph are fine.
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.runnables import RunnableConfig

from apps.query.inference_client import InferenceClient, InferenceUnavailable

from .llm import call_slm
from .prompts import (
    FALLBACK_REPLY,
    FOLLOWUP_SYSTEM_PROMPT,
    STANDALONE_CHECK_SYSTEM,
    build_followup_user_prompt,
    build_smalltalk_system_prompt,
    build_standalone_check_user_prompt,
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

def _depends_on_history(message: str, history: list) -> bool:
    """Generic (non-keyword) second opinion for a "smalltalk" verdict when prior
    turns exist. Real users phrase referential follow-ups countless ways ("need
    more details", "aur bata", "what about the other one", ...) — no fixed word
    list generalizes to production traffic, so this asks the LLM the underlying
    semantic question directly instead of pattern-matching specific phrasings.
    Fails closed to False (trust the original "smalltalk" verdict) on any error,
    since this is only a second-opinion check, not the primary classifier."""
    user_prompt = build_standalone_check_user_prompt(message, history)
    verdict = call_ollama(STANDALONE_CHECK_SYSTEM, user_prompt, max_tokens=5)
    return bool(verdict) and "dependent" in verdict.strip().lower()

# Deterministic fast path for the overwhelming majority of smalltalk: pure
# greetings/thanks/farewells with nothing else in the message. Tight, anchored
# patterns (whole-message match, not substring) so they can never misfire on a
# real question that merely starts with "hi" or ends with "thanks" — and
# _DATA_QUESTION_HINTS is still checked as a second guard before trusting this.
# Skips the classify LLM call entirely (classify_node) and lets smalltalk_node
# skip its own LLM call too — on this deployment's hardware a single such call
# alone can take ~20s, so a bare "hi" was paying 20-40+ seconds of pure LLM
# round-trip time for something that should be instant.
_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|yo|good\s*(morning|afternoon|evening|day)|"
    r"how\s*(are\s*(you|u)|'?s\s*it\s*going)( doing)?)\s*[.,!?]*\s*$", re.IGNORECASE)
_THANKS_RE = re.compile(
    r"^\s*(thanks?( you)?( very much| so much| a lot)?|thx|ty|appreciate it|"
    r"much appreciated|cheers)\s*[.,!?]*\s*$", re.IGNORECASE)
_BYE_RE = re.compile(
    r"^\s*(bye|goodbye|see\s*(you|ya)( later| soon)?|take care|good\s*night)\s*[.,!?]*\s*$",
    re.IGNORECASE)


def _canned_smalltalk_reply(message: str) -> str | None:
    """Instant reply for the fast-path patterns above — None means "not a fast
    match, fall back to the LLM" (used by both classify_node and smalltalk_node
    so the two stay in lockstep on what counts as trivial smalltalk)."""
    if _DATA_QUESTION_HINTS.search(message):
        return None
    if _GREETING_RE.match(message):
        return FALLBACK_REPLY
    if _THANKS_RE.match(message):
        return "You're welcome! Let me know if you have any other data questions."
    if _BYE_RE.match(message):
        return "Goodbye! Come back anytime you have data questions."
    return None


def _emit(config: RunnableConfig | None, phase: str, message: str) -> None:
    """Best-effort progress callback — a broken/absent UI callback must never
    sink the turn it's merely reporting on. Callers stash `on_event` in
    config["configurable"] (see chatbot/run.py::run_chat_turn)."""
    on_event = ((config or {}).get("configurable") or {}).get("on_event")
    if on_event is None:
        return
    try:
        on_event(phase, message)
    except Exception:
        logger.exception("_emit: on_event callback raised for phase=%s", phase)


def _turn_delta(state: ChatState, assistant_reply: str) -> list:
    """[user, assistant] pair for THIS turn — appended (not replacing) to the
    checkpointed history via the Annotated[..., operator.add] reducer on
    ChatState.history (state.py). Only terminal nodes call this, once each,
    so a turn is recorded exactly once regardless of which path it took."""
    return [
        {"role": "user", "content": state["message"]},
        {"role": "assistant", "content": assistant_reply},
    ]


def classify_node(state: ChatState, config: RunnableConfig) -> dict:
    """Decide what kind of message this is. Defaults to 'answer' (route to the
    engine) on any failure — refuse-over-guess: never silently short-circuit
    a real data question as smalltalk just because the classifier is down."""
    message = state["message"]
    history = state.get("history", [])

    if _canned_smalltalk_reply(message) is not None:
        # Deterministic fast path: a bare "hi"/"thanks"/"bye" needs no LLM call
        # at all — skips both this classify round-trip AND smalltalk_node's own
        # (each ~20s on this deployment's hardware). No "thinking" event either:
        # there's nothing to think about for an instant, deterministic reply.
        action = "smalltalk"
        logger.info("classify_node: deterministic smalltalk match, message=%r", message)
    else:
        _emit(config, "supervisor_classify", "Understanding your message...")
        raw = call_slm(
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

    if action == "smalltalk" and history and _depends_on_history(message, history):
        logger.warning(
            "classify_node: LLM said smalltalk but message depends on the prior "
            "conversation, overriding to 'followup': %r", message,
        )
        action = "followup"
    elif action == "smalltalk" and _DATA_QUESTION_HINTS.search(message):
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
        "engine_unavailable": False,
    }


def smalltalk_node(state: ChatState) -> dict:
    """Direct reply for greetings/thanks/chit-chat — engine bypassed entirely."""
    message = state["message"]
    reply = _canned_smalltalk_reply(message)
    if reply is None:
        # classify_node routed here via the LLM (not the deterministic fast
        # path above) — some smalltalk beyond a bare greeting/thanks/bye, so
        # this is the one case that still needs its own LLM call.
        reply = call_slm(
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
        "engine_unavailable": False,
        "history": _turn_delta(state, reply),
    }


def resolve_followup_node(state: ChatState, config: RunnableConfig) -> dict:
    """Rewrite the message as a fully self-contained question using prior turns.
    Used for both 'followup' and 'clarify_reply' actions — same mechanics:
    merge new input with context, then treat like a normal question."""
    _emit(config, "supervisor_followup", "Resolving your follow-up context...")
    message = state["message"]
    history = state.get("history", [])

    rewritten = call_slm(
        FOLLOWUP_SYSTEM_PROMPT,
        build_followup_user_prompt(message, history),
        max_tokens=80,
    )
    resolved = (rewritten or message).strip().strip('"')
    logger.info("resolve_followup_node: %r -> %r", message, resolved)
    return {"resolved_query": resolved}


def _extract_engine_result(payload: dict) -> tuple[dict, str]:
    """Walk MultiResult's wire shape ({"result": {"items": [{"result": {...}}]}}
    — same shape apps/query/inference_client.py's run_hybrid_query/
    stream_hybrid_query hand back, per inference/routes/hybrid.py's _serialize()).

    NOTE: item0 itself carries a SubResult-level status ("ok"/"refused"/"error",
    veda_core/query/multi_result.py) — that is NOT the status we want here. The
    pipeline-level status ("answered"/"refuse"/"clarify"/"no_table"/...,
    veda_core/veda/pipeline.py::_done) is one level deeper, at
    item0["result"]["status"]. Do not "simplify" this by reading item0["status"].
    """
    result = (payload or {}).get("result") or {}
    items = result.get("items") or []
    item0 = items[0] if items and isinstance(items[0], dict) else {}
    res0 = item0.get("result") or {}
    if not isinstance(res0, dict):
        res0 = {}
    return res0, res0.get("status", "error")


def call_engine_node(state: ChatState, config: RunnableConfig) -> dict:
    """Calls the inference tier over HTTP via apps.query.inference_client —
    same client/contract every other apps/ caller uses. The api tier never
    imports veda_core directly (see InferenceClient's own docstring); chatbot/
    now runs inside the api container's process (apps/chat/services.py), so it
    is subject to that same boundary.

    Forwards the inference tier's own stage-progress events (classify/
    decompose/route/answer/...) live via _emit as they arrive off the SSE
    stream — the loop below iterates the generator synchronously, so this
    naturally streams to the caller rather than batching.

    A transport/infra failure (InferenceUnavailable, or a mid-stream "error"
    event) is reported as status="unavailable"/engine_unavailable=True — kept
    DISTINCT from a reachable engine's own legitimate refusal, so callers
    (apps/chat/services.py) can surface a genuine outage as an error instead
    of a misleading "please clarify" chat reply.
    """
    query = state.get("resolved_query") or state["message"]
    client = InferenceClient()
    res0: dict = {}
    status = "error"

    try:
        for kind, data in client.stream_hybrid_query(
            query,
            source_id=state.get("source_id"),
            tenant=state.get("tenant"),
            request_id=state.get("request_id"),
        ):
            if kind == "progress":
                _emit(config, data.get("phase", "progress"), data.get("message", ""))
            elif kind == "error":
                logger.warning("call_engine_node: inference stream error for query=%r: %s", query, data)
                return {"engine_result": {}, "status": "unavailable", "engine_unavailable": True}
            elif kind == "result":
                res0, status = _extract_engine_result(data)
    except InferenceUnavailable as exc:
        logger.warning("call_engine_node: inference unavailable for query=%r: %s", query, exc)
        return {"engine_result": {}, "status": "unavailable", "engine_unavailable": True}
    except Exception:
        logger.exception("call_engine_node: unexpected failure for query=%r", query)
        return {"engine_result": {}, "status": "error", "engine_unavailable": False}

    logger.info("call_engine_node: status=%s query=%r", status, query)
    return {"engine_result": res0, "status": status, "engine_unavailable": False}


def ask_clarification_node(state: ChatState) -> dict:
    """Turn a refusal into a conversational clarifying question — reuses the
    engine's own deterministic explanation (already computed server-side by
    veda_core/veda/pipeline.py's _feedback()/explain_failure() for every
    non-"answered" status, and embedded at res0["feedback"]["text"]), never
    invents reasons of its own (refuse-over-guess, same as the rest of the
    codebase).

    status == "unavailable" (a transport/infra failure, not a real engine
    refusal — see call_engine_node) gets its own honest reply instead of the
    generic clarification text, and is not recorded into checkpointed history
    since a transient outage isn't real conversation content.
    """
    res0 = state.get("engine_result", {})
    status = state.get("status", "refuse")

    feedback = res0.get("feedback")
    if feedback:
        question = feedback.get("text") or "Could you clarify what you're asking about?"
    else:
        # Rare fallback (e.g. FEEDBACK_ENABLED=False in the engine, so it
        # never built one) — best-effort only, since we don't have the
        # per-status context (missing/column/value/candidates) the engine
        # itself has.
        try:
            from veda_core.veda.feedback import explain_failure
            explanation = explain_failure(status, res0.get("sm"), msg=res0.get("refusal"))
            question = explanation.get("text") or "Could you clarify what you're asking about?"
        except Exception:
            logger.exception("ask_clarification_node: explain_failure failed for status=%r", status)
            question = "Could you clarify what you're asking about?"

    update = {
        "reply_text": question,
        "needs_clarification": not unavailable,
        "clarification_question": None if unavailable else question,
        "engine_unavailable": unavailable,
    }
    if not unavailable:
        update["history"] = _turn_delta(state, question)
    return update


def format_reply_node(state: ChatState) -> dict:
    """Final assembly for the 'answered' path."""
    res0 = state.get("engine_result", {})
    answer = res0.get("answer") or "Here's what I found."
    return {
        "reply_text": answer,
        "needs_clarification": False,
        "sql": res0.get("sql"),
        "rows": res0.get("rows"),
        "engine_unavailable": False,
        "history": _turn_delta(state, answer),
    }
