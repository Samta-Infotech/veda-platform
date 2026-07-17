"""chatbot.run — the ONE entrypoint. Called by apps/chat/services.py's
ConversationQueryService (production); also usable standalone.

CLI smoke test:
    python -m chatbot.run "how many incidents are escalated" mysession
    python -m chatbot.run "and waived ones?" mysession   # same session_id -> follow-up
"""
from __future__ import annotations

from typing import Callable, Optional

from .graph import get_graph
from .llm import collect_usage, usage_totals


def run_chat_turn(
    message: str,
    session_id: str,
    history: Optional[list] = None,
    tenant: str = "default",
    source_id: Optional[int] = None,
    source_ids: Optional[list] = None,
    request_id: str = "",
    on_event: Optional[Callable[[str, str, dict], None]] = None,
) -> dict:
    """The ONE function a caller (apps/chat) invokes per user turn.

    `session_id` -> LangGraph `thread_id`: the checkpointer persists this
    graph's state per session automatically (§ checkpointer.py).

    `on_event(phase, message, extra)`, if given, is stashed in the graph's
    config["configurable"] and invoked synchronously by nodes as the turn
    progresses (see nodes.py::_emit) — callers that want live progress (e.g.
    apps/chat/services.py's SSE stream) should run this on a background
    thread and drain events via a queue, since this call itself blocks until
    the whole turn is done. `extra` is the inference tier's own per-phase
    structured fields (route's intent=, sub_query's index=/total=, ...),
    forwarded verbatim — {} when a phase carries none.
    """
    graph = get_graph()
    # Wraps the WHOLE turn (classify/smalltalk/followup nodes' own SLM calls —
    # chatbot/llm.py — plus, nested inside, the engine's own collect_usage()
    # scope around run_query()/Tier-2/federated). Folded into engine_result's
    # "usage" below so the supervisor's token spend is never silently dropped —
    # previously only the engine side was captured.
    with collect_usage() as _chat_usage:
        result = graph.invoke(
            {
                "message": message,
                "history": history or [],
                "session_id": session_id,
                "tenant": tenant,
                "source_id": source_id,
                "source_ids": list(source_ids) if source_ids else None,
                "request_id": request_id,
            },
            config={"configurable": {"thread_id": session_id, "on_event": on_event}},
        )

    engine_result = dict(result.get("engine_result") or {})
    _chat_totals = usage_totals(_chat_usage.calls())
    if _chat_totals["total_tokens"]:
        _engine_usage = engine_result.get("usage") or {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        engine_result["usage"] = {
            "prompt_tokens": _engine_usage.get("prompt_tokens", 0) + _chat_totals["prompt_tokens"],
            "completion_tokens": _engine_usage.get("completion_tokens", 0) + _chat_totals["completion_tokens"],
            "total_tokens": _engine_usage.get("total_tokens", 0) + _chat_totals["total_tokens"],
        }

    return {
        "session_id": session_id,
        "answer_text": result.get("reply_text"),
        "reply_text": result.get("reply_text"),
        "needs_clarification": result.get("needs_clarification", False),
        "clarification_question": result.get("clarification_question"),
        "sql": result.get("sql"),
        "rows": result.get("rows"),
        # result.get("status") or ... (not .get(key, default)): classify_node
        # explicitly resets status to None every turn, so a missing-vs-None
        # distinction would break this fallback for smalltalk turns.
        "status": result.get("status") or ("smalltalk" if result.get("action") == "smalltalk" else "answered"),
        "engine_unavailable": result.get("engine_unavailable", False),
        "engine_result": engine_result,
    }


if __name__ == "__main__":
    import json
    import logging
    import sys

    # Configured here, not at module level: this file is meant to be imported
    # (by chat_cli.py, apps/chat) as a plain library — a module-level
    # basicConfig() would silently reconfigure the importing app's root logger
    # (e.g. Django's) the moment `chatbot.run` is imported anywhere.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | [%(name)s] %(message)s")

    msg = sys.argv[1] if len(sys.argv) > 1 else "hi"
    sid = sys.argv[2] if len(sys.argv) > 2 else "cli-test-session"
    response = run_chat_turn(msg, sid)
    print(json.dumps(response, indent=2, default=str))
