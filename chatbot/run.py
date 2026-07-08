"""chatbot.run — the ONE entrypoint. Called by apps/chat/services.py's
ConversationQueryService (production); also usable standalone.

CLI smoke test:
    python -m chatbot.run "how many incidents are escalated" mysession
    python -m chatbot.run "and waived ones?" mysession   # same session_id -> follow-up
"""
from __future__ import annotations

from typing import Callable, Optional

from .graph import get_graph


def run_chat_turn(
    message: str,
    session_id: str,
    history: Optional[list] = None,
    tenant: str = "default",
    source_id: Optional[int] = None,
    request_id: str = "",
    on_event: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """The ONE function a caller (apps/chat) invokes per user turn.

    `session_id` -> LangGraph `thread_id`: the checkpointer persists this
    graph's state per session automatically (§ checkpointer.py).

    `on_event(phase, message)`, if given, is stashed in the graph's
    config["configurable"] and invoked synchronously by nodes as the turn
    progresses (see nodes.py::_emit) — callers that want live progress (e.g.
    apps/chat/services.py's SSE stream) should run this on a background
    thread and drain events via a queue, since this call itself blocks until
    the whole turn is done.
    """
    graph = get_graph()
    result = graph.invoke(
        {
            "message": message,
            "history": history or [],
            "session_id": session_id,
            "tenant": tenant,
            "source_id": source_id,
            "request_id": request_id,
        },
        config={"configurable": {"thread_id": session_id, "on_event": on_event}},
    )

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
        "engine_result": result.get("engine_result") or {},
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
