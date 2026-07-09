"""chatbot.run — the ONE entrypoint. Standalone for now; apps/chat will call
this instead of hitting the engine directly, once tested.

CLI smoke test:
    python -m chatbot.run "how many incidents are escalated" mysession
    python -m chatbot.run "and waived ones?" mysession   # same session_id -> follow-up
"""
from __future__ import annotations

from typing import Optional

from .graph import get_graph


def run_chat_turn(
    message: str,
    session_id: str,
    history: Optional[list] = None,
    tenant: str = "default",
) -> dict:
    """The ONE function a caller (later: apps/chat) invokes per user turn.

    `session_id` -> LangGraph `thread_id`: the checkpointer persists this
    graph's state per session automatically (§ checkpointer.py).
    """
    graph = get_graph()
    result = graph.invoke(
        {
            "message": message,
            "history": history or [],
            "session_id": session_id,
            "tenant": tenant,
        },
        config={"configurable": {"thread_id": session_id}},
    )

    return {
        "session_id": session_id,
        "answer_text": result.get("reply_text"),
        "needs_clarification": result.get("needs_clarification", False),
        "clarification_question": result.get("clarification_question"),
        "sql": result.get("sql"),
        "rows": result.get("rows"),
        # result.get("status") or ... (not .get(key, default)): classify_node
        # explicitly resets status to None every turn, so a missing-vs-None
        # distinction would break this fallback for smalltalk turns.
        "status": result.get("status") or ("smalltalk" if result.get("action") == "smalltalk" else "answered"),
    }


if __name__ == "__main__":
    import json
    import logging
    import sys

    # Configured here, not at module level: this file is meant to be imported
    # (by chat_cli.py, later by apps/chat) as a plain library — a module-level
    # basicConfig() would silently reconfigure the importing app's root logger
    # (e.g. Django's) the moment `chatbot.run` is imported anywhere.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | [%(name)s] %(message)s")

    msg = sys.argv[1] if len(sys.argv) > 1 else "hi"
    sid = sys.argv[2] if len(sys.argv) > 2 else "cli-test-session"
    response = run_chat_turn(msg, sid)
    print(json.dumps(response, indent=2, default=str))
