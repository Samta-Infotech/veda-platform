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
from .memory.store import session_turn_lock


def run_chat_turn(
    message: str,
    session_id: str,
    history: Optional[list] = None,
    tenant: str = "default",
    source_id: Optional[int] = None,
    source_ids: Optional[list] = None,
    request_id: str = "",
    on_event: Optional[Callable[[str, str, dict], None]] = None,
    data_scope: Optional[dict] = None,
    source_profiles: Optional[dict] = None,
    no_cache: bool = False,
    data_vocabulary: Optional[list] = None,
    message_mentions_data: Optional[bool] = None,
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

    `data_scope` (Gate 1, User Story 3, Task 15) is the caller's precomputed RBAC
    data-scope payload (apps.access_management.services.serialize_data_scope) — a
    plain JSON-safe dict, never a Django object, since this package must stay
    Django-free. Forwarded verbatim to call_engine_node, which is the only node
    that reaches the inference tier. None (RBAC off, staff, or a caller that
    predates Task 15) means no narrowing, identical to before this parameter
    existed.

    `source_profiles` is the caller's per-source routing metadata
    (apps.query.scope.source_profiles_for — source_type/is_canonical/domain_tags/
    description), forwarded the same way and for the same reason: it is what tells
    the engine a source is datalake/document rather than relational. Without it the
    engine skips the datalake-isolated semantic model and plans SQL against the
    primary source's schema, which does not contain the datalake tables at all.
    /api/v1/query has always sent these; chat did not, which is why the same
    question answered there and clarified here. None = identical to before.
    """
    graph = get_graph()
    # Wraps the WHOLE turn (classify/smalltalk/followup nodes' own SLM calls —
    # chatbot/llm.py — plus, nested inside, the engine's own collect_usage()
    # scope around run_query()/Tier-2/federated). Folded into engine_result's
    # "usage" below so the supervisor's token spend is never silently dropped —
    # previously only the engine side was captured.
    _chat_calls = []
    # One chat's turns run one at a time. They are causally ordered by definition — turn
    # N+1 is a follow-up to turn N — so two at once is a race, not a workload: measured
    # 2026-09-18, 4 concurrent turns on one session left 2 of 8 history entries and a
    # frame at version 1 instead of 4, with nothing anywhere reporting the loss.
    # Serialization rather than optimistic retry: LangGraph's checkpoint carries no
    # version to retry against, so a retry could only ever repair the frame (which
    # already aborts correctly on conflict) and never the history the checkpointer lost.
    # Degrades to no locking if Redis is unreachable (chatbot/memory/store.py), which is
    # exactly how every turn behaved before this existed.
    with session_turn_lock(tenant, session_id), collect_usage() as _chat_usage:
        result = graph.invoke(
            {
                "message": message,
                "history": history or [],
                "session_id": session_id,
                "tenant": tenant,
                "source_id": source_id,
                "source_ids": list(source_ids) if source_ids else None,
                "request_id": request_id,
                "data_scope": data_scope,
                "source_profiles": source_profiles,
                "no_cache": bool(no_cache),   # → call_engine_node → X-Veda-No-Cache
                "data_vocabulary": data_vocabulary,
                "message_mentions_data": message_mentions_data,
            },
            config={"configurable": {"thread_id": session_id, "on_event": on_event}},
        )
        # MUST read calls() INSIDE the with block — collect_usage().__exit__()
        # clears the thread-local buffer the instant this block ends, so reading
        # it after always returns empty (same bug class fixed in
        # veda_hybrid.py::_maybe_federated() — see that fix's commit for the
        # full mechanism).
        _chat_calls = _chat_usage.calls()

    engine_result = dict(result.get("engine_result") or {})
    _chat_totals = usage_totals(_chat_calls)
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
        # M4 (C.6): the SUPERVISOR's own SLM calls this turn — classify / followup /
        # standalone-check. They go through chatbot/llm.py, a deliberately separate client
        # from the engine's (see that module's docstring), so they never appear in the
        # engine's explain trace and a trace-derived count cannot see them at all. This is
        # the number the turn budget is about: the deterministic delta layer's whole job
        # is to drive it to 0 on a follow-up, leaving only the engine's prose summary.
        "supervisor_slm_calls": len(_chat_calls),
        "supervisor_slm_purposes": [c.get("purpose") for c in _chat_calls],
        "answer_text": result.get("reply_text"),
        "reply_text": result.get("reply_text"),
        "needs_clarification": result.get("needs_clarification", False),
        "clarification_question": result.get("clarification_question"),
        "sql": result.get("sql"),
        "rows": result.get("rows"),
        # M4 (C.7): the IR-derived surfaces format_reply_node computed. Both are
        # deterministic and may legitimately be absent (a refusal has no result to
        # describe), so callers must treat None as "nothing to show", not as an error.
        "context_strip": result.get("context_strip"),
        "follow_up_questions": result.get("follow_up_questions"),
        # result.get("status") or ... (not .get(key, default)): classify_node
        # explicitly resets status to None every turn, so a missing-vs-None
        # distinction would break this fallback for smalltalk turns.
        "status": result.get("status") or ("smalltalk" if result.get("action") == "smalltalk" else "answered"),
        # Which path the conversation layer took this turn — "answer"/"followup" reach
        # the engine, everything else ("smalltalk", "recall", "represent", "no_match",
        # "reset", "runtime_context") is answered here without it. Returned for
        # telemetry only; nothing in the pipeline branches on it. Without this the only
        # recorded signal was the ENGINE's status, so a turn the conversation layer
        # handled entirely was indistinguishable from one that never ran.
        "action": result.get("action"),
        # What the conversation layer carried into this turn — the remembered entity and
        # filters, the operation applied, and the text actually sent to the engine. None
        # on a turn that used no context (a first question, smalltalk, recall), which is
        # exactly when there is nothing honest to show. Built by the graph from facts the
        # turn already produced (chatbot/nodes.py::_context_used) — never re-derived here.
        "context_used": result.get("context_used"),
        # Turn Entry Gate reporting (chatbot/nodes.py::classify_with_entry_gate) —
        # which tier settled the turn and what the classification itself cost, so the
        # latency of a direct answer can be told apart from the engine's.
        "entry_path": result.get("entry_path"),
        "classification_latency_ms": result.get("classification_latency_ms"),
        "requires_veda": result.get("requires_veda"),
        "requires_context": result.get("requires_context"),
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
