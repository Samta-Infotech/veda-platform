"""One decision record per turn — "why did VEDA interpret this follow-up this way?"

Every bug fixed in the conversation layer on 2026-09-24/25 was found by reconstructing,
by hand, four facts from logs, checkpoints and engine traces: what memory held when the
turn started, how the turn was classified (and which evidence overrode the model), what
was actually sent to the engine, and whether the result was committed. This puts those
four facts in ONE structured record, keyed by the request id that already joins the engine
trace and the QueryLog audit row.

WHAT IS DELIBERATELY NOT IN IT: filter VALUES, row identities, the message text, SQL, or
reply prose. Those can carry personal data; the record carries columns, ids, flags and
counts, which is what reconstructing a decision needs. Built only from the turn's own final
state (plus the `memory_in` summary memory_read_node recorded) — nothing is re-derived and
nothing here affects the turn.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def memory_summary(frame: Optional[Dict[str, Any]], stack: Optional[List[Any]],
                   reference: Optional[Dict[str, Any]], topics: Optional[List[Any]],
                   previous: Optional[List[Any]]) -> Dict[str, Any]:
    """What memory held at the START of the turn — recorded by memory_read_node, because
    by the end of the turn the frame may have been replaced by a newer one."""
    frame = frame or {}
    return {
        "entity": frame.get("entity") or None,
        "source_id": frame.get("source_id"),
        "version": frame.get("version"),
        "filter_columns": sorted({str(f.get("column") or f.get("field"))
                                  for f in (frame.get("filters") or []) if isinstance(f, dict)
                                  and (f.get("column") or f.get("field"))}),
        "group_by": list(frame.get("group_by") or []),
        "drill_depth": len(stack or []),
        "reference": ({"kind": reference.get("kind"),
                       "items": len(reference.get("items") or [])} if reference else None),
        "topics": len(topics or []),
        "earlier_session_topics": len(previous or []),
    }


def decision_record(state: Dict[str, Any], *, request_id: str = "") -> Dict[str, Any]:
    """The record for a finished turn, from its final graph state."""
    ctx = state.get("conversation_context") or {}
    er = state.get("engine_result") or {}
    frame = state.get("frame") or {}
    mem_in = state.get("memory_in") or {}
    restore = state.get("topic_restore") or {}
    status = state.get("status")
    committed = bool(status == "answered" and frame.get("version") is not None
                     and frame.get("version") != mem_in.get("version"))
    return {
        "request_id": request_id or state.get("request_id") or "",
        "session_id": state.get("session_id"),
        "memory_in": mem_in or None,
        "interpretation": {
            "action": state.get("action"),
            "delta_type": state.get("delta_type"),
            "delta_field": state.get("delta_field") or None,
            # Evidence the conversation layer used IN PLACE of, or on top of, the model's
            # label. Each is a fact the turn carried, not a guess about which rule fired.
            "evidence": {
                "names_only_values": state.get("message_names_only_values"),
                "mentions_data": state.get("message_mentions_data"),
                "topic_restore": restore.get("kind"),
                "restored_from_earlier_session": bool(
                    (restore.get("topic") or {}).get("from_session")),
                "result_pointer_terms": len(ctx.get("resolved_terms") or []),
            },
        },
        "context_sent": ({
            "carried": bool(ctx.get("entity_table")),
            "entity_table": ctx.get("entity_table"),
            "source_id": ctx.get("source_id"),
            "operation": ctx.get("operation"),
            "filter_columns": sorted({str(f.get("column")) for f in (ctx.get("filters") or [])
                                      if isinstance(f, dict) and f.get("column")}),
            "group_by": list(ctx.get("group_by") or []),
            "replayed_earlier_question": bool(ctx.get("user_message")
                                              and ctx.get("user_message") != state.get("message")),
        } if ctx else None),
        "outcome": {
            "status": status,
            "engine_route": er.get("_route") or er.get("route"),
            "table": er.get("table"),
            "rows": len(state.get("rows") or er.get("rows") or []),
            "engine_called": bool(er),
            "memory_committed": committed,
            "memory_version": frame.get("version"),
        },
    }
