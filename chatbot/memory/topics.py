"""Topic index — the bounded list of earlier topics a session can return to.

WHY THIS EXISTS. The frame is per SOURCE, so within one source a new entity overwrites the
old one: "distribution of properties by facing" -> "only the Nagpur ones" -> "show payment
transactions by transaction type" left source 2's frame on payments, and the properties
topic — its Nagpur filter and its drill stack — was simply gone. "go back to the
properties" had nothing to return to (measured, VEDA_MEMORY_LAYER_PLAN.md M3).

WHAT IS KEPT. One entry per (source_id, entity), most recent first, at most _MAX_TOPICS.
Each entry is a SNAPSHOT of what an answered turn left behind for that topic — enough to
re-ask it, nothing more:

    source_id, entity, entity_display, base_query (the user's own root question),
    filters ({field, column, operator, value}), group_by, measures, aggregation,
    order_by, limit, drill_stack, version, updated_at

Never rows, never SQL text: a restored topic is always RE-EXECUTED under the current turn's
authorisation, exactly like a drill-up replay. The snapshot is context, never a cache.

Invariants (the same ones the frame keeps):
  · written ONLY by memory_write_node, on an answered turn, and cleared by every reset path
    (chatbot/memory/store.py::MemoryStore.reset — whole session, or one source);
  · read by memory_read_node and filtered to the CURRENT turn's authorised sources before
    anything else sees it, so a topic from a revoked source is invisible and unrestorable;
  · a document conversation IS indexed, as a smaller entry (see _document_snapshot): its
    "entity" is a document name, it has no filters or drill stack, and it is returned to by
    NAMING the document (chatbot/nodes.py::_match_named_document), not by replaying a SQL
    shape. Before 2026-09-25 it was not indexed at all, so "what is the fee for repair in
    the maintenance policy?" asked after a database question had no memory of the policy
    conversation and was answered from a structured fee table (demo X1).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from . import frame as memory_frame

# Small on purpose: a person returns to one of the last few things they looked at, and every
# entry is a candidate the return-to-topic matcher must disambiguate against.
_MAX_TOPICS = 5
_MAX_FILTERS = 20
_MAX_LIST = 20
_MAX_VALUE_LEN = 200


def _key(source_id: Any, entity: Any) -> tuple:
    return (None if source_id is None else str(source_id), str(entity or ""))


def topic_key(entry_or_frame: Optional[Dict[str, Any]]) -> tuple:
    """(source_id as str or None, entity) — the identity of a topic."""
    e = entry_or_frame or {}
    return _key(e.get("source_id"), e.get("entity"))


def _short(v: Any) -> Any:
    if isinstance(v, str):
        return v[:_MAX_VALUE_LEN]
    return v


def snapshot(frame: Optional[Dict[str, Any]],
             stack: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The index entry for the topic this frame is on, or None when it cannot be one.

    A document frame gets _document_snapshot (see module docstring). None for: no entity;
    a table topic with no base_query — without the user's own root question there is
    nothing to re-ask, and inventing one from the table name is exactly what the boundary
    forbids."""
    if not frame or not frame.get("entity"):
        return None
    if memory_frame.is_document_frame(frame):
        return _document_snapshot(frame)
    base_query = str(frame.get("base_query") or "").strip()
    if not base_query:
        return None
    # EVERY filter the frame holds, a value-less one included ("transaction_type IS NOT
    # NULL" is harvested with value None). The restored candidate must equal the frame as it
    # was: memory_write_node pushes a drill level for any filter the replay's SQL carries
    # that the candidate lacks, so dropping one here made an answered restore push a bogus
    # level — measured live 2026-09-25, ('Transaction Type', None). ConversationContext
    # still sends only filters with a column and a value; that rule lives there.
    filters = []
    for f in (frame.get("filters") or [])[:_MAX_FILTERS]:
        if not isinstance(f, dict):
            continue
        filters.append({"field": _short(f.get("field")), "column": _short(f.get("column")),
                        "operator": _short(f.get("operator") or "equals"),
                        "value": _short(f.get("value"))})
    return {
        "source_id": frame.get("source_id"),
        "entity": frame.get("entity"),
        "entity_display": frame.get("entity_display"),
        "base_query": base_query[:_MAX_VALUE_LEN * 2],
        "filters": filters,
        "group_by": [_short(g) for g in (frame.get("group_by") or [])[:_MAX_LIST]],
        "measures": [_short(m) for m in (frame.get("measures") or [])[:_MAX_LIST]],
        "aggregation": frame.get("aggregation"),
        "order_by": [o for o in (frame.get("order_by") or [])[:_MAX_LIST]
                     if isinstance(o, dict)],
        "limit": frame.get("limit"),
        "route": frame.get("route") or "",
        "entity_is_document": False,
        "drill_stack": [dict(lvl) for lvl in (stack or [])[:_MAX_LIST]
                        if isinstance(lvl, dict)],
        "version": frame.get("version"),
        "updated_at": frame.get("updated_at"),
    }


def _document_snapshot(frame: Dict[str, Any]) -> Dict[str, Any]:
    """A document topic: which document, from which source, by which head — and the user's
    own first question about it when the frame recorded one. Nothing else exists to keep:
    a document answer has no filters, grouping or drill stack."""
    return {
        "source_id": frame.get("source_id"),
        "entity": frame.get("entity"),
        "entity_display": frame.get("entity_display"),
        "base_query": str(frame.get("base_query") or "").strip()[:_MAX_VALUE_LEN * 2],
        "filters": [], "group_by": [], "measures": [], "aggregation": None,
        "order_by": [], "limit": None,
        "route": frame.get("route") or "",
        "entity_is_document": True,
        "drill_stack": [],
        "version": frame.get("version"),
        "updated_at": frame.get("updated_at"),
    }


def upsert(index: Optional[List[Dict[str, Any]]],
           entry: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`entry` at the front, replacing any older entry for the same topic, bounded.

    A topic the conversation moved away from keeps the snapshot it had when it was last
    answered — that is the whole point: it is what "go back to it" restores."""
    kept = [e for e in (index or []) if isinstance(e, dict) and e.get("entity")]
    if not entry:
        return kept[:_MAX_TOPICS]
    k = topic_key(entry)
    return ([entry] + [e for e in kept if topic_key(e) != k])[:_MAX_TOPICS]


def without_source(index: Optional[List[Dict[str, Any]]],
                   source_id: Any) -> List[Dict[str, Any]]:
    """Every entry except those of `source_id` — what a per-source reset leaves behind."""
    return [e for e in (index or []) if isinstance(e, dict)
            and str(e.get("source_id")) != str(source_id)]


def frame_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A candidate QueryFrame rebuilt from a snapshot. Only the fields the snapshot holds;
    everything else is filled from the engine's fresh evidence once the replay answers."""
    return {
        "entity": entry.get("entity"),
        "entity_display": entry.get("entity_display"),
        "source_id": entry.get("source_id"),
        "base_query": entry.get("base_query"),
        "filters": [dict(f) for f in (entry.get("filters") or [])],
        "group_by": list(entry.get("group_by") or []),
        "measures": list(entry.get("measures") or []),
        "aggregation": entry.get("aggregation"),
        "order_by": [dict(o) for o in (entry.get("order_by") or [])],
        "limit": entry.get("limit"),
        "route": entry.get("route") or "",
        "entity_is_document": bool(entry.get("entity_is_document")),
        "datasets": [entry.get("entity")] if entry.get("entity_is_document") else [],
        "drill_path": [dict(lvl) for lvl in (entry.get("drill_stack") or [])],
    }


def display_name(entry: Dict[str, Any]) -> str:
    """How a topic is named back to the user: the business name plus the question that
    started it, which is what tells two topics on similar tables apart."""
    name = entry.get("entity_display") or entry.get("entity") or "an earlier topic"
    asked = str(entry.get("base_query") or "").strip()
    filters = [str(f.get("value")) for f in (entry.get("filters") or [])
               if f.get("value") is not None]
    out = f"{name} — \"{asked}\"" if asked else str(name)
    if filters:
        out += f" (narrowed to {', '.join(filters)})"
    return out


# ── across sessions (step 6) ────────────────────────────────────────────────────────────
# Every other memory key is per SESSION, so a new chat starts with nothing: "what did we
# look at yesterday?" or "continue the property analysis" had no answer. What crosses
# sessions is deliberately the same thing a topic already is — a snapshot of an answered
# question (entity, root question, filters, shape), never rows and never an answer. A
# topic restored from an earlier session is re-executed exactly like one from this
# session, under this turn's authorisation.
_MAX_SESSIONS = 10          # sessions remembered per user
_MAX_PREVIOUS = 10          # earlier-session topics offered to one turn


def session_summary(session_id: str, topics: List[Dict[str, Any]],
                    updated_at: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """This session's compact summary: its topic snapshots (already bounded) and when it
    last answered. None when there is nothing worth carrying forward."""
    kept = [dict(t) for t in (topics or []) if isinstance(t, dict) and t.get("entity")]
    if not session_id or not kept:
        return None
    return {"session_id": str(session_id), "updated_at": int(updated_at or time.time()),
            "topics": kept[:_MAX_TOPICS]}


def merge_sessions(sessions: List[Dict[str, Any]],
                   summary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The user's session list with `summary` replacing its own session's entry, moved to
    the front, bounded to _MAX_SESSIONS."""
    rest = [s for s in (sessions or []) if isinstance(s, dict)
            and s.get("session_id") != (summary or {}).get("session_id")]
    return ([summary] if summary else []) + rest[:_MAX_SESSIONS - (1 if summary else 0)]


def without_session(sessions: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
    """A "start over" in a session forgets it here too — the user asked to forget it."""
    return [s for s in (sessions or []) if isinstance(s, dict)
            and s.get("session_id") != str(session_id)]


def previous_topics(sessions: List[Dict[str, Any]],
                    current_session_id: str) -> List[Dict[str, Any]]:
    """Topics from the user's OTHER sessions, most recent first, one per (source, entity),
    each tagged with the session it came from and when. Authorisation is NOT decided here —
    the caller filters every entry by the current turn's scope, exactly as it does the
    in-session index."""
    out, seen = [], set()
    for s in sessions or []:
        if not isinstance(s, dict) or s.get("session_id") == str(current_session_id):
            continue
        for t in s.get("topics") or []:
            if not isinstance(t, dict) or not t.get("entity"):
                continue
            key = topic_key(t)
            if key in seen:
                continue
            seen.add(key)
            out.append({**t, "from_session": s.get("session_id"),
                        "session_updated_at": s.get("updated_at")})
            if len(out) >= _MAX_PREVIOUS:
                return out
    return out
