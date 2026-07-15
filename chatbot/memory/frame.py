"""chatbot.memory.frame — QueryFrame: structured analytical memory (§3/§6/§7/§9
of docs/MEMORY_ARCHITECTURE.md).

Everything in this module is a PURE function — no I/O, no LLM calls, no Redis,
no veda_core import (chatbot/ runs in the api container, which must never
import veda_core directly — same boundary chatbot/llm.py and
apps/query/inference_client.py already document). That boundary is exactly
why validation here is deliberately light: real schema/column/value
validation already happens server-side, in the inference tier's own
deterministic pipeline (veda_core/veda/pipeline.py's L6a-L6c checks), on
EVERY call regardless of whether memory supplied the query text. Memory's job
is to hand that pipeline a BETTER-GROUNDED input than raw ellipsis text —
never to bypass or duplicate its validation.

Every field that ends up in a stored QueryFrame comes from ONE of exactly two
places:
  1. harvest_frame() — extracted from an engine_result that already has
     status == "answered", i.e. it already passed L6a-L6c server-side.
  2. The user's own current message, copied verbatim (render_frame_as_query
     never invents a filter value that isn't either an old proven fact or a
     substring the user just typed).
Nothing here is ever an LLM's free invention.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, TypedDict


class FilterFact(TypedDict, total=False):
    field: str          # business-facing field name (from business_explain.py)
    operator: str
    value: Any
    source: str          # "executed_sql" — the only source memory ever writes


class DrillLevel(TypedDict, total=False):
    dimension: str
    value: Any


class QueryFrame(TypedDict, total=False):
    version: int
    tenant: str
    session_id: str
    entity: Optional[str]           # raw table name (engine_result["table"])
    entity_display: Optional[str]   # humanized dataset name (explain.data_used.datasets[0])
    understanding: Optional[str]    # engine's own deterministic summary sentence
                                     # (business_explain.build_explain, NOT LLM prose)
    filters: List[FilterFact]
    group_by: List[str]
    drill_path: List[DrillLevel]
    last_sql: Optional[str]
    last_row_count: Optional[int]
    last_status: str
    confidence: float
    updated_at: str
    turn_index: int


_MAX_DRILL_DEPTH = 10


def empty_frame(tenant: str, session_id: str) -> QueryFrame:
    return {
        "version": 0,
        "tenant": tenant,
        "session_id": session_id,
        "entity": None,
        "entity_display": None,
        "understanding": None,
        "filters": [],
        "group_by": [],
        "drill_path": [],
        "last_sql": None,
        "last_row_count": None,
        "last_status": "",
        "confidence": 0.0,
        "updated_at": _now(),
        "turn_index": 0,
    }


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def harvest_frame(engine_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pure extraction, NO LLM: turn an "answered" engine_result (the full
    dict chatbot/nodes.py::call_engine_node stores at state["engine_result"],
    i.e. veda_core/veda/pipeline.py's _done() payload forwarded verbatim over
    the wire) into the harvestable facts for a QueryFrame. Returns None when
    there's nothing safe to harvest (wrong status, or business_explain itself
    failed server-side — pipeline.py already logs+degrades that case, we just
    skip the write rather than harvest partial/absent data).
    """
    if not engine_result or engine_result.get("status") != "answered":
        return None
    explain = engine_result.get("explain")
    if not explain:
        return None

    data_used = explain.get("data_used") or {}
    datasets = data_used.get("datasets") or []
    filters_applied = (explain.get("filters") or {}).get("applied") or []
    operations = explain.get("operations") or []
    understanding = (explain.get("understanding") or {}).get("summary")

    group_by = [op["summary"][len("Group by "):] for op in operations
                if op.get("type") == "group" and op.get("summary", "").startswith("Group by ")]

    rows = engine_result.get("rows")
    row_count = len(rows) if isinstance(rows, list) else None

    return {
        "entity": engine_result.get("table"),
        "entity_display": datasets[0] if datasets else None,
        "understanding": understanding,
        "filters": [
            {"field": f.get("field"), "operator": f.get("operator"), "value": f.get("value"),
             "source": "executed_sql"}
            for f in filters_applied if f.get("field")
        ],
        "group_by": group_by,
        "last_sql": engine_result.get("sql"),
        "last_row_count": row_count,
    }


def is_topic_switch(frame: Optional[QueryFrame], harvested: Optional[Dict[str, Any]]) -> bool:
    """Deterministic reset detector (§10) — reuses the engine's OWN independent
    table-routing decision (already computed for every query by the 5-signal
    retrieval engine in veda_core/retrieval/, with zero knowledge of the
    current frame) instead of asking an LLM "is this a new topic". No new
    embedding/vector lookup — this is a free byproduct of a call that already
    happened."""
    if not frame or not frame.get("entity") or not harvested or not harvested.get("entity"):
        return False
    return harvested["entity"] != frame["entity"]


def merge_frame_post_execution(
    prev: Optional[QueryFrame], harvested: Dict[str, Any], delta_type: str,
    tenant: str, session_id: str,
) -> QueryFrame:
    """POST-execution write (§6, memory_write_node). `harvested` always wins
    over `prev` for every field it supplies — it is strictly newer, executed,
    validated evidence; `prev` only fills in what harvested doesn't carry
    (e.g. drill_path, which harvest_frame can't derive on its own)."""
    reset = delta_type == "new_topic" or is_topic_switch(prev, harvested) or not prev
    base = empty_frame(tenant, session_id) if reset else dict(prev)  # type: ignore[arg-type]

    merged: QueryFrame = {**base, **harvested}  # type: ignore[typeddict-item]
    merged["tenant"] = tenant
    merged["session_id"] = session_id
    merged["version"] = (base.get("version") or 0) + 1
    merged["turn_index"] = (base.get("turn_index") or 0) + 1
    merged["last_status"] = "answered"
    merged["confidence"] = 1.0
    merged["updated_at"] = _now()
    merged["drill_path"] = [] if reset else list(base.get("drill_path") or [])
    return merged


def push_drill(stack: List[DrillLevel], harvested: Dict[str, Any]) -> List[DrillLevel]:
    """Append the most granular NEW filter as a drill level. Heuristic: the
    LAST filter in the freshly-executed query's filter list (deterministic,
    not LLM-chosen) — good enough for the linear drill-down shape in the
    prompt's own example (Region -> NA -> California -> Los Angeles)."""
    filters = harvested.get("filters") or []
    if not filters:
        return stack
    last = filters[-1]
    if not last.get("field"):
        return stack
    level: DrillLevel = {"dimension": last["field"], "value": last.get("value")}
    return (stack + [level])[-_MAX_DRILL_DEPTH:]


def pop_drill(stack: List[DrillLevel]) -> List[DrillLevel]:
    """"Go back" — pop one level. No-op (never errors) at the root."""
    return stack[:-1] if stack else stack


def _describe_frame(frame: QueryFrame) -> str:
    """Deterministic text reconstruction from ALREADY-PROVEN facts only —
    never an LLM paraphrase. Filters are copied verbatim from the last
    EXECUTED query's own explain output (business_explain.py, zero LLM).

    Deliberately a compact "entity + filters" noun phrase, NOT the full
    `frame["understanding"]` sentence — that sentence also carries the PRIOR
    turn's own operation verbs ("Find the top 100 ... by ...", sort/limit
    language), and concatenating a full second sentence ahead of the new
    message was observed (live testing) to confuse the engine's join planner
    into treating the repeated entity mention as a second, unnecessary join
    target. A short noun-phrase prefix carries the same grounding (which
    table, which filters already apply) without re-asserting operations the
    new message isn't asking to repeat.

    NOTE (audit C2 fix): this used to take its own `drop_last_filter` flag
    for the drill_up case, applied ON TOP OF the caller (context_resolve_node)
    already popping the drill stack and calling rebuild_frame_from_stack()
    first — a double decrement that silently dropped TWO context levels for
    a single "go back". `frame.filters` is now trusted as-is: whoever calls
    this (render_frame_as_query) is responsible for handing in an
    ALREADY-correct frame (rebuilt first, on drill_up), never for this
    function to second-guess it.
    """
    # Both the business-facing display name AND the raw table name — display
    # name alone (e.g. "Payment Transactions") was observed (2026-07 live
    # testing) to be ambiguous enough that a bare "go back" re-resolved to a
    # DIFFERENT, similarly-named table (reminders_reminderpaymenttransaction
    # instead of accounts_paymenttransaction) — the raw table name is the
    # exact, unambiguous identifier retrieval already indexes on; the display
    # name stays too since it's what makes the phrase read naturally.
    raw_entity, display = frame.get("entity"), frame.get("entity_display")
    if display and raw_entity and display.lower() != raw_entity.lower():
        entity = f"{display} ({raw_entity})"
    else:
        entity = display or raw_entity or ""
    filters = list(frame.get("filters") or [])
    filter_phrases = [f"{f['field']} {f.get('operator', 'equals')} {f['value']}"
                      for f in filters if f.get("field") and f.get("value") is not None]
    parts: List[str] = ([entity] if entity else []) + filter_phrases
    return ", ".join(p for p in parts if p)


def render_frame_as_query(frame: Optional[QueryFrame], message: str, delta_type: str) -> str:
    """PRE-execution: build the resolved_query text handed to the engine
    (chatbot/nodes.py::context_resolve_node), combining the frame's own
    previously-proven facts with the user's new message VERBATIM. The engine
    still independently re-validates everything from scratch (L6a-L6c) —
    this only gives it a better-grounded input, never a shortcut around that
    validation.

    For "drill_up", the caller MUST pass an already-rebuilt frame (see
    chatbot/nodes.py::context_resolve_node — pop_drill() +
    rebuild_frame_from_stack() BEFORE calling this) — this function no
    longer drops a filter of its own (audit C2 fix: that was a double-pop).

    "drill_up" is the one delta_type whose `message` ("go back", "go back
    again", ...) carries NO data content of its own — it's a pure navigation
    trigger, not a fragment to combine with the frame. Gluing it onto `ctx`
    (e.g. "go back (for accounts_paymenttransaction, ...)")  sent the literal
    words "back"/"again" to the engine as if they were part of the question,
    which then tried (and failed) to match them against columns/values. The
    resolved query for drill_up is just the popped frame's own restatement —
    the user's exact words never carried the intent, the stack pop already
    did.
    """
    if not frame or not frame.get("entity") or delta_type in ("new_topic", "ambiguous"):
        return message
    ctx = _describe_frame(frame)
    if not ctx:
        return message
    if delta_type == "drill_up":
        return ctx
    return f"{message} (for {ctx})".strip()


def rebuild_frame_from_stack(frame: QueryFrame, stack: List[DrillLevel]) -> QueryFrame:
    """After a drill_up pop, re-derive filters from the (now shorter) stack so
    frame.filters stays consistent with drill_path for the NEXT turn's
    render/prompt — the actual authoritative filters still get overwritten by
    harvest_frame() once the engine re-executes and returns fresh evidence;
    this only keeps the pre-call view honest in the interim."""
    filters: List[FilterFact] = [
        {"field": lvl["dimension"], "operator": "equals", "value": lvl.get("value"),
         "source": "executed_sql"}
        for lvl in stack
    ]
    return {**frame, "filters": filters, "drill_path": stack}
