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
import json
from typing import Any, Dict, List, Optional, TypedDict


class FilterFact(TypedDict, total=False):
    field: str          # business-facing field name (from business_explain.py)
    operator: str
    value: Any
    source: str          # "executed_sql" — the only source memory ever writes


class DrillLevel(TypedDict, total=False):
    dimension: str
    value: Any


class FrameEntry(TypedDict, total=False):
    """ONE answered turn, as structure rather than as a sentence (M4).

    The stack of these IS the conversation's analytical memory: a follow-up applies a
    DELTA to one of these entries' `ir` to produce the next question, instead of the
    pre-M4 path of restating the previous question in English and re-deriving its intent
    from that text on the server.

    `ir` is the engine's own QueryIR (veda/ir.py) as it crossed the wire on
    `engine_result["ir"]` — never re-derived here. When it is absent or `ir_partial`,
    this entry cannot be reliably edited slot-wise and the caller falls back to
    render_frame_as_query (see stack_is_structured).
    """
    ir: Optional[Dict[str, Any]]
    scope: List[int]
    result: Dict[str, Any]          # {row_count, shape, columns, top_values{dim:[...]}}
    drill_options: Dict[str, Any]   # {dimensions[], measures[], filters_applicable[]}
    turn_index: int
    question: str                   # the user's own words, for switch_frame matching
    sql: Optional[str]              # evidence only — never re-executed from here


class QueryFrame(TypedDict, total=False):
    version: int
    tenant: str
    session_id: str
    entity: Optional[str]           # raw table name (engine_result["table"])
    source_id: Optional[int]        # the source this turn's answer came from (P4, 2026-09-18);
                                    # a drill-down runs on THIS source, not the whole scope
    source_ids: List[int]           # the FULL scope that answered this turn. A cross-source
                                    # answer has no single source_id, and carrying only the
                                    # scalar meant such a turn inherited NOTHING and the next
                                    # turn re-routed from scratch — the one case where losing
                                    # the thread is most expensive (2026-09-21).
    primary_source_id: Optional[int]  # the anchor within a multi-source scope
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
    # M4 IR stack. Kept ALONGSIDE the flat fields above rather than replacing them: the
    # flat frame + render_frame_as_query remain the documented fallback for turns whose
    # IR is partial, so both representations have to coexist regardless. `cursor` is the
    # entry a follow-up applies to — normally the newest, moved by drill_up/switch_frame.
    stack: List[FrameEntry]
    cursor: int


_MAX_DRILL_DEPTH = 10
_MAX_STACK = 10


def empty_frame(tenant: str, session_id: str) -> QueryFrame:
    return {
        "version": 0,
        "tenant": tenant,
        "session_id": session_id,
        "entity": None,
        "source_id": None,
        "source_ids": [],
        "primary_source_id": None,
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
        "stack": [],
        "cursor": -1,
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
        # A DOCUMENT (RAG) answer has no `explain` — there is no SQL to explain. The
        # guard below was written for the SQL path, where a missing explain means
        # business_explain failed server-side and the data is untrustworthy; applied to a
        # document answer it meant document sources had NO session memory whatsoever, so
        # every follow-up ("what about the notice period") was re-routed as a new topic.
        # A document turn still has the two facts memory needs — which source answered and
        # which document it came from — so harvest those and nothing else.
        return _harvest_document_frame(engine_result)

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
        # the source this turn's answer came from (P4 / 2026-09-18): a drill-down keeps
        # working on THIS source instead of the whole scope re-deciding. None when the
        # engine result carries no source (a federated or independent multi-source answer)
        # — which is exactly why `source_ids` is harvested alongside it (2026-09-21): a
        # federated answer still has a definite SCOPE even with no single source, and the
        # next turn should stay inside it.
        "source_id": engine_result.get("source_id"),
        "source_ids": _harvest_scope(engine_result),
        "primary_source_id": (engine_result.get("primary_source_id")
                              or engine_result.get("source_id")),
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


def _cited_document(engine_result: Dict[str, Any]) -> Optional[str]:
    """The document a RAG answer drew on, from its first citation. Citations are
    "doc_name (p.N)" strings (query/rag_layer.py), so the page suffix is stripped."""
    for c in (engine_result.get("citations") or []):
        name = str(c or "").split(" (p.")[0].strip()
        if name:
            return name
    return None


def _harvest_document_frame(engine_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Harvestable facts for a DOCUMENT answer. Deliberately minimal — a document turn
    has no table, no filters and no grouping, so the only durable facts are the source
    and the document. That is enough for the two things memory must do on this path:
    keep the session on the same source, and let render_frame_as_query ground the next
    fragment against the document that just answered."""
    doc = _cited_document(engine_result)
    return {
        "entity": doc,                       # the document IS the entity here
        "source_id": engine_result.get("source_id"),
        "source_ids": _harvest_scope(engine_result),
        "primary_source_id": engine_result.get("source_id"),
        "entity_display": doc,
        "understanding": None,
        "filters": [],
        "group_by": [],
        "last_sql": None,
        "last_row_count": None,
    }


def _harvest_scope(engine_result: Dict[str, Any]) -> List[int]:
    """The set of sources that actually answered this turn, as ints.

    Reads `source_ids` when the engine reports a multi-source answer, otherwise the
    single `source_id`. Non-numeric / absent entries are dropped rather than guessed —
    an unparseable scope must degrade to "no inherited scope" (the next turn re-routes),
    never to a wrong one.
    """
    raw = engine_result.get("source_ids")
    if not isinstance(raw, (list, tuple)):
        raw = [engine_result.get("source_id")]
    out: List[int] = []
    for s in raw:
        try:
            v = int(s)
        except (TypeError, ValueError):
            continue
        if v not in out:
            out.append(v)
    return out


# ── M4: the IR stack ──────────────────────────────────────────────────────────
def harvest_entry(engine_result: Dict[str, Any], question: str,
                  turn_index: int) -> Optional[FrameEntry]:
    """One answered engine_result → a FrameEntry. Pure extraction, no LLM.

    Reads only fields the engine already computed and shipped: `ir` (veda/ir.py, built by
    the head that answered and checked by the firewall) and `analytics`
    (veda/result_analyzer.analytics_summary — the single post-execution analysis pass).
    Nothing here re-derives a dimension, a measure or a value from rows.
    """
    if not engine_result or engine_result.get("status") != "answered":
        return None
    an = engine_result.get("analytics") or {}
    stats = an.get("column_stats") or []
    if not an and not engine_result.get("explain"):
        # Document answer: no analytics pass ran (there are no rows to analyse) and no
        # IR (no SQL was compiled). The entry carries the scope and the document so the
        # NEXT turn inherits both; `ir` stays None, so entry_is_structured() is False and
        # the caller correctly uses the text-restatement path rather than slot-editing.
        return {
            "ir": None,
            "scope": _harvest_scope(engine_result),
            "result": {"row_count": None, "shape": "document",
                       "columns": [], "top_values": {}},
            "drill_options": {"dimensions": [], "measures": [], "filters_applicable": []},
            "turn_index": turn_index,
            "question": str(question or "")[:300],
            "document": _cited_document(engine_result),
            "sql": None,
        }

    # top_values per DIMENSION column — what a value-narrowing follow-up is matched
    # against by the deterministic delta layer.
    top_values: Dict[str, List[str]] = {}
    for st in stats:
        if st.get("role") in ("dimension", "boolean", "text") and st.get("top_values"):
            top_values[st["name"]] = [str(v) for v in st["top_values"][:12]]

    rows = engine_result.get("rows")
    cols = list(engine_result.get("cols") or [])
    return {
        "ir": engine_result.get("ir"),
        "scope": _harvest_scope(engine_result),
        "result": {
            "row_count": (len(rows) if isinstance(rows, list) else an.get("row_count")),
            "shape": an.get("result_shape"),
            "columns": cols[:20],
            "top_values": top_values,
        },
        "drill_options": {
            "dimensions": list(an.get("available_dimensions") or [])[:12],
            "measures": list(an.get("available_measures") or [])[:12],
            # a filter is "applicable" when we know real values for it — i.e. exactly the
            # dimensions we captured top_values for. Offering a dimension we have no
            # values for produces a follow-up suggestion nothing can ground.
            "filters_applicable": sorted(top_values),
        },
        "turn_index": turn_index,
        "question": str(question or "")[:300],
        "sql": engine_result.get("sql"),
    }


def _compact_entry(entry: FrameEntry) -> FrameEntry:
    """Drop the per-result detail, keep the structure. Applied to entries that fall past
    _MAX_STACK: a 30-turn session must not grow without bound, but an old entry's IR and
    scope are still what `switch_frame` ("go back to the vendors question") needs to
    return to it. Deterministic — never an LLM summarisation."""
    return {
        "ir": entry.get("ir"),
        "scope": list(entry.get("scope") or []),
        "result": {"row_count": (entry.get("result") or {}).get("row_count")},
        "drill_options": {},
        "turn_index": entry.get("turn_index"),
        "question": entry.get("question"),
        "sql": None,
        "compacted": True,
    }


def push_entry(stack: List[FrameEntry], entry: FrameEntry) -> List[FrameEntry]:
    """Append one answered turn. Entries beyond _MAX_STACK are compacted in place rather
    than dropped, so the oldest topic in a long session is still reachable."""
    out = list(stack or []) + [entry]
    if len(out) > _MAX_STACK:
        head, tail = out[:-_MAX_STACK], out[-_MAX_STACK:]
        out = [_compact_entry(e) for e in head] + tail
    return out


def stack_top(frame: Optional[QueryFrame]) -> Optional[FrameEntry]:
    """The entry a follow-up applies to: `cursor` when it points somewhere valid, else
    the newest. None for an empty stack."""
    st = list((frame or {}).get("stack") or [])
    if not st:
        return None
    cur = (frame or {}).get("cursor", -1)
    if isinstance(cur, int) and -len(st) <= cur < len(st):
        return st[cur]
    return st[-1]


def entry_is_structured(entry: Optional[FrameEntry]) -> bool:
    """True when this entry's IR can be edited slot-wise.

    An absent IR, or one the engine marked `ir_partial`, means the slots' provenance is
    unknown (see veda/ir.from_sql_facts) — applying a delta to it would produce a
    confidently-wrong question. Those frames fall back to render_frame_as_query, and the
    COUNT of that fallback is the measure of how much of the stack is genuinely
    structured (logged by the caller)."""
    ir = (entry or {}).get("ir")
    return bool(ir) and not ir.get("ir_partial")


def describe_ir(ir: Optional[Dict[str, Any]]) -> str:
    """One deterministic line naming what an IR asks — the user-facing context strip
    ("Vendors in Kochi · grouped by category"). Built from slots, never from prose."""
    if not ir:
        return ""
    parts: List[str] = []
    m = ir.get("measure") or {}
    agg = (m.get("aggregation") or "").lower()
    anchor = ir.get("anchor") or ""
    if agg and agg not in ("none",):
        col = m.get("column")
        parts.append(f"{agg} of {col}" if col else f"{agg} of {anchor}".strip())
    elif anchor:
        parts.append(str(anchor))
    fs = [f for f in (ir.get("filters") or []) if f.get("column")]
    if fs:
        parts.append(" and ".join(
            f"{f['column']} {f.get('op', '=')} {f.get('value')}" for f in fs[:3]))
    if ir.get("group_keys"):
        parts.append("grouped by " + ", ".join(str(g) for g in ir["group_keys"][:3]))
    order = ir.get("order") or {}
    if ir.get("limit"):
        parts.append(f"top {ir['limit']}" + (f" by {order['column']}" if order.get("column") else ""))
    return " · ".join(p for p in parts if p)


def compact_stack(frame: Optional[QueryFrame], max_entries: int = 10) -> List[Dict[str, Any]]:
    """The stack as the SLM classifier sees it — ≤ ~250 tokens per entry.

    Only what a classifier needs to pick a TARGET and an OP: what each turn asked, which
    slots it has, and which dimensions/measures are available to move to. Never rows,
    never SQL, never the prose answer."""
    out = []
    for i, e in enumerate(list((frame or {}).get("stack") or [])[-max_entries:]):
        ir = e.get("ir") or {}
        opts = e.get("drill_options") or {}
        out.append({
            "index": i,
            "asked": e.get("question"),
            "about": describe_ir(ir) or (ir.get("anchor") or ""),
            "rows": (e.get("result") or {}).get("row_count"),
            "can_group_by": list(opts.get("dimensions") or [])[:6],
            "can_measure": list(opts.get("measures") or [])[:4],
        })
    return out


def apply_delta(ir: Optional[Dict[str, Any]], delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The previous turn's IR + one delta → the NEXT turn's IR. Pure dict surgery.

    This is the core of M4: a follow-up edits a SLOT of a structured question, instead of
    the server re-deriving the whole intent from a restated English sentence. Only the
    named slot changes; every other slot — anchor, scope, the filters already applied —
    carries forward untouched, which is exactly the property the sentence-restatement
    path could not guarantee.

    Returns None when there is nothing to edit; the caller then falls back.
    """
    if not ir:
        return None
    nxt = json.loads(json.dumps(ir, default=str))     # deep copy, JSON-safe throughout
    op = delta.get("op")
    concept, value = delta.get("concept"), delta.get("value")

    if op == "add_filter" and concept:
        fs = [f for f in (nxt.get("filters") or []) if f.get("column") != concept]
        fs.append({"table": nxt.get("anchor"), "column": concept, "op": "=",
                   "value": value, "grounding": "session_top_values", "concept": concept})
        nxt["filters"] = fs
    elif op == "change_group" and concept:
        # REPLACES the grouping rather than appending: "by city instead" means instead.
        nxt["group_keys"] = [concept]
    elif op == "change_measure":
        agg = value or "count"
        nxt["measure"] = {"aggregation": agg, "column": concept,
                          "table": nxt.get("anchor"), "distinct": False}
    elif op == "change_order":
        if isinstance(value, int):
            nxt["limit"] = value
        if concept:
            nxt["order"] = {"column": concept,
                            "direction": (nxt.get("order") or {}).get("direction") or "desc"}
    else:
        return None
    nxt["head"] = "session_delta"
    return nxt


_AGG_PHRASE = {"count": "how many", "sum": "total", "avg": "average",
               "min": "minimum", "max": "maximum"}


def ir_to_question(ir: Optional[Dict[str, Any]]) -> str:
    """A modified IR → an English question the engine can answer.

    The engine has no compile-from-IR entry point yet (M3 checkpoint 2), so the IR is
    restated as text and the engine re-grounds it exactly as it would any question. That
    is deliberate: this changes WHERE the next question's structure comes from — the
    previous turn's validated slots rather than a paraphrase of its English — without
    bypassing a single one of the engine's own grounding or firewall checks.

    Built purely from slots; it never copies the user's sentence, so it cannot carry a
    pronoun the engine would have to resolve.
    """
    if not ir:
        return ""
    anchor = str(ir.get("anchor") or "").replace("_", " ").strip()
    m = ir.get("measure") or {}
    agg = str(m.get("aggregation") or "").lower()
    col = m.get("column")

    if agg in ("count", "") or not agg:
        head = f"how many {anchor}" if anchor else "how many"
    else:
        phrase = _AGG_PHRASE.get(agg, agg)
        head = (f"{phrase} {str(col).replace('_', ' ')}" if col else f"{phrase}")
        if anchor:
            head += f" of {anchor}"

    parts = [head]
    fs = [f for f in (ir.get("filters") or []) if f.get("column") and f.get("value") is not None]
    if fs:
        parts.append("where " + " and ".join(
            f"{str(f['column']).replace('_', ' ')} is {f['value']}" for f in fs[:4]))
    if ir.get("group_keys"):
        parts.append("per " + ", ".join(str(g).replace("_", " ") for g in ir["group_keys"][:2]))
    order = ir.get("order") or {}
    if ir.get("limit"):
        q = f"top {ir['limit']}"
        if order.get("column"):
            q += f" by {str(order['column']).replace('_', ' ')}"
        parts.append(q)
    return " ".join(parts).strip()


def follow_up_questions(entry: Optional[FrameEntry], limit: int = 3) -> List[str]:
    """Deterministic, LLM-free next-question suggestions from THIS result's own
    drill_options — the IR-derived replacement for the SLM-generated follow-ups. Every
    suggestion names a dimension or measure the result actually has, so each one is
    answerable by construction."""
    if not entry:
        return []
    opts = entry.get("drill_options") or {}
    ir = entry.get("ir") or {}
    grouped = {str(g) for g in (ir.get("group_keys") or [])}
    filtered = {str(f.get("column")) for f in (ir.get("filters") or [])}
    out: List[str] = []
    for d in (opts.get("dimensions") or []):
        # skip a dimension already grouped by, and one already pinned to a single value
        # by a filter — grouping by a column that can only take one value yields one row
        # and answers nothing.
        if str(d) not in grouped and str(d) not in filtered:
            out.append(f"Break this down by {str(d).replace('_', ' ')}")
        if len(out) >= limit:
            return out[:limit]
    for msr in (opts.get("measures") or []):
        out.append(f"Show the total {str(msr).replace('_', ' ')}")
        if len(out) >= limit:
            return out[:limit]
    for c in (opts.get("filters_applicable") or []):
        if str(c) not in filtered:
            vals = (entry.get("result") or {}).get("top_values", {}).get(c) or []
            if vals:
                out.append(f"Only the {vals[0]} ones")
        if len(out) >= limit:
            break
    return out[:limit]


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


def newly_added_filter(prev: Optional[Dict[str, Any]],
                       harvested: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The filter THIS turn added relative to `prev`, or None if it added none.

    Deterministic drill detection, and the reason the DrillStack was always empty. The LLM
    delta classifier labels the ordinary value-narrowing follow-ups — "only the open ones",
    "just the Repair ones" — as `refine`, not `drill_down` (measured on this deployment), and
    memory_write_node only pushed a level for `drill_down`. So no breadcrumb was ever recorded
    and "go back" had nothing to pop, even though the user had plainly drilled in.

    Evidence beats the label here, which is the same principle harvest_frame already applies:
    if the query we actually EXECUTED carries a filter the previous one did not, this turn
    narrowed — whatever the classifier chose to call it. Compares (field, operator, value) so
    re-running the same filter is not mistaken for a new level.
    """
    def _key(f):
        return (f.get("field"), f.get("operator"), str(f.get("value")))
    prev_keys = {_key(f) for f in ((prev or {}).get("filters") or [])}
    for f in (harvested.get("filters") or []):
        if f.get("field") and _key(f) not in prev_keys:
            return f
    return None


def push_drill_level(stack: List[DrillLevel], filt: Dict[str, Any]) -> List[DrillLevel]:
    """Push ONE explicitly-identified filter as a drill level (see newly_added_filter).

    push_drill() below guesses "the last filter" because a `drill_down` turn does not say which
    filter is new; when the caller already KNOWS which one was added, record that instead of
    guessing — with several filters in play the newest is not necessarily last.
    """
    if not filt or not filt.get("field"):
        return stack
    level: DrillLevel = {"dimension": filt["field"], "value": filt.get("value")}
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
