"""inference hybrid route — POST /v1/run_hybrid_query (migration_plan.md §8.2).

Calls ``veda_core.veda_hybrid.run_hybrid_query`` VERBATIM (the single front door:
decompose / route / fan-out / firewall) and returns the MultiResult with the
terminal ``status`` preserved exactly (§19 item 1). The heavy sync flow runs in a
thread pool via ``run_in_threadpool_with_context`` so the event loop stays
responsive and the tenant context is carried in (§4.1, §5.3).

Also exposes POST /v1/run_hybrid_query/stream: the SAME pipeline, but the
sync call's ``on_event`` hook is wired to an SSE stream so a caller sees real
stage-progress (classify / decompose / route / answer) AS the pipeline
advances, instead of blocking silently for the whole call. The sync pipeline
runs on its own daemon thread (not the thread pool — it must emit while still
running, which a pool call that awaits completion can't do); the ambient
(source, tenant) context is captured in the request coroutine and re-bound in
that thread via ``veda_core.context.with_context`` (§4.1).

# LINT: raw run_in_threadpool / ThreadPoolExecutor.submit is banned here —
# use inference.concurrency.run_in_threadpool_with_context (§4.1)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import threading
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    APIRouter = None
    Request = None
    StreamingResponse = None
    BaseModel = object


def _incoming_trace_id(request) -> "str | None":
    """Reuse the api-tier correlation id (apps/core/middleware.RequestIdMiddleware
    sets X-Request-Id, forwarded by apps/query/inference_client.py) as the ONE
    query trace_id — §1 "don't introduce a redundant identifier". None → the engine
    mints its own, so direct/CLI callers still get a trace_id."""
    try:
        return request.headers.get("x-request-id") or None
    except Exception:
        return None


# Internal-only keys that must never reach an HTTP caller. "context" is
# veda.execution_state.ExecutionState (Tier1→Tier2 propagation — explicitly
# internal-only, never an API field). "trace" is the full debug trace (already
# deliberately excluded from the chat/SSE path — see apps/chat/services.py's own
# "never sent over SSE" comment); stripped here too so EVERY caller of this route
# (not just the chat path) gets the same guarantee, not just the ones that happen
# to allowlist their own fields. "_debug" is the same idea for paths that have no
# ExplainTrace to record onto (e.g. Tier-2's _tier2_finish() noting an Insight
# Engine failure so a zero-token usage reading is distinguishable from "no LLM
# call was needed" — veda_hybrid.py).
_INTERNAL_ONLY_KEYS = frozenset({"context", "trace", "_debug"})



#: Keys whose VALUE can carry a raw engine error. The chat path shows safe copy for
#: these, but this route hands them to the caller verbatim — measured on the direct
#: endpoint, `refuse_reason` and `result.error` carried a full DuckDB failure:
#:
#:     Catalog Error: Table with name assets_amenitycategory does not exist!
#:     Did you mean "amenities_catalog"?
#:     ... COUNT(DISTINCT "id") AS "assets_amenitycategory_count" ... GROUP BY ...
#:     Did you mean "pg_settings"?
#:
#: That is raw table names, a SQL fragment with column aliases, and a hint naming
#: the storage engine — every category the standing rule says never crosses to a
#: caller. Sanitised HERE because _serialize is the ONE boundary every head result
#: passes through before the wire (see its docstring).
_ERROR_BEARING_KEYS = frozenset({"error", "refuse_reason"})

#: Fingerprints of an engine-internal error. Deliberately narrow: a message that
#: matches none of these is a business-level refusal already written for a user
#: ("I couldn't map 'maintenance' to any column…") and is passed through unchanged.
_INTERNAL_ERROR_MARKS = (
    "catalog error", "syntax error at", "binder error", "parser error",
    "conversion error", "psycopg2", "sqlstate",
    "relation ", "column \"", "select ", " from ", "group by", "pg_",
    # DuckDB's hint is `Did you mean "identifier"?` — the QUOTE is what makes it an
    # engine hint. A bare "did you mean" also appears in this platform's OWN
    # user-facing clarify copy ("more than one grouping fits what you asked for —
    # did you mean status or loe status?"), which is guidance the reader needs and
    # must survive. Matching on the quoted form keeps them apart.
    'did you mean "',
)

_SAFE_ERROR_TEXT = ("The query could not be completed against this data source. "
                    "Please rephrase, or contact your administrator.")


def _sanitise_error(value):
    """Replace an engine-internal error with safe copy. Never raises."""
    try:
        if not isinstance(value, str) or not value.strip():
            return value
        low = value.lower()
        if any(m in low for m in _INTERNAL_ERROR_MARKS):
            return _SAFE_ERROR_TEXT
    except Exception:
        pass
    return value

def _verbose() -> bool:
    """Container-log verbosity for the query pipeline, controlled by env.

    VEDA_INFERENCE_VERBOSE=1 (default) → run_hybrid_query(verbose=True): the full
    stage-by-stage detail (classify, routing, tier decisions, reuse logging) prints
    to stdout where `docker logs inference` captures it. Set 0 to quiet it down.
    Read per-request (not at import) so it can be flipped without a code change —
    just restart the container with the new env value."""
    import os
    return os.environ.get("VEDA_INFERENCE_VERBOSE", "1") not in ("0", "false", "False")


def _agent_memory(trace) -> "dict | None":
    """The one part of the (stripped) debug trace the NEXT chat turn needs: when the
    planner agent answered, its plan in draft form, its tool log and the question.

    chatbot/memory/frame.py::_harvest_agent records these on the IR stack so a follow-up
    edits the plan instead of re-planning (veda/understanding/frame_path.py::
    _agent_follow_up). They lived only in `trace.sections.agent`, which this module strips
    from every result — so over the wire the harvest never found them and agent
    follow-ups could not happen from chat at all. Lifted here as a named, bounded field;
    the rest of the trace stays internal."""
    try:
        if not isinstance(trace, dict) and hasattr(trace, "to_dict"):
            trace = trace.to_dict()
        sec = ((trace or {}).get("sections") or {}).get("agent") or {}
        if not isinstance(sec, dict) or sec.get("kind") != "sql" or not isinstance(sec.get("draft"), dict):
            return None
        calls = [{"tool": c.get("tool"), "args": c.get("args") or {}, "result": c.get("result") or {}}
                 for c in (sec.get("tool_calls") or []) if isinstance(c, dict) and c.get("tool")]
        return _serialize({"kind": "sql", "draft": sec["draft"], "tool_calls": calls[-12:],
                           "question": str(sec.get("question") or "")[:300]})
    except Exception:
        return None


def _serialize(obj: Any) -> Any:
    """Best-effort JSON-safe conversion that preserves the MultiResult shape.
    Strips _INTERNAL_ONLY_KEYS from any dict encountered, at any nesting depth —
    this is the ONE place every head result (SQL/Tier-2/RAG/hybrid/NoSQL) passes
    through before crossing the wire, so it's the correct place to enforce this."""
    if dataclasses.is_dataclass(obj):
        return _serialize(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        out = {}
        if "trace" in obj and "agent_memory" not in obj:
            _am = _agent_memory(obj.get("trace"))
            if _am:
                out["agent_memory"] = _am
        for k, v in obj.items():
            if k in _INTERNAL_ONLY_KEYS:
                continue
            if k in _ERROR_BEARING_KEYS:
                out[k] = _sanitise_error(v)
            elif k == "explain" and v == {}:
                # A failed head (`exec_error`) has nothing to explain, and shipped
                # `explain: {}` — a TRUTHY empty object in both Python and JS, so a
                # client's `if (explain)` opened an explainability panel with every
                # block missing. The key STAYS (removing it would change the shape a
                # client is being built against); its value becomes the falsy null
                # that already means "this turn produced no explainability".
                out[k] = None
            else:
                out[k] = _serialize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Decimal):
        # psycopg2 returns Decimal for NUMERIC/SUM/AVG columns (e.g. monetary
        # "amount" fields) — falling through to the generic str(obj) below
        # turned every such value into a STRING on the wire (e.g. "423.000"),
        # which silently broke every downstream numeric check that expects a
        # real number: apps/chat/visualization.py's _is_numeric()/_to_number()
        # (its own comment already assumes it receives a Decimal to convert,
        # not a pre-stringified one) — no chart was ever produced for a query
        # whose measure was a NUMERIC/DECIMAL column, only INTEGER aggregates
        # (e.g. COUNT(*), which survive as native JSON ints) worked. float()
        # matches what that downstream code already does with a real Decimal.
        return float(obj)
    return str(obj)



#: The engine is imported under BOTH names — bare ``context`` (cwd=veda_core) and
#: ``veda_core.context`` (inference tier, PYTHONPATH=/app) — and Python loads those
#: as TWO module objects. Verified in the running container: their
#: ``_source_profiles`` ContextVars are DIFFERENT objects, so a value set through
#: one name is invisible to a reader going through the other.
#:
#:     copy_context() snapshot, profiles set via veda_core.context:
#:         veda_core.context.current_source_profiles()  -> {'2': {...}}
#:         context.current_source_profiles()            -> {}          <-- lost
#:
#: `copy_context()` carries whatever the writer actually set, which is why it fixed
#: the measured "every source resolved to 'a data source'" bug. These two helpers
#: are the belt for the other half of the problem: they re-bind the captured scope
#: through EVERY name, so a reader on either one sees it. `veda_hybrid._current_ctx`
#: already does the equivalent for reads of the RequestContext; this covers the
#: writes, and source profiles, which have no such fallback.
_CONTEXT_MODULES = ("veda_core.context", "context")


def _capture_scope() -> dict:
    """The request scope, read through whichever module name holds it. Never raises."""
    import importlib
    out = {"ctx": None, "profiles": None}
    for name in _CONTEXT_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        try:
            if out["ctx"] is None:
                out["ctx"] = mod.try_current()
        except Exception:
            pass
        try:
            if not out["profiles"]:
                out["profiles"] = mod.current_source_profiles() or None
        except Exception:
            pass
    return out


def _rebind_scope(scope: dict) -> None:
    """Re-bind a captured scope through EVERY context module name.

    Idempotent — re-setting the value a copy_context() snapshot already carries
    changes nothing. Best-effort by design: a failure here must never cost the
    request, because the snapshot is the primary mechanism and this is the belt.
    """
    import importlib
    for name in _CONTEXT_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if scope.get("ctx") is not None:
            try:
                mod.set_context(scope["ctx"])
            except Exception:
                pass
        if scope.get("profiles"):
            try:
                mod.set_source_profiles(scope["profiles"])
            except Exception:
                pass

def _validated_conversation_context(flags) -> "dict | None":
    """The conversation context the api tier sent, coerced to a known shape.

    NOTHING here is trusted verbatim: this is a process boundary, and the engine must not
    be handed an unbounded or wrongly-typed structure because a client asked it to. Only
    the keys the engine actually consumes survive, each coerced and capped; anything else
    is dropped silently, and a malformed payload degrades to None (= no context), which is
    exactly how a first turn already behaves.

    Note what is NOT in the accepted set: the frame's `entity_display`. That display label
    is what contaminated the natural-language query in the first place, and it has no
    execution meaning — so it cannot cross this boundary even if a client sends it.
    """
    if not isinstance(flags, dict):
        return None
    raw = flags.get("conversation_context")
    if not isinstance(raw, dict):
        return None

    def _strs(key, cap=20):
        vals = raw.get(key)
        if not isinstance(vals, list):
            return []
        out = []
        for v in vals:
            if isinstance(v, (str, int, float)) and str(v).strip():
                sv = str(v).strip()[:200]
                if sv not in out:
                    out.append(sv)
            if len(out) >= cap:
                break
        return out

    def _int(key):
        try:
            v = raw.get(key)
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _filters(cap=20):
        """Structured remembered filters — each must know its own COLUMN, or it is not a
        filter this side can apply. Values stay strings; the engine grounds them the same
        way it grounds a literal from the query itself."""
        vals = raw.get("filters")
        if not isinstance(vals, list):
            return []
        out_f = []
        for f in vals:
            if not isinstance(f, dict):
                continue
            col, val = f.get("column"), f.get("value")
            if not isinstance(col, str) or not col.strip() or val is None:
                continue
            op = f.get("operator")
            out_f.append({"column": col.strip()[:200],
                          "operator": (op if isinstance(op, str) and op.strip() else "equals")[:32],
                          "value": str(val)[:200]})
            if len(out_f) >= cap:
                break
        return out_f

    out = {}
    _f = _filters()
    if _f:
        out["filters"] = _f
    tbl = raw.get("entity_table")
    if isinstance(tbl, str) and tbl.strip():
        out["entity_table"] = tbl.strip()[:200]
    for k in ("filter_values", "group_by", "measures", "order_by"):
        v = _strs(k)
        if v:
            out[k] = v
    for k in ("source_id", "limit", "drill_depth"):
        v = _int(k)
        if v is not None:
            out[k] = v
    agg = raw.get("aggregation")
    if isinstance(agg, str) and agg.strip():
        out["aggregation"] = agg.strip()[:32]
    route = raw.get("route")
    if isinstance(route, str) and route.strip():
        out["route"] = route.strip()[:64]
    um = raw.get("user_message")
    if isinstance(um, str):
        out["user_message"] = um

    # The planner agent's remembered turn (chatbot/memory/frame.py::_harvest_agent):
    # its plan in draft form, its tool log and the question it answered. Consumed by
    # veda/understanding/frame_path.py::_agent_follow_up -> veda/agent/planner.run_planner.
    # All three travel together or not at all — a log without its plan is noise.
    plan = _agent_plan(raw.get("agent_plan")) if "agent_plan" in raw else None
    if plan is not None:
        out["agent_plan"] = plan
        out["agent_log"] = _agent_log(raw.get("agent_log"))
        aq = raw.get("agent_question")
        if isinstance(aq, str) and aq.strip():
            out["agent_question"] = aq.strip()[:300]

    # THIS turn's delta on the previous query (chatbot/memory/context.py::wire_delta) —
    # applied structurally by the engine's continuity lane (veda/understanding/continuity.py).
    dl = _delta(raw.get("delta")) if "delta" in raw else None
    if dl is not None:
        out["delta"] = dl

    cmp_ = _comparison(raw.get("comparison")) if "comparison" in raw else None
    if cmp_:
        out["comparison"] = cmp_
    ep = raw.get("entry_path")
    if isinstance(ep, str) and ep.strip():
        out["entry_path"] = ep.strip()[:32]
    elif ep is not None:
        _drop("entry_path", ep)
    lat = raw.get("classification_latency_ms")
    if isinstance(lat, (int, float)) and not isinstance(lat, bool) and 0 <= lat <= 3_600_000:
        out["classification_latency_ms"] = float(lat)
    elif lat is not None:
        _drop("classification_latency_ms", lat)
    for k, lo, hi in (("target_frame_index", -_MAX_STACK_REF, _MAX_STACK_REF),
                      ("part_index", 0, _MAX_STACK_REF)):
        v = raw.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi:
            out[k] = v
        elif v is not None:
            _drop(k, v)
    return out or None


# ── shape checks for the structured (non-scalar) context keys ─────────────────
# Bounds, not business rules: the engine is never handed an unbounded structure, and
# a value of the wrong shape is DROPPED with a log line — never a 500, never passed
# through for a consumer to trip over halfway through a pipeline run.
_MAX_STACK_REF = 20
_MAX_PLAN_BYTES = 16_000
_MAX_LOG_ENTRIES = 12
_MAX_LOG_ENTRY_BYTES = 16_000
_MAX_LOG_BYTES = 64_000
_SCALAR = (str, int, float, bool, type(None))


def _drop(key, value, why: str = "wrong shape") -> None:
    logger.warning("conversation_context: dropped %s (%s): %.200r", key, why, value)


def _is_str_list(v, cap=20) -> bool:
    return (isinstance(v, list) and len(v) <= cap
            and all(isinstance(x, str) and 0 < len(x) <= 200 for x in v))


def _json_size(v) -> "int | None":
    try:
        return len(json.dumps(v, separators=(",", ":"), ensure_ascii=False))
    except (TypeError, ValueError):
        return None


_DELTA_OPS = ("add_filter", "remove_filter", "change_group", "change_measure", "change_order",
              "drill_up", "switch_frame", "replace", "ambiguous", "new_topic", "compare")


def _delta(v) -> "dict | None":
    """{op, slot?, concept?, value?, confidence?, applied?} — the op from the closed set,
    strings capped, value a scalar, confidence in [0, 1]. Anything else rejects the WHOLE
    delta (the engine then declines the continuity lane and answers as before)."""
    if not isinstance(v, dict):
        if v is not None:
            _drop("delta", v)
        return None
    unknown = set(v) - {"op", "slot", "concept", "value", "confidence", "applied"}
    if unknown:
        _drop("delta", v, f"unknown keys {sorted(unknown)}")
        return None
    if v.get("op") not in _DELTA_OPS:
        _drop("delta", v, "unknown op")
        return None
    out = {"op": v["op"]}
    for k in ("slot", "concept"):
        x = v.get(k)
        if x is None:
            continue
        if not isinstance(x, str) or len(x) > 200:
            _drop("delta", v, f"malformed {k}")
            return None
        if x.strip():
            out[k] = x.strip()
    val = v.get("value")
    if val is not None:
        if isinstance(val, bool) or not isinstance(val, (str, int, float)) \
                or (isinstance(val, str) and len(val) > 200):
            _drop("delta", v, "malformed value")
            return None
        out["value"] = val.strip() if isinstance(val, str) else val
    conf = v.get("confidence")
    if conf is not None:
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            _drop("delta", v, "malformed confidence")
            return None
        out["confidence"] = float(conf)
    ap = v.get("applied")
    if ap is not None and not isinstance(ap, bool):
        _drop("delta", v, "malformed applied")
        return None
    out["applied"] = bool(ap)
    return out


def _agent_plan(v) -> "dict | None":
    """veda/understanding/frame_path.py::_draft_of's shape, exactly. The planner puts the
    draft in its prompt and edits it; frame_path only requires a non-empty `tables`. Any
    unknown or malformed key rejects the WHOLE draft — a plan with a filter silently
    removed is a different question — and the engine plans afresh, which is exactly what
    it did before drafts crossed this boundary."""
    def scalar(x):
        return isinstance(x, _SCALAR) and not (isinstance(x, str) and len(x) > 500)

    def filt(f):
        val = f.get("value") if isinstance(f, dict) else None
        return (isinstance(f, dict) and set(f) <= {"col", "op", "value"}
                and isinstance(f.get("col"), str) and isinstance(f.get("op"), str)
                and len(f["op"]) <= 32
                and (scalar(val) or (isinstance(val, list) and len(val) <= 50
                                     and all(scalar(x) for x in val))))
    checks = {
        "tables": lambda x: _is_str_list(x) and len(x) > 0,
        "joins": _is_str_list,
        "select": lambda x: _is_str_list(x, 50),
        "filters": lambda x: isinstance(x, list) and len(x) <= 20 and all(filt(f) for f in x),
        "group_by": _is_str_list,
        "aggregates": lambda x: isinstance(x, list) and len(x) <= 20 and all(
            isinstance(a, dict) and set(a) <= {"fn", "col"}
            and isinstance(a.get("fn"), str) and isinstance(a.get("col"), str) for a in x),
        "order": lambda x: isinstance(x, list) and len(x) <= 10 and all(
            isinstance(o, dict) and set(o) <= {"by", "dir"}
            and isinstance(o.get("by"), str) and o.get("dir") in ("asc", "desc") for o in x),
        "limit": lambda x: isinstance(x, int) and not isinstance(x, bool) and 0 < x <= 100_000,
        "distinct": lambda x: isinstance(x, bool),
        "time": lambda x: (isinstance(x, dict) and set(x) <= {"col", "from", "to"}
                           and isinstance(x.get("col"), str)
                           and all(isinstance(x.get(k), (str, type(None))) for k in ("from", "to"))),
    }
    if not isinstance(v, dict):
        if v is not None:
            _drop("agent_plan", v)
        return None
    unknown = set(v) - set(checks)
    if unknown:
        _drop("agent_plan", v, f"unknown keys {sorted(unknown)}")
        return None
    if "tables" not in v:
        _drop("agent_plan", v, "no tables")
        return None
    bad = [k for k, ok in checks.items() if k in v and not ok(v[k])]
    if bad:
        _drop("agent_plan", v, f"malformed {bad}")
        return None
    size = _json_size(v)
    if size is None or size > _MAX_PLAN_BYTES:
        _drop("agent_plan", v, f"size {size}")
        return None
    return json.loads(json.dumps(v))


def _agent_log(v) -> list:
    """{tool, args, result} per call, as the planner's ToolBox log records it. `result`
    is replayed as an observation; a join-path result's `routes` are also re-read
    structurally (veda/agent/plan.py::route_edges needs each route's `id` and `path`), so
    a malformed route drops its entry rather than raising inside the planner."""
    if v is None:
        return []
    if not isinstance(v, list):
        _drop("agent_log", v)
        return []
    out, total = [], 0
    for e in v[-_MAX_LOG_ENTRIES:]:
        ok = (isinstance(e, dict) and isinstance(e.get("tool"), str) and 0 < len(e["tool"]) <= 64
              and isinstance(e.get("args", {}), dict) and isinstance(e.get("result", {}), dict))
        routes = (e.get("result") or {}).get("routes") if ok else None
        if ok and routes is not None:
            ok = isinstance(routes, list) and all(
                isinstance(r, dict) and isinstance(r.get("id"), (str, int))
                and _is_str_list(r.get("path") or [], 10) for r in routes)
        entry = ({"tool": e["tool"], "args": e.get("args") or {}, "result": e.get("result") or {}}
                 if ok else None)
        size = _json_size(entry) if entry is not None else None
        if entry is None or size is None or size > _MAX_LOG_ENTRY_BYTES or total + size > _MAX_LOG_BYTES:
            _drop("agent_log entry", e, "wrong shape or too large")
            continue
        total += size
        out.append(json.loads(json.dumps(entry)))
    return out


def _comparison(v) -> "dict | None":
    """chatbot/memory/frame.py::build_comparison, reduced to what has execution meaning:
    each side's table, label and source, and the dimension compared on. `entity_display`
    is excluded for the same reason as everywhere else in this payload."""
    if not isinstance(v, dict):
        if v is not None:
            _drop("comparison", v)
        return None

    def side(s):
        if not isinstance(s, dict):
            return None
        o = {}
        for k in ("entity", "label"):
            if isinstance(s.get(k), str) and s[k].strip():
                o[k] = s[k].strip()[:200]
        sid = s.get("source_id")
        try:
            if sid is not None:
                o["source_id"] = int(sid)
        except (TypeError, ValueError):
            pass
        return o or None

    a, b = side(v.get("primary")), side(v.get("comparison"))
    if not (a and b):
        _drop("comparison", v, "needs both sides")
        return None
    out = {"primary": a, "comparison": b}
    if isinstance(v.get("dimension"), str) and v["dimension"].strip():
        out["dimension"] = v["dimension"].strip()[:200]
    return out


if APIRouter is not None:
    router = APIRouter(prefix="/v1")

    class HybridRequest(BaseModel):
        query: str
        source_id: int | None = None
        tenant: str | None = None
        flags: dict | None = None

    @router.post("/run_hybrid_query")
    async def run_hybrid_query_route(req: "HybridRequest", request: Request):
        from inference.concurrency import run_in_threadpool_with_context
        from veda_core.veda_hybrid import run_hybrid_query

        _tid = _incoming_trace_id(request)
        result = await run_in_threadpool_with_context(
            run_hybrid_query, req.query, verbose=_verbose(), trace_id=_tid,
            conversation_context=_validated_conversation_context(req.flags))
        payload = _serialize(result)
        # Surface a top-level status for callers that don't walk items (§19 item 1).
        items = payload.get("items") if isinstance(payload, dict) else None
        top_status = (
            items[0].get("status") if items and isinstance(items[0], dict) else "unknown"
        )
        # trace_id surfaced top-level so a client can grep the full trace by it.
        _trace_id = payload.get("trace_id") if isinstance(payload, dict) else None
        return {"status": top_status, "trace_id": _trace_id, "result": payload}

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    @router.post("/run_hybrid_query/stream")
    async def run_hybrid_query_stream_route(req: "HybridRequest", request: Request):
        import asyncio
        from contextvars import copy_context

        from veda_core.veda_hybrid import run_hybrid_query

        loop = asyncio.get_event_loop()
        events: "asyncio.Queue[tuple[str, dict] | None]" = asyncio.Queue()
        # Snapshot the WHOLE context, not just the RequestContext.
        #
        # This used to be `with_context(try_current(), _run)`, which re-binds only
        # the (source, tenant) RequestContext. `_source_profiles` is a SEPARATE
        # ContextVar, so it was silently dropped in the worker thread — measured on
        # the real chat path: every source resolved to the generic "a data source"
        # label (`known: false`) even though the api tier had sent correct names,
        # and the routing coordinator's canonical tie-break saw an empty profile map.
        #
        # copy_context() carries EVERY ContextVar, which is what
        # inference.concurrency.run_in_threadpool_with_context already does for the
        # non-streaming route — so the two paths now behave identically, and adding
        # a third request-scoped ContextVar cannot reintroduce this class of bug.
        parent_ctx = copy_context()
        # Captured separately from the snapshot: see _capture_scope for why the
        # snapshot alone can leave one module view of the scope empty.
        _scope = _capture_scope()
        _tid = _incoming_trace_id(request)
        # Validated on the request thread, before the worker starts — a malformed payload
        # must fail here, not halfway through a pipeline run.
        _conv_ctx = _validated_conversation_context(req.flags)
        # the chat→engine contract, observable per request: WHICH keys arrived (never the
        # filter values) — "context never sent" and "context ignored" look identical otherwise
        # print, like the engine's own stage lines: nothing configures app loggers in this
        # process, so logger.info() would be dropped at the root's WARNING level
        print(f"[inference] conversation_context trace_id={_tid} "
              f"keys={sorted((_conv_ctx or {}).keys()) or None} "
              f"entity_table={(_conv_ctx or {}).get('entity_table')} "
              f"source_id={(_conv_ctx or {}).get('source_id')} "
              f"agent_plan={bool((_conv_ctx or {}).get('agent_plan'))} "
              f"delta={((_conv_ctx or {}).get('delta') or {}).get('op')}", flush=True)

        def on_event(phase: str, message: str, extra: dict):
            loop.call_soon_threadsafe(
                events.put_nowait, ("progress", {"phase": phase, "message": message, **extra})
            )

        def _run():
            _rebind_scope(_scope)          # belt; the snapshot is the primary path
            try:
                result = run_hybrid_query(
                    req.query, verbose=_verbose(), on_event=on_event, trace_id=_tid,
                    conversation_context=_conv_ctx)
                payload = _serialize(result)
                items = payload.get("items") if isinstance(payload, dict) else None
                top_status = (
                    items[0].get("status") if items and isinstance(items[0], dict) else "unknown"
                )
                _trace_id = payload.get("trace_id") if isinstance(payload, dict) else None
                loop.call_soon_threadsafe(
                    events.put_nowait, ("result", {"status": top_status,
                                                   "trace_id": _trace_id, "result": payload})
                )
            except Exception as exc:  # never leave the stream hanging on a crash
                loop.call_soon_threadsafe(
                    events.put_nowait, ("error", {"message": f"{type(exc).__name__}: {exc}"})
                )
            finally:
                loop.call_soon_threadsafe(events.put_nowait, None)

        # MERGE RESOLUTION (2026-09-10) — both sides were fixing the SAME bug and
        # both diagnoses were right; this keeps the working mechanism from one and
        # the concern from the other.
        #
        # The bug: this used to be `with_context(try_current(), _run)`, which
        # re-binds ONLY the (source, tenant) RequestContext. `source_profiles` is a
        # SEPARATE ContextVar and was silently dropped in the worker. Measured
        # consequences on the real chat path: every source resolved to the generic
        # "a data source" label (`known: false`) though the api tier had sent the
        # names; the routing coordinator's canonical tie-break saw an empty profile
        # map; and `_is_datalake_source()` returned False for a datalake source, so
        # the datalake-isolated semantic model was never loaded and a vendor
        # question got planned against the primary source's 178-table schema.
        #
        # `copy_context()` carries EVERY ContextVar, which is what
        # inference/concurrency.py::run_in_threadpool_with_context already does for
        # the non-streaming route — so both routes now behave identically and a
        # ContextVar added later propagates without touching this line again.
        #
        # The other side ALSO wrapped `with_context(...)` inside the snapshot, for
        # the dual-import case (bare `context` vs `veda_core.context` hold separate
        # vars — veda_hybrid.py:71-81). That concern is real and verified, but the
        # wrapper is not the way to address it here: `with_context` is not imported
        # in this module, and `parent_ctx` is a contextvars.Context rather than the
        # RequestContext that `set_context` expects. `_rebind_scope()` at the top of
        # `_run` covers the same ground correctly, for profiles as well as the
        # RequestContext, and is idempotent when the snapshot already carried them.
        threading.Thread(target=lambda: parent_ctx.run(_run), daemon=True).start()

        async def gen():
            while True:
                item = await events.get()
                if item is None:
                    return
                yield _sse(*item)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})
else:  # pragma: no cover
    router = None