# =============================================================================
# veda/safe_projection.py
# VEDA — the SAFE PROJECTION layer (traceability Phase 1, Parts 4/5/9/10/13/16).
#
#       Pipeline → internal ExplainTrace → [THIS MODULE] → SSE + explainability
#
# The internal trace stays internal. This is the ONLY place that turns it into
# something a user may see, which is what makes the safety rule auditable: if a
# field is not constructed here, it cannot reach the frontend.
#
# WHAT IS DELIBERATELY NEVER READ FROM THE TRACE
#   retrieval / rrf / reranking / graph_expansion  — candidate lists + scores
#   entity_selection / anchor_selection            — internal ranking
#   slm / llm_usage                                — model names, prompts, purposes
#   sql_planning.action, join_planning.join_path   — internal planner grammar
#   rbac_filter.before/after                       — discloses that N things were
#                                                    hidden AND how many
#   any *.candidates / *.rejected / column_names   — verbose debug payloads
#
# Every projection function is total: a missing/odd-shaped section yields an
# empty or omitted block, never an exception, because an observability failure
# must never fail a query.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

from veda import lifecycle as lc

# ── routing reason_code → user-facing copy (Part 13) ─────────────────────────
# The codes are query/routing_contracts.py's stable enum. Anything not listed
# falls back to the generic sentence — a NEW internal code must never leak as
# raw text just because this map wasn't updated.
_ROUTING_COPY = {
    "SINGLE_CANDIDATE":
        "One data source contains the information this question needs.",
    "CANONICAL_SELECTED":
        "More than one data source could answer this; the authoritative one was used.",
    "RELATIONSHIP_EDGE":
        "The answer needed data from more than one source, joined on a known relationship.",
    "SLM_RESOLVED":
        "Multiple data sources were relevant; the best match for this question was used.",
    "AMBIGUOUS_SOURCE_SELECTION":
        "More than one data source could answer this question.",
    "NO_EVIDENCE":
        "No available data source matched this question.",
    "INVALID_SLM_DECISION":
        "The data source could not be determined reliably.",
}
_ROUTING_GENERIC = "The data source was selected because it contains the information required."

_MODE_COPY = {"SINGLE": "single", "MULTI": "multi", "NONE": "none"}

# Execution-plan strategy → the user-facing mode word (Part 5).
_STRATEGY_MODE = {"single": "single_source", "federated": "federated",
                  "independent": "multi_source"}


def _sec(trace, name: str) -> Dict[str, Any]:
    """One trace section as a dict, or {} — never raises."""
    try:
        sections = getattr(trace, "sections", None)
        if sections is None and isinstance(trace, dict):
            sections = trace.get("sections")
        return (sections or {}).get(name) or {}
    except Exception:
        return {}


# ── Part 1/2: the timeline ───────────────────────────────────────────────────
def build_timeline(trace) -> List[Dict[str, Any]]:
    """The recorded lifecycle events, already safe by construction.

    Read back from the trace rather than from the live Timeline object so the
    final payload and the stream can never disagree — both derive from the one
    recorded list (the single-source-of-truth rule).
    """
    events = _sec(trace, lc.TRACE_SECTION).get("events") or []
    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        out.append({
            "phase": e.get("phase"),
            "status": e.get("status"),
            "title": e.get("title"),
            "message": e.get("message"),
            "elapsed_ms": e.get("elapsed_ms"),
        })
    return out


def build_timeline_summary(trace) -> List[Dict[str, Any]]:
    """The Level-1 "How VEDA answered this" checklist (Part 17): one row per phase
    that actually ran, collapsed to its worst status."""
    events = _sec(trace, lc.TRACE_SECTION).get("events") or []
    order: List[str] = []
    worst: Dict[str, str] = {}
    rank = {lc.STATUS_STARTED: 0, lc.STATUS_COMPLETED: 1,
            lc.STATUS_WARNING: 2, lc.STATUS_FAILED: 3}
    for e in events:
        if not isinstance(e, dict):
            continue
        ph, st = e.get("phase"), e.get("status")
        if ph not in lc.PHASES:
            continue
        if ph not in worst:
            order.append(ph)
            worst[ph] = st
        elif rank.get(st, 0) > rank.get(worst[ph], 0):
            worst[ph] = st
    return [{"phase": p, "title": lc.PHASE_TITLES.get(p, ""), "status": worst[p]}
            for p in order]


# ── Part 13: routing ─────────────────────────────────────────────────────────
def build_routing(trace) -> Optional[Dict[str, Any]]:
    """Safe routing summary. Reads ONLY status/mode/reason_code/source_ids —
    never candidate_sources, evidence_summary, scores or the SLM tie-break."""
    r = _sec(trace, "routing")
    if not r:
        return None
    mode_raw = str(r.get("mode") or "")
    # SHADOW decisions describe what the routing policy WOULD have chosen; they do
    # not drive execution (MULTISOURCE_ROUTING_SHADOW). Reporting one as fact told a
    # user "The answer needed data from more than one source, joined on a known
    # relationship" for an answer whose own `data_used` listed exactly ONE dataset
    # and where no join happened. An observe-only measurement is not an explanation
    # of this answer, so no routing block is produced for it.
    if r.get("shadow"):
        return None
    reason_code = str(r.get("reason_code") or "")
    sids = list(r.get("source_ids") or [])

    mode = _MODE_COPY.get(mode_raw, "none")
    if mode == "multi" and _sec(trace, "execution_plan").get("strategy") == "federated":
        mode = "federated"

    summary = _ROUTING_COPY.get(reason_code, _ROUTING_GENERIC)
    if mode in ("multi", "federated") and reason_code not in _ROUTING_COPY:
        summary = "Multiple data sources were required to answer the question."

    out: Dict[str, Any] = {"mode": mode, "summary": summary, "source_count": len(sids)}
    # `reason_code` is a stable, non-sensitive enum — useful for a client that
    # wants its own copy. The human sentence is always present regardless.
    if reason_code:
        out["reason_code"] = reason_code
    return out


# ── Part 5: execution plan ───────────────────────────────────────────────────
def build_execution_plan(trace) -> Optional[Dict[str, Any]]:
    """User-facing projection of the ExecutionPlan.

    HONESTY (Part 25, Defect 2): `execution` reports what the runtime ACTUALLY
    did. The planner's own `mode` field says PARALLEL, but source_coordinator
    executes steps in a sequential loop, so the recorder stamps `executed_mode`
    and that is what is projected. We never tell a user (or an admin) that work
    ran in parallel when it did not.
    """
    p = _sec(trace, "execution_plan")
    if not p:
        return None
    strategy = str(p.get("strategy") or "")
    steps = p.get("steps") or []
    from veda import source_names as sn

    projected = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        sid = st.get("source_id")
        projected.append({"name": sn.display_name(sid), "type": sn.display_type(sid),
                          "required": bool(st.get("required", True))})
    n = len(projected)
    if n > 1:
        summary = f"Used {n} data sources to answer this question"
    elif n == 1:
        summary = f"Used {projected[0]['name']} to answer this question"
    else:
        summary = "No data source was required"
    return {"summary": summary,
            "mode": _STRATEGY_MODE.get(strategy, "single_source"),
            "executed_mode": p.get("executed_mode") or "sequential",
            "steps": projected}


# ── Part 7/8: per-source execution ───────────────────────────────────────────
def build_execution(trace) -> Optional[Dict[str, Any]]:
    """Per-source execution summary, from the already-safe record projections."""
    from veda import exec_records as er
    sec = _sec(trace, er.TRACE_SECTION)
    records = sec.get("records") or []
    if not records:
        return None

    safe: List[Dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        rec = er.SourceExecutionRecord(**{k: v for k, v in raw.items()
                                          if k in er.SourceExecutionRecord.__dataclass_fields__})
        safe.append(rec.as_safe_dict())

    done = sum(1 for s in safe if s.get("status") == er.COMPLETED)
    total = len(safe)
    if total and done == total:
        status, summary = "complete", (
            f"Retrieved data from {total} source{'s' if total > 1 else ''}.")
    elif done:
        status, summary = "partial", (
            f"Retrieved data from {done} of {total} sources; the rest did not respond.")
    else:
        status, summary = "failed", "No data source returned results."
    return {"summary": summary, "status": status, "sources": safe}


# ── Part 6: which sources were used ──────────────────────────────────────────
def build_data_sources(trace) -> List[Dict[str, Any]]:
    """The sources that ACTUALLY participated, named safely.

    Prefers the execution records (proof of participation) and falls back to the
    routing decision's selected ids. Never lists a candidate or rejected source.
    """
    from veda import exec_records as er
    from veda import source_names as sn

    records = _sec(trace, er.TRACE_SECTION).get("records") or []
    ids = [r.get("source_id") for r in records if isinstance(r, dict) and r.get("source_id")]
    if not ids:
        # FEDERATED execution. The coordinator's independent strategy records each
        # source it runs, but the federated strategy returns before that loop, so a
        # cross-source answer had NO records — measured live: "There are 7 invoices
        # compared to 96 assets", an answer that demonstrably combined two sources,
        # shipped `sources: null`.
        #
        # The federation section is proof of participation in its own right: the
        # federated SQL was validated against, and executed over, exactly these
        # catalogs. Read only when the federation actually ran (`used`).
        _f = _sec(trace, "federation")
        if _f.get("used"):
            ids = [str(x) for x in (_f.get("source_ids") or [])]
    if not ids:
        # Fall back to the routing decision ONLY when it actually chose the sources.
        # Under MULTISOURCE_ROUTING_SHADOW the decision is observe-only, so naming
        # its sources here claimed a source had contributed data when it had not —
        # measured: `sources: [invoices_csv, homzhub]` on an answer whose figures
        # came entirely from one CSV. With no records and a shadow decision we do
        # not know which source answered, and an empty list says that honestly.
        _r = _sec(trace, "routing")
        if not _r.get("shadow"):
            ids = list(_r.get("source_ids") or [])
    if not ids:
        # A DENIAL never got to look anywhere. The scoped-source fallback below is
        # "this is where we looked", and on a denial that is not true — we refused
        # before searching, so naming the source alongside "you don't have
        # permission to access this data" reads as "we used it". Nothing to name.
        if _access_denied(trace):
            return []
        ids = _scoped_single_source() or _executed_source(trace)
    out = sn.describe_all(ids)
    _attach_contribution(out, records, trace)
    return out


#: The engine is imported under BOTH names — bare ``context`` and
#: ``veda_core.context`` — and Python loads those as two module objects with two
#: separate ContextVars. Read through both or the value set by the other half of
#: the process is invisible (the same trap documented in inference/routes/hybrid.py).
_CONTEXT_MODULE_NAMES = ("veda_core.context", "context")


def _executed_source(trace) -> List[str]:
    """The source the execution actually connected to, when work actually ran.

    With SEVERAL sources in scope, `_scoped_single_source` correctly declines to
    guess — but a Tier-1/Tier-2 answer still executed against exactly ONE of them,
    and the request context holds which: `storage_adapters.reader` resolves the
    connection from that id and fail-closes when it is unset, so it is the source
    of truth for what was queried, not an assumption. Without this a perfectly
    ordinary answer with the default all-sources scope named nothing at all.

    GATED ON EXECUTION. The context id is set for the whole turn, including turns
    that refuse before touching anything — naming a source there would claim we
    queried data we never read. A recorded row count is the proof that something
    ran; with none, this says nothing. Deliberately AFTER the federation check, so
    a cross-source answer is never narrowed to its primary source.
    """
    if _sane_count(_sec(trace, "execution").get("row_count")) is None:
        return []
    import importlib
    for name in _CONTEXT_MODULE_NAMES:
        try:
            ctx = importlib.import_module(name).try_current()
        except Exception:
            continue
        sid = getattr(ctx, "source_id", None) if ctx is not None else None
        if sid:
            return [str(sid)]
    return []


def _access_denied(trace) -> bool:
    """Did the authorization check FAIL for this turn? Read from the timeline the
    engine actually wrote, not from any reply text."""
    try:
        return any(e.get("phase") == "access_check" and e.get("status") == "failed"
                   for e in build_timeline(trace))
    except Exception:
        return False


def _scoped_single_source() -> List[str]:
    """The one source the request was scoped to, when there is exactly one.

    LAST RESORT, and deliberately narrow. The `sources` block is built from proof
    of participation: an execution record, or a routing decision that actually
    drove execution. A plain single-source query produces NEITHER — only the
    cross-source coordinator writes execution records, and the routing decision is
    observe-only under shadow mode — so the commonest query in the system answered
    with no `sources` block at all, and the reader could not see where the answer
    came from (measured: a Tier-1 relational answer, `sources: null`).

    With exactly ONE source in scope there is no ambiguity: the answer came from
    it. With two or more and no records, we genuinely do not know which one
    answered, and an empty block says that honestly rather than naming a guess.
    """
    import importlib
    for name in _CONTEXT_MODULE_NAMES:
        try:
            profiles = importlib.import_module(name).current_source_profiles()
        except Exception:
            continue
        if isinstance(profiles, dict) and len(profiles) == 1:
            return [str(next(iter(profiles)))]
    return []


def _attach_contribution(entries, records, trace) -> None:
    """Say what each source CONTRIBUTED, so the block answers "where did this come
    from" rather than only "who took part".

    Only from a figure that was actually recorded: a per-source execution record's
    own row count, or — when a single source answered the whole query — the
    query's row count. Never divided, never estimated, and omitted entirely when
    no figure exists, because "0 rows" and "not reported" are different claims.
    """
    try:
        by_id = {}
        for r in (records or []):
            if isinstance(r, dict) and r.get("source_id") is not None:
                by_id[str(r["source_id"])] = _sane_count(r.get("row_count"))
        single = len(entries) == 1
        total = _sane_count(_sec(trace, "execution").get("row_count"))
        for e in entries:
            rows = by_id.get(str(e.get("id")))
            if rows is None and single:
                rows = total
            if isinstance(rows, int) and rows >= 0:
                e["rows"] = rows
    except Exception:
        pass


# ── Part 9/10: cross-source ──────────────────────────────────────────────────
def build_cross_source(trace) -> Optional[Dict[str, Any]]:
    """Federation facts. Returns None for a single-source query so the key is
    simply absent rather than a misleading `used: false` on every answer."""
    f = _sec(trace, "federation")
    if not f or not f.get("used"):
        return None
    from veda import source_names as sn

    out: Dict[str, Any] = {
        "used": True,
        "sources": [d["name"] for d in sn.describe_all(f.get("source_ids") or [])],
        "operation": f.get("operation") or "combined",
        "result_status": f.get("result_status") or "complete",
    }
    # Join facts (Part 10): counts + rate only. The raw join keys and the SQL
    # predicate stay internal — an admin/debug view may read them off the trace.
    join = f.get("join") or {}
    if join.get("join_used"):
        matched = join.get("matched_count")
        unmatched = join.get("unmatched_count")
        jd: Dict[str, Any] = {
            "used": True,
            "summary": ("Records from the data sources were matched using the "
                        "available relationship between the datasets."),
        }
        if isinstance(matched, int) and isinstance(unmatched, int) and (matched + unmatched):
            rate = round(100.0 * matched / (matched + unmatched), 1)
            jd["match_rate_pct"] = rate
            jd["summary"] += f" {rate}% of relevant records were successfully matched."
        out["join"] = jd
    return out


# ── Part 11: warnings + limitations ──────────────────────────────────────────
def build_warnings(trace) -> List[Dict[str, Any]]:
    """User-safe warnings, in discovery order. `user_safe=False` entries — which
    nothing currently produces — are filtered defensively."""
    from veda import warnings as vw
    out = []
    for w in (_sec(trace, vw.TRACE_SECTION).get("items") or []):
        if not isinstance(w, dict) or w.get("user_safe") is False:
            continue
        out.append({"code": w.get("code"), "severity": w.get("severity"),
                    "message": w.get("message")})
    return out


def build_limitations(trace) -> List[str]:
    """The subset of warnings that constrain how the ANSWER should be read, as
    plain sentences — for a client that wants a short "caveats" list separate
    from the coded warning array."""
    from veda import warnings as vw
    # LOW_EVIDENCE belongs here too: "there was limited matching data, so the answer
    # may be incomplete" constrains how the answer should be READ, which is exactly
    # what this list is for. It was missing, so a low-confidence answer produced a
    # warning but an empty `limitations` array (caught while verifying EXP-B3).
    # FALLBACK_USED is deliberately NOT limiting — an alternate path still produced a
    # complete answer; it is informational, not a caveat on the result.
    # NO_RESULTS is the strongest caveat there is — there is no answer to read —
    # so it belongs here above all the others.
    limiting = {vw.NO_RESULTS, vw.RESULT_TRUNCATED, vw.RESTRICTED_DATA,
                vw.PARTIAL_SOURCE_FAILURE, vw.SOURCE_CONFLICT,
                vw.UNMATCHED_RECORDS, vw.LOW_EVIDENCE}
    return [w["message"] for w in build_warnings(trace) if w.get("code") in limiting]


# ── result metadata ──────────────────────────────────────────────────────────

def _sane_count(value) -> Optional[int]:
    """A row/record count fit to show a user, or None for "unknown".

    A nonsensical value becomes UNKNOWN, not 0. In practice row_count is a len()
    so it cannot be negative — but this module is the boundary that makes trace
    content safe to display, and passing through whatever the trace happens to
    hold is not that. `0` is deliberately NOT the fallback: it would claim there
    were no rows, which is a different statement from "the count makes no sense".
    `bool` is excluded explicitly because in Python it is an int, and `True` would
    otherwise render as "1 row returned".
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value

def build_result_meta(trace) -> Dict[str, Any]:
    """Row count / truncation / partiality — all already recorded."""
    ex = _sec(trace, "execution")
    from veda import warnings as vw
    codes = {w.get("code") for w in build_warnings(trace)}
    return {
        "row_count": _sane_count(ex.get("row_count")),
        "truncated": bool(ex.get("truncated")),
        "partial": bool(codes & {vw.PARTIAL_SOURCE_FAILURE, vw.RESTRICTED_DATA}),
        "reused_verified_query": bool(ex.get("from_cache")),
    }


#: The one user-facing sentence about a replayed answer. Deliberately says WHAT
#: happened without naming the mechanism — no "cache", no SQL, no identifiers.
_REUSED_SUMMARY = ("This reused a query already verified for a very similar question, "
                   "rather than working it out again.")


def build_provenance(trace) -> Optional[Dict[str, Any]]:
    """How this answer was arrived at, when that is not the ordinary path.

    Returns None on a normal query — an always-present block saying "nothing
    special happened" is noise. Today the only non-ordinary provenance is a
    verified-query replay, which matters because the replayed query was verified
    under possibly-older code than the one now answering.
    """
    ex = _sec(trace, "execution")
    if not ex.get("from_cache"):
        return None
    return {"reused_verified_query": True, "summary": _REUSED_SUMMARY}



# ── Part §10: "How this answer was generated" ────────────────────────────────
#: The stages of the flow, in the order a reader follows them. A CLOSED list: a
#: flow can omit a stage it has no evidence for, but it cannot invent one.
FLOW_STAGES = ("request", "access", "sources", "evidence", "validation",
               "operations", "result", "answer")


def _flow_stage(kind: str, label: str, items=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"stage": kind, "label": label}
    if items:
        out["items"] = [str(i)[:120] for i in items if i]
    return out


def build_flow(trace, *, operations=None,
               validation=None) -> Optional[Dict[str, Any]]:
    """The end-to-end story of this answer, assembled from what actually happened.

    One shape for every execution type — the reader always follows
    request → access → sources → evidence → … → answer — while the STAGES PRESENT
    vary with what the turn really did. A document answer has passages and no query
    validation; a database answer has validation and operations and no passages; a
    federated answer has several sources and a reconciliation step. None of that is
    templated per route: each stage appears only when the trace holds evidence for
    it, so an execution type nobody has written a template for still renders.

    Returns None when there is nothing beyond the request to show, rather than a
    skeleton that implies work no one can point to.
    """
    from veda import exec_records as er

    stages: List[Dict[str, Any]] = [_flow_stage("request", "Your request")]

    # ACCESS — only when the timeline actually recorded the check.
    tl = {e.get("phase"): e.get("status") for e in build_timeline(trace)}
    access = tl.get("access_check")
    if access == "completed":
        stages.append(_flow_stage("access", "Access verified"))
    elif access in ("failed", "warning"):
        stages.append(_flow_stage("access", "Access could not be confirmed"))

    # SOURCES — by display name, never an id.
    names = [d.get("name") for d in build_data_sources(trace) if d.get("name")]
    if names:
        label = "Data source" if len(names) == 1 else f"{len(names)} data sources"
        stages.append(_flow_stage("sources", label, names))

    # EVIDENCE — counted, never scored.
    ex = _sec(trace, "execution")
    records = _sec(trace, er.TRACE_SECTION).get("records") or []
    rows = _sane_count(ex.get("row_count"))
    retrieved = sum(_sane_count(r.get("rows_returned")) or 0
                    for r in records if isinstance(r, dict))
    if rows is not None:
        stages.append(_flow_stage(
            "evidence", f"{rows} row{'' if rows == 1 else 's'} returned"))
    elif retrieved:
        stages.append(_flow_stage(
            "evidence", f"{retrieved} record{'' if retrieved == 1 else 's'} retrieved"))

    # VALIDATION — counted from the LIST THE USER IS SHOWN, passed in from the
    # payload, not from the trace's own check ledger.
    #
    # The two are not the same number: build_explain expands one trace check into
    # SEVERAL user-facing labels (`_CHECK_LABELS`), so a turn with 4 trace checks
    # shows 5 statements. Counting the trace here put "4 checks passed" in the flow
    # directly above a list of 5 — both numbers correct in their own terms, and
    # contradictory side by side. The number the reader can count must be the
    # number they are told.
    items = [c for c in ((validation or {}).get("checks") or []) if isinstance(c, dict)]
    passed = [c for c in items if c.get("passed")]
    if items:
        stages.append(_flow_stage(
            "validation",
            f"{len(passed)} of {len(items)} checks passed" if len(passed) != len(items)
            else f"{len(items)} check{'' if len(items) == 1 else 's'} passed"))

    # OPERATIONS — the semantic steps, in order, as the answer payload states them.
    # Passed in rather than read from the trace: operations are derived from the
    # executed SQL by build_explain and live only in the payload.
    ops = [o.get("summary") for o in (operations or [])
           if isinstance(o, dict) and o.get("summary")]
    if ops:
        stages.append(_flow_stage("operations", "Operations applied", ops[:8]))

    # CROSS-SOURCE — present only when a federation actually ran.
    cs = build_cross_source(trace)
    if cs:
        stages.append(_flow_stage("operations", "Combined across sources"))

    if len(stages) == 1:
        return None
    stages.append(_flow_stage("answer", "Answer"))
    return {"stages": stages}


def demote_source_ids(out: Dict[str, Any]) -> None:
    """Move raw source IDENTIFIERS out of the user-facing blocks into `audit`.

    §3/§9: a source id is an internal key, not something a normal client renders —
    the display NAME is what a reader needs. It is MOVED, not dropped, because the
    audit row records which sources took part and apps/query/audit.py reads the ids
    from exactly this block.

    Runs LAST, after every block that can carry a source has been assembled: an
    earlier pass missed the top-level `sources` list, which is built further down.

    PUBLIC because there are TWO assemblers, not one — the answered path
    (build_explain_extension) and the refusal path (_apply_v2_refusal). When the
    refusal path started emitting `sources` it shipped the raw ids, because this
    ran only inside the other one. Same "wired to one path" mistake as EXP-B1/B4/B5,
    the terminal step frame and the refusal timeline_summary; caught here by the
    invariant checker rather than in production.
    """
    try:
        ids: List[Dict[str, Any]] = []
        blocks = [(out.get("execution") or {}).get("sources") or [],
                  out.get("sources") or []]
        for block in blocks:
            for entry in block:
                if isinstance(entry, dict) and entry.get("id"):
                    if not any(x["id"] == entry["id"] for x in ids):
                        ids.append({"id": entry["id"], "name": entry.get("name")})
        for block in blocks:
            for entry in block:
                if isinstance(entry, dict):
                    entry.pop("id", None)
        if ids:
            out.setdefault("audit", {})["sources"] = ids
    except Exception:
        pass

# ── the whole v2 extension ───────────────────────────────────────────────────
def build_explain_extension(trace, *, trace_id: str = "",
                            operations=None, validation=None) -> Dict[str, Any]:
    """Every v2 block, assembled from the trace. Keys whose block does not apply
    to this query are OMITTED rather than emitted as null, except `warnings` /
    `limitations` (always a list, so a client can render unconditionally) and
    `result` (always present).

    Never raises: each block is independently guarded, so one malformed section
    cannot cost the caller the whole extension.
    """
    out: Dict[str, Any] = {}
    for key, fn in (
        ("routing", build_routing),
        ("execution_plan", build_execution_plan),
        ("execution", build_execution),
        ("provenance", build_provenance),
        ("cross_source", build_cross_source),
    ):
        try:
            v = fn(trace)
            if v:
                out[key] = v
        except Exception:
            pass
    try:
        _flow = build_flow(trace, operations=operations, validation=validation)
        if _flow:
            out["flow"] = _flow
    except Exception:
        pass
    # LEVEL 3 (§9): the technical record. Grouped under `audit` rather than left at
    # the top level because these are the only blocks that carry RAW BACKEND PHASE
    # NAMES (`source_selection`, `data_retrieval`, `result_preparation`), and the
    # spec is explicit that those must not appear in the primary experience. A
    # Level-1/2 client renders `steps` + `flow` + `warnings` and never opens this;
    # a client that wants the audit trail asks for it by name.
    audit: Dict[str, Any] = {}
    for key, fn in (("timeline", build_timeline),
                    ("timeline_summary", build_timeline_summary)):
        try:
            audit[key] = fn(trace)
        except Exception:
            audit[key] = []
    if any(audit.values()):
        out["audit"] = audit

    for key, fn, default in (
        ("warnings", build_warnings, []),
        ("limitations", build_limitations, []),
        ("result", build_result_meta, {}),
    ):
        try:
            out[key] = fn(trace)
        except Exception:
            out[key] = default
    try:
        sources = build_data_sources(trace)
        if sources:
            out["sources"] = sources
    except Exception:
        pass
    if trace_id:
        out["support"] = {"trace_id": trace_id}
    demote_source_ids(out)
    return out
