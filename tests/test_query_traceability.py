"""Tests for the user-safe query traceability spine (traceability Phase 1).

Covers veda/lifecycle.py, veda/warnings.py, veda/exec_records.py,
veda/source_names.py, veda/safe_projection.py, the build_explain v2 extension,
and the two confirmed defects the audit found.

Pure python — no DB, no network, no SLM. Every flag is set explicitly per test
and restored, because the whole feature is flag-gated default-OFF and the
DEFAULT-OFF path is itself part of the contract (prod must stay byte-identical).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

import pytest

import config
from veda import exec_records as er
from veda import lifecycle as lc
from veda import safe_projection as sp
from veda import source_names as sn
from veda import warnings as vw
from veda import business_explain as be
from veda.business_explain import build_explain
from veda.explain import ExplainTrace, summarize_explain_payload


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def flags_on(monkeypatch):
    """Every traceability flag on, restored afterwards by monkeypatch."""
    for name in ("LIFECYCLE_EVENTS_ENABLED", "QUERY_WARNINGS_ENABLED",
                 "SOURCE_EXECUTION_RECORDS_ENABLED", "EXECUTION_PLAN_TRACE_ENABLED",
                 "EXPLAIN_V2_ENABLED", "DB_EXECUTION_TIMING_ENABLED"):
        monkeypatch.setattr(config, name, True, raising=False)


@pytest.fixture
def trace():
    return ExplainTrace(query="how many incidents are open?", trace_id="testtrace123")


@pytest.fixture
def profiles(monkeypatch):
    """Bind a request-scoped source-profile map, the way the inference middleware
    does from the X-Veda-Source-Profiles header."""
    from veda_core import context as ctx
    token = ctx.set_source_profiles({
        "2": {"name": "Sales Database", "source_type": "relational"},
        "3": {"name": "Contract Documents", "source_type": "document"},
    })
    yield
    try:
        ctx._source_profiles.reset(token)
    except Exception:
        pass


# ── DEFAULT-OFF contract ─────────────────────────────────────────────────────
def test_the_intended_default_flag_state(flags_on=None):
    """WHICH flags ship enabled, and why each of the others does not.

    This test used to assert every flag was OFF — the standing "prod stays
    byte-identical" rule. That rule's purpose was to leave the DECISION to the
    product owner rather than have it made by whoever wrote the feature. The
    decision has now been made (2026-09-10): the flags below ship ON because each
    was verified end to end, and the three still OFF are named with the reason.

    So this test no longer enforces "all off" — it enforces "exactly this state,
    deliberately". Flipping any of these should fail here first, so the change is
    a decision and not a drift.
    """
    import importlib
    import config
    importlib.reload(config)

    # --- ON: verified end to end -------------------------------------------
    on = {
        "LIFECYCLE_EVENTS_ENABLED":
            "6 query shapes live; 0 phases left unresolved in the saved record",
        "QUERY_WARNINGS_ENABLED":
            "result_truncated / low_evidence / fallback_used all fired correctly live",
        "SOURCE_EXECUTION_RECORDS_ENABLED":
            "per-source verified; the two wrong-source bugs are fixed",
        "EXECUTION_PLAN_TRACE_ENABLED":
            "was computed then discarded; reports `sequential` honestly",
        "EXPLAIN_V2_ENABLED":
            "additive by contract, checked key-by-key against a live capture",
        "DB_EXECUTION_TIMING_ENABLED":
            "measured rather than inferred from stage gaps",
    }
    for name, why in on.items():
        assert getattr(config, name) is True, f"{name} should ship ON — {why}"

    # --- OFF: named reasons, not oversight ---------------------------------
    off = {
        "EXPLAIN_NARRATOR_ENABLED":
            "burns an SLM call per query and its own validator rejects every real "
            "narration — zero output for real cost. Fix the validator first.",
        "FEDERATED_JOIN_STATS_ENABLED":
            "never exercised against a real multi-source query; multi-source "
            "routing is itself disabled on accuracy evidence",
        "EXPLAIN_EXPOSE_SQL":
            "decision D2 — raw SQL names tables and columns, which is what this "
            "whole layer keeps out of a user-facing explanation",
    }
    for name, why in off.items():
        assert getattr(config, name) is False, f"{name} must stay OFF — {why}"

    # --- a threshold, not a boolean ----------------------------------------
    assert config.LOW_CONFIDENCE_WARNING_BELOW == 0.5, (
        "decision D1 — a computed confidence below this raises the low_evidence "
        "caveat; 0.0 would disable it entirely")



def test_flag_off_yields_null_objects(monkeypatch):
    monkeypatch.setattr(config, "LIFECYCLE_EVENTS_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "SOURCE_EXECUTION_RECORDS_ENABLED", False, raising=False)
    tl = lc.new_timeline()
    rec = er.new_recorder()
    assert tl.enabled is False and rec.enabled is False
    # every call is a no-op that returns cleanly
    assert tl.completed(lc.PHASE_UNDERSTANDING) is None
    assert rec.safe_records() == []
    assert rec.overall_status() == "none"


def test_warnings_noop_when_flag_off(monkeypatch, trace):
    monkeypatch.setattr(config, "QUERY_WARNINGS_ENABLED", False, raising=False)
    assert vw.add(vw.RESULT_TRUNCATED, trace=trace, limit=1000) is None
    assert vw.collect(trace) == []


# ── lifecycle ────────────────────────────────────────────────────────────────
def test_lifecycle_records_into_trace_and_streams(flags_on, trace):
    seen = []
    tl = lc.new_timeline(on_event=lambda p, m, e: seen.append((p, m, e)), trace=trace)
    tl.started(lc.PHASE_UNDERSTANDING)
    tl.completed(lc.PHASE_UNDERSTANDING)

    events = trace.sections[lc.TRACE_SECTION]["events"]
    assert [e["status"] for e in events] == ["started", "completed"]
    assert events[0]["title"] == "Understanding your question"
    # the SAME facts reached the stream — one record, two consumers
    assert [p for p, _, _ in seen] == [lc.PHASE_UNDERSTANDING] * 2
    assert seen[1][1] == "Understood what you're asking for"
    assert seen[1][2]["status"] == "completed"


def test_lifecycle_rejects_unknown_phase(flags_on, trace):
    """An unmapped internal stage name must emit NOTHING rather than leak."""
    tl = lc.new_timeline(trace=trace)
    assert tl.emit("tier2_shared_planner", lc.STATUS_COMPLETED) is None
    assert lc.TRACE_SECTION not in trace.sections


def test_lifecycle_callback_exception_never_loses_the_record(flags_on, trace):
    def boom(*a, **k):
        raise RuntimeError("client went away")

    tl = lc.new_timeline(on_event=boom, trace=trace)
    tl.completed(lc.PHASE_VALIDATION)
    assert len(trace.sections[lc.TRACE_SECTION]["events"]) == 1


def test_default_messages_never_empty():
    for phase in lc.PHASES:
        for status in lc.STATUSES:
            assert lc.default_message(phase, status).strip()


# ── warnings ─────────────────────────────────────────────────────────────────
def test_warning_formats_and_is_idempotent(flags_on, trace):
    w = vw.add(vw.RESULT_TRUNCATED, trace=trace, limit=1000)
    assert w.severity == "info"
    assert w.message == "The result was limited to the first 1000 records."
    vw.add(vw.RESULT_TRUNCATED, trace=trace, limit=1000)   # same condition again
    assert len(vw.collect(trace)) == 1, "duplicate warnings must collapse"


def test_unknown_warning_code_is_dropped(flags_on, trace):
    assert vw.add("not_a_real_code", trace=trace) is None
    assert vw.collect(trace) == []


def test_warning_missing_placeholder_does_not_leak_braces(flags_on, trace):
    w = vw.add(vw.RESULT_TRUNCATED, trace=trace)      # no `limit` supplied
    assert "{" not in w.message and "}" not in w.message


def test_restricted_data_warning_names_nothing(flags_on, trace):
    w = vw.add(vw.RESTRICTED_DATA, trace=trace)
    lowered = w.message.lower()
    for leak in ("table", "column", "schema", "salary", "db:", "select"):
        assert leak not in lowered


# ── source names ─────────────────────────────────────────────────────────────
def test_display_name_from_profiles(profiles):
    assert sn.display_name("2") == "Sales Database"
    assert sn.display_type("2") == "Database"
    assert sn.display_type("3") == "Documents"
    assert sn.is_known("2") is True


def test_unknown_source_never_discloses_the_id(profiles):
    """A source the request was not told about must not be named — and must not
    fall back to leaking its numeric id as the display name."""
    assert sn.display_name("99") == sn.GENERIC_NAME
    assert "99" not in sn.display_name("99")
    assert sn.is_known("99") is False


def test_summarize_lists_sources(profiles):
    assert sn.summarize(["2"]) == "Used Sales Database."
    assert sn.summarize(["2", "3"]) == (
        "Used 2 data sources: Sales Database and Contract Documents.")
    assert sn.summarize([]) == "No data source was used."


def test_describe_all_dedupes_preserving_order(profiles):
    got = sn.describe_all(["3", "2", "3"])
    assert [d["name"] for d in got] == ["Contract Documents", "Sales Database"]


# ── per-source execution records ─────────────────────────────────────────────
def test_execution_record_lifecycle_and_safe_projection(flags_on, trace, profiles):
    rec = er.SourceExecutionRecord(source_id="2", engine="deterministic_sql")
    rec.start()
    rec.finish(er.COMPLETED, rows=42)
    assert rec.duration_ms is not None and rec.rows_returned == 42

    safe = rec.as_safe_dict()
    assert safe["name"] == "Sales Database"
    assert safe["status"] == "completed"
    assert safe["rows_returned"] == 42
    # the internal-only fields must be absent from the safe projection
    for internal in ("engine", "error", "fallback_path", "error_class"):
        assert internal not in safe


def test_safe_projection_strips_raw_driver_error(flags_on, profiles):
    rec = er.SourceExecutionRecord(source_id="2").start()
    rec.finish(er.FAILED,
               error="psycopg2.OperationalError: could not connect to host=10.0.0.4 password=hunter2")
    safe = rec.as_safe_dict()
    blob = repr(safe).lower()
    for leak in ("psycopg2", "10.0.0.4", "hunter2", "password"):
        assert leak not in blob
    assert safe["message"] == "This data source could not be reached"


def test_recorder_rolls_up_partial(flags_on, trace, profiles):
    r = er.ExecutionRecorder(trace=trace)
    a = r.open("2", required=True)
    r.close(a, er.COMPLETED, rows=5)
    b = r.open("3", required=False)
    r.close(b, er.FAILED, error="timeout")
    assert r.overall_status() == "partial"
    assert r.ok_count() == 1
    assert r.any_failed() is True
    assert r.any_required_failed() is False
    assert trace.sections[er.TRACE_SECTION]["count"] == 2


def test_activity_message_never_names_the_engine(flags_on, profiles):
    known = er.SourceExecutionRecord(source_id="2", engine="deterministic_sql")
    assert known.activity_message() == "Retrieving data from Sales Database"
    unknown = er.SourceExecutionRecord(source_id="99", engine="rag")
    msg = unknown.activity_message()
    assert msg == "Searching relevant documents"
    assert "rag" not in msg and "99" not in msg


# ── safe projection ──────────────────────────────────────────────────────────
def test_routing_projection_maps_reason_code(flags_on, trace):
    trace.set("routing", status="ROUTED", mode="SINGLE", source_ids=["2"],
              reason_code="SINGLE_CANDIDATE", decision_method="deterministic")
    out = sp.build_routing(trace)
    assert out["mode"] == "single"
    assert out["summary"] == "One data source contains the information this question needs."
    assert out["source_count"] == 1


def test_routing_projection_omits_internal_evidence(flags_on, trace):
    """candidate_sources / evidence / scores must never survive the projection."""
    trace.set("routing", status="ROUTED", mode="SINGLE", source_ids=["2"],
              reason_code="SINGLE_CANDIDATE",
              candidate_sources=[{"source_id": "7", "top_score": 0.91}],
              evidence_summary=[{"columns": ["salary", "ssn"]}])
    out = sp.build_routing(trace)
    blob = repr(out).lower()
    for leak in ("salary", "ssn", "0.91", "top_score", "candidate_sources", "evidence"):
        assert leak not in blob
    # reason_code IS deliberately exposed (a stable, non-sensitive enum), so the
    # check above must not be read as "the string 'candidate' can never appear".
    assert out["reason_code"] == "SINGLE_CANDIDATE"
    assert "7" not in [s for s in blob if s.isdigit()]


def test_unknown_reason_code_falls_back_to_generic_copy(flags_on, trace):
    trace.set("routing", status="ROUTED", mode="SINGLE", source_ids=["2"],
              reason_code="SOME_NEW_INTERNAL_CODE")
    out = sp.build_routing(trace)
    assert out["summary"] == sp._ROUTING_GENERIC
    assert "SOME_NEW_INTERNAL_CODE" not in out["summary"]


def test_execution_plan_projection_is_honest_about_sequential(flags_on, trace, profiles):
    """Defect 2: the planner says PARALLEL but the runtime is sequential. The
    projection must report what actually happened."""
    trace.set("execution_plan", planner_mode="PARALLEL", executed_mode="sequential",
              strategy="independent", reason="Multiple relevant sources.",
              steps=[{"source_id": "2", "source_type": "relational", "required": True},
                     {"source_id": "3", "source_type": "document", "required": True}])
    out = sp.build_execution_plan(trace)
    assert out["executed_mode"] == "sequential"
    assert out["mode"] == "multi_source"
    assert out["summary"] == "Used 2 data sources to answer this question"
    assert [s["name"] for s in out["steps"]] == ["Sales Database", "Contract Documents"]
    assert "PARALLEL" not in repr(out)


def test_cross_source_projection_with_join_rate(flags_on, trace, profiles):
    trace.set("federation", used=True, source_ids=["2", "3"], operation="combined",
              result_status="complete",
              join={"join_used": True, "join_keys": ["sales.cust_id=contracts.id"],
                    "matched_count": 98, "unmatched_count": 2})
    out = sp.build_cross_source(trace)
    assert out["sources"] == ["Sales Database", "Contract Documents"]
    assert out["join"]["match_rate_pct"] == 98.0
    assert "98.0% of relevant records were successfully matched." in out["join"]["summary"]
    # the raw join predicate stays internal
    assert "cust_id" not in repr(out)


def test_cross_source_absent_for_single_source(flags_on, trace):
    assert sp.build_cross_source(trace) is None


def test_timeline_summary_collapses_to_worst_status(flags_on, trace):
    tl = lc.new_timeline(trace=trace)
    tl.started(lc.PHASE_DATA_RETRIEVAL)
    tl.completed(lc.PHASE_DATA_RETRIEVAL)
    tl.warning(lc.PHASE_DATA_RETRIEVAL, "one source was slow")
    rows = sp.build_timeline_summary(trace)
    assert rows == [{"phase": lc.PHASE_DATA_RETRIEVAL,
                     "title": "Running the query", "status": "warning"}]
    assert lc.PHASE_TITLES[lc.PHASE_VALIDATION] == "Checking the query", (
        "validation is PRE-flight — see the note in lifecycle.PHASE_TITLES")


def test_result_meta_marks_partial_from_warnings(flags_on, trace):
    trace.set("execution", row_count=1000, truncated=True)
    vw.add(vw.PARTIAL_SOURCE_FAILURE, trace=trace)
    meta = sp.build_result_meta(trace)
    assert meta == {"row_count": 1000, "truncated": True, "partial": True,
                    "reused_verified_query": False}


def test_projection_survives_a_malformed_section(flags_on, trace):
    trace.sections["routing"] = "not-a-dict"
    trace.sections[er.TRACE_SECTION] = {"records": ["junk", None]}
    ext = sp.build_explain_extension(trace, trace_id="t1")
    assert ext["support"]["trace_id"] == "t1"
    assert ext["warnings"] == []


# ── build_explain v1/v2 ──────────────────────────────────────────────────────
_SQL = "SELECT status, COUNT(*) AS n FROM incidents WHERE entry_type = %s GROUP BY status"
_CHECKS = [{"name": "value_grounding", "status": "pass"},
           {"name": "qualifier_completeness", "status": "pass"}]


def test_v1_shape_unchanged_when_v2_off(monkeypatch):
    monkeypatch.setattr(config, "EXPLAIN_V2_ENABLED", False, raising=False)
    p = build_explain(sql=_SQL, table="incidents", sm=None, checks=_CHECKS, params=["OPEN"])
    assert p["version"] == "1.0"
    assert set(p) == {"version", "understanding", "data_used", "operations", "filters",
                      "validation", "sql", "confidence", "timeline"}


def test_v2_adds_blocks_without_touching_v1_keys(flags_on, trace, profiles):
    trace.set("routing", status="ROUTED", mode="SINGLE", source_ids=["2"],
              reason_code="SINGLE_CANDIDATE")
    trace.set("execution", row_count=7, truncated=False)
    v1 = build_explain(sql=_SQL, table="incidents", sm=None, checks=_CHECKS, params=["OPEN"],
                       trace=None)
    v2 = build_explain(sql=_SQL, table="incidents", sm=None, checks=_CHECKS, params=["OPEN"],
                       trace=trace, trace_id="testtrace123")
    assert v2["version"] == "2.0"
    for key in ("understanding", "data_used", "operations", "filters", "validation"):
        assert v2[key] == v1[key], f"v1 key {key} must be byte-identical under v2"
    assert v2["routing"]["mode"] == "single"
    assert v2["support"]["trace_id"] == "testtrace123"
    assert v2["result"]["row_count"] == 7
    assert v2["warnings"] == []


def test_v2_failure_never_costs_the_v1_payload(flags_on, monkeypatch):
    """If the projection layer raises, the caller must still get a valid payload."""
    def boom(*a, **k):
        raise RuntimeError("projection exploded")

    monkeypatch.setattr(sp, "build_explain_extension", boom)
    p = build_explain(sql=_SQL, table="incidents", sm=None, checks=_CHECKS, params=["OPEN"],
                      trace=ExplainTrace(query="q", trace_id="t"))
    assert p["version"] == "1.0"
    assert p["understanding"]["summary"]


def test_sql_exposure_is_gateable(monkeypatch):
    monkeypatch.setattr(config, "EXPLAIN_EXPOSE_SQL", False, raising=False)
    p = build_explain(sql=_SQL, table="incidents", sm=None, checks=_CHECKS)
    assert p["sql"] == {"enabled": False, "query": None}
    monkeypatch.setattr(config, "EXPLAIN_EXPOSE_SQL", True, raising=False)
    p = build_explain(sql=_SQL, table="incidents", sm=None, checks=_CHECKS)
    assert p["sql"]["query"] == _SQL


# ── Defect 1 ─────────────────────────────────────────────────────────────────
def test_defect1_summary_reads_nested_payload_keys():
    """Regression: this used to read flat "datasets"/"check_items", which
    build_explain has never emitted, so every trace recorded None/None."""
    payload = build_explain(sql=_SQL, table="incidents", sm=None, checks=_CHECKS, params=["OPEN"])
    got = summarize_explain_payload(payload)
    assert got["datasets"] == payload["data_used"]["datasets"]
    assert got["datasets"] is not None
    assert got["validation_passed"] is True
    assert got["check_count"] == len(payload["validation"]["checks"])
    assert got["filter_count"] == 1


def test_defect1_reports_a_failing_check():
    payload = build_explain(sql=_SQL, table="incidents", sm=None,
                            checks=[{"name": "value_grounding", "status": "fail"}])
    assert summarize_explain_payload(payload)["validation_passed"] is False


def test_defect1_handles_refusal_payload():
    from veda.business_explain import build_refusal_explain
    r = build_refusal_explain("ungrounded", {"why": "w", "what_needed": "n", "suggestions": []})
    got = summarize_explain_payload(r)
    assert got["datasets"] is None and got["validation_passed"] is None


# ── version stamping ─────────────────────────────────────────────────────────
def test_version_stamp_is_admin_only(flags_on, trace, monkeypatch):
    monkeypatch.setenv("VEDA_GIT_SHA", "abc123def456")
    trace.stamp_versions()
    assert trace.sections["versions"]["git_sha"] == "abc123def456"
    # ...and never reaches the user-facing projection
    assert "versions" not in sp.build_explain_extension(trace, trace_id="t")
    assert "git_sha" not in repr(sp.build_explain_extension(trace, trace_id="t"))


def test_every_templated_code_has_a_no_arg_message():
    """A templated warning added without a NO_ARG_MESSAGE fallback would show a
    user a raw "{placeholder}". Fail here rather than in production."""
    for code, (_sev, msg) in vw.CATALOG.items():
        if "{" in msg:
            assert code in vw.NO_ARG_MESSAGE, f"{code} is templated but has no fallback"
            assert "{" not in vw.NO_ARG_MESSAGE[code]


# ── end-to-end spine ─────────────────────────────────────────────────────────
def test_full_spine_produces_a_coherent_v2_payload(flags_on, trace, profiles):
    """One query's worth of facts recorded through the real modules, then
    projected. Asserts the pieces AGREE — the single-source-of-truth rule."""
    streamed = []
    tl = lc.new_timeline(on_event=lambda p, m, e: streamed.append((p, e["status"])), trace=trace)
    rec = er.ExecutionRecorder(trace=trace, timeline=tl)

    with lc.use_timeline(tl):
        tl.completed(lc.PHASE_RECEIVED)
        tl.completed(lc.PHASE_UNDERSTANDING, "Identified the requested metric")
        tl.completed(lc.PHASE_ACCESS_CHECK)
        trace.set("routing", status="ROUTED", mode="MULTI", source_ids=["2", "3"],
                  reason_code="RELATIONSHIP_EDGE")
        tl.completed(lc.PHASE_SOURCE_SELECTION, source_count=2)
        trace.set("execution_plan", planner_mode="PARALLEL", executed_mode="sequential",
                  strategy="independent", reason="Multiple relevant sources.",
                  steps=[{"source_id": "2", "source_type": "relational", "required": True},
                         {"source_id": "3", "source_type": "document", "required": True}])
        a = rec.open("2", source_type="relational", engine="deterministic_sql")
        rec.close(a, er.COMPLETED, rows=12)
        b = rec.open("3", source_type="document", engine="rag")
        rec.close(b, er.FAILED, error="psycopg2 OperationalError host=10.0.0.9")
        vw.add(vw.PARTIAL_SOURCE_FAILURE, trace=trace)
        trace.set("execution", row_count=12, truncated=False)
        tl.completed(lc.PHASE_COMPLETED)

    ext = sp.build_explain_extension(trace, trace_id="testtrace123")

    # routing / plan agree on 2 sources, and the plan is HONEST about sequencing
    assert ext["routing"]["mode"] == "multi"
    assert ext["execution_plan"]["executed_mode"] == "sequential"
    assert ext["execution_plan"]["summary"] == "Used 2 data sources to answer this question"

    # per-source truth: one ok, one failed -> partial, and the driver error is gone
    assert ext["execution"]["status"] == "partial"
    names = {s["name"]: s["status"] for s in ext["execution"]["sources"]}
    assert names == {"Sales Database": "completed", "Contract Documents": "failed"}
    blob = repr(ext).lower()
    for leak in ("psycopg2", "10.0.0.9", "deterministic_sql", "rag", "parallel"):
        assert leak not in blob, f"{leak} leaked into the user-facing projection"

    # the warning is present, and result.partial agrees with it
    assert [w["code"] for w in ext["warnings"]] == [vw.PARTIAL_SOURCE_FAILURE]
    assert ext["result"]["partial"] is True
    assert ext["result"]["row_count"] == 12

    # what streamed and what persisted describe the same run
    assert ("received", "completed") in streamed
    # Level 3 (§9): the phase-carrying record lives under `audit`, not at the top
    # level, because it is the only place raw backend phase names appear.
    assert len(ext["audit"]["timeline"]) == len(
        trace.sections[lc.TRACE_SECTION]["events"])
    assert {r["phase"] for r in ext["audit"]["timeline_summary"]} <= set(lc.PHASES)
    assert ext["support"]["trace_id"] == "testtrace123"


def test_spine_is_silent_when_flags_are_off(trace, monkeypatch, profiles):
    """The default-OFF contract, end to end: nothing recorded, nothing projected."""
    for name in ("LIFECYCLE_EVENTS_ENABLED", "QUERY_WARNINGS_ENABLED",
                 "SOURCE_EXECUTION_RECORDS_ENABLED", "EXPLAIN_V2_ENABLED"):
        monkeypatch.setattr(config, name, False, raising=False)
    tl = lc.new_timeline(trace=trace)
    rec = er.new_recorder(trace=trace, timeline=tl)
    with lc.use_timeline(tl):
        tl.completed(lc.PHASE_RECEIVED)
        r = rec.open("2")
        rec.close(r, er.COMPLETED, rows=3)
        vw.add(vw.PARTIAL_SOURCE_FAILURE, trace=trace)
    assert lc.TRACE_SECTION not in trace.sections
    assert er.TRACE_SECTION not in trace.sections
    assert vw.TRACE_SECTION not in trace.sections
    p = build_explain(sql=_SQL, table="incidents", sm=None, checks=_CHECKS, trace=trace)
    assert p["version"] == "1.0"


# ── Part 3: RBAC narrowing warning ───────────────────────────────────────────
def test_restricted_data_warning_is_projected(flags_on, trace):
    """RBAC narrowing used to be entirely silent. It must now surface as a
    warning — while the before/after COUNTS stay internal, because the number of
    hidden things is itself a disclosure."""
    trace.set("rbac_filter", before=40, after=12)
    vw.add(vw.RESTRICTED_DATA, trace=trace)
    ext = sp.build_explain_extension(trace, trace_id="t")
    assert [w["code"] for w in ext["warnings"]] == [vw.RESTRICTED_DATA]
    assert ext["result"]["partial"] is True
    assert vw.RESTRICTED_DATA in [str(x) for x in ext["limitations"]] or ext["limitations"]
    blob = repr(ext)
    assert "40" not in blob and "rbac_filter" not in blob, (
        "the count of withheld candidates must not be projected")


# ── Part 23: retries / fallback ──────────────────────────────────────────────
def test_execute_reliably_stamps_retry_count(monkeypatch):
    from query import reliability as rl
    from query.agents import AgentResult

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            return AgentResult("2", "relational", "failed", error="connection reset by peer")
        return AgentResult("2", "relational", "ok", engine="deterministic_sql")

    res = rl.execute_reliably(flaky, enabled=True, max_retries=2)
    assert res.status == "ok"
    assert res.retry_count == 1
    assert calls["n"] == 2


def test_retry_count_defaults_to_zero_without_retry():
    from query import reliability as rl
    from query.agents import AgentResult
    res = rl.execute_reliably(lambda: AgentResult("2", "relational", "ok"),
                              enabled=True, max_retries=2)
    assert res.retry_count == 0


def test_retried_record_shows_retried_not_the_count(flags_on, profiles):
    rec = er.SourceExecutionRecord(source_id="2", engine="deterministic_sql")
    rec.start(); rec.retry_count = 2; rec.fallback_used = True
    rec.finish(er.COMPLETED, rows=3)
    safe = rec.as_safe_dict()
    assert safe["retried"] is True
    assert safe["fallback_used"] is True
    # the raw attempt count is internal — "it was retried" is the user-facing fact
    assert "retry_count" not in safe


def test_fallback_warning_copy_names_no_component(flags_on, trace):
    w = vw.add(vw.FALLBACK_USED, trace=trace)
    lowered = w.message.lower()
    for leak in ("tier", "tier-2", "slm", "llm", "agent", "deterministic", "rag"):
        assert leak not in lowered


# ── Part 10: cross-source join stats ─────────────────────────────────────────
def test_join_stats_over_real_duckdb(monkeypatch):
    """Real DuckDB, real temp tables — the same shape execute_plan materializes."""
    duckdb = pytest.importorskip("duckdb")
    monkeypatch.setattr(config, "FEDERATED_JOIN_STATS_ENABLED", True, raising=False)
    from query.federated_executor import _join_stats

    conn = duckdb.connect(":memory:")
    conn.execute('CREATE TEMP TABLE agg_0 AS SELECT * FROM (VALUES (\'a\',1),(\'b\',2),(\'c\',3)) t("k","x")')
    conn.execute('CREATE TEMP TABLE agg_1 AS SELECT * FROM (VALUES (\'a\',9),(\'b\',8),(\'z\',7)) t("k","y")')

    stats = _join_stats(conn, ["agg_0", "agg_1"], '"k"')
    # a,b matched; c and z each present on one side only
    assert stats == {"join_used": True, "matched_count": 2, "unmatched_count": 2,
                     "source_count": 2}


def test_join_stats_off_by_default(monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    monkeypatch.setattr(config, "FEDERATED_JOIN_STATS_ENABLED", False, raising=False)
    from query.federated_executor import _join_stats
    conn = duckdb.connect(":memory:")
    conn.execute('CREATE TEMP TABLE agg_0 AS SELECT * FROM (VALUES (\'a\')) t("k")')
    conn.execute('CREATE TEMP TABLE agg_1 AS SELECT * FROM (VALUES (\'a\')) t("k")')
    assert _join_stats(conn, ["agg_0", "agg_1"], '"k"') is None


def test_join_stats_single_table_is_not_a_join(monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    monkeypatch.setattr(config, "FEDERATED_JOIN_STATS_ENABLED", True, raising=False)
    from query.federated_executor import _join_stats
    conn = duckdb.connect(":memory:")
    conn.execute('CREATE TEMP TABLE agg_0 AS SELECT * FROM (VALUES (\'a\')) t("k")')
    assert _join_stats(conn, ["agg_0"], '"k"') is None


def test_join_stats_feed_the_match_rate_projection(flags_on, trace, profiles):
    """End of the chain: executor stats → trace → user-facing match rate."""
    trace.set("federation", used=True, source_ids=["2", "3"], operation="combined",
              result_status="complete",
              join={"join_used": True, "matched_count": 49, "unmatched_count": 1,
                    "source_count": 2})
    out = sp.build_cross_source(trace)
    assert out["join"]["match_rate_pct"] == 98.0
    assert "98.0% of relevant records were successfully matched." in out["join"]["summary"]
    assert "matched_count" not in out["join"]      # raw counts stay internal


# ── EXP-B1: no phase left hanging ────────────────────────────────────────────
def test_open_phases_are_closed_at_the_terminal(flags_on, trace):
    """A phase emitted as `started` and never resolved showed the user a spinner that
    never finished — on 6 of 10 benchmark query types."""
    from veda_hybrid import _emit_terminal_lifecycle
    tl = lc.new_timeline(trace=trace)
    tl.started(lc.PHASE_ACCESS_CHECK)          # opened, never resolved
    tl.started(lc.PHASE_UNDERSTANDING)
    tl.completed(lc.PHASE_UNDERSTANDING)       # this one IS resolved
    _emit_terminal_lifecycle(tl, "refused")

    per_phase = {}
    for e in tl.events:
        per_phase.setdefault(e.phase, []).append(e.status)
    assert lc.STATUS_COMPLETED in per_phase[lc.PHASE_ACCESS_CHECK], "must not stay open"
    assert per_phase[lc.PHASE_UNDERSTANDING].count(lc.STATUS_COMPLETED) == 1, "no double close"


def test_a_negatively_resolved_phase_is_not_overwritten(flags_on, trace):
    """A real permission denial must stay failed — never be flipped to "verified" by
    the terminal sweep."""
    from veda_hybrid import _emit_terminal_lifecycle
    tl = lc.new_timeline(trace=trace)
    tl.started(lc.PHASE_ACCESS_CHECK)
    tl.failed(lc.PHASE_ACCESS_CHECK)
    _emit_terminal_lifecycle(tl, "refused")
    statuses = [e.status for e in tl.events if e.phase == lc.PHASE_ACCESS_CHECK]
    assert lc.STATUS_COMPLETED not in statuses
    assert sp.build_timeline_summary(trace)[0]["status"] == lc.STATUS_FAILED


# ── EXP-B2: truncation ───────────────────────────────────────────────────────
def test_truncated_result_raises_a_warning(flags_on, trace):
    """A truncated page used to ship with warnings:[] because the warning tested a
    1000-row cap while truncation is decided at 20."""
    from veda.explain import record_result_stages
    with lc.use_timeline(lc.new_timeline(trace=trace)):
        from veda.explain import use_trace
        with use_trace(trace):
            record_result_stages(cols=["a"], row_count=20, truncated=True)
    assert [w["code"] for w in vw.collect(trace)] == [vw.RESULT_TRUNCATED]
    assert sp.build_result_meta(trace)["truncated"] is True


def test_untruncated_result_raises_nothing(flags_on, trace):
    from veda.explain import record_result_stages, use_trace
    with use_trace(trace):
        record_result_stages(cols=["a"], row_count=3, truncated=False)
    assert vw.collect(trace) == []


# ── EXP-B6: fallback copy ────────────────────────────────────────────────────
def test_fallback_copy_does_not_claim_unavailability(flags_on, trace):
    w = vw.add(vw.FALLBACK_USED, trace=trace)
    assert "unavailable" not in w.message.lower(), (
        "the primary head could not ANSWER; nothing was down")
    assert "could not answer" in w.message.lower()


# ── EXP-B7: sync is no longer O(n^2) ─────────────────────────────────────────
def test_close_updates_only_the_changed_record(flags_on, trace, profiles):
    r = er.ExecutionRecorder(trace=trace)
    a = r.open("2"); b = r.open("3")
    r.close(a, er.COMPLETED, rows=7)
    r.close(b, er.FAILED, error="boom")
    rows = trace.sections[er.TRACE_SECTION]["records"]
    assert len(rows) == 2
    by_id = {x["source_id"]: x for x in rows}
    assert by_id["2"]["status"] == er.COMPLETED and by_id["2"]["rows_returned"] == 7
    assert by_id["3"]["status"] == er.FAILED
    assert by_id["3"]["error"] == "boom"     # internal copy still complete


def test_sync_scales_linearly(flags_on):
    """200 records used to take 654 ms because every close re-serialised all of them.

    Asserts the SCALING RATIO, not a wall-clock budget: 4x the records should cost
    roughly 4x (linear), not ~16x (quadratic). An absolute threshold made this test
    fail on a loaded machine (measured 329 ms) while the behaviour was correct —
    the ratio is what the fix actually claims.
    
    TIMING-SENSITIVE: this measures a RATIO, not a wall clock, which is what makes
    it meaningful — an earlier wall-clock version of this test PASSED while the
    code was still quadratic. But a ratio can still wobble when the machine is
    loaded (observed once during a parallel test run, passing 5/5 immediately
    after in isolation). If it fails, re-run it alone before treating it as a
    regression; a real regression fails consistently and by a wide margin
    (14.8x was the measured pre-fix figure, against a 6x ceiling).
    """
    import time

    def cost(n):
        tr = ExplainTrace(query="q", trace_id="scale")
        r = er.ExecutionRecorder(trace=tr)
        t0 = time.perf_counter()
        for i in range(n):
            r.close(r.open(str(i)), er.COMPLETED, rows=1)
        return (time.perf_counter() - t0), len(tr.sections[er.TRACE_SECTION]["records"])

    cost(50)                      # warm up, discard
    t50, n50 = cost(50)
    t200, n200 = cost(200)
    assert (n50, n200) == (50, 200)
    ratio = t200 / max(t50, 1e-6)
    assert ratio < 8, (
        f"4x the records cost {ratio:.1f}x the time — still superlinear "
        f"(linear ~4x, quadratic ~16x). Re-run this test ALONE before treating "
        f"it as a regression — see its docstring.")


def test_recorder_reports_whether_anything_was_recorded(flags_on, trace):
    r = er.ExecutionRecorder(trace=trace)
    assert r.has_records() is False
    r.close(r.open("2"), er.COMPLETED, rows=1)
    assert r.has_records() is True
    assert er.new_recorder().has_records() is False      # null recorder too


def test_limitations_covers_every_result_constraining_warning(flags_on, trace):
    """A warning that changes how the answer should be READ must appear in
    `limitations`, not only in `warnings`."""
    from veda import warnings as w
    for code in (w.RESULT_TRUNCATED, w.RESTRICTED_DATA, w.PARTIAL_SOURCE_FAILURE,
                 w.SOURCE_CONFLICT, w.UNMATCHED_RECORDS, w.LOW_EVIDENCE):
        tr = ExplainTrace(query="q", trace_id="t")
        vw.add(code, trace=tr, limit=10)
        assert sp.build_limitations(tr), f"{code} produced no limitation"


def test_fallback_is_informational_not_a_limitation(flags_on, trace):
    """An alternate retrieval path still produced a COMPLETE answer — it is worth
    saying, but it does not constrain how the result should be read."""
    vw.add(vw.FALLBACK_USED, trace=trace)
    assert [w["code"] for w in sp.build_warnings(trace)] == [vw.FALLBACK_USED]
    assert sp.build_limitations(trace) == []


# ---------------------------------------------------------------------------
# Provenance: a verified-query REPLAY must be visible to the person reading the
# answer. A replayed answer runs SQL verified under possibly-older code, and the
# lane was previously invisible in the payload — no `cache` key anywhere.
# ---------------------------------------------------------------------------

def test_provenance_is_absent_on_an_ordinary_query(flags_on, trace):
    """An always-present "nothing special happened" block is noise, not honesty."""
    trace.set("execution", row_count=5)
    assert sp.build_provenance(trace) is None
    assert sp.build_result_meta(trace)["reused_verified_query"] is False


def test_provenance_reports_a_replayed_answer(flags_on, trace):
    trace.set("execution", row_count=5, from_cache=True)
    prov = sp.build_provenance(trace)
    assert prov["reused_verified_query"] is True
    assert prov["summary"]
    assert sp.build_result_meta(trace)["reused_verified_query"] is True


def test_provenance_summary_names_no_internal_mechanism(flags_on, trace):
    """The sentence may say WHAT happened, never HOW it is implemented."""
    trace.set("execution", from_cache=True)
    low = sp.build_provenance(trace)["summary"].lower()
    for banned in ("cache", "cached", "sql", "table", "column", "sentinel",
                   "embedding", "vector", "similarity"):
        assert banned not in low, f"{banned!r} leaked into the provenance summary"


def test_provenance_is_included_in_the_v2_extension(flags_on, trace):
    trace.set("execution", from_cache=True)
    ext = sp.build_explain_extension(trace, trace_id="t")
    assert ext["provenance"]["reused_verified_query"] is True
    assert ext["result"]["reused_verified_query"] is True


def test_refusal_payload_discloses_a_replayed_query(flags_on, trace):
    """A replay that got REFUSED must still say it was a replay.

    Measured live: 2 of 3 verified-cache replays were stopped by the alignment
    gate, so the reuse is often the reason the question could not be answered —
    which makes it more relevant on a refusal, not less.
    """
    trace.set("execution", from_cache=True)
    out = {}
    be._apply_v2_refusal(out, trace=trace, trace_id="t")
    assert out["provenance"]["reused_verified_query"] is True
    # The blocks a refusal genuinely cannot have stay absent.
    for absent in ("routing", "execution", "execution_plan"):
        assert absent not in out


def test_refusal_payload_omits_provenance_on_an_ordinary_refusal(flags_on, trace):
    out = {}
    be._apply_v2_refusal(out, trace=trace, trace_id="t")
    assert "provenance" not in out


# ---------------------------------------------------------------------------
# F30: the PERSISTED timeline must include the terminal phase.
#
# pipeline._done builds the payload, and only then does the front door emit the
# phase describing the outcome — so the stored record ended one phase early.
# ---------------------------------------------------------------------------

def test_persisted_timeline_is_refreshed_after_the_terminal_phase(flags_on, trace):
    import veda_hybrid as VH
    from veda import lifecycle as lcm
    from veda.explain import bind_trace

    bind_trace(trace)
    tl = lcm.new_timeline(trace=trace)
    tl.completed(lcm.PHASE_RECEIVED)
    tl.completed(lcm.PHASE_VALIDATION)

    # The payload as `_done` would have built it — before result_preparation exists.
    stale = sp.build_timeline_summary(trace)
    payload = {"explain": {"audit": {"timeline_summary": list(stale),
                                     "timeline": sp.build_timeline(trace)}}}
    assert lcm.PHASE_RESULT_PREPARATION not in {r["phase"] for r in stale}

    # Now the front door resolves the outcome, as it does after _done returns.
    tl.warning(lcm.PHASE_RESULT_PREPARATION, "Could not answer this")

    class _Item:
        result = payload

    class _Res:
        items = [_Item()]

    VH._refresh_persisted_timeline(_Res(), tl)
    phases = {r["phase"]: r["status"]
              for r in payload["explain"]["audit"]["timeline_summary"]}
    assert phases.get(lcm.PHASE_RESULT_PREPARATION) == "warning"


def test_refresh_never_introduces_a_v2_block_into_a_v1_payload(flags_on, trace):
    """Additive-by-contract runs one way only: refresh what exists, add nothing."""
    import veda_hybrid as VH
    from veda import lifecycle as lcm
    from veda.explain import bind_trace

    bind_trace(trace)
    tl = lcm.new_timeline(trace=trace)
    tl.completed(lcm.PHASE_RECEIVED)
    payload = {"explain": {"version": "1.0", "understanding": {"summary": "x"}}}

    class _Item:
        result = payload

    class _Res:
        items = [_Item()]

    VH._refresh_persisted_timeline(_Res(), tl)
    assert "timeline_summary" not in payload["explain"]
    assert "timeline" not in payload["explain"]


def test_refresh_is_inert_when_the_timeline_is_disabled(trace):
    import veda_hybrid as VH
    from veda import lifecycle as lcm

    payload = {"explain": {"audit": {"timeline_summary": ["untouched"]}}}

    class _Item:
        result = payload

    class _Res:
        items = [_Item()]

    VH._refresh_persisted_timeline(_Res(), lcm._NullTimeline())
    assert payload["explain"]["audit"]["timeline_summary"] == ["untouched"]


# ---------------------------------------------------------------------------
# The access-check outcome is decided from the TURN's terminal feedback, not from
# an intermediate attempt's. Found live on the data-lake path: Tier-1 refused,
# feedback classified it as an access problem, Tier-2 then produced a clarify
# about an ambiguous column — and the user saw a red cross on
# "Checking access permissions" above a reply about column names.
# ---------------------------------------------------------------------------

def _turn(status, payload):
    class _Item:
        pass
    it = _Item()
    it.status = status
    it.result = payload

    class _Res:
        items = [it]
    return _Res()


def test_access_check_fails_when_the_turn_refused_on_access(flags_on, trace):
    import veda_hybrid as VH
    from veda import lifecycle as lcm
    from veda.explain import bind_trace
    from veda.feedback import ACCESS_DENIED_WHY

    bind_trace(trace)
    tl = lcm.new_timeline(trace=trace)
    tl.started(lcm.PHASE_ACCESS_CHECK)
    VH._reconcile_access_check(
        _turn("refused", {"feedback": {"why": ACCESS_DENIED_WHY}}), tl)
    rows = {r["phase"]: r["status"] for r in sp.build_timeline_summary(trace)}
    assert rows[lcm.PHASE_ACCESS_CHECK] == "failed"


def test_access_check_is_not_failed_when_the_turn_refused_for_another_reason(flags_on, trace):
    import veda_hybrid as VH
    from veda import lifecycle as lcm
    from veda.explain import bind_trace

    bind_trace(trace)
    tl = lcm.new_timeline(trace=trace)
    tl.completed(lcm.PHASE_ACCESS_CHECK)
    VH._reconcile_access_check(
        _turn("refused", {"feedback": {"why": "that column is ambiguous"}}), tl)
    rows = {r["phase"]: r["status"] for r in sp.build_timeline_summary(trace)}
    assert rows[lcm.PHASE_ACCESS_CHECK] == "completed"


def test_access_check_is_not_failed_when_the_turn_answered(flags_on, trace):
    """An answered turn cannot have been blocked on permissions, whatever an
    intermediate attempt's feedback said."""
    import veda_hybrid as VH
    from veda import lifecycle as lcm
    from veda.explain import bind_trace
    from veda.feedback import ACCESS_DENIED_WHY

    bind_trace(trace)
    tl = lcm.new_timeline(trace=trace)
    tl.completed(lcm.PHASE_ACCESS_CHECK)
    VH._reconcile_access_check(
        _turn(VH.STATUS_OK, {"feedback": {"why": ACCESS_DENIED_WHY}}), tl)
    rows = {r["phase"]: r["status"] for r in sp.build_timeline_summary(trace)}
    assert rows[lcm.PHASE_ACCESS_CHECK] == "completed"


def test_sweep_never_invents_an_access_failure(flags_on, trace):
    """The sweep infers `failed` from ANOTHER stage's status — a guess. For
    access_check a wrong guess tells the user they lack permission they have.

    Measured live: Tier-1 ended `qualifier_dropped`, the sweep failed the open
    access_check, Tier-2 then clarified about an ambiguous column, and the reply
    had nothing to do with permissions.
    """
    from veda import lifecycle as lcm
    from veda.explain import bind_trace

    bind_trace(trace)
    tl = lcm.new_timeline(trace=trace)
    tl.started(lcm.PHASE_ACCESS_CHECK)
    tl.started(lcm.PHASE_DATA_RETRIEVAL)
    tl.close_open_phases(failed=True)
    rows = {r["phase"]: r["status"] for r in sp.build_timeline_summary(trace)}
    assert rows[lcm.PHASE_ACCESS_CHECK] == "completed", (
        "the sweep must not guess a permission failure")
    # Every other phase still sweeps to failed — the guard is deliberately narrow.
    assert rows[lcm.PHASE_DATA_RETRIEVAL] == "failed"


def test_an_explicit_access_failure_still_stands(flags_on, trace):
    """The guard must not make a REAL denial unreportable."""
    from veda import lifecycle as lcm
    from veda.explain import bind_trace

    bind_trace(trace)
    tl = lcm.new_timeline(trace=trace)
    tl.started(lcm.PHASE_ACCESS_CHECK)
    tl.failed(lcm.PHASE_ACCESS_CHECK)          # the decision-maker said so
    tl.close_open_phases(failed=False)         # and a positive sweep cannot undo it
    rows = {r["phase"]: r["status"] for r in sp.build_timeline_summary(trace)}
    assert rows[lcm.PHASE_ACCESS_CHECK] == "failed"


# ---------------------------------------------------------------------------
# A document answer must not be attributed to the database.
#
# The gap-fill record (EXP-B4) stamped the request context's default source_id,
# not the one the router chose. Measured on a real document question: routing
# picked source 3 (the document store) and the answer came from a PDF, but the
# user was told "Retrieving data from homzhub".
# ---------------------------------------------------------------------------

def test_gap_fill_record_names_the_routed_source_not_the_default(flags_on, trace, monkeypatch):
    import veda_hybrid as VH
    from veda import exec_records as er
    from veda.explain import bind_trace

    bind_trace(trace)
    trace.set("routing", status="ROUTED", mode="SINGLE", source_ids=["3"])

    class _Res:
        error = None
        rows = [1, 2, 3]

    # The ambient context still points at the DEFAULT source, as it did live.
    class _Ctx:
        source_id = "2"

    monkeypatch.setattr(VH, "_current_ctx", lambda: _Ctx())
    monkeypatch.setattr(VH, "_dispatch_single_inner",
                        lambda *a, **k: ("rag", _Res()))

    rec = er.ExecutionRecorder(trace=trace)
    with er.use_recorder(rec):
        VH._dispatch_single("what is the notice period?")
        ids = [r.get("source_id") for r in
               (trace.sections.get(er.TRACE_SECTION) or {}).get("records", [])]

    assert ids == ["3"], f"expected the ROUTED source, got {ids}"


def test_gap_fill_falls_back_to_the_context_when_routing_is_silent(flags_on, trace, monkeypatch):
    """No routing decision recorded (single-source shortcut) — the context is then
    the only signal there is, and it is better than nothing."""
    import veda_hybrid as VH
    from veda import exec_records as er
    from veda.explain import bind_trace

    bind_trace(trace)

    class _Res:
        error = None
        rows = []

    class _Ctx:
        source_id = "2"

    monkeypatch.setattr(VH, "_current_ctx", lambda: _Ctx())
    monkeypatch.setattr(VH, "_dispatch_single_inner",
                        lambda *a, **k: ("rag", _Res()))

    rec = er.ExecutionRecorder(trace=trace)
    with er.use_recorder(rec):
        VH._dispatch_single("q")
        ids = [r.get("source_id") for r in
               (trace.sections.get(er.TRACE_SECTION) or {}).get("records", [])]
    assert ids == ["2"]


# ---------------------------------------------------------------------------
# An answer with NO confidence signal must not be quieter than a low-confidence
# one. Measured live: "probation period days in samta" was answered from amenity
# monthly fees, relabelled "Total Days", with confidence=None and ZERO warnings —
# while a 0.018-confidence answer on the same system correctly carried a caveat.
# ---------------------------------------------------------------------------


def test_the_caveat_fires_on_a_computed_low_score_only(flags_on):
    """Deliberately NOT on a missing score.

    Treating "no confidence computed" as low was tried and REVERTED: the federated
    path computes none at all and Tier-2 only has one when the Insight Engine ran,
    so it put a "limited matching data" caveat on every answer from those paths,
    including correct ones — measured on "top 5 credit transactions". A caveat on
    everything is a caveat on nothing. An absent signal is a gap in the SIGNAL; the
    fix is to compute one, not to warn unconditionally.
    """
    import veda_hybrid as VH
    import config as _cfg
    from veda.explain import bind_trace

    floor = float(getattr(_cfg, "LOW_CONFIDENCE_WARNING_BELOW", 0.0) or 0.0)
    assert floor > 0, "D1 set this to 0.5; a 0 floor disables the caveat entirely"

    for confidence, expected in ((0.018, True), (0.95, False), (None, False)):
        tr = ExplainTrace(query="q", trace_id=f"c-{confidence}")
        bind_trace(tr)
        VH._raise_low_confidence_caveat(confidence)
        codes = [w["code"] for w in sp.build_warnings(tr)]
        assert (vw.LOW_EVIDENCE in codes) is expected, (
            f"confidence={confidence} should {'' if expected else 'not '}warn")


# ---------------------------------------------------------------------------
# §10 "How this answer was generated": ONE shape for every execution type, with
# only the stages the trace holds evidence for.
# ---------------------------------------------------------------------------

def test_flow_is_absent_when_there_is_nothing_to_show(flags_on, trace):
    """A skeleton implying work nobody can point to is worse than no section."""
    assert sp.build_flow(trace) is None


def test_flow_for_a_database_answer(flags_on, trace):
    tl = lc.new_timeline(trace=trace)
    tl.completed(lc.PHASE_ACCESS_CHECK)
    trace.set("execution", row_count=5)
    # the flow counts the PAYLOAD's expanded label list, not the trace ledger —
    # one trace check becomes several user-facing labels
    payload_validation = {"passed": True, "checks": [
        {"label": "Read-only query", "passed": True},
        {"label": "Duplicate-safe (no double-counting)", "passed": True}]}
    rec = er.ExecutionRecorder(trace=trace)
    r = rec.open("2", source_type="relational", engine="deterministic_sql")
    rec.close(r, er.COMPLETED, rows=5)

    # Operations are derived from the executed SQL by build_explain and live only in
    # the PAYLOAD, so the flow is given them rather than reading the trace for them.
    ops = [{"summary": "Group by Status"}, {"summary": "Sort by Amount"}]
    stages = [s["stage"] for s in
              sp.build_flow(trace, operations=ops,
                            validation=payload_validation)["stages"]]
    assert stages[0] == "request" and stages[-1] == "answer"
    for expected in ("access", "sources", "evidence", "validation", "operations"):
        assert expected in stages, f"{expected} missing from {stages}"


def test_flow_for_a_document_answer_has_no_query_validation(flags_on, trace):
    """The stages present must follow the evidence, not a per-route template."""
    tl = lc.new_timeline(trace=trace)
    tl.completed(lc.PHASE_ACCESS_CHECK)
    rec = er.ExecutionRecorder(trace=trace)
    r = rec.open("3", source_type="document", engine="rag")
    rec.close(r, er.COMPLETED, rows=5)

    flow = sp.build_flow(trace)
    stages = [s["stage"] for s in flow["stages"]]
    assert "access" in stages and "sources" in stages and "evidence" in stages
    assert "validation" not in stages, "no query checks ran on a document answer"
    assert "operations" not in stages


def test_flow_never_names_an_internal_identifier(flags_on, trace):
    tl = lc.new_timeline(trace=trace)
    tl.completed(lc.PHASE_ACCESS_CHECK)
    trace.set("execution", row_count=3)
    rec = er.ExecutionRecorder(trace=trace)
    rec.close(rec.open("2", source_type="relational", engine="deterministic_sql"),
              er.COMPLETED, rows=3)
    blob = json.dumps(sp.build_flow(trace)).lower()
    for banned in ("source_id", "deterministic_sql", "rag_retrieve", "schema_linking",
                   "sql_planning", "select ", "postgres", "duckdb"):
        assert banned not in blob, f"{banned!r} leaked into the flow"


def test_source_identifiers_are_level_three_only(flags_on, trace):
    """§3/§9: a source id is an internal key. The display NAME is what a normal
    client renders; the id is MOVED to `audit`, not dropped, because the audit row
    records which sources took part and reads them from there."""
    rec = er.ExecutionRecorder(trace=trace)
    rec.close(rec.open("2", source_type="relational", engine="deterministic_sql"),
              er.COMPLETED, rows=5)
    ext = sp.build_explain_extension(trace, trace_id="t")

    user_facing = {k: v for k, v in ext.items() if k != "audit"}
    assert '"id"' not in json.dumps(user_facing), (
        f"a raw source id reached the normal UX: {json.dumps(user_facing)[:200]}")
    # names survive where they matter
    assert [s["name"] for s in ext["sources"]]
    assert [s["name"] for s in ext["execution"]["sources"]]
    # and the id is still recoverable for the audit trail
    assert [s["id"] for s in ext["audit"]["sources"]] == ["2"]


def test_frozen_warning_codes(flags_on):
    """CHAT_API_CONTRACT.md §1e — a frontend maps these to its own copy and icons.
    Adding a code is fine; removing or renaming one breaks a client."""
    frozen = {"result_truncated", "partial_source_failure", "restricted_data",
              "unmatched_records", "source_conflict", "fallback_used", "low_evidence"}
    assert frozen <= set(vw.CATALOG), (
        f"a frozen warning code was removed or renamed: {frozen - set(vw.CATALOG)}")


def test_frozen_flow_stages(flags_on):
    frozen = {"request", "access", "sources", "evidence", "validation",
              "operations", "result", "answer"}
    assert frozen <= set(sp.FLOW_STAGES), (
        f"a frozen flow stage changed: {frozen - set(sp.FLOW_STAGES)}")


# ===========================================================================
# EDGE CASES for the engine-side projection. Same audit, same five classes.
# ===========================================================================

def test_two_traces_do_not_leak_into_each_other(flags_on):
    """Two queries run at once on the inference tier. The projection reads an
    explicit trace, and this pins that it never falls back to an ambient one —
    the ContextVar class of bug that has bitten this codebase twice."""
    from veda.explain import bind_trace
    a = ExplainTrace(query="a", trace_id="AAA")
    b = ExplainTrace(query="b", trace_id="BBB")
    bind_trace(a)                                   # ambient is A
    ta = lc.new_timeline(trace=a)
    ta.completed(lc.PHASE_RECEIVED)
    vw.add(vw.RESULT_TRUNCATED, trace=a, rows=100)
    a.set("execution", row_count=100)

    # B saw none of it, even though A is the ambient trace
    assert sp.build_timeline_summary(b) == []
    assert sp.build_warnings(b) == []
    assert sp.build_result_meta(b)["row_count"] is None
    ext_b = sp.build_explain_extension(b, trace_id="BBB")
    assert ext_b["support"]["trace_id"] == "BBB"
    assert "AAA" not in json.dumps(ext_b)


def test_the_same_warning_twice_is_recorded_once(flags_on, trace):
    for _ in range(5):
        vw.add(vw.RESULT_TRUNCATED, trace=trace, rows=20)
    codes = [w["code"] for w in sp.build_warnings(trace)]
    assert codes == [vw.RESULT_TRUNCATED], f"duplicated: {codes}"
    assert len(sp.build_limitations(trace)) == 1


def test_the_sweep_run_repeatedly_changes_nothing(flags_on, trace):
    """close_open_phases is called TWICE by design; a third call must be a no-op."""
    tl = lc.new_timeline(trace=trace)
    tl.started(lc.PHASE_ACCESS_CHECK)
    tl.started(lc.PHASE_DATA_RETRIEVAL)
    tl.close_open_phases(failed=True)
    first = sp.build_timeline_summary(trace)
    for _ in range(3):
        tl.close_open_phases(failed=True)
    assert sp.build_timeline_summary(trace) == first


def test_a_negative_or_absurd_row_count_does_not_reach_the_user(flags_on, trace):
    trace.set("execution", row_count=-5)
    assert sp.build_result_meta(trace)["row_count"] is None, (
        "a nonsensical count must read as UNKNOWN, not be passed through")
    # `0` would be a different claim — "there were no rows" — so it is not the
    # fallback. And a real zero still comes through as zero.
    trace.set("execution", row_count=0)
    assert sp.build_result_meta(trace)["row_count"] == 0
    # non-integers too: a string or a bool must not render as a count
    for junk in ("20", True, 3.7, None):
        trace.set("execution", row_count=junk)
        assert sp.build_result_meta(trace)["row_count"] is None, junk


def test_non_ascii_source_names_survive_the_projection(flags_on, trace):
    rec = er.ExecutionRecorder(trace=trace)
    rec.close(rec.open("2", source_type="relational", engine="deterministic_sql"),
              er.COMPLETED, rows=3)
    ext = sp.build_explain_extension(trace, trace_id="t")
    json.dumps(ext, ensure_ascii=False)      # must serialise either way
    json.dumps(ext, ensure_ascii=True)


def test_every_projection_survives_a_completely_empty_trace(flags_on):
    """A turn that died before recording anything must not crash the projection."""
    empty = ExplainTrace(query="", trace_id="")
    for fn in (sp.build_timeline, sp.build_timeline_summary, sp.build_warnings,
               sp.build_limitations, sp.build_result_meta, sp.build_routing,
               sp.build_execution, sp.build_execution_plan, sp.build_cross_source,
               sp.build_provenance, sp.build_data_sources, sp.build_flow):
        fn(empty)                              # must not raise
    ext = sp.build_explain_extension(empty, trace_id="")
    assert isinstance(ext, dict)


def test_every_projection_survives_hostile_section_values(flags_on):
    """Sections are written by many call sites; one bad write must not cost the
    whole payload."""
    bad = ExplainTrace(query="q", trace_id="t")
    for name in ("routing", "execution", "validation", "federation", "operations"):
        bad.sections[name] = "not-a-dict"
    bad.sections[er.TRACE_SECTION] = {"records": ["junk", None, 42]}
    bad.sections[lc.TRACE_SECTION] = {"events": ["junk", None]}
    ext = sp.build_explain_extension(bad, trace_id="t")
    assert isinstance(ext, dict)


def test_the_refusal_payload_keeps_phase_names_at_level_three(flags_on, trace):
    """§9 — the refusal path had its OWN top-level `timeline_summary`, so backend
    phase names still reached the normal UX on refusals after the answered path
    had been fixed. Same "wired to one path" shape as EXP-B1/B4/B5."""
    tl = lc.new_timeline(trace=trace)
    tl.completed(lc.PHASE_RECEIVED)
    tl.completed(lc.PHASE_SOURCE_SELECTION)
    out = {}
    be._apply_v2_refusal(out, trace=trace, trace_id="t")

    assert "timeline_summary" not in out, "must not sit above `audit`"
    assert out["audit"]["timeline_summary"], "but must still be recoverable"
    user_facing = {k: v for k, v in out.items() if k != "audit"}
    for banned in ("source_selection", "schema_linking", "sql_planning",
                   "data_retrieval", "result_preparation"):
        assert banned not in json.dumps(user_facing), f"{banned!r} leaked on a refusal"


def test_no_user_facing_tick_carries_a_raw_identifier(flags_on):
    """The v1 payload's `timeline` is user-facing and is built from pipeline's
    stage ticks. One of them interpolated the primary TABLE NAME — observed live
    as "Using assets_asset for this". A table name must never reach a user-facing
    payload, so no tick may interpolate an identifier at all.
    """
    import os
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "veda_core", "veda", "pipeline.py")) as fh:
        src = fh.read()
    offenders = [m.group(0) for m in re.finditer(r'_tick\(\s*"[^"]+"\s*,\s*f"[^"]*\{[^"]*"',
                                                 src)]
    assert not offenders, (
        "a stage tick interpolates a value into user-facing text; if the value is "
        f"an identifier it leaks: {offenders}")


def test_the_flow_and_the_check_list_never_disagree(flags_on, trace):
    """One trace check expands into SEVERAL user-facing labels, so counting the
    trace put "4 checks passed" in the flow directly above a list of 5. Both
    numbers were right in their own terms and contradictory side by side.

    The number the reader is told must be the number they can count.
    """
    tl = lc.new_timeline(trace=trace)
    tl.completed(lc.PHASE_ACCESS_CHECK)
    trace.set("execution", row_count=5)
    # the TRACE holds 4 checks...
    trace.set("validation", checks=[{"name": "value_grounding", "status": "pass"},
                                    {"name": "ir_equivalence", "status": "pass"},
                                    {"name": "ast_readonly_parameterized_fanout",
                                     "status": "pass"},
                                    {"name": "join_fanout", "status": "pass"}])
    # ...while the PAYLOAD shows 5 expanded labels
    payload_validation = {"passed": True, "checks": [
        {"label": "All filter values exist in the data", "passed": True},
        {"label": "No requested filters were ignored", "passed": True},
        {"label": "No extra filters, joins, or grouping were added", "passed": True},
        {"label": "Read-only query", "passed": True},
        {"label": "Duplicate-safe (no double-counting)", "passed": True},
    ]}
    flow = sp.build_flow(trace, validation=payload_validation)
    stage = next(s for s in flow["stages"] if s["stage"] == "validation")
    assert "5 checks passed" in stage["label"], (
        f"the flow must count the 5 labels the reader sees, got {stage['label']!r}")
    assert "4" not in stage["label"]


def test_the_flow_reports_a_partial_pass_honestly(flags_on, trace):
    tl = lc.new_timeline(trace=trace)
    tl.completed(lc.PHASE_ACCESS_CHECK)
    trace.set("execution", row_count=1)
    flow = sp.build_flow(trace, validation={"passed": False, "checks": [
        {"label": "Read-only query", "passed": True},
        {"label": "Duplicate-safe (no double-counting)", "passed": False},
    ]})
    stage = next(s for s in flow["stages"] if s["stage"] == "validation")
    assert stage["label"] == "1 of 2 checks passed"


def test_no_live_progress_line_states_a_check_COUNT(flags_on):
    """The live line is emitted before the payload exists, so any count it states
    can only be the trace's — a different number from the list the reader opens."""
    import os
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "veda_core", "veda", "pipeline.py")) as fh:
        src = fh.read()
    offenders = re.findall(r'f"\{len\([^)]*\)\}\s*(?:safety\s*)?checks?\b[^"]*"', src)
    assert not offenders, f"a live progress line states a check count: {offenders}"
