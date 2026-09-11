"""Tests for the four-step live thinking UX.

  apps/chat/thinking_steps.py    :: ThinkingStepTracker  (mapping, timing, states)
  apps/chat/thinking_context.py  :: ThinkingContext      (sentences, details)
  veda_core/veda/narrator.py     :: start / validate     (async SLM narrator)

Pure python — no Django, no DB, no network, no SLM. The two api-tier modules are
deliberately dependency-free for exactly this reason (same rationale as
apps/chat/turn_events.py's own docstring).
"""
import importlib.util
import os
import sys
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, "veda_core"))


def _load(name, relpath):
    """Load an api-tier module standalone, without importing Django.

    Registered under a PRIVATE name (`_standalone_*`), never the real dotted path:
    inserting `apps.chat.thinking_steps` into sys.modules poisoned Django's own
    import of it, so this suite and tests/test_query_governance.py could not run in
    the same session. `thinking_context` imports its sibling as
    `from . import thinking_steps`, so the package alias below is what satisfies
    that — a throwaway package object, not a shadow of the real one.
    """
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# A private package so `from . import thinking_steps` resolves inside the loaded
# module without touching the real `apps.chat` package.
import types  # noqa: E402

_pkg = types.ModuleType("_standalone_chat")
_pkg.__path__ = [os.path.join(_ROOT, "apps", "chat")]
sys.modules["_standalone_chat"] = _pkg

ts = _load("_standalone_chat.thinking_steps", "apps/chat/thinking_steps.py")
tc = _load("_standalone_chat.thinking_context", "apps/chat/thinking_context.py")
from veda import narrator as nar  # noqa: E402



def rows(step: dict, kind: str | None = None) -> list:
    """The evidence rows inside a rendered step, optionally of one type.

    The contract folds authorization/validation checks and the semantic evidence
    into ONE ordered `details` list, so a test asks for what it means rather than
    which internal bucket the row came from.
    """
    out = step.get("details") or []
    return [r for r in out if kind is None or r.get("type") == kind]


def labels(step: dict, kind: str | None = None) -> list:
    return [r.get("label") for r in rows(step, kind)]

def ev(phase, message="", status=None, ts_ms=None, **details):
    p = {"phase": phase, "message": message}
    if status:
        p["status"] = status
    if ts_ms is not None:
        p["timestamp_ms"] = ts_ms
    if details:
        p["details"] = details
    return p


# ── UX: the four fixed steps ─────────────────────────────────────────────────
def test_exactly_four_steps_in_a_fixed_order():
    assert ts.STEP_ORDER == ("understanding", "finding", "analyzing", "preparing")
    assert [ts.STEP_TITLES[k] for k in ts.STEP_ORDER] == [
        "Understanding your request",
        "Finding the right information",
        "Analyzing the information",
        "Preparing your answer",
    ]


def test_no_step_is_engine_or_output_shaped():
    """The four names must read correctly for SQL, documents, NoSQL, charts and
    summaries alike — so none may name an engine or an output type."""
    blob = " ".join(ts.STEP_TITLES.values()).lower()
    for leak in ("sql", "document", "rag", "nosql", "chart", "table", "summary",
                 "source", "authorization", "permission", "visualization"):
        assert leak not in blob


def test_every_real_emitted_phase_is_mapped():
    """Every phase the repository actually emits must map to a step, or it silently
    disappears from the UI. This list is the measured inventory, not a guess."""
    real = {
        # chatbot/nodes.py
        "supervisor_classify", "supervisor_followup",
        # veda_hybrid.py
        "classify", "route", "sql_probe", "decompose", "sub_query", "tier2",
        "rag", "hybrid", "nosql", "nosql_build", "answer",
        # query/rag_layer.py
        "rag_retrieve", "rag_synthesize", "hybrid_retrieve", "hybrid_synthesize",
        # query/lg_nodes.py
        "tier2_intent", "tier2_entity", "tier2_columns", "tier2_filters",
        "tier2_assemble",
        # veda/pipeline.py _tick
        "schema_linking", "sql_planning", "output",
        # apps/chat/services.py
        "visualization_prep",
        # veda/lifecycle.py
        "received", "understanding", "access_check", "source_selection",
        "execution_plan", "data_retrieval", "cross_source_processing",
        "validation", "result_preparation", "completed",
    }
    missing = sorted(real - set(ts.PHASE_TO_STEP))
    assert not missing, f"unmapped phases would be invisible: {missing}"


def test_unknown_phase_is_ignored_not_shown():
    """A newly-added internal phase must not leak its raw name into the UI."""
    t = ts.ThinkingStepTracker()
    assert t.consume(ev("some_brand_new_internal_phase")) is False
    assert t.current_step() is None


# ── UX: progression and state ────────────────────────────────────────────────
def test_steps_progress_and_expose_active_completed_pending():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    snap = {s["id"]: s["state"] for s in t.snapshot()}
    assert snap == {"understanding": "active", "finding": "pending",
                    "analyzing": "pending", "preparing": "pending"}

    t.consume(ev("source_selection", status="completed"))
    snap = {s["id"]: s["state"] for s in t.snapshot()}
    assert snap["understanding"] == "completed"
    assert snap["finding"] == "active"

    t.consume(ev("sql_planning"))
    t.consume(ev("answer"))
    t.finish()
    snap = {s["id"]: s["state"] for s in t.snapshot()}
    assert all(v == "completed" for v in snap.values()), snap


def test_steps_never_go_backwards():
    """Tier-2 genuinely re-does intent work late in the turn. A finished step must
    not reopen — that reads as a bug."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("source_selection", status="completed"))
    t.consume(ev("sql_planning"))
    assert t.current_step() == "analyzing"
    t.consume(ev("tier2_intent"))          # maps to analyzing by design
    t.consume(ev("supervisor_classify"))   # maps to UNDERSTANDING — already closed
    assert t.current_step() == "analyzing", "must not reopen a finished step"
    assert t.steps["understanding"].state == "completed"


def test_a_step_that_never_ran_stays_pending_with_no_duration():
    """A null duration renders as "—". Reporting 0 s would imply it ran instantly."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    snap = {s["id"]: s for s in t.snapshot()}
    assert snap["preparing"]["state"] == "pending"
    assert snap["preparing"]["duration_ms"] is None


def test_failed_status_marks_the_step_failed():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("source_selection", status="failed"))
    assert t.steps["finding"].state == "failed"


# ── Timing: real, never simulated ────────────────────────────────────────────
def test_duration_uses_engine_timestamps_when_present():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed", ts_ms=1_000_000))
    t.consume(ev("source_selection", status="completed", ts_ms=1_000_420))
    t.consume(ev("sql_planning", ts_ms=1_002_250))
    assert t.steps["understanding"].duration_ms() == 420
    assert t.steps["finding"].duration_ms() == 1830


def test_active_duration_is_live_then_frozen_on_completion():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    first = t.steps["understanding"].duration_ms()
    time.sleep(0.05)
    second = t.steps["understanding"].duration_ms()
    assert second >= first, "an active step's elapsed time must advance"

    t.finish()
    frozen = t.steps["understanding"].duration_ms()
    time.sleep(0.03)
    assert t.steps["understanding"].duration_ms() == frozen, "must freeze on completion"


def test_total_duration_spans_the_whole_turn():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed", ts_ms=5_000_000))
    t.consume(ev("answer", ts_ms=5_005_200))
    t.finish()
    assert t.total_duration_ms() >= 5200


def test_narration_never_touches_a_timestamp():
    """SLM latency must not land inside any step duration — narration only ever
    replaces TEXT."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed", ts_ms=2_000_000))
    t.consume(ev("sql_planning", ts_ms=2_001_000))
    before = [s["duration_ms"] for s in t.snapshot()]
    t.set_context("analyzing", "Comparing values across months.", from_narrator=True)
    after = [s["duration_ms"] for s in t.snapshot()]
    assert before == after


# ── Authorization: a timed sub-check inside Finding ──────────────────────────
def test_access_check_is_a_sub_check_of_finding_not_a_top_level_step():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("access_check", status="started", ts_ms=3_000_000))
    t.consume(ev("access_check", status="completed", ts_ms=3_000_180))
    assert "access" not in [s["id"] for s in t.snapshot()]
    checks = t.steps["finding"].sub_checks
    assert len(checks) == 1
    assert checks[0]["label"] == "Checking access permissions"
    assert checks[0]["state"] == "completed"
    assert checks[0]["duration_ms"] == 180


def test_access_sub_check_does_not_complete_the_finding_step():
    """Otherwise the access check would look like the whole of "Finding"."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("access_check", status="completed"))
    assert t.steps["finding"].state == "active"


@pytest.mark.parametrize("status,expect", [
    ("completed", "You have permission to access the required information"),
    ("warning", "Access is limited for some information."),
    ("failed", "The required information isn't available with your current access."),
])
def test_access_copy_per_outcome(status, expect):
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("access_check", status=status))
    msg = t.steps["finding"].sub_checks[0]["message"]
    assert expect in msg


def test_access_copy_leaks_no_authorization_internals():
    for status in ("completed", "warning", "failed", "active"):
        msg = (ts._ACCESS_COPY.get(status) or "").lower()
        for leak in ("role", "policy", "grant", "rbac", "permission id", "token",
                     "db:", "scope", "resource_path"):
            assert leak not in msg


# ── Contextual sentences: confirmed facts only ───────────────────────────────
def test_sentences_use_only_confirmed_facts():
    c = tc.ThinkingContext()
    c.absorb(ev("understanding", status="completed",
                intent="ranking", period="2026-04-01 to 2026-06-30"))
    assert c.sentence("understanding") == (
        "You're asking for a ranking, for 2026-04-01 to 2026-06-30.")


def test_sentence_omits_what_is_not_known_rather_than_inventing():
    c = tc.ThinkingContext()
    assert c.sentence("understanding") == "Working out what you're asking for."
    assert c.sentence("finding") == "Looking for the information that answers this."
    assert c.details("understanding") == [], "no facts must mean no fabricated detail"


def test_multi_source_sentence_counts_sources_without_naming_them():
    c = tc.ThinkingContext()
    c.absorb(ev("source_selection", status="completed", source_count=3))
    s = c.sentence("finding")
    assert "3 sources" in s
    for leak in ("homzhub", "db:", "postgres", "parquet"):
        assert leak not in s.lower()


def test_details_never_carry_engine_or_schema_vocabulary():
    """NOTE the banned list deliberately does NOT contain the bare words "table" or
    "column": "Result table" is a legitimate OUTPUT name the product asked for. The
    leak that matters is a schema identifier or an engine name, so the check is on
    the displayed VALUES and targets those specifically."""
    c = tc.ThinkingContext()
    c.absorb(ev("understanding", status="completed", intent="count", grouped=True,
                period="Q2"))
    c.absorb(ev("source_selection", status="completed", source_count=2))
    c.absorb(ev("data_retrieval", status="completed", row_count=12))

    shown = []
    for k in ts.STEP_ORDER:
        for v in [r.get("label") for r in (c.details(k) or [])]:
            shown.extend(v if isinstance(v, list) else [v])
    blob = " ".join(str(v) for v in shown).lower()

    for leak in ("sql", "schema", "rag", "tier2", "nosql", "postgres", "duckdb",
                 "select ", " from ", "anchor", "db:", "assets_", "accounts_",
                 "table name", "column name", "deterministic_sql"):
        assert leak not in blob, f"{leak!r} leaked into displayed detail"
    # ...and nothing that looks like a schema identifier
    import re
    assert not re.search(r"\b[a-z]+_[a-z_]{4,}\b", blob), (
        f"an identifier-shaped token reached the user: {blob!r}")


# ── Output types: same four steps adapt ──────────────────────────────────────
@pytest.mark.parametrize("output,expect", [
    ("chart", "a chart"),
    ("table", "a table"),
    ("summary", "a summary"),
    ("chart+summary", "a chart with a short summary"),
])
def test_preparing_sentence_adapts_to_the_requested_output(output, expect):
    c = tc.ThinkingContext()
    c.absorb(ev("result_preparation", status="completed", output=output))
    assert expect in c.sentence("preparing")


def test_chart_output_adds_a_charting_operation_to_analyzing():
    c = tc.ThinkingContext()
    c.absorb(ev("understanding", status="completed", intent="trend", output="chart"))
    s = c.sentence("analyzing").lower()
    assert "over time" in s and "charted" in s


def test_output_shape_never_becomes_its_own_step():
    """A chart or summary must not create a fifth top-level step."""
    t = ts.ThinkingStepTracker()
    for ph in ("received", "source_selection", "sql_planning",
               "visualization_prep", "answer"):
        t.consume(ev(ph, status="completed"))
    assert [s["id"] for s in t.snapshot()] == list(ts.STEP_ORDER)


@pytest.mark.parametrize("phases,expected_step", [
    (["rag", "rag_retrieve"], "finding"),          # document question
    (["rag_synthesize"], "analyzing"),
    (["nosql", "nosql_build"], "analyzing"),       # nosql_build is the planning half
    (["hybrid", "hybrid_retrieve"], "finding"),
    (["hybrid_synthesize"], "analyzing"),
    (["cross_source_processing"], "analyzing"),    # federated
])
def test_every_engine_path_lands_in_the_four_steps(phases, expected_step):
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    for ph in phases:
        t.consume(ev(ph))
    assert t.current_step() == expected_step
    assert [s["id"] for s in t.snapshot()] == list(ts.STEP_ORDER)


# ── SLM narrator ─────────────────────────────────────────────────────────────
_FACTS = {"intent": "trend", "time_period": "Q2 2026", "requested_output": "chart"}


def test_valid_narration_is_accepted():
    got = nar.validate("Reviewing the trend over Q2 2026.", _FACTS)
    assert got == "Reviewing the trend over Q2 2026."


def test_narration_missing_full_stop_is_completed():
    assert nar.validate("Reviewing the trend", _FACTS).endswith(".")


@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "Running SELECT count(*) from sales.",          # SQL
    "Querying the sales table.",                     # schema vocabulary
    "Tier-2 is planning the query.",                 # internal component
    "Using RAG to search the documents.",            # engine name
    "Checking your role and policy grants.",         # authorization internals
    "First I will compare. Then I will rank.",       # multi-sentence
    'Comparing "sales" across regions.',             # quoting
    "Comparing sales across 14 regions.",            # invented number
    "x" * 200,                                        # oversized
])
def test_invalid_narration_is_rejected(bad):
    assert nar.validate(bad, _FACTS) is None


def test_number_present_in_the_facts_is_allowed():
    facts = {"source_count": 3}
    assert nar.validate("Gathering information across 3 sources.", facts)


def test_ungrounded_content_word_is_rejected():
    """A narrator that introduces a concept the facts never mentioned is inventing."""
    assert nar.validate("Reviewing the quarterly headcount attrition.", _FACTS) is None


def test_narrator_disabled_returns_an_inert_handle(monkeypatch):
    import config
    monkeypatch.setattr(config, "EXPLAIN_NARRATOR_ENABLED", False, raising=False)
    calls = []
    h = nar.start(_FACTS, "analyzing", lambda *a: calls.append(a),
                  slm_call=lambda u, s: "should never run")
    time.sleep(0.05)
    assert calls == [] and h.result is None


def test_narrator_delivers_when_valid(monkeypatch):
    import config
    monkeypatch.setattr(config, "EXPLAIN_NARRATOR_ENABLED", True, raising=False)
    got = []
    h = nar.start(_FACTS, "analyzing", lambda step, s: got.append((step, s)),
                  slm_call=lambda u, s: "Reviewing the trend over Q2 2026.")
    for _ in range(100):
        if got:
            break
        time.sleep(0.02)
    assert got and got[0][0] == "analyzing"
    assert h.result == "Reviewing the trend over Q2 2026."


@pytest.mark.parametrize("slm", [
    lambda u, s: (_ for _ in ()).throw(RuntimeError("SLM unreachable")),   # failure
    lambda u, s: (time.sleep(0.4) or "Reviewing the trend."),              # slow
    lambda u, s: "SELECT * FROM sales",                                     # invalid
    lambda u, s: None,                                                      # empty
])
def test_narrator_failure_modes_are_all_silent(monkeypatch, slm):
    import config
    monkeypatch.setattr(config, "EXPLAIN_NARRATOR_ENABLED", True, raising=False)
    got = []
    nar.start(_FACTS, "analyzing", lambda *a: got.append(a), slm_call=slm)
    time.sleep(0.15)
    assert got == [], "a failing/slow/invalid narration must deliver nothing"


def test_cancelled_narration_is_discarded_not_delivered(monkeypatch):
    """The answer always wins the race — a narration that arrives after the turn
    ended must be dropped, never shown."""
    import config
    monkeypatch.setattr(config, "EXPLAIN_NARRATOR_ENABLED", True, raising=False)
    got = []

    def slow(u, s):
        time.sleep(0.2)
        return "Reviewing the trend over Q2 2026."

    h = nar.start(_FACTS, "analyzing", lambda *a: got.append(a), slm_call=slow)
    h.cancel()                       # the answer finished first
    time.sleep(0.35)
    assert got == []


def test_cancel_does_not_block():
    """cancel() must never join the thread — waiting is the one thing this may not do."""
    import config
    config.EXPLAIN_NARRATOR_ENABLED = True
    try:
        h = nar.start(_FACTS, "analyzing", lambda *a: None,
                      slm_call=lambda u, s: (time.sleep(1.0) or "Reviewing."))
        t0 = time.perf_counter()
        h.cancel()
        assert (time.perf_counter() - t0) < 0.05, "cancel() must return immediately"
    finally:
        config.EXPLAIN_NARRATOR_ENABLED = False


def test_narrator_only_ever_sees_display_cleared_facts():
    c = tc.ThinkingContext()
    c.absorb(ev("understanding", status="completed", intent="count", period="Q2"))
    c.absorb(ev("source_selection", status="completed", source_count=2))
    facts = c.narrator_facts()
    assert set(facts) <= {"intent", "time_period", "breakdown", "requested_output",
                          "source_count", "completed_operations"}
    blob = repr(facts).lower()
    for leak in ("sql", "table", "column", "schema", "homzhub", "select"):
        assert leak not in blob


# ── Regression: the legacy contract ──────────────────────────────────────────
def test_step_payload_is_additive_only():
    """The `steps` block is extra; `phase` and `message` keep their exact meaning so
    an existing client is unaffected."""
    t = ts.ThinkingStepTracker()
    payload = ev("source_selection", "Found relevant data", status="completed")
    t.consume(payload)
    payload["steps"] = t.as_payload()
    assert payload["phase"] == "source_selection"
    assert payload["message"] == "Found relevant data"
    # `steps` reveals PROGRESSIVELY — a step appears when the turn reaches it — so
    # its length is the progress so far, not the size of the model. `total_steps`
    # is the denominator a client needs for "step 2 of 4".
    assert payload["steps"]["total_steps"] == 4
    assert len(payload["steps"]["steps"]) == 2
    assert [s["id"] for s in payload["steps"]["steps"]] == ["understanding", "finding"]


def test_tracker_never_raises_on_malformed_input():
    t = ts.ThinkingStepTracker()
    for bad in ({}, {"phase": None}, {"phase": 123}, {"phase": "received",
                 "timestamp_ms": "nonsense"}, {"details": "not-a-dict"}):
        t.consume(bad)          # must not raise
    t.finish()
    assert len(t.snapshot()) == 4


def test_context_never_raises_on_malformed_input():
    c = tc.ThinkingContext()
    for bad in ({}, {"details": None}, {"details": "x"}, {"phase": None}):
        c.absorb(bad)
    for k in ts.STEP_ORDER:
        assert isinstance(c.sentence(k), str) and c.sentence(k)
        assert isinstance(c.details(k), list)


def test_sub_check_attaches_to_its_mapped_step_not_the_current_one():
    """Observed live: the access check's `completed` arrives AFTER validation, so
    routing it through the monotonic guard filed the start under Finding and the
    completion under Preparing — the same check shown twice, once unresolved, and
    authorization displayed under the wrong heading."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("access_check", status="started"))
    t.consume(ev("source_selection", status="completed"))
    t.consume(ev("sql_planning"))
    t.consume(ev("validation", status="completed"))
    t.consume(ev("answer"))
    t.consume(ev("access_check", status="completed"))   # late, as it really arrives
    t.finish()

    by_step = {s["id"]: rows(s, ts.DETAIL_ACCESS) for s in t.snapshot()}
    access = [c for cs in by_step.values() for c in cs]
    assert len(access) == 1, "the access check must appear exactly once"
    assert access[0]["state"] == "completed"
    assert [c["type"] for c in by_step["finding"]] == [ts.DETAIL_ACCESS]
    assert not [c for c in by_step["preparing"] if c["kind"] == "access"], (
        "authorization must never be shown under Preparing")


def test_late_sub_check_does_not_reopen_a_finished_step():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("source_selection", status="completed"))
    t.consume(ev("answer"))
    t.finish()
    frozen = t.steps["finding"].duration_ms()
    t.consume(ev("access_check", status="completed"))
    assert t.steps["finding"].duration_ms() == frozen
    assert t.steps["finding"].state == "completed"


# ── period humanising ────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expect", [
    ("2024-01-01T00:00:00 to 2024-12-31T23:59:59", "2024"),
    ("2026-04-01T00:00:00 to 2026-04-30T23:59:59", "April 2026"),
    ("2026-03-05T00:00:00 to 2026-03-05T23:59:59", "2026-03-05"),
    ("2026-01-15T00:00:00 to 2026-02-20T00:00:00", "2026-01-15 to 2026-02-20"),
])
def test_period_is_humanised(raw, expect):
    assert tc._humanize_period(raw) == expect


def test_unparseable_period_is_passed_through_not_guessed():
    assert tc._humanize_period("last quarter") == "last quarter"


# ── intent recovered from the executed operations ────────────────────────────
def test_intent_is_recovered_from_operations_when_the_live_stream_lacked_it():
    """"how many …" is detected by the deterministic fast path, not the aggregation
    grammar, so `aggregation.op` is None and intent was absent mid-flight."""
    c = tc.ThinkingContext()
    assert c.intent is None
    c.absorb_explain({"operations": [{"type": "count", "summary": "Count records"}]})
    assert c.intent == "count"
    assert "a count" in c.sentence("understanding")


def test_ranking_intent_inferred_from_sort_plus_limit():
    c = tc.ThinkingContext()
    c.absorb_explain({"operations": [{"type": "sort", "summary": "Sort by x"},
                                     {"type": "limit", "summary": "Return top 5"}]})
    assert c.intent == "ranking"


def test_group_operation_sets_the_breakdown_flag():
    c = tc.ThinkingContext()
    c.absorb_explain({"operations": [{"type": "group", "summary": "Group by status"}]})
    assert c.grouped is True


def test_live_intent_is_not_overwritten_by_the_late_recovery():
    c = tc.ThinkingContext()
    c.absorb(ev("understanding", status="completed", intent="ranking"))
    c.absorb_explain({"operations": [{"type": "count", "summary": "Count records"}]})
    assert c.intent == "ranking", "a confirmed live fact must win over the inference"


def test_partial_month_is_not_labelled_as_the_whole_month():
    """"1 Apr to 15 Apr" is NOT "April 2026" — that would report a different period
    from the one the user asked for."""
    assert tc._humanize_period(
        "2026-04-01T00:00:00 to 2026-04-15T23:59:59") == "2026-04-01 to 2026-04-15"


def test_measured_access_duration_wins_over_the_event_gap():
    """The engine's access_check span is "when access could be confirmed", not how
    long checking took — it reported 26.6 s live for sub-millisecond work. A duration
    measured where the RBAC resolution actually happens must override it."""
    t = ts.ThinkingStepTracker()
    t.set_sub_check_duration(ts.STEP_FINDING, "access", 18)
    t.consume(ev("received", status="completed", ts_ms=1_000_000))
    t.consume(ev("access_check", status="started", ts_ms=1_000_100))
    t.consume(ev("access_check", status="completed", ts_ms=1_026_700))  # 26.6s gap
    check = t.steps["finding"].sub_checks[0]
    assert check["duration_ms"] == 18, "the measured value must not be overwritten"
    assert check["state"] == "completed"


def test_event_gap_is_used_when_nothing_was_measured():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed", ts_ms=2_000_000))
    t.consume(ev("access_check", status="started", ts_ms=2_000_000))
    t.consume(ev("access_check", status="completed", ts_ms=2_000_180))
    assert t.steps["finding"].sub_checks[0]["duration_ms"] == 180


def test_sub_check_private_bookkeeping_is_not_exposed():
    t = ts.ThinkingStepTracker()
    t.set_sub_check_duration(ts.STEP_FINDING, "access", 18)
    t.consume(ev("access_check", status="completed"))
    for s in t.snapshot():
        for c in rows(s):
            assert not [k for k in c if k.startswith("_")], f"leaked internals: {c}"


def test_access_sub_check_advances_the_step_it_opens():
    """Measured live: the access check opened Finding but left `_current` on
    Understanding, so Understanding accrued 71 s of routing time."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("supervisor_classify", ts_ms=3_000_000))
    t.consume(ev("access_check", status="started", ts_ms=3_000_200))
    t.consume(ev("route", ts_ms=3_071_000))          # routing is genuinely slow
    assert t.steps["understanding"].duration_ms() == 200, (
        "Understanding must close when Finding opens, not 71 s later")
    assert t.current_step() == "finding"


def test_narrator_thread_carries_the_trace_so_its_call_is_visible(monkeypatch):
    """A fresh thread starts with an empty contextvars context, so without an
    explicit hand-off the narrator's SLM call records nothing and burns latency and
    tokens invisibly."""
    import config
    from veda.explain import ExplainTrace, use_trace
    monkeypatch.setattr(config, "EXPLAIN_NARRATOR_ENABLED", True, raising=False)

    tr = ExplainTrace(query="q", trace_id="narrtrace")
    seen = {}

    def fake_slm(user, system):
        from veda.explain import current_trace
        seen["trace_id"] = getattr(current_trace(), "trace_id", None)
        return "Reviewing the trend over Q2 2026."

    with use_trace(tr):
        nar.start({"intent": "trend", "time_period": "Q2 2026"}, "analyzing",
                  lambda *a: None, slm_call=fake_slm)
    for _ in range(100):
        if seen:
            break
        time.sleep(0.02)
    assert seen.get("trace_id") == "narrtrace", (
        "the narrator thread must see the query's trace, or its SLM call is invisible")


# ── validator calibration (learned from real SLM output) ────────────────────
@pytest.mark.parametrize("text,facts", [
    # These are VERBATIM outputs the real SLM produced. The first version of the
    # validator rejected 100% of them over ordinary words like "entire" and
    # "occurrences" — a filter that rejects everything is a disabled feature, not a
    # safety property.
    ("Analyzing data for the entire year of 2024.",
     {"period": "2024-01-01T00:00:00 to 2024-12-31T23:59:59"}),
    ("Analyzing data to count occurrences in 2024.",
     {"intent": "count", "grouped": True, "period": "2024"}),
])
def test_real_slm_output_is_accepted(text, facts):
    assert nar.validate(text, facts) == text


@pytest.mark.parametrize("text,facts", [
    # A DOMAIN noun absent from the facts is the dangerous case — a user can act on
    # an invented metric. Still rejected.
    ("Comparing headcount attrition across departments.", {"intent": "count"}),
    ("Reviewing the monthly revenue trend.", {"intent": "trend", "period": "2026"}),
    ("Querying the assets_leaselisting table.", {"intent": "count"}),
    ("Found 4821 matching records.", {"intent": "count"}),
])
def test_invented_domain_nouns_and_leaks_still_rejected(text, facts):
    assert nar.validate(text, facts) is None


def test_free_vocabulary_contains_no_business_metrics():
    """The general-English allowlist must never acquire a domain noun, or the
    invented-metric protection silently disappears."""
    for domain in ("revenue", "sales", "profit", "headcount", "attrition", "churn",
                   "rent", "listing", "asset", "invoice", "vendor", "customer",
                   "negotiation", "maintenance", "amenity", "tenant"):
        assert domain not in nar._FREE, f"{domain!r} must not be freely usable"


# ── ORDERING: no thinking event may ever follow the answer ──────────────────
def test_late_narration_is_discarded_not_emitted_after_the_answer():
    """The hard requirement: no explainability operation may delay, reorder or follow
    a terminal answer event.

    This reproduces the real consumer-loop contract in
    ``ConversationQueryService._run_streamed``: a worker thread enqueues progress
    items and, in its `finally`, a ("done", None) sentinel. The consumer yields
    thinking events until it sees that sentinel, and ONLY THEN does the caller emit
    content. So a narration that arrives after the sentinel is left in the queue and
    silently dropped — which is exactly the specified behaviour ("if the answer
    finishes before the SLM, discard the pending output"), and is a property of the
    control flow rather than of timing.
    """
    import queue

    q: "queue.Queue" = queue.Queue()
    q.put(("thinking", {"phase": "received"}))
    q.put(("thinking", {"phase": "source_selection"}))
    q.put(("result", {"ok": True}))
    q.put(("done", None))
    q.put(("thinking", {"phase": "narration"}))   # loses the race

    emitted, result = [], None
    while True:                                   # mirrors _run_streamed's loop
        kind, payload = q.get()
        if kind == "done":
            break
        if kind == "result":
            result = payload
        else:
            emitted.append(("thinking", payload["phase"]))
    emitted.append(("content", "the answer"))     # only reachable after the loop

    assert result == {"ok": True}
    first_content = [i for i, (k, _) in enumerate(emitted) if k == "content"][0]
    assert not [k for k, _ in emitted[first_content + 1:] if k == "thinking"]
    assert "narration" not in [v for _, v in emitted], "late narration must be dropped"


def test_terminal_cancel_prevents_delivery_even_mid_call(monkeypatch):
    """The other half of the guarantee: pipeline._done cancels every in-flight
    narration, and the narrator re-checks `cancelled` after the SLM returns and
    before invoking the callback."""
    import config
    monkeypatch.setattr(config, "EXPLAIN_NARRATOR_ENABLED", True, raising=False)
    delivered, started = [], threading_event()

    def slow(u, s):
        started.set()
        time.sleep(0.25)
        return "Analyzing data for the entire year of 2024."

    h = nar.start({"period": "2024"}, "analyzing",
                  lambda *a: delivered.append(a), slm_call=slow)
    started.wait(2)
    h.cancel()                       # the answer landed while the SLM was mid-flight
    time.sleep(0.4)
    assert delivered == []


def threading_event():
    import threading
    return threading.Event()


# ── multi-query findings ─────────────────────────────────────────────────────
def test_operations_are_plain_language_not_sql_vocabulary():
    """build_explain's own summaries read as SQL ("Group by Listing Status",
    "Return top 100"). Fine in the technical payload where they already ship; the
    live progress UI must not use that vocabulary. Caught on the `grouped` query in
    the 12-query run."""
    c = tc.ThinkingContext()
    c.absorb_explain({"operations": [
        {"type": "count", "summary": "Count distinct records"},
        {"type": "group", "summary": "Group by Listing Status"},
        {"type": "sort", "summary": "Sort by Transaction Identifier (highest first)"},
        {"type": "limit", "summary": "Return top 100"}]})
    assert c.operations == ["Counting the records",
                            "Broken down by Listing Status",
                            "Ordered by Transaction Identifier",
                            "Limited to the top results"]
    blob = " ".join(c.operations).lower()
    for sqlish in ("group by", "sort by", "select", "return top", "distinct records"):
        assert sqlish not in blob


def test_unmapped_operation_keeps_the_fact_but_strips_the_sql_verb():
    c = tc.ThinkingContext()
    c.absorb_explain({"operations": [{"type": "somethingnew",
                                      "summary": "Group by Region"}]})
    assert c.operations == ["Region"], "the fact survives, the SQL verb does not"


def test_measured_access_check_is_not_left_spinning():
    """A measured duration proves the api tier already completed the RBAC
    resolution. Leaving it `active` showed a permanently spinning access check on
    the smalltalk path, where no engine access_check event ever arrives."""
    t = ts.ThinkingStepTracker()
    t.set_sub_check_duration(ts.STEP_FINDING, "access", 35)
    c = t.steps[ts.STEP_FINDING].sub_checks[0]
    assert c["state"] == "completed"
    assert c["message"] == ts._ACCESS_COPY["completed"]


def test_sub_checks_are_dropped_from_a_step_that_never_ran():
    """Smalltalk bypasses the engine, so "Finding" never starts — but the api tier
    still measured its RBAC resolution and attached a check to it. A check displayed
    under a step that did not happen is wrong."""
    t = ts.ThinkingStepTracker()
    t.set_sub_check_duration(ts.STEP_FINDING, "access", 35)
    t.consume(ev("supervisor_classify"))          # smalltalk: understanding only
    t.finish()
    snap = {s["id"]: s for s in t.snapshot()}
    assert snap["finding"]["state"] == "pending"
    assert snap["finding"]["duration_ms"] is None
    assert not rows(snap["finding"]), "must not show a check that never ran"


def test_sub_checks_are_kept_on_a_step_that_did_run():
    t = ts.ThinkingStepTracker()
    t.set_sub_check_duration(ts.STEP_FINDING, "access", 20)
    t.consume(ev("received", status="completed"))
    t.consume(ev("source_selection", status="completed"))
    t.consume(ev("answer"))
    t.finish()
    snap = {s["id"]: s for s in t.snapshot()}
    assert len(rows(snap["finding"], ts.DETAIL_ACCESS)) == 1
    assert rows(snap["finding"], ts.DETAIL_ACCESS)[0]["duration_ms"] == 20


# ---------------------------------------------------------------------------
# A turn that produced NO answer must not render as four green ticks.
#
# Found live on the verified-cache lane: 2 of 3 cache hits were stopped by the
# intent/SQL alignment gate, and the user got four completed steps — including
# "Checking the result is complete and safe · passed" — directly above a reply
# saying the question could not be answered. Two separate causes, one per test.
# ---------------------------------------------------------------------------

def test_clarify_marks_the_preparing_step_as_warning_not_completed():
    """`result_preparation` resolving as `warning` must reach step 4's state.

    veda_hybrid emits `warning` (not `completed`) for a refusal/clarify. The
    message there was always honest; the STATUS was not, and the status is what
    the UI renders.
    """
    t = ts.ThinkingStepTracker()
    for phase in ("received", "understanding", "source_selection", "data_retrieval"):
        t.consume(ev(phase, status="completed"))
    t.consume(ev("result_preparation", "Could not answer this from the available data",
                 status="warning"))
    t.finish()
    steps = {s["id"]: s for s in t.snapshot()}
    assert steps["preparing"]["state"] == ts.STATE_WARNING
    # And it must NOT have been silently upgraded by the terminal sweep.
    assert steps["preparing"]["state"] != ts.STATE_COMPLETED


def test_a_failed_phase_is_never_downgraded_to_warning():
    """`warning` must not overwrite a real failure — only the reverse is safe."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("result_preparation", status="failed"))
    t.consume(ev("result_preparation", status="warning"))
    t.finish()
    steps = {s["id"]: s for s in t.snapshot()}
    assert steps["preparing"]["state"] == ts.STATE_FAILED


def test_validation_warning_does_not_read_as_a_passed_check():
    """A `warning` validation phase must not present as a satisfied check.

    The engine's ledger holds the AST/read-only/fan-out checks and those really do
    pass on a clarify — but a later correctness gate stopped the turn, so the
    step may not claim the result is complete and safe.
    """
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("source_selection", status="completed"))
    t.consume(ev("validation", "This result did not pass a completeness check",
                 status="warning"))
    t.finish()
    step = {s["id"]: s for s in t.snapshot()}["analyzing"]
    checks = rows(step, ts.DETAIL_VALIDATION)
    assert checks, "the validation sub-check should still be shown"
    assert checks[0]["state"] != ts.STATE_COMPLETED


# ---------------------------------------------------------------------------
# A refusal must not describe itself as assembling an answer, and a replayed
# answer must say so. Both found on the verified-cache lane.
# ---------------------------------------------------------------------------

def test_preparing_copy_is_honest_when_no_answer_was_produced():
    c = tc.ThinkingContext()
    c.output = "summary"
    assert "Putting" in c.sentence(ts.STEP_PREPARING)     # the ordinary case
    c.no_answer = True
    s = c.sentence(ts.STEP_PREPARING)
    assert "putting" not in s.lower(), s
    assert "couldn't answer" in s.lower(), s


def test_preparing_details_claim_no_output_when_there_is_none():
    c = tc.ThinkingContext()
    c.output = "summary"
    c.row_count = 5
    assert c.details(ts.STEP_PREPARING), "the ordinary case still lists output"
    c.no_answer = True
    d = c.details(ts.STEP_PREPARING)
    assert [r["label"] for r in d] == [
        "No answer could be produced for this question."]
    assert d[0]["state"] == ts.STATE_WARNING


def test_a_replayed_answer_is_disclosed_under_finding():
    c = tc.ThinkingContext()
    c.found_something = True
    assert not any("Reused" in r["label"] for r in c.details(ts.STEP_FINDING))
    c.from_cache = True
    d = c.details(ts.STEP_FINDING)
    reuse = [r for r in d if "Reused" in r["label"]]
    assert reuse, "the replay must be disclosed"
    low = reuse[0]["label"].lower()
    for banned in ("cache", "sql", "table", "column", "similarity"):
        assert banned not in low, f"{banned!r} leaked into the reuse disclosure"


def test_provenance_is_absorbed_from_either_payload_shape():
    for payload in ({"provenance": {"reused_verified_query": True}},
                    {"result": {"reused_verified_query": True}}):
        c = tc.ThinkingContext()
        c.absorb_explain(payload)
        assert c.from_cache is True, payload
    c = tc.ThinkingContext()
    c.absorb_explain({"result": {"row_count": 3}})
    assert c.from_cache is False


# ---------------------------------------------------------------------------
# A step that never ran must not sit `pending` between two completed ones.
#
# Found live on the document/RAG head, whose phase stream is
# source_selection -> access_check -> result_preparation: nothing maps to
# "Analyzing", so it stayed ○ while "Preparing" was already ✓.
# ---------------------------------------------------------------------------

def test_unreported_step_with_content_completes_without_inventing_a_duration():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("source_selection", status="completed"))
    t.consume(ev("result_preparation", status="completed"))   # skips over analyzing
    # Content recovered from the final payload proves the work happened.
    t.set_details(ts.STEP_ANALYZING,
                  [{"type": ts.DETAIL_OPERATION, "label": "Working out the total"}],
                  terminal=True)
    t.finish()
    st = {s["id"]: s for s in t.snapshot()}["analyzing"]
    assert st["state"] == ts.STATE_COMPLETED
    assert st.get("duration_ms") is None, "no duration was measured — do not invent one"


def test_unreported_step_without_content_is_skipped_not_pending():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("source_selection", status="completed"))
    t.consume(ev("result_preparation", status="completed"))
    t.finish()
    st = {s["id"]: s for s in t.snapshot()}["analyzing"]
    assert st["state"] == ts.STATE_SKIPPED
    assert st["state"] != ts.STATE_PENDING


def test_no_pending_step_ever_precedes_a_resolved_one():
    """The invariant the two tests above exist to protect."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("result_preparation", status="completed"))
    t.finish()
    seen_resolved = False
    for s in reversed(t.snapshot()):
        if s["state"] in (ts.STATE_COMPLETED, ts.STATE_WARNING,
                          ts.STATE_FAILED, ts.STATE_SKIPPED):
            seen_resolved = True
        elif s["state"] == ts.STATE_PENDING:
            assert not seen_resolved, f"{s['id']} is pending after a resolved step"


def test_smalltalk_still_leaves_later_steps_pending():
    """The opposite case must not regress: when NOTHING after a step ran, pending
    is the correct answer — smalltalk bypasses the engine entirely."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("access_check", status="completed"))   # sub-check under Finding
    t.finish()
    st = {s["id"]: s for s in t.snapshot()}
    assert st["analyzing"]["state"] == ts.STATE_PENDING
    assert st["preparing"]["state"] == ts.STATE_PENDING


def test_a_skipped_step_shows_nothing_inside_it():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("access_check", status="completed"))
    t.consume(ev("result_preparation", status="completed"))
    t.finish()
    st = {s["id"]: s for s in t.snapshot()}["analyzing"]
    assert st["state"] == ts.STATE_SKIPPED
    assert not st.get("details")
    assert not rows(st)


# ---------------------------------------------------------------------------
# Three defects visible in one real streamed turn the user pasted.
# ---------------------------------------------------------------------------

def test_a_pending_step_does_not_advertise_content_it_may_never_produce():
    """The live loop refreshes all four steps every frame, and the Preparing
    details always list at least a "Supporting summary" — so at
    total_duration_ms=0, before anything ran, step 4 sat `pending` while already
    claiming an output."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("supervisor_classify"))          # only step 1 has begun
    assert t.set_details(ts.STEP_PREPARING,
                         [{"type": ts.DETAIL_OUTPUT, "label": "Preparing summary"}]) is False
    snap = {s["id"]: s for s in t.snapshot()}
    assert not snap["preparing"].get("details")
    assert snap["preparing"]["state"] == ts.STATE_PENDING


def test_the_terminal_frame_may_still_attach_content_to_an_unreported_step():
    """The guard above must not break the document/RAG case, where content on a
    never-started step is how unreported work is recognised."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("result_preparation", status="completed"))
    assert t.set_details(ts.STEP_ANALYZING,
                         [{"type": ts.DETAIL_OPERATION, "label": "Counting"}],
                         terminal=True) is True
    t.finish()
    snap = {s["id"]: s for s in t.snapshot()}
    assert snap["analyzing"]["state"] == ts.STATE_COMPLETED


def test_a_measured_access_check_is_hidden_until_its_step_opens():
    """The api tier measures RBAC before the engine runs, so the sub-check lands on
    the very first frame — and a resolved ✓ check inside a ○ step reads as broken.

    Hidden, not discarded: the measurement surfaces the moment the step opens. The
    alternative (opening the step early) would overclaim on the smalltalk path,
    where RBAC is still measured but the engine is bypassed entirely.
    """
    t = ts.ThinkingStepTracker()
    t.consume(ev("supervisor_classify"))
    t.set_sub_check_duration(ts.STEP_FINDING, "access", 52)
    hidden = {s["id"]: s for s in t.snapshot()}["finding"]
    assert hidden["state"] == ts.STATE_PENDING
    assert not rows(hidden)
    assert hidden["expandable"] is False

    t.consume(ev("access_check", status="started"))          # now Finding opens
    shown = {s["id"]: s for s in t.snapshot()}["finding"]
    assert shown["state"] != ts.STATE_PENDING
    sub = rows(shown, ts.DETAIL_ACCESS)[0]
    assert sub["state"] == ts.STATE_COMPLETED and sub["duration_ms"] == 52


def test_no_step_is_left_active_when_the_turn_fails():
    """The outage path returned without emitting a terminal frame, leaving
    "Analyzing" spinning next to an error saying the assistant was down."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("source_selection", status="completed"))
    t.consume(ev("sql_planning"))
    assert {s["id"]: s for s in t.snapshot()}["analyzing"]["state"] == ts.STATE_ACTIVE
    t.finish(failed=True)
    for s in t.snapshot():
        assert s["state"] != ts.STATE_ACTIVE, f"{s['id']} left spinning after a failure"
    assert {s["id"]: s for s in t.snapshot()}["analyzing"]["state"] == ts.STATE_FAILED


# ---------------------------------------------------------------------------
# A turn that bypassed the engine has no progress to show.
#
# Observed on the canned-greeting fast path: answered in 104 ms, no phase ever
# emitted, and the payload still carried four `pending` steps under
# `status: "completed"` — four empty circles above a finished answer, claiming
# completion of work that never happened.
# ---------------------------------------------------------------------------

def test_a_turn_that_bypassed_the_engine_reports_no_progress():
    t = ts.ThinkingStepTracker()
    assert t.has_progress() is False
    t.finish()
    assert t.has_progress() is False, "finishing must not manufacture progress"
    # every step is still honestly pending — the model is right, it just must
    # not be RENDERED (services.py omits the block)
    assert {s["state"] for s in t.snapshot()} == {ts.STATE_PENDING}


def test_a_measured_sub_check_alone_is_not_progress():
    """The api tier times RBAC before the engine runs, so a measurement can exist
    on a turn where no step ever began. That is not progress."""
    t = ts.ThinkingStepTracker()
    t.set_sub_check_duration(ts.STEP_FINDING, "access", 14)
    assert t.has_progress() is False


def test_a_real_turn_reports_progress():
    t = ts.ThinkingStepTracker()
    t.consume(ev("supervisor_classify"))
    assert t.has_progress() is True


# ---------------------------------------------------------------------------
# THE FROZEN CONTRACT (CHAT_API_CONTRACT.md §1e).
#
# A frontend is being built against these. They are written as LITERALS on
# purpose: if someone renames a step id, a state, a detail type or a warning
# code, this test fails and they have to decide, consciously, that they are
# shipping a breaking change — rather than discovering it from a broken client.
#
# ADDING a value to a closed set is allowed and does not break a client (§1e says
# to treat an unknown value as "something new"). REMOVING or RENAMING one is the
# breaking change these assertions exist to catch, so each check is a subset test
# in the direction that matters.
# ---------------------------------------------------------------------------

def test_frozen_step_ids_and_order():
    assert list(ts.STEP_ORDER) == ["understanding", "finding", "analyzing", "preparing"], (
        "the four step ids and their order are frozen — a client renders by id")
    assert [ts.STEP_TITLES[k] for k in ts.STEP_ORDER] == [
        "Understanding your request",
        "Finding the right information",
        "Analyzing the information",
        "Preparing your answer",
    ], "step TITLES are copy and may be reworded — but the product spec fixes these four"


def test_frozen_state_values():
    frozen = {"pending", "active", "completed", "warning", "failed", "skipped"}
    actual = {ts.STATE_PENDING, ts.STATE_ACTIVE, ts.STATE_COMPLETED,
              ts.STATE_WARNING, ts.STATE_FAILED, ts.STATE_SKIPPED}
    assert frozen <= actual, f"a frozen state value was removed or renamed: {frozen - actual}"


def test_frozen_detail_types():
    frozen = {"access", "source", "evidence", "operation", "validation", "output"}
    assert frozen <= set(ts.DETAIL_TYPES), (
        f"a frozen detail type was removed or renamed: {frozen - set(ts.DETAIL_TYPES)}")


def test_frozen_execution_types():
    frozen = {"sql", "documents", "multi_source", "unknown"}
    actual = {ts.EXEC_SQL, ts.EXEC_DOCUMENTS, ts.EXEC_MULTI_SOURCE, ts.EXEC_UNKNOWN}
    assert frozen <= actual, f"a frozen execution type changed: {frozen - actual}"


def test_frozen_top_level_model_keys():
    t = ts.ThinkingStepTracker()
    t.consume(ev("supervisor_classify"))
    payload = t.as_payload()
    for key in ("type", "status", "current_step", "steps", "evidence",
                "execution", "timing"):
        assert key in payload, f"frozen model key `{key}` is missing"
    assert payload["type"] == "thinking"
    for key in ("id", "index", "title", "state", "duration_ms", "summary",
                "details", "expandable"):
        assert key in payload["steps"][0], f"frozen step key `{key}` is missing"


# ===========================================================================
# EDGE CASES the suite did not cover. Written after an audit found these five
# classes untested. Each one is a thing a real backend can do to this model.
# ===========================================================================

# -- duplicate / repeated events -------------------------------------------

def test_the_same_phase_arriving_twice_does_not_duplicate_anything():
    """The engine re-emits phases: `understanding` is emitted started, then
    completed, and Tier-2 re-does intent work later in the turn."""
    t = ts.ThinkingStepTracker()
    for _ in range(4):
        t.consume(ev("source_selection", status="completed"))
    t.set_details(ts.STEP_FINDING,
                  [{"type": ts.DETAIL_SOURCE, "label": "homzhub"}])
    t.set_details(ts.STEP_FINDING,
                  [{"type": ts.DETAIL_SOURCE, "label": "homzhub"}])
    t.finish()
    step = {s["id"]: s for s in t.snapshot()}["finding"]
    assert labels(step, ts.DETAIL_SOURCE) == ["homzhub"], "a repeated row must not duplicate"
    assert len([s for s in t.snapshot() if s["id"] == "finding"]) == 1


def test_a_repeated_sub_check_resolves_once():
    t = ts.ThinkingStepTracker()
    t.consume(ev("access_check", status="started"))
    t.consume(ev("access_check", status="completed"))
    t.consume(ev("access_check", status="completed"))
    t.finish()
    step = {s["id"]: s for s in t.snapshot()}["finding"]
    assert len(rows(step, ts.DETAIL_ACCESS)) == 1


# -- concurrency / isolation ------------------------------------------------

def test_two_turns_do_not_share_state():
    """Two requests are in flight at once on a threaded server. The tracker holds
    no module-level state, and this pins that: it is the class of bug that has
    bitten this codebase twice through ContextVars."""
    a, b = ts.ThinkingStepTracker(), ts.ThinkingStepTracker()
    a.consume(ev("source_selection", status="completed"))
    a.set_evidence(rows=5)
    a.set_details(ts.STEP_FINDING, [{"type": ts.DETAIL_SOURCE, "label": "homzhub"}])

    assert b.as_payload()["evidence"] == {}, "turn B must not see turn A's evidence"
    assert b.current_step() is None
    assert all(not (s.get("details") or []) for s in b.snapshot())
    a.finish()
    assert b.finished is False and b.status == "active", "finishing A must not finish B"


def test_trackers_stay_independent_under_real_threads():
    import threading
    results = {}

    def run(name, phases):
        t = ts.ThinkingStepTracker()
        for p in phases:
            t.consume(ev(p, status="completed"))
        t.finish()
        results[name] = [s["state"] for s in t.snapshot()]

    threads = [threading.Thread(target=run, args=("short", ["received"])),
               threading.Thread(target=run, args=("long",
                   ["received", "source_selection", "sql_planning", "result_preparation"]))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results["short"][1:] == [ts.STATE_PENDING] * 3, "the short turn kept its own shape"
    assert results["long"][3] != ts.STATE_PENDING, "the long turn reached its last step"


# -- unicode / non-ASCII ----------------------------------------------------

def test_non_ascii_content_is_carried_and_capped_without_corruption():
    t = ts.ThinkingStepTracker()
    t.consume(ev("source_selection", status="completed"))
    long_label = "देखें " * 80 + "अंत"        # well over the 160-char cap
    t.set_details(ts.STEP_FINDING,
                  [{"type": ts.DETAIL_SOURCE, "label": long_label}])
    t.set_context(ts.STEP_FINDING, "स्रोत खोज रहे हैं — 5 दस्तावेज़")
    step = {s["id"]: s for s in t.snapshot()}["finding"]
    got = labels(step, ts.DETAIL_SOURCE)[0]
    assert len(got) <= 160
    assert got == long_label[:160], "the cap must slice codepoints, not bytes"
    assert step["summary"] == "स्रोत खोज रहे हैं — 5 दस्तावेज़"
    import json
    json.dumps(t.as_payload())          # must serialise for the wire


def test_narrator_validator_handles_non_ascii_without_raising():
    for text in ("देखें", "résumé data", "日本語", "emoji 🎉 here"):
        assert nar.validate(text, {"intent": "ranking"}) in (None, text, text + ".")


# -- extreme / hostile values ----------------------------------------------

def test_a_backwards_clock_never_yields_a_negative_duration():
    """Engine timestamps come off a different clock than the api tier's."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("source_selection", status="completed", ts_ms=5_000_000))
    t.consume(ev("sql_planning", ts_ms=1_000_000))      # earlier than the last
    t.finish()
    for s in t.snapshot():
        d = s.get("duration_ms")
        assert d is None or d >= 0, f"{s['id']} reported {d}"
    assert t.total_duration_ms() >= 0


def test_many_operations_do_not_grow_the_payload_without_bound():
    t = ts.ThinkingStepTracker()
    t.consume(ev("sql_planning"))
    t.set_details(ts.STEP_ANALYZING,
                  [{"type": ts.DETAIL_OPERATION, "label": f"op {i}"} for i in range(500)])
    step = {s["id"]: s for s in t.snapshot()}["analyzing"]
    got = len(rows(step, ts.DETAIL_OPERATION))
    assert got == 500, (
        f"the tracker stores what it is given ({got}); the CAP is the caller's job "
        "— thinking_context slices operations to 6 before this point")


def test_junk_detail_rows_are_dropped_not_rendered():
    t = ts.ThinkingStepTracker()
    t.consume(ev("source_selection", status="completed"))
    assert t.set_details(ts.STEP_FINDING, [
        {"type": "made_up_type", "label": "x"},      # not in the closed vocabulary
        {"type": ts.DETAIL_SOURCE},                   # no label
        {"type": ts.DETAIL_SOURCE, "label": ""},      # empty label
        "not-a-dict",
        None,
    ]) is False
    assert not rows({s["id"]: s for s in t.snapshot()}["finding"])


# -- absent is not zero ----------------------------------------------------

def test_an_evidence_count_of_zero_is_reported_but_an_unreported_one_is_absent():
    """§1e tells clients not to substitute 0 for a missing key, so the two cases
    must actually be distinguishable on the wire."""
    t = ts.ThinkingStepTracker()
    # `chunks` arrives at the TOP level of the event, not inside `details` — that is
    # how query/rag_layer.py emits it (verified against a real stream), whereas
    # `source_count` arrives nested. The helper puts kwargs in `details`, so this
    # one is built by hand to match the wire.
    t.consume({"phase": "rag_retrieve", "message": "", "chunks": 0})
    ev_block = t.as_payload()["evidence"]
    assert ev_block["passages"] == 0, "a reported zero is a fact and must be shown"
    assert "rows" not in ev_block, "an unreported count must be ABSENT, not 0"

    t2 = ts.ThinkingStepTracker()
    t2.consume(ev("source_selection", status="completed"))
    assert "passages" not in t2.as_payload()["evidence"]


def test_a_negative_count_is_rejected_rather_than_shown():
    t = ts.ThinkingStepTracker()
    t.set_evidence(rows=-1, sources=3)
    assert t.as_payload()["evidence"] == {"sources": 3}


def test_an_engine_denial_overrides_the_measured_access_completion():
    """The api tier measures RBAC before the engine runs and marks the check
    complete from that duration — it knows the check HAPPENED, not that it
    PASSED. When the engine later denies, the denial must win.

    Observed live: `access ✓ "You have permission to access the required
    information"` (6 ms) directly above an answer saying the opposite.
    """
    t = ts.ThinkingStepTracker()
    t.set_sub_check_duration(ts.STEP_FINDING, "access", 6)
    t.consume(ev("access_check", status="started"))
    assert rows({s["id"]: s for s in t.snapshot()}["finding"],
                ts.DETAIL_ACCESS)[0]["state"] == ts.STATE_COMPLETED

    t.consume(ev("access_check", status="failed"))
    check = rows({s["id"]: s for s in t.snapshot()}["finding"], ts.DETAIL_ACCESS)[0]
    assert check["state"] == ts.STATE_FAILED
    msg = (check.get("message") or "").lower()
    # The copy must convey a denial. It deliberately says "access" rather than
    # "permission" (that wording is in _ACCESS_COPY) — what matters is that the
    # optimistic sentence is GONE, because that is the one that contradicted the
    # answer the user was reading.
    assert "you have permission" not in msg, msg
    assert "isn't available" in msg or "not available" in msg, msg


# ---------------------------------------------------------------------------
# A hybrid turn: the SQL head's validation FAILS, the documents answer, and the
# answer is correct. Observed live — the panel showed
# "✗ The result did not pass a safety check" above a good handbook answer.
# ---------------------------------------------------------------------------

def test_a_superseded_validation_failure_does_not_describe_a_delivered_answer():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("sql_planning"))
    t.consume(ev("validation", status="failed"))          # the SQL attempt failed
    t.consume(ev("result_preparation", status="completed"))  # documents answered
    t.finish(answered_without_result=False)

    check = rows({s["id"]: s for s in t.snapshot()}["analyzing"], ts.DETAIL_VALIDATION)[0]
    assert check["state"] == ts.STATE_WARNING, "a delivered answer did not fail a check"
    msg = check["message"].lower()
    assert "did not pass" not in msg, msg
    assert "set aside" in msg, msg


def test_a_validation_failure_on_a_REFUSAL_stays_a_failure():
    """The downgrade must not soften the case where the failure IS the outcome."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("sql_planning"))
    t.consume(ev("validation", status="failed"))
    t.finish(answered_without_result=True)

    check = rows({s["id"]: s for s in t.snapshot()}["analyzing"], ts.DETAIL_VALIDATION)[0]
    assert check["state"] == ts.STATE_FAILED
    assert "did not pass" in check["message"].lower()


def test_a_failed_turn_keeps_its_validation_failure():
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume(ev("sql_planning"))
    t.consume(ev("validation", status="failed"))
    t.finish(failed=True)
    check = rows({s["id"]: s for s in t.snapshot()}["analyzing"], ts.DETAIL_VALIDATION)[0]
    assert check["state"] == ts.STATE_FAILED


def test_hybrid_doc_chunks_are_counted_as_passages():
    """`hybrid_retrieve` names the count `doc_chunks`, not `chunks`. Reading only
    the latter reported `evidence: {"rows": 0}` for an answer built from 5
    passages — a count that was not merely absent but wrong."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("received", status="completed"))
    t.consume({"phase": "hybrid_retrieve", "message": "", "sql_cols": 0, "doc_chunks": 5})
    assert t.as_payload()["evidence"].get("passages") == 5


# ------------------------------------------------- execution shape from evidence
def test_two_named_sources_are_a_multi_source_execution():
    """The cross-source lane emits no `intent`, so a federated answer reported
    execution.type "unknown" while naming the sources it had just combined."""
    assert ts.execution_shape_from_evidence(
        source_names=["sales_lake", "homzhub"]) == ts.EXEC_MULTI_SOURCE
    assert ts.execution_shape_from_evidence(source_count=3) == ts.EXEC_MULTI_SOURCE


def test_one_source_with_passages_is_a_document_execution():
    assert ts.execution_shape_from_evidence(
        source_count=1, passages=6) == ts.EXEC_DOCUMENTS


def test_rows_alone_are_a_sql_execution():
    assert ts.execution_shape_from_evidence(has_rows=True) == ts.EXEC_SQL


def test_multi_source_wins_over_the_rows_a_federated_answer_also_returns():
    assert ts.execution_shape_from_evidence(
        source_count=2, has_rows=True) == ts.EXEC_MULTI_SOURCE


def test_zero_passages_is_not_evidence_of_a_document_answer():
    """0 retrieved passages means retrieval found nothing — not that this was a
    document turn. Same for a null count, which means 'not reported'."""
    assert ts.execution_shape_from_evidence(passages=0) is None
    assert ts.execution_shape_from_evidence(passages=None) is None


def test_no_facts_at_all_leaves_the_shape_honestly_unknown():
    assert ts.execution_shape_from_evidence() is None
    assert ts.execution_shape_from_evidence(source_count=1, source_names=["x"]) is None


# --------------------------------------------------------- progressive reveal
def test_the_first_frame_shows_only_the_step_that_actually_started():
    """All four steps were listed from the first frame — that is a PLAN, not
    progress, and on a turn that never reaches step 3 the plan is simply wrong."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("supervisor_classify", "Understanding your question"))
    p = t.as_payload()
    assert [s["id"] for s in p["steps"]] == ["understanding"]
    assert p["total_steps"] == 4


def test_the_next_step_appears_only_once_the_turn_reaches_it():
    t = ts.ThinkingStepTracker()
    t.consume(ev("supervisor_classify", "Understanding your question"))
    assert len(t.as_payload()["steps"]) == 1
    t.consume(ev("source_selection", "Finding the data"))
    ids = [s["id"] for s in t.as_payload()["steps"]]
    assert ids == ["understanding", "finding"]
    assert ids[0] == "understanding" and ids[-1] == "finding"


def test_a_future_step_is_never_listed_while_the_turn_is_running():
    t = ts.ThinkingStepTracker()
    t.consume(ev("supervisor_classify", "Understanding your question"))
    ids = {s["id"] for s in t.as_payload()["steps"]}
    assert "analyzing" not in ids and "preparing" not in ids, (
        "advertising a step that has not run claims work that may never happen")


def test_a_step_the_turn_moved_past_is_shown_resolved_not_pending():
    """A pending circle sitting between two ticks reads as broken. The document
    path emits no phase mapping to 'Analyzing', so this is the live case."""
    t = ts.ThinkingStepTracker()
    t.consume(ev("result_preparation", "Putting your answer together"))
    steps = {s["id"]: s for s in t.as_payload()["steps"]}
    assert steps["understanding"]["state"] in (ts.STATE_SKIPPED, ts.STATE_COMPLETED)
    assert steps["understanding"]["state"] != ts.STATE_PENDING


def test_a_skipped_step_carries_no_content():
    t = ts.ThinkingStepTracker()
    t.consume(ev("result_preparation", "Putting your answer together"))
    for s in t.as_payload()["steps"]:
        if s["state"] == ts.STATE_SKIPPED:
            assert s["details"] == [] and s["expandable"] is False


def test_a_turn_that_never_started_lists_no_steps_at_all():
    """Smalltalk bypasses the engine — four empty circles above a finished answer
    is not progress, it is a progress display full of blanks."""
    t = ts.ThinkingStepTracker()
    t.finish()
    assert t.as_payload()["steps"] == []
    assert t.has_progress() is False


def test_total_steps_is_stable_while_the_list_grows():
    t = ts.ThinkingStepTracker()
    seen = []
    for phase in ("supervisor_classify", "source_selection",
                  "data_retrieval", "result_preparation"):
        t.consume(ev(phase, "…"))
        p = t.as_payload()
        assert p["total_steps"] == 4
        seen.append(len(p["steps"]))
    assert seen == sorted(seen), "the list may only grow, never shrink"
    assert seen[-1] == 4


def test_the_running_step_always_has_a_summary_to_fall_back_to():
    """The legacy top-level `message` falls back to the running step's summary,
    because most phases carry no user-facing copy and shipped `message: ""` — a
    status line that appeared, blanked and reappeared several times per turn.
    That fallback is only safe while a summary is guaranteed to exist."""
    for phase in ("received", "source_selection", "data_retrieval",
                  "result_preparation"):
        t = ts.ThinkingStepTracker()
        t.consume(ev(phase, ""))
        cur = t.current_step()
        assert cur, f"{phase} opened no step"
        assert t.steps[cur].summary or ts._GENERIC_CONTEXT[cur], (
            f"{phase} leaves the running step with nothing to say")
        rendered = {s["id"]: s for s in t.as_payload()["steps"]}
        assert rendered[cur]["summary"].strip(), "a rendered step must never be blank"


# ------------------------------------------------- document answer, no duplicates
def test_a_document_answer_lists_each_operation_once():
    """The placeholder operations shown while the real ones are unknown must be
    SUPERSEDED, not kept. Measured live on a contract question: the step listed
    'Reading relevant passages' AND 'Read the relevant passages', and
    'Synthesizing the retrieved information' AND 'Synthesized the retrieved
    information' — the same work twice, in two tenses."""
    from apps.chat.thinking_context import ThinkingContext
    c = ThinkingContext()
    c.execution_type = ts.EXEC_DOCUMENTS
    c.passages = 5
    live = c.details("analyzing")                    # nothing known yet
    assert [r["label"] for r in live] == ["Reading relevant passages",
                                          "Synthesizing the retrieved information"]
    assert all(r.get("_generic") for r in live), "placeholders must be supersedable"

    t = ts.ThinkingStepTracker()
    t.consume(ev("rag_synthesize", "…"))
    t.set_details("analyzing", live)
    # the real operations arrive with the final payload
    c.operations = ["Retrieved 5 relevant passages", "Read the relevant passages",
                    "Synthesized the retrieved information"]
    t.set_details("analyzing", c.details("analyzing"), terminal=True)
    labels = [r["label"] for r in
              {s["id"]: s for s in t.as_payload()["steps"]}["analyzing"]["details"]]
    # CHANGED (Phase 2): the assertion used to be `labels == c.operations`, pinning
    # all three of the document head's operations. Two of them are now dropped as
    # duplicates — the retrieval row repeats the Finding step's passage count, and
    # "Read the relevant passages" is not an event separable from synthesising them
    # (see TestTheDocumentStepStatesEachEventOnce). What this test exists to prove is
    # unchanged and still proved: the live PLACEHOLDERS are superseded rather than
    # kept alongside the real operations.
    assert labels == ["Synthesized the retrieved information"], (
        f"expected the real operations only, got {labels}")
    assert "Reading relevant passages" not in labels
    assert "Synthesizing the retrieved information" not in labels


# ------------------------------------------- multi_source is a checkable claim
def test_a_hybrid_document_answer_is_not_called_multi_source():
    """The hybrid head reports `hybrid` — "database first, then documents" — which
    is one source and two attempts, not several sources. A handbook question
    answered from one PDF reported execution.type "multi_source" while its own
    sources block named a single source (measured live)."""
    assert ts.correct_multi_source_claim(
        ts.EXEC_MULTI_SOURCE, source_count=1, passages=5) == ts.EXEC_DOCUMENTS


def test_a_hybrid_answer_that_came_from_rows_is_sql():
    assert ts.correct_multi_source_claim(
        ts.EXEC_MULTI_SOURCE, source_count=1, has_rows=True) == ts.EXEC_SQL


def test_a_genuine_multi_source_answer_keeps_its_shape():
    assert ts.correct_multi_source_claim(
        ts.EXEC_MULTI_SOURCE, source_count=2, passages=5) == ts.EXEC_MULTI_SOURCE
    assert ts.correct_multi_source_claim(
        ts.EXEC_MULTI_SOURCE,
        source_names=["homzhub", "invoices_csv"]) == ts.EXEC_MULTI_SOURCE


def test_with_nothing_to_re_derive_from_the_shape_is_left_alone():
    """Downgrading to `unknown` would throw away the only thing we were told."""
    assert ts.correct_multi_source_claim(
        ts.EXEC_MULTI_SOURCE, source_count=1) == ts.EXEC_MULTI_SOURCE


def test_a_shape_is_upgraded_when_several_sources_actually_contributed():
    """CHANGED 2026-09-11: the correction is two-way. The federated lane reports
    the SQL it built rather than the federation, so a cross-source answer that
    named two participating sources in its own payload still reported
    execution.type "sql" (measured live: "There are 7 invoices compared to 96
    assets"). The count is the authority for a countable claim."""
    for shape in (ts.EXEC_SQL, ts.EXEC_DOCUMENTS, ts.EXEC_UNKNOWN):
        assert ts.correct_multi_source_claim(
            shape, source_count=2) == ts.EXEC_MULTI_SOURCE
        assert ts.correct_multi_source_claim(
            shape, source_names=["homzhub", "invoices_csv"]) == ts.EXEC_MULTI_SOURCE


def test_a_shape_is_left_alone_when_one_source_served_the_turn():
    for shape in (ts.EXEC_SQL, ts.EXEC_DOCUMENTS, ts.EXEC_UNKNOWN):
        assert ts.correct_multi_source_claim(
            shape, source_count=1, passages=99, has_rows=True) == shape
        assert ts.correct_multi_source_claim(shape) == shape, (
            "with no source evidence at all, nothing may be upgraded")


# ================================================= one normalized terminal state
class TestTerminalOutcome:
    """Three clarifications of identical shape — an unscoped question, a filter on
    a value that does not exist, and a prompt-injection attempt — came back
    `failed`, while three others came back `completed`. The rule was an allow-list
    of engine statuses that named `clarify` but not `refuse`/`qualifier_dropped`."""

    def test_an_answered_turn_is_not_a_failure(self):
        assert ts.terminal_outcome(ok=True, engine_status="answered") == (False, None)

    def test_every_refusal_vocabulary_the_engine_uses_is_completed(self):
        for st in ("clarify", "refuse", "qualifier_dropped", "no_answer",
                   "something_new_next_year"):
            failed, code = ts.terminal_outcome(ok=False, engine_status=st)
            assert failed is False, f"{st!r} is a refusal, not a crash"
            assert code is None

    def test_the_ways_the_engine_can_break_are_failures(self):
        for st in ("exec_error", "tier2_exec_error", "invalid", "error",
                   "internal_error", "EXEC_ERROR"):
            failed, code = ts.terminal_outcome(ok=False, engine_status=st)
            assert failed is True, f"{st!r} is a real error"
            assert code is None

    def test_a_refusal_that_explained_itself_is_never_a_failure(self):
        """Even an error-shaped status: if the payload carries why/what_would_help,
        the reader was given guidance, not a crash."""
        assert ts.terminal_outcome(ok=False, engine_status="exec_error",
                                   has_refusal_explanation=True) == (False, None)

    def test_a_denial_is_a_failure_with_a_code_whatever_else_is_true(self):
        """The denial is delivered as an ordinary reply, which is exactly how it
        went out looking like a completed answer. It is checked first."""
        for ok in (True, False):
            for st in ("answered", "exec_error", "clarify", None):
                assert ts.terminal_outcome(
                    ok=ok, engine_status=st, has_refusal_explanation=True,
                    access_denied=True) == (True, ts.ERROR_ACCESS_DENIED)

    def test_a_missing_status_is_not_invented_into_a_failure(self):
        assert ts.terminal_outcome(ok=False, engine_status=None) == (False, None)
        assert ts.terminal_outcome(ok=False, engine_status="") == (False, None)


# ============================================ severity never silently downgrades
class TestAccessDenialSurvives:
    """The engine emits the access check TWICE on a denial — `failed`, then
    `warning` — and the second overwrote the first, so a DENIAL displayed as a
    partial-access warning under a step marked completed (measured live)."""

    def _denied(self):
        t = ts.ThinkingStepTracker()
        t.consume(ev("access_check", "Checking your access", status="started"))
        t.consume(ev("access_check", "…", status="failed"))
        t.consume(ev("access_check", "…", status="warning"))
        return t

    def test_a_later_warning_cannot_soften_a_failed_check(self):
        t = self._denied()
        acc = [c for s in t.steps.values() for c in s.sub_checks
               if c["kind"] == "access"]
        assert [c["state"] for c in acc] == [ts.STATE_FAILED]

    def test_the_denial_is_reported_by_the_tracker(self):
        assert self._denied().access_denied() is True

    def test_an_ordinary_turn_reports_no_denial(self):
        t = ts.ThinkingStepTracker()
        t.consume(ev("access_check", "…", status="started"))
        t.consume(ev("access_check", "…", status="completed"))
        assert t.access_denied() is False

    def test_a_partial_access_warning_is_not_a_denial(self):
        t = ts.ThinkingStepTracker()
        t.consume(ev("access_check", "…", status="started"))
        t.consume(ev("access_check", "…", status="warning"))
        assert t.access_denied() is False

    def test_the_step_holding_a_failed_check_is_not_shown_completed(self):
        t = self._denied()
        t.consume(ev("result_preparation", "…", status="completed"))
        t.finish()
        finding = {s["id"]: s for s in t.as_payload()["steps"]}["finding"]
        assert finding["state"] == ts.STATE_WARNING, (
            "a green tick over a failed access check reads as granted")


# =========================================== Phase 1: the panel must not lie
class TestTheTerminalFrameIsAuthoritative:
    """Merging the terminal builder's rows into what accumulated during the turn
    kept rows the outcome had already invalidated. Measured on 8 of 8 refusals:
    the step asserted "Preparing summary ✓" and "No answer could be produced ⚠"
    at the same time."""

    def _turn(self):
        t = ts.ThinkingStepTracker()
        t.consume(ev("result_preparation", "…"))
        t.set_details("preparing", [{"type": ts.DETAIL_OUTPUT,
                                     "label": "Preparing summary",
                                     "state": ts.STATE_COMPLETED}])
        return t

    def test_a_row_the_outcome_invalidated_is_dropped(self):
        t = self._turn()
        assert "Preparing summary" in [r["label"] for r in t.steps["preparing"].details]
        t.set_details("preparing", [{"type": ts.DETAIL_OUTPUT,
                                     "label": "No answer could be produced for this question.",
                                     "state": ts.STATE_WARNING}], terminal=True)
        labels = [r["label"] for r in t.steps["preparing"].details]
        assert labels == ["No answer could be produced for this question."]
        assert "Preparing summary" not in labels

    def test_sub_checks_survive_the_replacement(self):
        """Authorization and validation live in sub_checks, not details."""
        t = ts.ThinkingStepTracker()
        t.consume(ev("access_check", "…", status="started"))
        t.consume(ev("access_check", "…", status="completed"))
        t.consume(ev("result_preparation", "…"))
        t.set_details("preparing", [{"type": ts.DETAIL_OUTPUT, "label": "x",
                                     "state": ts.STATE_COMPLETED}], terminal=True)
        t.finish()
        finding = {s["id"]: s for s in t.as_payload()["steps"]}["finding"]
        assert any(r["type"] == "access" for r in finding["details"])

    def test_an_empty_terminal_list_does_not_wipe_the_step(self):
        t = self._turn()
        t.set_details("preparing", [], terminal=True)
        assert [r["label"] for r in t.steps["preparing"].details] == ["Preparing summary"]


class TestAnEmptyResultReadsAsEmpty:
    """The mainstream SQL path showed four green ticks, "Checks passed" and an
    empty warning list above a reply reading "No results found."."""

    def _ctx(self, **kw):
        from apps.chat.thinking_context import ThinkingContext
        c = ThinkingContext()
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def test_found_nothing_gets_its_own_sentence(self):
        assert self._ctx(found_nothing=True, no_answer=True).sentence("preparing") == \
            "No matching data was found for this question."

    def test_a_refusal_keeps_the_guidance_wording(self):
        """"See the reply for what's needed" is right for a refusal, which can say
        what would help — and wrong for an empty result, where nothing would."""
        s = self._ctx(no_answer=True).sentence("preparing")
        assert "what's needed" in s and "No matching data" not in s

    def test_the_only_preparing_row_is_the_warning(self):
        rows = self._ctx(found_nothing=True, no_answer=True).details("preparing")
        assert [r["state"] for r in rows] == [ts.STATE_WARNING]
        assert "Nothing matched" in rows[0]["label"]

    def test_an_ordinary_answer_is_unaffected(self):
        s = self._ctx(row_count=5).sentence("preparing")
        assert "No matching data" not in s and "what's needed" not in s


class TestAFailedSourceIsNotATick:
    """A datalake turn's payload said `status: failed, message: "This data source
    could not be reached"` while the Finding step showed the source completed."""

    def _ctx_with(self, status, message="This data source could not be reached"):
        from apps.chat.thinking_context import ThinkingContext
        c = ThinkingContext()
        c.absorb_explain({"sources": [{"name": "catalog_parquet"}],
                          "execution": {"sources": [{"name": "catalog_parquet",
                                                     "status": status,
                                                     "message": message}]}})
        return c

    def test_a_failed_source_renders_as_a_warning_with_the_reason(self):
        rows = self._ctx_with("failed").details("finding")
        src = [r for r in rows if r["type"] == ts.DETAIL_SOURCE]
        assert len(src) == 1
        assert src[0]["state"] == ts.STATE_WARNING
        assert "could not be reached" in src[0]["label"]

    def test_a_completed_source_is_still_a_plain_tick(self):
        rows = self._ctx_with("completed").details("finding")
        src = [r for r in rows if r["type"] == ts.DETAIL_SOURCE]
        assert src[0]["state"] == ts.STATE_COMPLETED
        assert src[0]["label"] == "catalog_parquet"

    def test_nothing_claims_information_was_available_when_every_source_failed(self):
        labels = [r["label"] for r in self._ctx_with("failed").details("finding")]
        assert not any("Relevant information available" in l for l in labels)


class TestALimitThatCannotTruncateIsNotStated:
    """The deterministic head appends LIMIT 100 to every statement, so a COUNT that
    can only return one row was telling the reader it had been cut short."""

    def _ctx(self, truncated):
        from apps.chat.thinking_context import ThinkingContext
        c = ThinkingContext()
        c.operations = ["Counting the records", "Limited to the top results"]
        c.truncated = truncated
        return c

    def test_the_row_is_dropped_when_nothing_was_truncated(self):
        assert [r["label"] for r in self._ctx(False).details("analyzing")] == \
            ["Counting the records"]

    def test_the_row_survives_a_real_truncation(self):
        assert "Limited to the top results" in \
            [r["label"] for r in self._ctx(True).details("analyzing")]


# ============================== Phase 3: show what the payload already knows
class TestThePayloadIsActuallyRead:
    """`absorb_explain` read 6 keys out of a payload carrying roughly twenty, and
    one of the six (`warnings`) was stored and never rendered anywhere."""

    def _ctx(self, explain, **kw):
        from apps.chat.thinking_context import ThinkingContext
        c = ThinkingContext()
        for k, v in kw.items():
            setattr(c, k, v)
        c.absorb_explain(explain)
        return c

    def _labels(self, c, step):
        return [r["label"] for r in c.details(step)]

    def test_the_safety_checks_are_named(self):
        c = self._ctx({"validation": {"passed": True, "checks": [
            {"label": "Read-only query", "passed": True},
            {"label": "Duplicate-safe (no double-counting)", "passed": True}]}})
        assert "Read-only query" in self._labels(c, "analyzing")

    def test_a_failed_check_is_named_and_flagged(self):
        """Knowing WHICH check failed is the case where the name matters most."""
        c = self._ctx({"validation": {"checks": [
            {"label": "No requested filters were ignored", "passed": False}]}})
        row = [r for r in c.details("analyzing") if r["type"] == ts.DETAIL_VALIDATION][0]
        assert row["state"] == ts.STATE_WARNING

    def test_the_collapsed_line_goes_once_the_checks_are_named(self):
        t = ts.ThinkingStepTracker()
        t.consume(ev("validation", "…", status="completed"))
        t.set_details("analyzing", [{"type": ts.DETAIL_VALIDATION,
                                     "label": "Read-only query",
                                     "state": ts.STATE_COMPLETED}], terminal=True)
        labels = [r["label"] for r in
                  {s["id"]: s for s in t.as_payload()["steps"]}["analyzing"]["details"]]
        assert "Read-only query" in labels
        assert "Checking the result is complete and safe" not in labels, (
            "the collapsed line says only that checking happened")

    def test_filters_are_shown(self):
        c = self._ctx({"filters": {"applied": [
            {"field": "City Name", "operator": "is", "value": "Pune"}]}})
        assert "Filtered to City Name is Pune" in self._labels(c, "understanding")

    def test_document_names_are_shown_for_a_document_answer(self):
        c = self._ctx({"data_used": {"datasets": ["msa green tower"]}},
                      execution_type=ts.EXEC_DOCUMENTS)
        assert "msa green tower" in self._labels(c, "finding")

    def test_a_sql_turn_never_shows_its_table_name_as_a_source(self):
        """The same key holds the humanized TABLE name on a SQL turn ("Assets") —
        engine vocabulary, and it appeared as a source row beside `homzhub`."""
        c = self._ctx({"data_used": {"datasets": ["Assets", "Maintenances"]}},
                      execution_type=ts.EXEC_SQL)
        labels = self._labels(c, "finding")
        assert "Assets" not in labels and "Maintenances" not in labels

    def test_each_source_contribution_is_shown_on_a_cross_source_answer(self):
        c = self._ctx({"sources": [{"name": "homzhub", "rows": 96},
                                   {"name": "invoices_csv", "rows": 7}]})
        labels = self._labels(c, "analyzing")
        assert "homzhub contributed 96 records" in labels
        assert "invoices_csv contributed 7 records" in labels
        assert "Combined across sources" in labels

    def test_a_single_source_contribution_is_not_narrated(self):
        c = self._ctx({"sources": [{"name": "homzhub", "rows": 96}]})
        assert not any("contributed" in l for l in self._labels(c, "analyzing"))

    def test_the_refusal_guidance_reaches_the_reader(self):
        c = self._ctx({"what_would_help": "Name the column to group by.",
                       "suggestions": ["Try one figure at a time."]},
                      no_answer=True)
        labels = self._labels(c, "preparing")
        assert "Name the column to group by." in labels
        assert labels[0].startswith("No answer could be produced")

    def test_an_empty_result_offers_no_guidance_because_there_is_none(self):
        c = self._ctx({"what_would_help": "x"}, found_nothing=True, no_answer=True)
        assert len(c.details("preparing")) == 1

    def test_the_chart_reason_replaces_the_bare_line(self):
        c = self._ctx({"visualization": {"type": "bar", "reason": "Bar chart selected "
                                         "because the query compares a measure."}})
        labels = self._labels(c, "preparing")
        assert any("Bar chart selected" in l for l in labels)
        assert "Preparing visualization" not in labels

    def test_warning_sentences_are_rendered_not_just_collected(self):
        c = self._ctx({"warnings": [{"code": "low_evidence",
                                     "message": "There was limited matching data."}]})
        assert "There was limited matching data." in self._labels(c, "preparing")

    def test_the_truncation_sentence_is_not_said_twice(self):
        c = self._ctx({"result": {"truncated": True, "row_count": 100},
                       "warnings": [{"code": "result_truncated",
                                     "message": "The result was limited to the first 100 records."}]})
        labels = self._labels(c, "preparing")
        assert sum(1 for l in labels if "first" in l.lower()) == 1


# ============================ Phase 2: each fact is stated once, in one place
class TestASourceThatIsNamedIsNotAlsoDescribedVaguely:
    """"Relevant information available" was measured in the FINAL frame of 12 of 14
    database turns, sitting directly beneath the named source row `homzhub`. If we
    can name the source we found, the vaguer row states nothing further — and it
    escaped the generic-supersede rule in `set_details`, which only replaces a
    generic row of the SAME type (this one is `evidence`, the source row is
    `source`)."""

    def _ctx(self, **kw):
        from apps.chat.thinking_context import ThinkingContext
        c = ThinkingContext()
        c.found_something = True
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def _labels(self, c):
        return [r["label"] for r in c.details("finding")]

    def test_the_vague_row_goes_once_the_source_is_named(self):
        labels = self._labels(self._ctx(source_names=["homzhub"]))
        assert "homzhub" in labels, "the survivor must be present in the same build"
        assert "Relevant information available" not in labels

    def test_the_vague_row_goes_once_passages_are_reported(self):
        labels = self._labels(self._ctx(passages=5))
        assert "5 relevant passages retrieved" in labels
        assert "Relevant information available" not in labels

    def test_the_vague_row_goes_when_a_document_is_named(self):
        """A document answer names its sources through `datasets`, not
        `source_names` — the attribute the old guard tested — so this build showed
        "msa green tower" and "Relevant information available" together."""
        labels = self._labels(self._ctx(execution_type=ts.EXEC_DOCUMENTS,
                                        datasets=["msa green tower"]))
        assert "msa green tower" in labels
        assert "Relevant information available" not in labels

    def test_it_survives_for_the_case_it_was_written_for(self):
        """Something WAS found and nothing about it can be named. There is no
        survivor to carry the fact, so the row is the only thing saying it."""
        assert self._labels(self._ctx()) == ["Relevant information available"]

    def test_a_failed_source_still_reads_as_a_warning(self):
        """A WARNING is an outcome, not noise: naming a source never silences it."""
        c = self._ctx(source_names=["catalog_parquet"],
                      failed_sources={"catalog_parquet": "This data source could "
                                                         "not be reached"})
        row = [r for r in c.details("finding") if r["type"] == ts.DETAIL_SOURCE][0]
        assert row["state"] == ts.STATE_WARNING


class TestTheSourceCountIsOnlyAPlaceholderForAName:
    """"1 relevant source found" is correctly superseded by the named source at the
    terminal frame, but within ONE build the document path emitted both: it names
    its sources through `datasets`, so `source_names` stayed empty and the `elif`
    that was meant to hold the count back never fired."""

    def _ctx(self, **kw):
        from apps.chat.thinking_context import ThinkingContext
        c = ThinkingContext()
        c.found_something = True
        c.source_count = 1
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def _labels(self, c):
        return [r["label"] for r in c.details("finding")]

    def test_the_count_goes_when_the_source_is_named_in_the_same_build(self):
        labels = self._labels(self._ctx(source_names=["homzhub"]))
        assert "homzhub" in labels
        assert "1 relevant source found" not in labels

    def test_the_count_goes_when_a_document_is_named_in_the_same_build(self):
        labels = self._labels(self._ctx(execution_type=ts.EXEC_DOCUMENTS,
                                        datasets=["msa green tower"]))
        assert "msa green tower" in labels
        assert "1 relevant source found" not in labels

    def test_the_count_survives_while_no_name_is_known(self):
        """The live stream reaches the panel long before the terminal payload names
        anything; until then the count is all there is to say."""
        labels = self._labels(self._ctx())
        assert labels[0] == "1 relevant source found"

    def test_the_counted_evidence_key_is_untouched(self):
        """`evidence.sources` states the same number a second time and STAYS: it is
        a documented part of the frontend contract that other clients read. The ROW
        is the half that goes."""
        t = ts.ThinkingStepTracker()
        t.consume(ev("source_selection", "…", source_count=1))
        assert t.as_payload()["evidence"]["sources"] == 1


class TestTheDocumentStepStatesEachEventOnce:
    """The `analyzing` step listed all three operations `_apply_document_v1` emits:
    "Retrieved 5 relevant passages", "Read the relevant passages", "Synthesized the
    retrieved information". The first is the Finding step's "5 relevant passages
    retrieved" — same number, same event, wording reversed — and reading is not an
    event separable from retrieving in this pipeline. Only the synthesis row names
    work the panel states nowhere else."""

    _TRIPLE = ["Retrieved 5 relevant passages", "Read the relevant passages",
               "Synthesized the retrieved information"]

    def _ctx(self, **kw):
        from apps.chat.thinking_context import ThinkingContext
        c = ThinkingContext()
        c.execution_type = ts.EXEC_DOCUMENTS
        c.passages = 5
        c.operations = list(self._TRIPLE)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def test_only_the_synthesis_row_survives(self):
        c = self._ctx()
        assert [r["label"] for r in c.details("analyzing")] == \
            ["Synthesized the retrieved information"]

    def test_the_finding_step_still_carries_the_passage_count(self):
        """The survivor for the dropped retrieval row lives in a DIFFERENT step, so
        it is checked on the same context rather than assumed."""
        assert "5 relevant passages retrieved" in \
            [r["label"] for r in self._ctx().details("finding")]

    def test_the_retrieval_row_stays_when_no_passage_count_was_reported(self):
        """`passages` is None when the live stream never carried `chunks` /
        `doc_chunks`. The Finding row is then not built at all, so dropping the
        retrieval row would delete the only mention of the passages."""
        c = self._ctx(passages=None)
        labels = [r["label"] for r in c.details("analyzing")]
        assert "Retrieved 5 relevant passages" in labels
        assert not any("passage" in r["label"]
                       for r in c.details("finding")), "no survivor in Finding"

    def test_reading_is_only_folded_into_a_synthesis_row_that_exists(self):
        c = self._ctx(operations=["Retrieved 5 relevant passages",
                                  "Read the relevant passages"])
        assert [r["label"] for r in c.details("analyzing")] == \
            ["Read the relevant passages"]

    def test_a_database_answers_operations_are_untouched(self):
        c = self._ctx(execution_type=ts.EXEC_SQL, passages=None,
                      operations=["Counting the records", "Broken down by City"])
        assert [r["label"] for r in c.details("analyzing")] == \
            ["Counting the records", "Broken down by City"]


class TestPreparingSummaryIsTheStepsOwnHeadline:
    """"Preparing summary" was emitted on 14 of 14 turns: it is true of every turn
    that produces any answer, so it distinguishes nothing — and the step's collapsed
    line already says it in the same words ("Putting your answer together.")."""

    def _ctx(self, **kw):
        from apps.chat.thinking_context import ThinkingContext
        c = ThinkingContext()
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def test_the_row_is_gone(self):
        assert "Preparing summary" not in \
            [r["label"] for r in self._ctx(row_count=5).details("preparing")]

    def test_the_collapsed_line_still_says_it(self):
        """The survivor is the step's own summary, not another row."""
        assert self._ctx().sentence("preparing") == "Putting your answer together."
        assert self._ctx(output="summary").sentence("preparing") == \
            "Putting together a summary."

    def test_the_rows_that_carry_a_real_fact_are_kept(self):
        labels = [r["label"] for r in
                  self._ctx(row_count=42, output="chart", truncated=True,
                            chart_reason="Bar chart selected because the query "
                                         "compares a measure.").details("preparing")]
        assert "Bar chart selected because the query compares a measure." in labels
        assert "Preparing table · 42 rows" in labels
        assert "Showing the first page of results only" in labels

    def test_a_warning_row_is_never_dropped(self):
        rows = self._ctx(truncated=True, row_count=100).details("preparing")
        assert any(r["state"] == ts.STATE_WARNING for r in rows)

    def test_a_step_that_never_started_is_unaffected_by_having_no_rows(self):
        """finish() picks `completed` vs `skipped` for a never-started step from
        whether it has details, and this was the only unconditional row here. The
        branch is unreachable for "preparing": it is gated on a LATER step having
        started, and preparing is last in STEP_ORDER — a never-started preparing is
        withheld from the snapshot entirely."""
        t = ts.ThinkingStepTracker()
        t.consume(ev("supervisor_classify", "…"))
        t.consume(ev("rag_retrieve", "…"))
        t.finish()
        assert t.steps["preparing"].state == ts.STATE_PENDING
        assert "preparing" not in {s["id"] for s in t.as_payload()["steps"]}

    def test_a_plain_summary_turn_leaves_the_step_unexpandable(self):
        """Which is the honest rendering: there is nothing inside the step that the
        collapsed line does not already say."""
        t = ts.ThinkingStepTracker()
        t.consume(ev("result_preparation", "…"))
        c = self._ctx(output="summary")
        t.set_context("preparing", c.sentence("preparing"))
        t.set_details("preparing", c.details("preparing"), terminal=True)
        t.finish()
        step = {s["id"]: s for s in t.as_payload()["steps"]}["preparing"]
        assert step["expandable"] is False
        assert step["summary"] == "Putting together a summary."


class TestATurnThatNeverReachedTheEngineExplainsNothing:
    """A canned greeting is answered in ~100 ms without the engine ever being
    called. It still shipped a `thinking` frame reading "Finalizing the results…"
    — there were no results — and a full `explainability` skeleton with every block
    empty, which a client renders as a "how this answer was generated" panel
    explaining nothing. The four steps were already suppressed for exactly this
    reason; the other two surfaces were left behind."""

    def test_no_progress_means_no_progress_event(self):
        t = ts.ThinkingStepTracker()
        t.finish()
        assert t.has_progress() is False, (
            "has_progress() is the gate both suppressions hang on")

    def test_a_turn_that_ran_still_reports_progress(self):
        t = ts.ThinkingStepTracker()
        t.consume(ev("supervisor_classify", "…"))
        t.finish()
        assert t.has_progress() is True

    def test_a_failed_turn_with_no_progress_still_reports(self):
        """The outage path has no progress either, and its frame is how the error
        code reaches the client — suppressing it would lose the failure."""
        t = ts.ThinkingStepTracker()
        t.finish(failed=True, error_code="LLM_UNAVAILABLE", retryable=True)
        p = t.as_payload()
        assert p["status"] == "failed"
        assert p["error"]["code"] == "LLM_UNAVAILABLE"
