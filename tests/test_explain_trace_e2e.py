"""End-to-end tests for the centralized query trace (veda/explain.py + the
slm/_call_slm.py ledger seam).

Covers the CORE tracing mechanism without a live SLM / DB / MLflow:
1. trace_id is minted (or reused from the caller) and ambient-reuse works —
   new_trace() inside a bound scope returns the SAME trace (one query, one trace).
2. The SLM ledger: call_slm() records every invocation (purpose/model/duration/ok)
   into the current trace, on success AND failure — via a monkeypatched backend.
3. finish() is a CHECKPOINT when an outer scope owns the trace (no premature
   persist); finalize() persists exactly once and forces the final status.
4. The final `totals` block is built from what stages recorded (durations,
   slm_call_count, row/col/chart counts).
5. record_result_stages() populates execution / result_analysis / summary /
   visualization / explainability from already-computed values.
6. Verbose gating + production safety: with verbose OFF, heavy candidate lists
   are NOT captured (cand() is a no-op), so no row/prompt dumps leak.
7. Disabled path returns a zero-cost _NullTrace and current_trace() is safe.

Run from repo root: ``pytest tests/test_explain_trace_e2e.py``
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


# ── 1. trace_id + ambient reuse ───────────────────────────────────────────────
def test_trace_id_minted_and_reused():
    from veda.explain import new_trace, use_trace, current_trace, _NullTrace

    # explicit id is honored
    t = new_trace("top query", trace_id="req-abc-123")
    assert t.trace_id == "req-abc-123"
    assert t.enabled is True

    # unbound: current_trace() is a safe NullTrace (never raises)
    assert isinstance(current_trace(), _NullTrace)

    with use_trace(t):
        # any stage grabbing the ambient trace gets THE trace
        assert current_trace() is t
        # Tier-1's own new_trace() reuses the ambient one — one query, one trace
        inner = new_trace("sub query")
        assert inner is t

    # after the scope, ambient is cleared again
    assert isinstance(current_trace(), _NullTrace)


def test_trace_id_auto_minted_when_absent():
    from veda.explain import new_trace
    t = new_trace("q")
    assert isinstance(t.trace_id, str) and len(t.trace_id) >= 8


# ── 2. the SLM ledger via the real call_slm() choke-point ─────────────────────
class _FakeBackend:
    name = "fake"
    model = "fake-model"

    def __init__(self, fail=False):
        self._fail = fail

    def call(self, user_message, **kw):
        if self._fail:
            raise RuntimeError("SLM unreachable (test)")
        return "ok", {"prompt_tokens": 5, "completion_tokens": 7}


def test_call_slm_records_ledger_success_and_failure(monkeypatch):
    import slm._call_slm as cs
    from veda.explain import new_trace, use_trace

    t = new_trace("ledger query", trace_id="led-1")
    with use_trace(t):
        # success
        monkeypatch.setattr(cs, "get_backend", lambda: _FakeBackend(fail=False))
        cs.call_slm("hi", purpose="classification", model="qwen-x")
        cs.call_slm("hi", purpose="sql_ir_generation")
        # failure still gets recorded (finally block), then re-raises
        monkeypatch.setattr(cs, "get_backend", lambda: _FakeBackend(fail=True))
        try:
            cs.call_slm("boom", purpose="nl_answer")
        except RuntimeError:
            pass

    calls = t.sections["slm"]["calls"]
    assert t.sections["slm"]["count"] == 3
    purposes = [c["purpose"] for c in calls]
    assert purposes == ["classification", "sql_ir_generation", "nl_answer"]
    assert calls[0]["model"] == "qwen-x"           # explicit model honored
    assert calls[1]["model"] == "fake-model"       # falls back to backend model
    assert all("duration_ms" in c for c in calls)
    assert calls[0]["ok"] is True and calls[2]["ok"] is False
    assert "error" in calls[2]                      # failure captured its error


# ── 3. checkpoint vs finalize (one persisted record per query) ────────────────
def test_finish_is_checkpoint_finalize_persists_once(tmp_path, monkeypatch):
    import veda.explain as ex
    from veda.explain import new_trace, use_trace

    log = tmp_path / "explain_trace.jsonl"
    monkeypatch.setattr(ex, "_TRACE_LOG", str(log))

    root = new_trace("q", trace_id="one-record")
    with use_trace(root):
        inner = new_trace("q")            # Tier-1 reuses ambient
        inner.set("query_understanding", intent="SIMPLE")
        inner.finish("refuse")            # inner (Tier-1) → checkpoint, must NOT persist
        assert not log.exists() or log.read_text() == ""
        root.set("execution", row_count=3, column_count=2)
        root.finalize("answered", route="tier2")   # owner persists once, forces status

    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1                 # exactly one record for the whole query
    assert root.sections["output"]["status"] == "answered"   # final status forced


# ── 4. totals block ───────────────────────────────────────────────────────────
def test_totals_summary(monkeypatch):
    import veda.explain as ex
    from veda.explain import new_trace, use_trace

    monkeypatch.setattr(ex, "_TRACE_LOG", os.devnull)
    t = new_trace("q", trace_id="tot-1")
    with use_trace(t):
        t.slm_call("classification", "m", 10.0, True)
        t.slm_call("sql_ir_generation", "m", 800.0, True)
        t.set("execution", row_count=5, column_count=4)
        t.set("visualization", selected_count=1)
        t.finalize("answered", route="sql")

    tot = t.sections["totals"]
    assert tot["trace_id"] == "tot-1"
    assert tot["status"] == "answered"
    assert tot["slm_call_count"] == 2
    assert tot["slm_total_duration_ms"] == 810.0
    assert tot["row_count"] == 5 and tot["column_count"] == 4
    assert tot["chart_count"] == 1
    assert "stage_durations_ms" in tot


# ── 5. record_result_stages reads already-computed values ─────────────────────
class _FakeIctx:
    result_shape = "RANKING"
    result_type = "ranking"
    dimensions = ["city"]
    measures = ["cnt"]
    entities = ["assets_asset"]
    patterns = []
    chart_candidates = [{"type": "bar", "x_axis": "city", "y_axis": "cnt", "confidence": 0.9}]


def test_record_result_stages(monkeypatch):
    from veda.explain import new_trace, use_trace, record_result_stages

    t = new_trace("top 5 cities", trace_id="rrs-1")
    with use_trace(t):
        record_result_stages(
            engine="run_nl_answer", cols=["city", "cnt"], row_count=5,
            ictx=_FakeIctx(), answer="Mumbai leads with 5.",
            summary_model="qwen2.5:7b-instruct", summary_ok=True,
            visualization={"type": "bar", "x_axis": "city", "y_axis": "cnt"},
            explain_payload={"datasets": ["assets_asset"], "operations": [{"type": "count"}],
                             "filters": [], "check_items": [{"status": "pass"}]})

    s = t.sections
    assert s["execution"]["row_count"] == 5 and s["execution"]["column_count"] == 2
    assert s["result_analysis"]["result_shape"] == "RANKING"
    assert s["result_analysis"]["dimensions"] == ["city"]
    assert s["summary"]["engine"] == "run_nl_answer"
    assert s["summary"]["answer_chars"] == len("Mumbai leads with 5.")
    assert s["visualization"]["selected_count"] == 1
    assert s["visualization"]["candidate_count"] == 1
    assert s["explainability"]["operation_count"] == 1
    assert s["explainability"]["validation_passed"] is True


# ── 6. verbose gating / production safety ─────────────────────────────────────
def test_verbose_off_suppresses_heavy_lists():
    from veda.explain import ExplainTrace
    t = ExplainTrace(query="q", verbose=False, trace_id="v-off")
    t.set("retrieval", n_columns=15)                 # scalar decision — kept
    t.cand("retrieval", "top_columns", {"col": "x", "secret_row": "PII"})  # heavy — dropped
    d = t.to_dict()
    assert d["sections"]["retrieval"]["n_columns"] == 15
    assert "top_columns" not in d["sections"]["retrieval"]   # no candidate/PII dump


def test_verbose_on_captures_candidates():
    from veda.explain import ExplainTrace
    t = ExplainTrace(query="q", verbose=True, trace_id="v-on")
    t.cand("rrf", "top_candidates", {"col_id": "t.c", "rrf_score": 0.5})
    assert t.to_dict()["sections"]["rrf"]["top_candidates"][0]["col_id"] == "t.c"


# ── 7. disabled path is a zero-cost NullTrace ─────────────────────────────────
def test_disabled_path_is_null(monkeypatch):
    import config as cfg
    from veda.explain import new_trace, _NullTrace
    monkeypatch.setattr(cfg, "EXPLAIN_TRACE_ENABLED", False)
    t = new_trace("q")
    assert isinstance(t, _NullTrace)
    # every method is a safe no-op
    t.set("x", a=1); t.slm_call("p", "m", 1.0, True); t.cand("x", "k", 1)
    assert t.to_dict() is None and t.finalize("answered") is None
