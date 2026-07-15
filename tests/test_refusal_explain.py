"""Item 1 regression coverage: veda/pipeline.py::run_query()'s refusal paths
must return a structured res["explain"] (not None), same as the "answered"
path already does — via veda/business_explain.py::build_refusal_explain().

Drives run_query() itself (not just build_refusal_explain() in isolation —
see tests/test_business_explain.py for that) to prove the actual _done()
wiring works, for two DIFFERENT refusal statuses reached via two different
early-return branches (no_table: schema linking finds nothing; clarify:
vet_primary's single-table ambiguity gate). Both hit before any SQL is
generated, so no DB/LLM/embedding model is needed — the embedding-backed
retrieve() call and vet_primary/select_primary_table's real (DB/model-backed)
implementations are monkeypatched out, same pattern as
tests/test_execution_state_reuse.py's dependency-level monkeypatching.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


class _FakeEngine:
    """Stands in for get_engine(sm) — an empty retrieval, no embedding model
    or DB touched."""
    def retrieve(self, **kwargs):
        return []


def _quiet_pipeline_deps(monkeypatch):
    """Disable every config-gated planner/rerank/expand branch that would
    otherwise try to reach a DB/embedding model before schema linking runs,
    and stub the two schema-linking calls themselves. Shared by both refusal
    tests below; only `vet_primary`'s return value differs between them."""
    import config
    import veda.pipeline as pipeline

    monkeypatch.setattr(pipeline, "verified_cache_lookup", lambda q: (None, 0.0))
    monkeypatch.setattr(pipeline, "get_engine", lambda sm: _FakeEngine())
    monkeypatch.setattr(config, "GRAPH_EXPAND_ENABLED", False)
    monkeypatch.setattr(config, "PRIMARY_RERANK_ENABLED", False)
    monkeypatch.setattr(config, "FAST_PATH_ENABLED", False)
    monkeypatch.setattr(config, "SUPERLATIVE_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "GROUPED_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "RATIO_PLAN_ENABLED", False)
    return pipeline


def test_run_query_no_table_refusal_has_structured_explain(monkeypatch):
    import veda.pipeline as pipeline
    pipeline = _quiet_pipeline_deps(monkeypatch)
    monkeypatch.setattr(pipeline, "select_primary_table", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "vet_primary", lambda *a, **k: None)

    res = pipeline.run_query("zzzz_no_such_query_xyz_12345", sm={}, all_cols=[],
                             return_result=True)

    assert res["status"] == "no_table"
    assert res["ok"] is False
    assert res["explain"] is not None
    assert res["explain"]["status"] == "no_table"
    assert "couldn't confidently match" in res["explain"]["why"]
    assert res["explain"]["understanding"]["summary"] == res["explain"]["why"]


def test_run_query_clarify_refusal_has_structured_explain(monkeypatch):
    """A second, differently-reached refusal status (vet_primary's ambiguity
    gate, not schema linking finding zero candidates) — proves the fix isn't
    special-cased to a single status."""
    import veda.pipeline as pipeline
    pipeline = _quiet_pipeline_deps(monkeypatch)
    monkeypatch.setattr(pipeline, "select_primary_table", lambda *a, **k: "ledger")
    monkeypatch.setattr(pipeline, "vet_primary",
                        lambda *a, **k: {"clarify": "which ledger do you mean?"})

    res = pipeline.run_query("zzzz_no_such_query_xyz_12345", sm={}, all_cols=[],
                             return_result=True)

    assert res["status"] == "clarify"
    assert res["ok"] is False
    assert res["explain"] is not None
    assert res["explain"]["status"] == "clarify"
    assert res["explain"]["why"] == "which ledger do you mean?"


def test_run_query_no_table_refusal_matches_prior_non_explain_fields(monkeypatch):
    """Regression guard: adding `explain` must not change any of the OTHER
    fields _done() already returned for a refusal before this fix (status,
    ok, feedback, msg, trace, context)."""
    import veda.pipeline as pipeline
    pipeline = _quiet_pipeline_deps(monkeypatch)
    monkeypatch.setattr(pipeline, "select_primary_table", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "vet_primary", lambda *a, **k: None)

    res = pipeline.run_query("zzzz_no_such_query_xyz_12345", sm={}, all_cols=[],
                             return_result=True)

    assert res["status"] == "no_table"
    assert res["ok"] is False
    assert res["msg"] == "no single table confidently matched the question"
    assert res["feedback"]["why"] == res["explain"]["why"]
    assert "trace" in res and "context" in res
