"""Pipeline-level proof that veda/pipeline.py's deterministic SELECT-list
construction was switched from `allowed_columns` (the validation allow-list)
to `recommended_projection(...)` (the new business-facing projection),
without changing SQL semantics or weakening validation.

Drives run_query() itself (not just recommended_projection() in isolation —
see tests/test_recommended_projection.py for that), through the
`temporal_only` deterministic branch, using the same dependency-level
monkeypatch pattern as tests/test_refusal_explain.py and
tests/test_verified_cache_qualifier_gate.py: no DB, no LLM, no embedding
model. execute_sql() itself is stubbed too (a fixed (cols, rows, err) triple)
so the run reaches "answered" without a live database — this test is about
what SQL TEXT gets built and validated, not about real row data.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

TABLE = "accounts_paymenttransaction"
ALL_COLS = [f"{TABLE}.{c}" for c in (
    "id", "payment_number", "amount", "status", "created_at",
    "updated_at", "created_by_id", "updated_by_id", "deleted_at",
    "payment_signature", "third_party_name", "third_party_email",
)]
SM = {"columns": {
    f"{TABLE}.id": {"importance_class": "LOW"},
    f"{TABLE}.payment_number": {"importance_class": "HIGH"},
    f"{TABLE}.amount": {"importance_class": "HIGH"},
    f"{TABLE}.status": {"importance_class": "HIGH"},
    f"{TABLE}.created_at": {"importance_class": "MEDIUM", "semantic_type": "TEMPORAL"},
    f"{TABLE}.updated_at": {"importance_class": "LOW"},
    f"{TABLE}.created_by_id": {"importance_class": "LOW"},
    f"{TABLE}.updated_by_id": {"importance_class": "LOW"},
    f"{TABLE}.deleted_at": {"importance_class": "LOW"},
    f"{TABLE}.payment_signature": {"importance_class": "LOW"},
    f"{TABLE}.third_party_name": {"importance_class": "LOW"},
    f"{TABLE}.third_party_email": {"importance_class": "LOW"},
}}


class _FakeResult:
    def __init__(self, col_id, score):
        self.col_id = col_id
        self.column_name = col_id.split(".", 1)[1]
        self.final_score = score


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def retrieve(self, **kwargs):
        return list(self._rows)


def _quiet_pipeline_deps(monkeypatch, retrieval_results):
    import config
    import veda.pipeline as pipeline

    monkeypatch.setattr(pipeline, "verified_cache_lookup", lambda q: (None, 0.0))
    monkeypatch.setattr(pipeline, "get_engine", lambda sm: _FakeEngine(retrieval_results))
    monkeypatch.setattr(pipeline, "select_primary_table", lambda *a, **k: TABLE)
    monkeypatch.setattr(pipeline, "vet_primary", lambda *a, **k: TABLE)
    monkeypatch.setattr(pipeline, "execute_sql",
                        lambda sql, params=None: (["created_at", "amount"],
                                                  [("2026-06-05", 500)], None))
    monkeypatch.setattr(config, "GRAPH_EXPAND_ENABLED", False)
    monkeypatch.setattr(config, "PRIMARY_RERANK_ENABLED", False)
    monkeypatch.setattr(config, "FAST_PATH_ENABLED", False)
    monkeypatch.setattr(config, "SUPERLATIVE_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "GROUPED_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "RATIO_PLAN_ENABLED", False)
    return pipeline


def test_temporal_only_branch_projects_business_columns_not_all_allowed(monkeypatch):
    results = [_FakeResult(f"{TABLE}.amount", 0.9), _FakeResult(f"{TABLE}.status", 0.85),
              _FakeResult(f"{TABLE}.created_at", 0.8)]
    pipeline = _quiet_pipeline_deps(monkeypatch, results)

    res = pipeline.run_query("transactions created last month", sm=SM, all_cols=ALL_COLS,
                             return_result=True)

    assert res["status"] == "answered"
    sql = res["sql"]
    # Business/relevant columns present in the actual SELECT clause.
    for expected in ("amount", "status", "created_at", "payment_number"):
        assert f'"{expected}"' in sql
    # Audit/internal columns — never retrieval-relevant nor HIGH-importance in
    # this fixture — must be ABSENT from the SELECT clause. This is the actual
    # behavior change: before this fix, every one of these would have appeared
    # (SELECT reused allowed_columns verbatim).
    for excluded in ("created_by_id", "updated_by_id", "deleted_at",
                    "third_party_name", "third_party_email"):
        assert f'"{excluded}"' not in sql
    # WHERE clause still correctly filters on the temporal column even though
    # it's not "recommended" as a HIGH-importance display column — proving
    # allowed_columns (extended with _tcol a few lines after _proj is built)
    # is untouched for validation/WHERE purposes.
    assert '"created_at" BETWEEN' in sql


def test_validation_allow_list_is_unaffected_by_projection_trimming(monkeypatch):
    """The other half of the responsibility split: allowed_columns (what the
    AST firewall permits) must still cover every original column, not just
    the trimmed recommended set — proven by requesting an audit column
    explicitly (forcing it into the SELECT via the safety override) and
    confirming validation still accepts it, i.e. it was never dropped from
    the allow-list just because it's rarely "recommended"."""
    results = [_FakeResult(f"{TABLE}.amount", 0.9)]
    pipeline = _quiet_pipeline_deps(monkeypatch, results)

    res = pipeline.run_query("show me the payment signature created last month",
                             sm=SM, all_cols=ALL_COLS, return_result=True)

    assert res["status"] == "answered"
    assert '"payment_signature"' in res["sql"]
