"""Tests for veda/business_explain.py's extract_sql_facts() — the public
wrapper onto the existing zero-LLM sqlglot AST pass, exposed for reuse by
veda/result_analyzer.py. Pure-python, no DB, no network."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


def test_extract_sql_facts_matches_private_extract():
    from veda.business_explain import extract_sql_facts, _extract
    sql = ('SELECT payer_name, SUM(amount) AS total FROM ledger '
           'WHERE entry_type = \'CREDIT\' GROUP BY payer_name '
           'ORDER BY total DESC LIMIT 5')
    assert extract_sql_facts(sql) == _extract(sql)


def test_extract_sql_facts_aggregation_and_grouping():
    from veda.business_explain import extract_sql_facts
    sql = 'SELECT status, COUNT(*) AS n FROM incidents GROUP BY status'
    facts = extract_sql_facts(sql)
    assert facts["entities"] == ["incidents"]
    assert facts["groupings"] == ["status"]
    assert ("COUNT", None) in facts["aggregations"]


def test_extract_sql_facts_orderings_and_limit():
    from veda.business_explain import extract_sql_facts
    sql = 'SELECT id FROM ledger ORDER BY amount DESC LIMIT 10'
    facts = extract_sql_facts(sql)
    assert facts["orderings"] == [("amount", True)]
    assert facts["limit"] == 10


def test_extract_sql_facts_filters():
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE amount > 100"
    facts = extract_sql_facts(sql)
    assert ("amount", "GT", "100") in facts["filters"]


def test_extract_sql_facts_invalid_sql_returns_safe_empty_shape():
    from veda.business_explain import extract_sql_facts
    facts = extract_sql_facts("not valid sql at all !!!")
    assert facts["entities"] == []
    assert facts["limit"] is None


# ---------------------------------------------------------------------------
# Phase 2 gap-fill: build_explain() surfaces the Insight Engine's validated
# visualization reasoning — additive only, omitted when None (existing
# callers/consumers unaffected).
# ---------------------------------------------------------------------------

def test_build_explain_omits_visualization_key_by_default():
    from veda.business_explain import build_explain
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None)
    assert "visualization" not in out


def test_build_explain_includes_validated_visualization():
    """Reasoning is deterministic/standardized (Final Polish, Section 9) — the
    SLM's own free-text "reason" is NOT surfaced verbatim; a known chart type
    always gets the same, LLM-free phrasing."""
    from veda.business_explain import build_explain
    sm = {"columns": {"ledger.total": {"business_role": "Total Amount"}}}
    out = build_explain(
        sql='SELECT payer_name, SUM(amount) AS total FROM ledger GROUP BY payer_name',
        table="ledger", sm=sm,
        visualization={"type": "bar", "x_axis": "payer_name", "y_axis": "total",
                       "reason": "categorical vs numeric comparison"},
    )
    assert out["visualization"]["type"] == "bar"
    assert out["visualization"]["reason"] == (
        "Bar chart selected because the query compares a numeric measure "
        "across discrete categories."
    )


def test_build_explain_unknown_chart_type_falls_back_to_slm_reason():
    from veda.business_explain import build_explain
    out = build_explain(
        sql='SELECT a FROM t', table="t", sm=None,
        visualization={"type": "scatter", "x_axis": None, "y_axis": None,
                       "reason": "a free-text reason with no deterministic template"},
    )
    assert out["visualization"]["reason"] == "a free-text reason with no deterministic template"
