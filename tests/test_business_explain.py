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


# ---------------------------------------------------------------------------
# Filter-value resolution from PARAMETERIZED sql: validate_and_parameterize()
# rewrites every filter literal into a %s placeholder (bound separately in
# `params`) for safe execution — _extract()/build_explain() must resolve the
# real value back from `params` by position, not just report None.
# ---------------------------------------------------------------------------

def test_extract_sql_facts_resolves_placeholder_value_from_params():
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE status = %s"
    facts = extract_sql_facts(sql, params=["open"])
    assert ("status", "EQ", "open") in facts["filters"]


def test_extract_sql_facts_without_params_still_degrades_to_none():
    """Regression guard: existing callers that don't pass `params` (the
    default) must see identical behavior to before this fix — None, not a
    crash or a changed shape."""
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE status = %s"
    facts = extract_sql_facts(sql)
    assert ("status", "EQ", None) in facts["filters"]


def test_extract_sql_facts_multiple_placeholders_resolve_by_position():
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE status = %s AND entry_type = %s"
    facts = extract_sql_facts(sql, params=["open", "DEBIT"])
    assert ("status", "EQ", "open") in facts["filters"]
    assert ("entry_type", "EQ", "DEBIT") in facts["filters"]


def test_extract_sql_facts_placeholder_index_out_of_range_degrades_to_none():
    """Fewer params than placeholders (shouldn't happen in practice, but must
    never crash) — degrades to None for the unresolvable one, same as today."""
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE status = %s"
    facts = extract_sql_facts(sql, params=[])
    assert ("status", "EQ", None) in facts["filters"]


def test_build_explain_filter_value_resolved_with_params():
    from veda.business_explain import build_explain
    out = build_explain(
        sql="SELECT id FROM ledger WHERE entry_type = %s", table="ledger", sm=None,
        params=["DEBIT"],
    )
    applied = out["filters"]["applied"]
    assert len(applied) == 1
    assert applied[0]["value"] == "DEBIT"
    assert "DEBIT" in out["filters"]["summary"]


def test_build_explain_filter_value_none_without_params_unchanged():
    """Regression guard: existing callers of build_explain() that don't pass
    `params` (every caller before this fix) see the exact same None-value
    behavior as before — this fix is additive/opt-in only."""
    from veda.business_explain import build_explain
    out = build_explain(sql="SELECT id FROM ledger WHERE entry_type = %s",
                        table="ledger", sm=None)
    applied = out["filters"]["applied"]
    assert len(applied) == 1
    assert applied[0]["value"] is None


# ---------------------------------------------------------------------------
# Explainability-gap fixes: refusal explain (Item 1), confidence placeholder
# (Item 2), timeline (Item 3) — all additive, existing fields/behavior
# unaffected.
# ---------------------------------------------------------------------------

def test_build_explain_confidence_key_always_present_and_none():
    """Item 2: schema-only placeholder — never a computed/fake number."""
    from veda.business_explain import build_explain
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None)
    assert "confidence" in out
    assert out["confidence"] is None


def test_build_explain_timeline_defaults_to_empty_list():
    """Item 3: omitting `timeline` must not change any existing caller's output
    shape beyond adding this one always-present, empty-by-default key."""
    from veda.business_explain import build_explain
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None)
    assert out["timeline"] == []


def test_build_explain_timeline_relays_ticks_verbatim_in_order():
    """Item 3: timeline is a passive relay of the run's own _tick() checkpoints
    — same messages, same order, no re-derivation."""
    from veda.business_explain import build_explain
    ticks = [("schema_linking", "Using ledger for this"),
             ("sql_planning", "Narrowing to that time period"),
             ("output", "Done — here's your answer")]
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None, timeline=ticks)
    assert out["timeline"] == [
        {"phase": "schema_linking", "message": "Using ledger for this"},
        {"phase": "sql_planning", "message": "Narrowing to that time period"},
        {"phase": "output", "message": "Done — here's your answer"},
    ]


def test_build_explain_existing_fields_unaffected_by_new_keys():
    """Regression guard: adding confidence/timeline must not change any
    existing field's value for a caller that predates both."""
    from veda.business_explain import build_explain
    sm = {"columns": {"ledger.total": {"business_role": "Total Amount"}}}
    out = build_explain(
        sql='SELECT payer_name, SUM(amount) AS total FROM ledger GROUP BY payer_name',
        table="ledger", sm=sm,
    )
    assert out["version"] == "1.0"
    assert out["data_used"]["datasets"] == ["Ledgers"]
    assert out["operations"] == [
        {"type": "total", "summary": "Calculate total Amount"},
        {"type": "group", "summary": "Group by Payer Name"},
    ]
    assert out["sql"]["query"].startswith("SELECT payer_name")


def test_build_refusal_explain_returns_none_without_feedback():
    """Item 1: no feedback dict (e.g. FEEDBACK_ENABLED=False, or the
    invalid/exec_error _done() call sites that never build one) -> None,
    same "no explain" signal the answered path already uses."""
    from veda.business_explain import build_refusal_explain
    assert build_refusal_explain("no_table", None) is None
    assert build_refusal_explain("invalid", {}) is None


def test_build_refusal_explain_no_table():
    from veda.business_explain import build_refusal_explain
    from veda.feedback import explain_failure
    fb = explain_failure("no_table", {}, candidates=["ledger", "invoice"])
    out = build_refusal_explain("no_table", fb)
    assert out["version"] == "1.0"
    assert out["status"] == "no_table"
    assert out["understanding"]["summary"] == fb["why"]
    assert "couldn't confidently match" in out["why"]
    assert out["what_would_help"] == fb["what_needed"]
    assert out["suggestions"] == ["ledger", "invoice"]


def test_build_refusal_explain_qualifier_dropped():
    """A second, differently-shaped refusal status — proves the fix isn't
    special-cased to just one status."""
    from veda.business_explain import build_refusal_explain
    from veda.feedback import explain_failure
    fb = explain_failure("qualifier_dropped", {"columns": {"ledger.status": {}}},
                         missing="pending")
    out = build_refusal_explain("qualifier_dropped", fb)
    assert out["status"] == "qualifier_dropped"
    assert "pending" in out["why"]
    assert out["what_would_help"]


# ---------------------------------------------------------------------------
# Item 4 (optional) — understanding.breakdown, additive alongside summary.
# ---------------------------------------------------------------------------

def test_build_explain_understanding_breakdown_is_additive():
    """`summary` (the single prose sentence) must stay byte-identical; the
    new `breakdown` list is assembled from the same operations/filter_phrases
    already computed, not a new derivation."""
    from veda.business_explain import build_explain
    sm = {"columns": {"ledger.status": {"business_role": "Status"}}}
    out = build_explain(
        sql="SELECT status, COUNT(*) AS n FROM ledger WHERE status = 'open' "
            "GROUP BY status ORDER BY n DESC LIMIT 5",
        table="ledger", sm=sm,
    )
    assert out["understanding"]["summary"] == (
        "Find the top 5 Statuses by N where Status equals open."
    )
    assert out["understanding"]["breakdown"] == [
        "Count records",
        "Group by Status",
        "Sort by N (highest first)",
        "Return top 5",
        "Status equals open",
    ]


def test_build_explain_understanding_breakdown_empty_for_bare_list():
    """No aggregation/grouping/ordering/limit/filter -> operations defaults
    to a single 'List <dataset>' entry, filter_phrases is empty -> breakdown
    is that one phrase, matching `operations` exactly."""
    from veda.business_explain import build_explain
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None)
    assert out["understanding"]["breakdown"] == ["List records"]
