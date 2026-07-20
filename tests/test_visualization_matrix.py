"""VEDA visualization — comprehensive MODEL-FREE matrix.

Exercises the REAL deterministic visualization components end-to-end:
  · veda_core result_analyzer: analyze_result → result_shape, roles,
    compute_chart_candidates (analytics_summary)
  · apps.chat.visualization.VisualizationRecommender.recommend (primary)
  · the final _build_visualizations precedence (recommender → suggestion →
    chart_candidate) — faithfully REPLICATED here because apps/chat/services.py is
    Django-coupled and cannot be imported model-free; the replica mirrors the fixed
    services logic (see _spec_from_suggestion / final_chart below).

No SLM / LLM / embeddings / Ollama / external services.

Covers shapes A–Q from the task, axis correctness, cross-tier consistency, schema
independence, and the two fixes applied this round:
  FIX-1  a RANKING leads with BAR, not a misleading PIE (part-of-whole).
  FIX-2  the recommender honors the semantic ROLE (a numeric-valued CATEGORY dimension
         charts as the category axis, not mistaken for a measure).
  FIX-3  _spec_from_suggestion returns ONE spec (not a list) — the list previously
         crashed _build_visualizations' `spec.to_dict()` on any bar/pie fallback.

Run from repo root: ``pytest tests/test_visualization_matrix.py``
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "veda_core"))


def _col(t, n, st, ar):
    return {"col_name": n, "table_name": t, "semantic_type": st, "analytics_role": ar}


def _recommender():
    from apps.chat.visualization import VisualizationRecommender
    return VisualizationRecommender()


def _analytics(sql, cols, rows, sm, table):
    from veda.result_analyzer import analyze_result, analytics_summary
    ctx = analyze_result("q", sql, cols, rows, sm=sm, table=table)
    return ctx, analytics_summary(ctx)


def _positional(cols, rows):
    return [[r.get(c) for c in cols] for r in rows]


# --- faithful replica of the FIXED services._spec_from_suggestion / _build_visualizations
def _spec_from_suggestion(rec, cols, rows, suggestion):
    if not suggestion or not isinstance(suggestion, dict):
        return None
    vtype, x, y = suggestion.get("type"), suggestion.get("x_axis"), suggestion.get("y_axis")
    if x not in cols or y not in cols:
        return None
    xi, yi = cols.index(x), cols.index(y)
    if vtype == "line":
        return rec._line(cols, rows, xi, yi)
    if vtype in ("bar", "pie"):
        specs = rec._category_numeric(cols, rows, xi, yi)
        if not specs:
            return None
        return next((s for s in specs if s.type.value == vtype), specs[0])
    return None


def _final_chart(rec, cols, prows, analytics, suggestion=None):
    """Mirror of _build_visualizations: recommender → SLM suggestion → chart_candidate."""
    specs = rec.recommend(cols, prows, analytics=analytics)
    if specs:
        return specs
    sp = _spec_from_suggestion(rec, cols, prows, suggestion)
    if sp:
        return [sp]
    cands = (analytics or {}).get("chart_candidates") or []
    sp = _spec_from_suggestion(rec, cols, prows, cands[0]) if cands else None
    return [sp] if sp else []


def _final(sql, cols, rows, sm, table="t", suggestion=None):
    rec = _recommender()
    ctx, a = _analytics(sql, cols, rows, sm, table)
    specs = _final_chart(rec, cols, _positional(cols, rows), a, suggestion)
    return ctx.result_shape, [s.type.value for s in specs], specs


SM = {"columns": {
    "t.label":    _col("t", "label", "CATEGORY", "DIMENSION"),
    "t.status":   _col("t", "status", "CATEGORY", "DIMENSION"),
    "t.numcat":   _col("t", "numcat", "CATEGORY", "DIMENSION"),      # numeric-valued category
    "t.amount":   _col("t", "amount", "MONETARY", "MEASURE"),
    "t.profit":   _col("t", "profit", "METRIC", "MEASURE"),
    "t.month":    _col("t", "month", "TEMPORAL", "TIME_DIMENSION"),
    "t.key":      _col("t", "key", "IDENTIFIER", "IDENTIFIER"),
    "t.email":    _col("t", "email", "FREE_TEXT", "ATTRIBUTE"),
}, "tables": {"t": {}}}


def _rows(pairs, a="label", b="amount"):
    return [{a: n, b: v} for n, v in pairs]


# ===========================================================================
# A–Q shape matrix (final chart via full precedence)
# ===========================================================================
def test_A_category_measure_bar_and_pie():
    shape, types, specs = _final(
        "SELECT label, SUM(amount) AS amount FROM t GROUP BY label",
        ["label", "amount"], _rows([("A", 900), ("B", 600), ("C", 300)]), SM)
    assert shape == "GROUPED"
    assert "bar" in types and "pie" in types
    bar = next(s for s in specs if s.type.value == "bar")
    assert bar.x_axis_title == "Label" and bar.y_axis_title == "Amount"   # axis correctness


def test_B_ranking_leads_bar_not_pie():           # FIX-1
    shape, types, specs = _final(
        "SELECT label, SUM(amount) AS amount FROM t GROUP BY label ORDER BY SUM(amount) DESC LIMIT 3",
        ["label", "amount"], _rows([("A", 900), ("B", 600), ("C", 300)]), SM)
    assert shape == "RANKING"
    assert types[0] == "bar" and "pie" not in types
    assert specs[0].x_axis_title == "Label" and specs[0].y_axis_title == "Amount"


def test_C_temporal_measure_line():
    shape, types, specs = _final(
        "SELECT month, SUM(amount) AS amount FROM t GROUP BY month ORDER BY month",
        ["month", "amount"], _rows([("2026-01-01", 100), ("2026-02-01", 150), ("2026-03-01", 220)],
                                   a="month"), SM)
    assert shape == "TREND"
    assert types[0] == "line"
    assert specs[0].x_axis_title == "Month" and specs[0].y_axis_title == "Amount"


def test_D_temporal_category_measure_series_dropped_limitation():
    # LIMITATION (documented, not a bug fix here): multi-series (month × region) is not
    # supported — it charts the measure over time and drops the second dimension.
    shape, types, specs = _final(
        "SELECT month, status, SUM(amount) AS amount FROM t GROUP BY month, status",
        ["month", "status", "amount"],
        [{"month": "2026-01-01", "status": "N", "amount": 100},
         {"month": "2026-02-01", "status": "S", "amount": 200}], SM)
    assert "line" in types
    assert specs[0].x_axis_title == "Month"          # series (status) is not an axis today


def test_E_category_two_measures_combo():
    shape, types, specs = _final(
        "SELECT status, SUM(amount) AS amount, SUM(profit) AS profit FROM t GROUP BY status",
        ["status", "amount", "profit"],
        [{"status": "N", "amount": 900, "profit": 100},
         {"status": "S", "amount": 600, "profit": 80}], SM)
    assert shape == "PIVOT"
    assert types == ["line_histogram"]               # two distinct measures → combo


def test_F_two_measures_no_dimension_no_chart():
    shape, types, _ = _final(
        "SELECT SUM(amount) AS amount, SUM(profit) AS profit FROM t",
        ["amount", "profit"], [{"amount": 900, "profit": 100}], SM)
    assert types == []


def test_G_identifier_measure_no_chart():
    shape, types, _ = _final(
        "SELECT key, SUM(amount) AS amount FROM t GROUP BY key",
        ["key", "amount"], [{"key": i, "amount": v} for i, v in [(1, 900), (2, 600), (3, 300)]], SM)
    assert types == []


def test_H_numeric_identifier_ranking_no_chart_current_behavior():
    # CURRENT behavior (reported): a RANKING whose only label is a NUMERIC identifier
    # yields no chart (the string-id RANKING rescue can't find a non-numeric label).
    shape, types, _ = _final(
        "SELECT key, SUM(amount) AS amount FROM t GROUP BY key ORDER BY SUM(amount) DESC LIMIT 3",
        ["key", "amount"], [{"key": i, "amount": v} for i, v in [(1, 900), (2, 600), (3, 300)]], SM)
    assert shape == "RANKING" and types == []


def test_I_detail_table_no_chart():
    _, types, _ = _final(
        "SELECT label, email, status FROM t LIMIT 100",
        ["label", "email", "status"],
        [{"label": "A", "email": "a@x.com", "status": "ok"},
         {"label": "B", "email": "b@x.com", "status": "no"}], SM)
    assert types == []


def test_J_scalar_no_chart():
    _, types, _ = _final("SELECT SUM(amount) AS amount FROM t", ["amount"], [{"amount": 12345}], SM)
    assert types == []


def test_K_high_cardinality_bucketed_single_bar():
    _, types, specs = _final(
        "SELECT label, SUM(amount) AS amount FROM t GROUP BY label",
        ["label", "amount"], [{"label": f"C{i}", "amount": i} for i in range(50)], SM)
    assert types == ["bar"]                          # top-N + Other, not 50 slices
    assert len(specs[0].chart_data["labels"]) <= 10


def test_L_single_row_no_chart():
    _, types, _ = _final(
        "SELECT label, SUM(amount) AS amount FROM t GROUP BY label",
        ["label", "amount"], [{"label": "Only", "amount": 500}], SM)
    assert types == []


def test_M_null_heavy_no_chart_no_crash():
    _, types, _ = _final(
        "SELECT label, SUM(amount) AS amount FROM t GROUP BY label",
        ["label", "amount"],
        [{"label": None, "amount": None}, {"label": "A", "amount": None}, {"label": None, "amount": 5}], SM)
    assert types == []                               # no misleading single-"None" chart, no crash


def test_N_empty_no_chart():
    rec = _recommender()
    assert rec.recommend(["label", "amount"], [], analytics={}) == []
    assert _final_chart(rec, ["label", "amount"], [], {}) == []


def test_O_distribution_pie():
    shape, types, specs = _final(
        "SELECT status, COUNT(*) AS n FROM t GROUP BY status",
        ["status", "n"], [{"status": "ok", "n": 10}, {"status": "no", "n": 3}, {"status": "meh", "n": 7}], SM)
    assert shape == "DISTRIBUTION"
    assert "pie" in types


def test_Q_multiple_dimensions_second_dropped_limitation():
    # LIMITATION: a second dimension is aggregated away / not a series today.
    shape, types, specs = _final(
        "SELECT label, status, SUM(amount) AS amount FROM t GROUP BY label, status",
        ["label", "status", "amount"],
        [{"label": "A", "status": "ok", "amount": 100}, {"label": "B", "status": "no", "amount": 200}], SM)
    assert shape == "GROUPED" and ("bar" in types or "pie" in types)


# ===========================================================================
# FIX-2 — role-aware binning (numeric-valued CATEGORY dimension)
# ===========================================================================
def test_numeric_category_dimension_charts_from_recommender():
    rec = _recommender()
    _, a = _analytics("SELECT numcat, SUM(amount) AS amount FROM t GROUP BY numcat",
                      ["numcat", "amount"], _rows([(10, 900), (20, 600), (30, 300)], a="numcat"),
                      SM, "t")
    prows = _positional(["numcat", "amount"], _rows([(10, 900), (20, 600), (30, 300)], a="numcat"))
    specs = rec.recommend(["numcat", "amount"], prows, analytics=a)
    assert specs, "recommender should chart a numeric-valued CATEGORY dimension (role, not kind)"
    bar = next((s for s in specs if s.type.value == "bar"), specs[0])
    assert bar.x_axis_title == "Numcat"              # the category is the X axis, not a measure


# ===========================================================================
# axis correctness — identifier is never chosen over a real display dimension
# ===========================================================================
def test_identifier_never_x_when_display_dimension_present():
    rec = _recommender()
    cols = ["key", "label", "amount"]
    rows = [{"key": 1, "label": "A", "amount": 900}, {"key": 2, "label": "B", "amount": 300}]
    _, a = _analytics("SELECT key, label, SUM(amount) AS amount FROM t GROUP BY key, label",
                      cols, rows, SM, "t")
    specs = rec.recommend(cols, _positional(cols, rows), analytics=a)
    assert specs
    bar = next((s for s in specs if s.type.value == "bar"), specs[0])
    assert bar.x_axis_title == "Label"               # display dimension, never the id


# ===========================================================================
# FIX-3 — the suggestion/candidate fallback returns ONE spec (no list.to_dict crash)
# ===========================================================================
def test_category_numeric_returns_list_but_suggestion_returns_single_spec():
    rec = _recommender()
    cols = ["label", "amount"]
    prows = _positional(cols, _rows([("A", 900), ("B", 600)]))
    # root of the old crash: _category_numeric returns a LIST
    assert isinstance(rec._category_numeric(cols, prows, 0, 1), list)
    # the fixed suggestion path returns a single spec with .to_dict() (never a list)
    sp = _spec_from_suggestion(rec, cols, prows, {"type": "bar", "x_axis": "label", "y_axis": "amount"})
    assert sp is not None and hasattr(sp, "to_dict")
    assert sp.to_dict()["type"] == "bar"


# ===========================================================================
# cross-tier consistency — output depends only on result + semantics, not origin
# ===========================================================================
def test_visualization_independent_of_sql_origin_tier():
    rec = _recommender()
    cols = ["label", "amount"]
    rows = _rows([("A", 900), ("B", 600), ("C", 300)])
    _, a = _analytics("SELECT label, SUM(amount) AS amount FROM t GROUP BY label", cols, rows, SM, "t")
    prows = _positional(cols, rows)
    out1 = [s.type.value for s in rec.recommend(cols, prows, analytics={**a, "origin": "tier1"})]
    out2 = [s.type.value for s in rec.recommend(cols, prows, analytics={**a, "origin": "langgraph"})]
    assert out1 == out2 and out1


# ===========================================================================
# schema independence — anti-convention names, same chart decisions
# ===========================================================================
def test_schema_independence_charts_by_role_not_names():
    rec = _recommender()
    sm = {"columns": {
        "x.object_key":      _col("x", "object_key", "IDENTIFIER", "IDENTIFIER"),
        "x.friendly_cap":    _col("x", "friendly_cap", "CATEGORY", "DIMENSION"),
        "x.metric_value":    _col("x", "metric_value", "METRIC", "MEASURE"),
    }, "tables": {"x": {}}}
    cols = ["friendly_cap", "metric_value"]
    rows = [{"friendly_cap": n, "metric_value": v} for n, v in [("A", 9), ("B", 3)]]
    _, a = _analytics("SELECT friendly_cap, SUM(metric_value) AS metric_value FROM x GROUP BY friendly_cap",
                      cols, rows, sm, "x")
    specs = rec.recommend(cols, _positional(cols, rows), analytics=a)
    assert specs
    bar = next((s for s in specs if s.type.value == "bar"), specs[0])
    assert bar.x_axis_title == "Friendly Cap" and bar.y_axis_title == "Metric Value"
    # identifier-only dimension in the same schema → no chart
    cols2 = ["object_key", "metric_value"]
    rows2 = [{"object_key": i, "metric_value": v} for i, v in [(1, 9), (2, 3)]]
    _, a2 = _analytics("SELECT object_key, SUM(metric_value) AS metric_value FROM x GROUP BY object_key",
                       cols2, rows2, sm, "x")
    assert _final_chart(rec, cols2, _positional(cols2, rows2), a2) == []
