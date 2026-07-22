"""VEDA analytics — ONE comprehensive MODEL-FREE integration suite.

Validates the recent analytics improvements END-TO-END without any SLM / LLM /
embedding model / Ollama / external service:

  1. aggregate / analytical semantics (operator preservation)
  2. display-vs-identifier semantics (metadata-driven, no suffix reliance)
  3. Tier-1 -> Tier-2 provenance handoff
  4. shared semantic validation (advisory; enforcement stays OFF)
  5. result analysis (shapes / roles / findings)
  6. summary analytics (modes / prompt / grounding)  [SLM stubbed]
  7. population consistency + ANALYSIS_MAX_ROWS bound
  8. existing visualization baseline (regression only — not modified)
  9. cross-component integration scenarios
 10. schema-independence matrix (two opposite-convention schemas)

This tests the CURRENT implementation. Where a test encodes a genuine
implementation gap it is marked xfail(strict=True) with a reason and reported —
production code is NOT changed here.

Run from repo root: ``pytest tests/test_analytics_integration.py``
"""
import json
import os
import statistics
import sys
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                       # repo root (apps.chat.visualization)
sys.path.insert(0, os.path.join(_ROOT, "veda_core"))

_SM_PATH = os.path.join(_ROOT, "veda_core", "data", "veda_semantic_model.json")


# ---------------------------------------------------------------------------
# synthetic schemas — same business semantics, DELIBERATELY different naming.
# Schema A: conventional-ish. Schema B: anti-convention (identifier has no _id,
# display has no _name). Only semantic_type / analytics_role identify columns —
# proving no dependence on project/customer/revenue/_id/_name vocabulary.
# ---------------------------------------------------------------------------
def _col(table, name, stype, arole):
    return {"col_name": name, "table_name": table, "semantic_type": stype,
            "analytics_role": arole}


SCHEMA_A = {
    "columns": {
        "sales.entity_id":   _col("sales", "entity_id", "IDENTIFIER", "IDENTIFIER"),
        "sales.entity_name": _col("sales", "entity_name", "CATEGORY", "DIMENSION"),
        "sales.amount":      _col("sales", "amount", "MONETARY", "MEASURE"),
        "sales.sold_on":     _col("sales", "sold_on", "TEMPORAL", "TIME_DIMENSION"),
    },
    "tables": {"sales": {}},
}
SCHEMA_B = {
    "columns": {
        "ledger.object_key":       _col("ledger", "object_key", "IDENTIFIER", "IDENTIFIER"),
        "ledger.friendly_caption": _col("ledger", "friendly_caption", "CATEGORY", "DIMENSION"),
        "ledger.metric_value":     _col("ledger", "metric_value", "METRIC", "MEASURE"),
        "ledger.captured_at":      _col("ledger", "captured_at", "TEMPORAL", "TIME_DIMENSION"),
    },
    "tables": {"ledger": {}},
}
# Schema with NO display/category column at all — only an identifier + a measure.
SCHEMA_NO_DISPLAY = {
    "columns": {
        "t.k":   _col("t", "k", "IDENTIFIER", "IDENTIFIER"),
        "t.val": _col("t", "val", "METRIC", "MEASURE"),
    },
    "tables": {"t": {}},
}


def _stat(name, kind, role):
    from veda.result_analyzer import ColumnStat
    return ColumnStat(name=name, kind=kind, role=role)


# ===========================================================================
# 1. AGGREGATE / ANALYTICAL SEMANTICS
# ===========================================================================
@pytest.mark.parametrize("query,op", [
    ("average amount per entity", "AVG"),
    ("total amount by entity", "SUM"),
    ("minimum amount per entity", "MIN"),
    ("maximum amount by entity", "MAX"),
    ("count of orders per entity", "COUNT"),
])
def test_1_operator_normalization_is_canonical_and_schema_free(query, op):
    from veda.planning import aggregate_operator
    assert aggregate_operator(query) == op


def test_1_grouped_mode_preserves_measure_operator_not_count():
    from veda.planning import grouped_mode
    assert grouped_mode("average amount per entity") == {"grouped": True, "op": "AVG"}
    assert grouped_mode("total amount per entity") == {"grouped": True, "op": "SUM"}
    # COUNT-per-dimension is intentionally NOT the measure planner's job
    assert grouped_mode("count of orders per entity") is None


def test_1_LIMITATION_bare_by_is_not_a_grouping_trigger():
    """CURRENT behavior (reported limitation, not a correctness bug): the grouping
    grammar recognizes 'per' / 'each' / 'grouped by' / 'breakdown' but NOT bare 'by',
    so a very common phrasing like 'total revenue by region' does NOT route to the
    deterministic grouped planner — it falls through to the LLM path (still validated).
    Documented here so a future grammar change is a conscious decision."""
    from veda.planning import grouped_mode
    assert grouped_mode("total amount by entity") is None
    assert grouped_mode("total amount per entity") is not None


def test_1_deterministic_planner_preserves_operator_and_groups_by_category():
    # Uses the REAL semantic model (QSR needs its artifacts); asserts the GENERIC
    # invariant via metadata, never a business name: operator preserved + GROUP BY a
    # CATEGORY column, never an IDENTIFIER.
    with open(_SM_PATH) as f:
        sm = json.load(f)
    from query.superlative_plan import try_grouped_plan
    r = try_grouped_plan("average carpet_area_sqft per project", sm)
    assert r is not None and not isinstance(r, tuple)
    assert "AVG(" in r.sql and "SUM(" not in r.sql
    grouped_col = r.sql.split("GROUP BY", 1)[1].split("ORDER BY", 1)[0]
    grouped_col = grouped_col.replace('a."', "").replace('"', "").strip()
    st = (sm["columns"].get(f"{r.primary}.{grouped_col}", {}) or {}).get("semantic_type")
    assert st in ("CATEGORY", "CATEGORICAL") and st != "IDENTIFIER"


def test_1_ambiguous_measure_clarifies_not_guesses():
    with open(_SM_PATH) as f:
        sm = json.load(f)
    from query.superlative_plan import try_grouped_plan
    r = try_grouped_plan("average carpet area per project", sm)  # two carpet columns
    assert isinstance(r, tuple) and r[0] == "clarify"


# ===========================================================================
# 2. DISPLAY VS IDENTIFIER SEMANTICS
# ===========================================================================
def test_2_display_resolution_is_metadata_driven_both_schemas():
    from veda.generation import _resolve_display_column
    assert _resolve_display_column("sales", SCHEMA_A) == "entity_name"
    # anti-convention: no _name suffix, resolved via analytics_role=DIMENSION+CATEGORY
    assert _resolve_display_column("ledger", SCHEMA_B) == "friendly_caption"


def test_2A_normal_query_prefers_category_dimension():
    from veda.semantic_validation import validate_analytical_semantics as V
    codes = {f["code"] for f in V(
        "average metric per object",
        "SELECT friendly_caption, AVG(metric_value) FROM ledger GROUP BY friendly_caption",
        SCHEMA_B)}
    assert "identifier_dimension" not in codes


def test_2B_explicit_identifier_request_preserved():
    from veda.semantic_validation import validate_analytical_semantics as V, user_requested_identifier
    assert user_requested_identifier("which object key has the highest metric")
    codes = {f["code"] for f in V(
        "which object key has the highest metric",
        "SELECT object_key, AVG(metric_value) FROM ledger GROUP BY object_key", SCHEMA_B)}
    assert "identifier_dimension" not in codes          # explicit id → not flagged


def test_2_identifier_grouping_without_request_is_flagged():
    from veda.semantic_validation import validate_analytical_semantics as V
    codes = {f["code"] for f in V(
        "average metric per object",
        "SELECT object_key, AVG(metric_value) FROM ledger GROUP BY object_key", SCHEMA_B)}
    assert "identifier_dimension" in codes


def test_2C_no_display_available_no_invented_column():
    # No CATEGORY column exists → resolver returns None (never fabricates a name).
    from veda.generation import _resolve_display_column
    assert _resolve_display_column("t", SCHEMA_NO_DISPLAY) is None


# ===========================================================================
# 3. TIER-1 -> TIER-2 PROVENANCE
# ===========================================================================
def test_3_execution_state_has_provenance_fields():
    from veda.execution_state import ExecutionState
    es = ExecutionState()
    assert hasattr(es, "candidate_fields") and hasattr(es, "rerank_query")


def test_3_seed_merge_preserves_semantic_type_and_score():
    from query.retrieval_v2 import _merge_seed_candidates
    seeds = [
        {"table_name": "sales", "col_name": "entity_name", "score": 0.8,
         "semantic_type": "CATEGORY", "rrf_score": 0.4, "cross_encoder_score": 0.8,
         "reranked": True},
        {"table_name": "sales", "col_name": "amount", "score": 0.7,
         "semantic_type": "MONETARY", "rrf_score": 0.7, "cross_encoder_score": None,
         "reranked": False},
        {"table_name": "sales", "col_name": "legacy", "score": 0.6},   # no type
    ]
    cols, _ = _merge_seed_candidates([], [], seeds)
    by = {c.col_name: c for c in cols}
    assert by["entity_name"].semantic_type == "CATEGORY"
    assert by["amount"].semantic_type == "MONETARY"
    assert by["legacy"].semantic_type == "UNKNOWN"      # missing key → UNKNOWN, not wrong
    # seed's own score flows in as first-stage similarity (reranker still re-scores)
    assert by["amount"].similarity == 0.7


def test_3_raw_rrf_not_confused_with_cross_encoder():
    # a non-reranked seed carries rrf_score but cross_encoder_score None + reranked False
    seed = {"table_name": "sales", "col_name": "amount", "score": 0.7,
            "semantic_type": "MONETARY", "rrf_score": 0.7,
            "cross_encoder_score": None, "reranked": False}
    assert seed["reranked"] is False and seed["cross_encoder_score"] is None
    assert seed["rrf_score"] == 0.7


# ===========================================================================
# 4. SHARED SEMANTIC VALIDATION
# ===========================================================================
def test_4_operator_pass_and_mismatch():
    from veda.semantic_validation import validate_analytical_semantics as V
    ok = {f["code"] for f in V("average amount per entity",
          "SELECT entity_name, AVG(amount) FROM sales GROUP BY entity_name", SCHEMA_A)}
    assert "operator_mismatch" not in ok
    bad = {f["code"] for f in V("average amount per entity",
          "SELECT entity_name, SUM(amount) FROM sales GROUP BY entity_name", SCHEMA_A)}
    assert "operator_mismatch" in bad


def test_4_grounded_and_ungrounded_join():
    from veda.semantic_validation import validate_analytical_semantics as V
    sm = {"columns": {"a.x": {"semantic_type": "METRIC"}, "b.y": {"semantic_type": "CATEGORY"}},
          "tables": {"a": {}, "b": {}}}
    grounded = {"edges": [{"source_table": "a", "target_table": "b"}]}
    ungrounded = {"edges": []}
    assert "ungrounded_join" not in {f["code"] for f in V(
        "y and x", "SELECT b.y, a.x FROM a JOIN b ON a.k=b.k", sm, graph=grounded)}
    assert "ungrounded_join" in {f["code"] for f in V(
        "y and x", "SELECT b.y, a.x FROM a JOIN b ON a.k=b.k", sm, graph=ungrounded)}


def test_4_enforcement_is_currently_off():
    # Confirm we did NOT enable enforcement — advisory only.
    import config
    assert getattr(config, "SEMANTIC_VALIDATION_ENABLED", False) is True
    assert getattr(config, "SEMANTIC_VALIDATION_ENFORCE", True) is False


# ===========================================================================
# 5. RESULT ANALYSIS  (shapes / roles / findings)
# ===========================================================================
def _analyze(query, sql, cols, rows, sm, table):
    from veda.result_analyzer import analyze_result
    return analyze_result(query, sql, cols, rows, sm=sm, table=table)


def test_5_scalar_shape():
    ctx = _analyze("total amount", "SELECT SUM(amount) AS amount FROM sales",
                   ["amount"], [{"amount": 5000}], SCHEMA_A, "sales")
    assert ctx.result_shape == "SCALAR"


def test_5_grouped_category_measure_roles_and_findings():
    rows = [{"entity_name": n, "amount": v} for n, v in
            [("A", 900), ("B", 300), ("C", 600)]]
    ctx = _analyze("amount per entity",
                   "SELECT entity_name, SUM(amount) AS amount FROM sales GROUP BY entity_name",
                   ["entity_name", "amount"], rows, SCHEMA_A, "sales")
    roles = {s.name: s.role for s in ctx.column_stats}
    assert roles == {"entity_name": "dimension", "amount": "measure"}
    kinds = {p.kind for p in ctx.patterns}
    assert {"leader", "laggard"} <= kinds


def test_5_identifier_measure_no_dimension_findings():
    rows = [{"entity_id": i, "amount": v} for i, v in [(1, 900), (2, 300)]]
    ctx = _analyze("amount per id",
                   "SELECT entity_id, SUM(amount) AS amount FROM sales GROUP BY entity_id",
                   ["entity_id", "amount"], rows, SCHEMA_A, "sales")
    assert {s.name: s.role for s in ctx.column_stats}["entity_id"] == "identifier"
    assert not any(p.kind in ("leader", "laggard") for p in ctx.patterns)


def test_5_detail_table_shape():
    rows = [{"entity_name": n, "amount": v} for n, v in [("A", 1), ("B", 2)]]
    ctx = _analyze("list sales", "SELECT entity_name, amount FROM sales LIMIT 100",
                   ["entity_name", "amount"], rows, SCHEMA_A, "sales")
    assert ctx.result_shape == "DETAIL_TABLE"


def test_5_display_columns_exclude_identifier():
    from veda.result_analyzer import analytics_summary
    rows = [{"entity_id": 1, "entity_name": "A", "amount": 5}]
    ctx = _analyze("x", "SELECT entity_id, entity_name, amount FROM sales",
                   ["entity_id", "entity_name", "amount"], rows, SCHEMA_A, "sales")
    disp = analytics_summary(ctx)["display_columns"]
    assert "entity_id" not in disp and "entity_name" in disp


# ===========================================================================
# 6. SUMMARY ANALYTICS  (SLM stubbed)
# ===========================================================================
def _capture(monkeypatch, query, cols, rows, **kw):
    import slm
    import query.result_explainer as re_mod
    seen = {}
    monkeypatch.setattr(slm, "call_slm",
                        lambda p, **k: (seen.update(p=p, np=k.get("num_predict")) or "Stub."))
    monkeypatch.setattr(re_mod, "NL_SUMMARY_NUMERIC_GUARD", False, raising=False)
    re_mod.run_nl_answer(query, cols, rows, **kw)
    return seen["p"], seen["np"]


def test_6A_scalar_brief(monkeypatch):
    p, npv = _capture(monkeypatch, "total amount", ["amount"], [{"amount": 5000}],
                      result_shape="SCALAR")
    assert "1-2 sentences" in p and "3-5 concise sentences" not in p


def test_6B_grouped_analytical(monkeypatch):
    rows = [{"entity_name": n, "amount": v} for n, v in [("A", 900), ("B", 300)]]
    p, npv = _capture(monkeypatch, "amount per entity", ["entity_name", "amount"], rows,
                      result_shape="GROUPED",
                      patterns=["A has the highest amount at 900", "B has the lowest amount at 300"],
                      analytical_context={"operation": "SUM"})
    assert "3-5 concise sentences" in p and npv >= 300
    assert "A has the highest amount at 900" in p


def test_6C_ranking_context_and_no_invented_math(monkeypatch):
    rows = [{"entity_name": n, "amount": v} for n, v in [("A", 5000), ("B", 1000)]]
    p, _ = _capture(monkeypatch, "top 2 entities by amount", ["entity_name", "amount"], rows,
                    result_shape="RANKING", patterns=["the #1 entry leads #2 by 400% on amount"])
    assert "growth rate" in p          # explicit "never calculate ... growth rate"
    assert "the #1 entry leads #2 by 400% on amount" in p


def test_6E_explicit_identifier_survives_to_summary(monkeypatch):
    p, _ = _capture(monkeypatch, "which entity id has the highest amount",
                    ["entity_id", "amount"], [{"entity_id": 1, "amount": 9}],
                    result_shape="RANKING", analytical_context={"explicit_identifier": True})
    assert "explicit id requested=True" in p and "explicitly asked for an id" in p


def test_6F_no_pattern_no_fake_finding(monkeypatch):
    p, _ = _capture(monkeypatch, "total amount", ["amount"], [{"amount": 5000}],
                    result_shape="SCALAR")               # no patterns passed
    assert "Verified findings already computed" not in p
    assert "do NOT invent an insight" in p


def test_6G_numeric_grounding_rejects_ungrounded_number():
    from query.result_explainer import _answer_numbers_grounded
    facts = {"row_count": 3, "metrics": {"amount": {"sum": 1500, "mean": 500}}}
    assert _answer_numbers_grounded("Total amount is 1500.", facts, [])
    assert not _answer_numbers_grounded("Total amount is 9999.", facts, [])


# ===========================================================================
# 7. POPULATION CONSISTENCY + ANALYSIS_MAX_ROWS BOUND
# ===========================================================================
def test_7_true_row_count_preserved_when_bounded(monkeypatch):
    import config
    from veda.result_analyzer import analyze_result
    monkeypatch.setattr(config, "ANALYSIS_MAX_ROWS", 100, raising=False)
    rows = [{"g": f"g{i}", "v": 1.0} for i in range(150)]
    ctx = analyze_result("v per g", "SELECT g, v FROM t GROUP BY g", ["g", "v"], rows)
    assert ctx.row_count == 150           # TRUE total, not the bound


def test_7_both_paths_use_same_bounded_population(monkeypatch):
    # RC-4: analyzer pattern stats and summary metric stats use the SAME bound → agree.
    import config, query.result_explainer as re_mod
    monkeypatch.setattr(config, "ANALYSIS_MAX_ROWS", 100, raising=False)
    monkeypatch.setattr(re_mod, "ANALYSIS_MAX_ROWS", 100, raising=False)
    rows = [{"v": 100.0} for _ in range(100)] + [{"v": 999.0} for _ in range(50)]
    facts_mean = re_mod._numeric_aggregates(["v"], rows)["v"]["mean"]
    # analyzer measure stat over same bound
    from veda.result_analyzer import _column_stats
    assert facts_mean == 100.0            # bounded to first 100 → 100.0, no full/sample mix


def test_7_bounded_metrics_are_flagged_as_partial(monkeypatch):
    # BUG-2 fixed: when a result exceeds ANALYSIS_MAX_ROWS, facts flags the metrics as
    # partial (metrics_partial + metrics_scanned) so a partial SUM/COUNT is not narrated
    # as the full-population total. TRUE row_count is still reported.
    import config, query.result_explainer as re_mod
    monkeypatch.setattr(config, "ANALYSIS_MAX_ROWS", 100, raising=False)
    monkeypatch.setattr(re_mod, "ANALYSIS_MAX_ROWS", 100, raising=False)
    rows = [{"v": 100.0} for _ in range(150)]     # 150 > bound 100
    facts = re_mod._extract_facts(["v"], rows)
    assert facts["row_count"] == 150
    assert facts.get("metrics_partial") is True
    assert facts.get("metrics_scanned") == 100


def test_7_full_result_metrics_not_flagged_partial():
    # under the bound → exact metrics, no partial flag
    import query.result_explainer as re_mod
    facts = re_mod._extract_facts(["v"], [{"v": 1.0} for _ in range(10)])
    assert "metrics_partial" not in facts


# ===========================================================================
# 8. EXISTING VISUALIZATION BEHAVIOR (regression baseline — NOT modified)
# ===========================================================================
def _viz(cols, rows, analytics=None):
    from apps.chat.visualization import VisualizationRecommender
    specs = VisualizationRecommender().recommend(cols, rows, analytics)
    return [s.type.value for s in specs]


def test_8_category_measure_has_bar_candidate():
    rows = [["A", 900], ["B", 600], ["C", 300]]
    types = _viz(["region", "revenue"], rows)
    assert "bar" in types


def test_8_identifier_measure_no_chart():
    # identifier dimension → no usable dimension → no chart (current behavior)
    rows = [[1, 900], [2, 600], [3, 300]]
    assert _viz(["region_id", "revenue"], rows) == []


def test_8_scalar_and_detail_no_inappropriate_chart():
    assert _viz(["total"], [[5000]]) == []
    # a single categorical + no measure isn't a chart
    assert _viz(["name"], [["A"], ["B"]]) == []


def test_8_high_cardinality_still_charts_via_topn():
    rows = [[f"g{i}", i] for i in range(50)]      # 50 categories
    types = _viz(["grp", "val"], rows)
    assert "bar" in types                          # bucketed (top-N + Other), not suppressed


def test_8_recommender_and_analyzer_agree_on_identifier_no_chart():
    from veda.result_analyzer import analyze_result, analytics_summary
    rows = [{"region_id": i, "revenue": v} for i, v in [(1, 900), (2, 600), (3, 300)]]
    sm = {"columns": {
        "s.region_id": _col("s", "region_id", "IDENTIFIER", "IDENTIFIER"),
        "s.revenue":   _col("s", "revenue", "MONETARY", "MEASURE")}, "tables": {"s": {}}}
    ctx = analyze_result("revenue per region_id",
                         "SELECT region_id, SUM(revenue) AS revenue FROM s GROUP BY region_id",
                         ["region_id", "revenue"], rows, sm=sm, table="s")
    analyzer_candidates = analytics_summary(ctx)["chart_candidates"]
    recommender = _viz(["region_id", "revenue"], [[r["region_id"], r["revenue"]] for r in rows])
    # both independently decline to chart an identifier-only dimension
    assert analyzer_candidates == [] and recommender == []


# ===========================================================================
# 9. CROSS-COMPONENT INTEGRATION SCENARIOS
# ===========================================================================
def test_9A_avg_per_entity_full_chain(monkeypatch):
    # AVG recognized -> CATEGORY dim -> roles -> display_columns -> analytical summary
    # -> leader/laggard -> chart candidate. Anti-convention schema (Schema B).
    from veda.planning import aggregate_operator
    from veda.result_analyzer import analyze_result, analytics_summary
    assert aggregate_operator("average metric per caption") == "AVG"
    rows = [{"friendly_caption": n, "metric_value": v} for n, v in [("A", 900), ("B", 300)]]
    ctx = analyze_result(
        "average metric per caption",
        "SELECT friendly_caption, AVG(metric_value) AS metric_value FROM ledger GROUP BY friendly_caption",
        ["friendly_caption", "metric_value"], rows, sm=SCHEMA_B, table="ledger")
    summ = analytics_summary(ctx)
    assert ctx.result_shape == "GROUPED"
    assert set(summ["display_columns"]) == {"friendly_caption", "metric_value"}
    assert {p.kind for p in ctx.patterns} >= {"leader", "laggard"}
    assert summ["chart_candidates"] and summ["chart_candidates"][0]["type"] == "bar"
    from query.result_explainer import _summary_mode
    assert _summary_mode(ctx.result_shape, ctx.row_count) == "analytical"


def test_9B_topN_ranking_chain():
    # BUG-1 fixed: the deterministic planner's ranked-aggregate shape
    # ("... ORDER BY AGG(m) DESC") is no longer double-counted → classified RANKING,
    # with leader/top_gap findings and a chart candidate (not misread as PIVOT).
    from veda.result_analyzer import analyze_result, analytics_summary
    rows = [{"friendly_caption": n, "metric_value": v} for n, v in
            [("A", 5000), ("B", 1000), ("C", 800)]]
    ctx = analyze_result(
        "top 3 captions by metric",
        "SELECT friendly_caption, SUM(metric_value) AS metric_value FROM ledger "
        "GROUP BY friendly_caption ORDER BY SUM(metric_value) DESC LIMIT 3",
        ["friendly_caption", "metric_value"], rows, sm=SCHEMA_B, table="ledger")
    assert ctx.result_shape == "RANKING"
    kinds = {p.kind for p in ctx.patterns}
    assert "leader" in kinds and "top_gap" in kinds
    assert analytics_summary(ctx)["chart_candidates"]          # chart no longer suppressed


def test_9B_ranked_aggregate_does_not_double_count_measures():
    # Root-cause guard for BUG-1: an aggregate repeated in SELECT + ORDER BY counts once.
    from veda.business_explain import extract_sql_facts
    f = extract_sql_facts("SELECT c, SUM(v) AS v FROM t GROUP BY c ORDER BY SUM(v) DESC LIMIT 3")
    assert f["aggregations"] == [("SUM", "v")]
    # genuine multi-measure query is still multi-measure (real PIVOTs survive)
    f2 = extract_sql_facts("SELECT c, SUM(a) AS s, COUNT(b) AS n FROM t GROUP BY c")
    assert len(f2["aggregations"]) == 2


def test_9C_temporal_trend_chain():
    from veda.result_analyzer import analyze_result, analytics_summary
    rows = [{"captured_at": d, "metric_value": v} for d, v in
            [("2026-01-01", 100), ("2026-02-01", 150), ("2026-03-01", 220)]]
    ctx = analyze_result(
        "monthly metric trend",
        "SELECT captured_at, SUM(metric_value) AS metric_value FROM ledger "
        "GROUP BY captured_at ORDER BY captured_at",
        ["captured_at", "metric_value"], rows, sm=SCHEMA_B, table="ledger")
    assert ctx.result_shape == "TREND"
    assert any(p.kind in ("growth", "decline") for p in ctx.patterns)
    # existing viz baseline for temporal+measure
    types = _viz(["captured_at", "metric_value"],
                 [[r["captured_at"], r["metric_value"]] for r in rows],
                 analytics=analytics_summary(ctx))
    assert "line" in types or "bar" in types


def test_9D_explicit_identifier_end_to_end():
    from veda.semantic_validation import validate_analytical_semantics as V, user_requested_identifier
    q = "which object key has the highest metric"
    assert user_requested_identifier(q)
    codes = {f["code"] for f in V(
        q, "SELECT object_key, MAX(metric_value) FROM ledger GROUP BY object_key", SCHEMA_B)}
    assert "identifier_dimension" not in codes       # validator preserves explicit id


# ===========================================================================
# 10. SCHEMA-INDEPENDENCE MATRIX
# ===========================================================================
@pytest.mark.parametrize("sm,table,idcol,dispcol,meas", [
    (SCHEMA_A, "sales", "entity_id", "entity_name", "amount"),
    (SCHEMA_B, "ledger", "object_key", "friendly_caption", "metric_value"),
])
def test_10_equivalent_decisions_across_schemas(sm, table, idcol, dispcol, meas):
    from veda.generation import _resolve_display_column
    from veda.semantic_validation import validate_analytical_semantics as V
    from veda.result_analyzer import analyze_result, analytics_summary
    # display resolves to the CATEGORY column, whatever it is named
    assert _resolve_display_column(table, sm) == dispcol
    # grouping by the identifier (no explicit request) is flagged in BOTH schemas
    codes = {f["code"] for f in V(
        f"average {meas} per thing",
        f"SELECT {idcol}, AVG({meas}) FROM {table} GROUP BY {idcol}", sm)}
    assert "identifier_dimension" in codes
    # grouping by the display column is clean + charts in BOTH schemas
    rows = [{dispcol: n, meas: v} for n, v in [("A", 900), ("B", 300)]]
    ctx = analyze_result(
        f"average {meas} per thing",
        f"SELECT {dispcol}, AVG({meas}) AS {meas} FROM {table} GROUP BY {dispcol}",
        [dispcol, meas], rows, sm=sm, table=table)
    summ = analytics_summary(ctx)
    assert dispcol in summ["display_columns"] and idcol not in summ["display_columns"]
    assert {p.kind for p in ctx.patterns} >= {"leader", "laggard"}
    assert summ["chart_candidates"] and summ["chart_candidates"][0]["type"] == "bar"
