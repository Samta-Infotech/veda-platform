"""Analytics regression suite — operator-preserving deterministic grouped
aggregation + summary population consistency.

Covers VEDA_NLSQL_ROOTCAUSE_AUDIT root causes:
  RC-1  "average/avg/mean … per …" must route to the deterministic grouped
        planner (not fall to the LLM) — grammar/operator coverage.
  RC-2/3  the grouped dimension must be the metadata-resolved DISPLAY dimension
        (semantic_type CATEGORY), never a technical IDENTIFIER — chosen from
        semantic metadata, NOT from name suffixes.
  operator preservation  AVG/SUM/MIN/MAX survive into the emitted SQL.
  RC-4  pattern/outlier statistics use the SAME population as the headline
        aggregate (full result), never a 200-row sample presented as the mean.

The operator/shape layer (aggregate_operator, grouped_mode) is pure language
layer — schema-independent by construction (it only sees the query string). The
planner tests use the real onboarded semantic model to prove the dimension is
picked from METADATA (semantic_type), with NO table/column/entity hardcoding.

Run from repo root: ``pytest tests/test_grouped_aggregation_operators.py``
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

_SM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "veda_core", "data", "veda_semantic_model.json")


@pytest.fixture(scope="module")
def sm():
    with open(_SM_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Generic operator normalizer (schema-independent — query string only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("average carpet area per project", "AVG"),
    ("avg revenue by region", "AVG"),
    ("mean price per type", "AVG"),
    ("total revenue per city", "SUM"),
    ("sum of amount per account", "SUM"),
    ("how much does each type contribute", "SUM"),
    ("minimum rent per project", "MIN"),
    ("maximum score per team", "MAX"),
    ("count of leads per source", "COUNT"),
    ("how many orders per customer", "COUNT"),
    ("show all assets", None),
    ("list properties in the UAE", None),
])
def test_aggregate_operator_is_canonical(query, expected):
    from veda.planning import aggregate_operator
    assert aggregate_operator(query) == expected


def test_aggregate_operator_prefers_longer_phrase():
    # "how much" (SUM) must not be shadowed by any bare token; "how many" is COUNT.
    from veda.planning import aggregate_operator
    assert aggregate_operator("how much revenue per type") == "SUM"
    assert aggregate_operator("how many payments per type") == "COUNT"


# ---------------------------------------------------------------------------
# grouped_mode preserves the operator; legacy SUM behavior unchanged; COUNT
# is intentionally left to the counting machinery.
# ---------------------------------------------------------------------------
def test_grouped_mode_preserves_avg():
    from veda.planning import grouped_mode
    assert grouped_mode("average carpet area per project") == {"grouped": True, "op": "AVG"}


def test_grouped_mode_legacy_sum_unchanged():
    from veda.planning import grouped_mode
    # the old measure_agg trigger words still route, now normalized to SUM
    assert grouped_mode("how much does each type contribute") == {"grouped": True, "op": "SUM"}
    assert grouped_mode("total amount per type") == {"grouped": True, "op": "SUM"}


def test_grouped_mode_requires_grouping_word():
    from veda.planning import grouped_mode
    # an aggregate with no grouping word is NOT a grouped breakdown
    assert grouped_mode("average carpet area") is None


def test_grouped_mode_count_falls_through():
    from veda.planning import grouped_mode
    # COUNT-per-dimension needs no measure column → handled elsewhere, not here
    assert grouped_mode("count of leads per source") is None
    assert grouped_mode("how many leads per source") is None


def test_grouped_mode_yields_to_superlative():
    from veda.planning import grouped_mode
    # interrogative ranking ("which … highest …") is owned by the superlative planner
    assert grouped_mode("which project has the highest carpet area") is None


# ---------------------------------------------------------------------------
# Deterministic grouped planner — operator preserved AND dimension is the
# metadata-resolved CATEGORY column, never the IDENTIFIER. No hardcoding: the
# assertions read semantic_type from the model to prove metadata drove the pick.
# ---------------------------------------------------------------------------
def _semantic_type(sm, table, col):
    return (sm.get("columns", {}).get(f"{table}.{col}", {}) or {}).get("semantic_type")


@pytest.mark.parametrize("query,op", [
    ("average carpet_area_sqft per project", "AVG"),
    ("total carpet_area_sqft per project", "SUM"),
    ("minimum carpet_area_sqft per project", "MIN"),
    ("maximum carpet_area_sqft per project", "MAX"),
])
def test_grouped_planner_preserves_operator_and_groups_by_display_dimension(sm, query, op):
    from query.superlative_plan import try_grouped_plan
    r = try_grouped_plan(query, sm)
    assert r is not None and not isinstance(r, tuple), f"expected a plan for {query!r}, got {r!r}"
    sql = r.sql
    # 1. requested operator survived into SQL (not assumed SUM)
    assert f"{op}(" in sql, f"operator {op} missing from SQL: {sql}"
    if op != "SUM":
        assert "SUM(" not in sql, f"SQL fell back to SUM for a {op} request: {sql}"
    # 2. GROUP BY is present and is a metadata CATEGORY dimension, not an IDENTIFIER.
    #    Prove it via semantic_type, not by matching a name suffix.
    assert "GROUP BY" in sql
    anchor = r.primary
    grouped_cols = [c for c in sql.split("GROUP BY", 1)[1].split("ORDER BY", 1)[0].replace('"', "").split(".")
                    if c.strip() and c.strip() != "a"]
    # the grouped column name is the last identifier token after 'a.' in GROUP BY
    grouped_col = sql.split("GROUP BY", 1)[1].split("ORDER BY", 1)[0]
    grouped_col = grouped_col.replace('a."', "").replace('"', "").strip()
    st = _semantic_type(sm, anchor, grouped_col)
    assert st in ("CATEGORY", "CATEGORICAL"), (
        f"grouped-by {anchor}.{grouped_col} has semantic_type {st!r}, expected a CATEGORY dimension")
    assert st != "IDENTIFIER", f"grouped by a technical identifier: {grouped_col}"


def test_ambiguous_measure_yields_grounded_clarify(sm):
    # "carpet area" matches two distinct measure columns (carpet_area vs
    # carpet_area_sqft, different units) → refuse-over-guess grounded clarify,
    # NOT a silent guess. This is correct escalation, not a regression.
    from query.superlative_plan import try_grouped_plan
    r = try_grouped_plan("average carpet area per project", sm)
    assert isinstance(r, tuple) and r[0] == "clarify"
    assert "carpet_area" in r[1] and "carpet_area_sqft" in r[1]


# ---------------------------------------------------------------------------
# RC-4 — summary population consistency: pattern/outlier statistics run over the
# FULL result, so the outlier "vs average X" mean equals the full-population mean
# the headline uses, never a 200-row-sample mean.
# ---------------------------------------------------------------------------
def test_patterns_use_full_population_not_sample():
    import statistics
    from veda.result_analyzer import analyze_result

    # 260 rows: the first 200 (the old sample window) are all 100.0; the tail
    # (rows 200-259) is where a single extreme outlier lives. Under 200-row
    # sampling the outlier is invisible AND the mean would be exactly 100.0;
    # over the full population the mean shifts and the outlier is detected.
    rows = [{"label": f"g{i}", "score": 100.0} for i in range(259)]
    rows.append({"label": "g259", "score": 100000.0})
    full_mean = round(statistics.fmean([r["score"] for r in rows]), 2)
    assert full_mean != 100.0  # sanity: full population differs from the 200-sample

    ctx = analyze_result(
        question="score per label",
        sql='SELECT label, score FROM t',
        columns=["label", "score"],
        rows=rows,
        max_rows=200,
    )
    outliers = [p for p in ctx.patterns if p.kind == "outlier" and p.column == "score"]
    assert outliers, "outlier in the tail was not detected → patterns did not scan the full result"
    # the reported comparison mean is the FULL mean, not the 100.0 sample mean
    assert f"vs average {full_mean}" in outliers[0].detail, outliers[0].detail


def test_missing_values_wording_not_sampled():
    from veda.result_analyzer import detect_patterns, ColumnStat
    stats = [ColumnStat(name="note", kind="categorical", role="dimension")]
    rows = [{"note": None} for _ in range(8)] + [{"note": "x"} for _ in range(2)]
    pats = detect_patterns("DETAIL_TABLE", stats, rows, filters=[], measures=[])
    miss = [p for p in pats if p.kind == "missing_values"]
    assert miss and "sampled" not in miss[0].detail
