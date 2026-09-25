"""Numeric comparisons in a follow-up: "only the ones with carpet area above 1000".

Measured 2026-09-26 (audit D1 level 5): refused — "couldn't map 'above'".

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_numeric_qualifier.py -q`
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from query.qualifier_grounding import ground_numeric  # noqa: E402
from query.value_arbiter import where_clause  # noqa: E402

SM = {"columns": {
    "assets_asset.carpet_area": {"allowed_aggregations": ["SUM", "AVG"],
                                 "aliases": ["area", "carpet area", "floor space"]},
    "assets_asset.facing": {"allowed_aggregations": ["COUNT", "GROUP_BY"],
                            "aliases": ["direction"]},
    "assets_asset.built_up_area": {"allowed_aggregations": ["SUM", "AVG"], "aliases": ["area"]},
}}


@pytest.mark.parametrize("msg,op,val", [
    ("only the ones with carpet area above 1000", ">", "1000"),
    ("carpet area over 1,500", ">", "1500"),
    ("carpet area at least 2k", ">=", "2000"),
    ("only those with carpet area less than or equal to 800", "<=", "800"),
    ("floor space below 750.5", "<", "750.5"),
])
def test_grounds_the_comparison(msg, op, val):
    f = ground_numeric(msg, "assets_asset", SM)
    assert [(x["column"], x["op"], x["value"]) for x in f] == [("carpet_area", op, val)]
    assert "above" in f[0]["consumed"] or op != ">" or "over" in f[0]["consumed"]


@pytest.mark.parametrize("msg", [
    "only the ones with area above 1000",        # "area" names two numeric columns
    "only the EAST ones",                          # no comparison
    "facing above 1000",                           # not a numeric column
    "carpet area above lots",                      # no number
    "above 1000",                                  # no column named
])
def test_anything_unclear_grounds_nothing(msg):
    assert ground_numeric(msg, "assets_asset", SM) == []


def test_the_sql_compares_as_a_number():
    sql = where_clause([{"column": "carpet_area", "op": ">", "value": "1000", "value_norm": "1000"},
                        {"column": "city_name", "op": "=", "value": "Nagpur", "value_norm": "nagpur"}])
    assert sql == 'lower("city_name"::text) = \'nagpur\' AND "carpet_area" > 1000'


def test_a_non_number_never_reaches_sql():
    assert where_clause([{"column": "c", "op": ">", "value": "1; drop", "value_norm": "x"}]) == ""


def test_a_threshold_measure_is_not_the_figure_to_rank_by(monkeypatch):
    from veda import intent_sql_alignment as A
    monkeypatch.setattr(A, "_enabled", lambda: True)
    sm = {"columns": {"assets_asset.carpet_area": {"analytics_role": "MEASURE"}}}
    sql = ('SELECT "facing", COUNT(*) AS "assets_asset_count" FROM "assets_asset" WHERE '
           '"carpet_area" > 1000 GROUP BY "facing" ORDER BY "assets_asset_count" DESC')
    assert A.entity_anchor_ok("only the ones with carpet area above 1000", sql, sm)[0]
    # ...while ranking BY it on another table is still refused
    bad = 'SELECT "unit", COUNT(*) AS n FROM "assets_carpetareaunit" GROUP BY "unit" ORDER BY n DESC'
    assert not A.entity_anchor_ok("highest carpet area", bad, sm)[0]
