"""Tests for veda/planning.py::_ranking_tie_break — the deterministic secondary
ORDER BY key for GROUPED rankings (2026-09-29, SQL_RANKING_TIE_BREAK_ENABLED,
default OFF). Pure-python, no DB."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))
import config
from veda.planning import _ranking_tie_break, build_aggregate_sql


def test_default_off_is_a_noop():
    assert config.SQL_RANKING_TIE_BREAK_ENABLED is False   # confirms the default itself
    assert _ranking_tie_break("city") == ""


def test_no_group_col_is_always_a_noop(monkeypatch):
    monkeypatch.setattr(config, "SQL_RANKING_TIE_BREAK_ENABLED", True)
    assert _ranking_tie_break(None) == ""
    assert _ranking_tie_break("") == ""


def test_enabled_appends_the_group_column_ascending(monkeypatch):
    monkeypatch.setattr(config, "SQL_RANKING_TIE_BREAK_ENABLED", True)
    assert _ranking_tie_break("city") == ', t0."city" ASC'


def test_build_aggregate_sql_single_anchor_grouped_unaffected_when_flag_off():
    sql, tables = build_aggregate_sql("orders", [], sm={}, measure_agg="COUNT",
                                      group_col="city", top_n=5, direction="desc")
    assert sql is not None
    assert ', t0."city" ASC' not in sql
    assert 'GROUP BY t0."city" ORDER BY count_result DESC LIMIT 5' in sql


def test_build_aggregate_sql_single_anchor_grouped_tie_break_when_flag_on(monkeypatch):
    monkeypatch.setattr(config, "SQL_RANKING_TIE_BREAK_ENABLED", True)
    sql, tables = build_aggregate_sql("orders", [], sm={}, measure_agg="COUNT",
                                      group_col="city", top_n=5, direction="desc")
    assert sql is not None
    assert 'ORDER BY count_result DESC, t0."city" ASC LIMIT 5' in sql


def test_build_aggregate_sql_ungrouped_untouched_by_flag(monkeypatch):
    """The ungrouped (ranking without a GROUP BY) case is deliberately NOT given a
    tie-break — no generically-safe secondary key is known here — regardless of
    the flag."""
    monkeypatch.setattr(config, "SQL_RANKING_TIE_BREAK_ENABLED", True)
    sql, tables = build_aggregate_sql("orders", [], sm={}, measure_agg="COUNT",
                                      group_col=None, top_n=None, direction="desc")
    assert sql is not None
    assert "ORDER BY" not in sql   # no group_col -> the single-anchor branch has no ORDER BY at all
