"""Unit tests for the SLM-free single-table SQL builder
(generation._deterministic_single_table_sql).

Deterministic, no DB / no SLM. Proves: safe queries (projection + date-range +
temporal rank) build valid SQL without the model; queries needing a measure/value
ranking or a categorical filter return None (defer to the SLM) so a dropped filter is
never silently answered.

Run from repo root: ``pytest tests/test_single_table_deterministic.py``
"""
import os, sys
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "veda_core"))


@dataclass
class _T:
    start: str = None
    end: str = None


def _build(query, table, cols, temporal=None, time_col=None, proj=None):
    from veda.generation import _deterministic_single_table_sql
    return _deterministic_single_table_sql(query, table, cols, temporal or _T(),
                                           time_col, proj if proj is not None else cols[:2])


# ── safe cases: build SLM-free ───────────────────────────────────────────────
def test_plain_projection_builds():
    sql = _build("list all tenants", "users_user", ["first_name", "last_name", "email"])
    assert sql == 'SELECT "first_name", "last_name" FROM "users_user" LIMIT 100'


def test_projection_only_uses_recommended():
    sql = _build("show properties", "assets_asset",
                 ["project_name", "building_name", "carpet_area"],
                 proj=["project_name"])
    assert sql == 'SELECT "project_name" FROM "assets_asset" LIMIT 100'


def test_date_range_filter_is_deterministic():
    sql = _build("payment transactions", "accounts_paymenttransaction",
                 ["id", "paid_amount", "created_at"],
                 temporal=_T("2026-06-27", "2026-07-27"), time_col="created_at")
    assert 'WHERE "created_at" BETWEEN \'2026-06-27\' AND \'2026-07-27\'' in sql
    assert sql.endswith("LIMIT 100")


def test_temporal_ranking_adds_order_by():
    sql = _build("latest 10 tickets", "worklists_ticket", ["id", "title", "created_at"],
                 time_col="created_at")
    assert 'ORDER BY "created_at" DESC' in sql
    assert sql.endswith("LIMIT 10")          # requested limit respected


# ── unsafe cases: must defer to the SLM (return None) ────────────────────────
def test_categorical_filter_defers():
    assert _build("show open tickets", "worklists_ticket", ["id", "status"]) is None
    assert _build("completed payment transactions", "accounts_paymenttransaction",
                  ["id", "status"]) is None


def test_measure_ranking_defers():
    assert _build("top 5 payment transactions by paid amount",
                  "accounts_paymenttransaction", ["id", "paid_amount"]) is None


# ── flag default OFF → production path unchanged (SLM still writes the SQL) ───
def test_flag_default_off():
    import config
    assert config.SINGLE_TABLE_DETERMINISTIC is False
