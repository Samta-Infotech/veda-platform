"""Unit tests for Phase-1 analytical spec (veda/analytical_spec.py).

Deterministic, no DB / no SLM. Validates: spec derivation from a GroundedIntent, correct
SQL shape (scalar vs grouped), COUNT/SUM/AVG, single-anchor scoping (defers multi-table),
and the flag default. Uses tiny stub intents + a minimal in-memory semantic model.
"""
import os, sys
from dataclasses import dataclass, field
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "veda_core"))


@dataclass
class _Measure:
    kind: str = ""
    concept: str = ""


@dataclass
class _GI:                      # stand-in for GroundedIntent
    intent: str
    anchor: Optional[str]
    secondaries: List[str] = field(default_factory=list)
    measure: Optional[_Measure] = None


# minimal semantic model: paymenttransaction(paid_amount numeric, payment_type_id, id),
# assets_asset(carpet_area_sqft numeric, project_name, id)
_SM = {"columns": {
    "accounts_paymenttransaction.id": {},
    "accounts_paymenttransaction.paid_amount": {"data_type": "numeric",
                                                "business_role": "Paid Amount"},
    "accounts_paymenttransaction.payment_type_id": {"business_role": "Payment Type"},
    "assets_asset.id": {},
    "assets_asset.carpet_area_sqft": {"data_type": "numeric", "business_role": "Carpet Area"},
    "assets_asset.project_name": {"business_role": "Project Name"},
}}


def _derive(intent, anchor, query, measure_concept=None, secondaries=None):
    from veda.analytical_spec import derive_spec
    gi = _GI(intent=intent, anchor=anchor, secondaries=secondaries or [],
             measure=_Measure(concept=measure_concept) if measure_concept else None)
    return derive_spec(gi, query, _SM)


# ── scalar COUNT — emit_sql REUSES build_aggregate_sql (no parallel builder) ──
def test_scalar_count():
    from veda.analytical_spec import emit_sql
    s = _derive("count", "assets_asset", "how many properties are there")
    assert s and s.output_shape == "scalar" and s.aggregation == "count"
    sql = emit_sql(s, _SM)
    assert "COUNT(*)" in sql and '"assets_asset"' in sql and "GROUP BY" not in sql


# ── scalar SUM with a grounded measure column ────────────────────────────────
def test_scalar_sum_grounds_measure():
    from veda.analytical_spec import emit_sql
    s = _derive("sum", "accounts_paymenttransaction",
                "total paid amount of payment transactions", measure_concept="paid amount")
    assert s and s.aggregation == "sum" and s.measure_column == "paid_amount"
    sql = emit_sql(s, _SM)
    assert 'SUM("paid_amount")' in sql and "GROUP BY" not in sql


# ── grouped GROUP BY on an anchor dimension ──────────────────────────────────
def test_grouped_by_dimension():
    from veda.analytical_spec import emit_sql
    s = _derive("count", "accounts_paymenttransaction",
                "number of payment transactions by payment type")
    assert s and s.output_shape == "grouped" and s.group_keys == ["payment_type_id"]
    sql = emit_sql(s, _SM)
    assert "GROUP BY" in sql and '"payment_type_id"' in sql and "COUNT(*)" in sql


# ── measure that can't ground → defer (None), never guess ────────────────────
def test_ungroundable_measure_defers():
    assert _derive("avg", "assets_asset", "average rocket fuel", measure_concept="rocket fuel") is None


# ── multi-entity (secondaries) → Phase 2, defer ──────────────────────────────
def test_multi_table_defers_to_phase2():
    s = _derive("sum", "assets_asset", "total paid per owner",
                measure_concept="paid amount", secondaries=["users_user"])
    assert s is None


# ── non-analytical intent → None ─────────────────────────────────────────────
def test_list_intent_not_analytical():
    assert _derive("list", "assets_asset", "show properties") is None


# ── flag default OFF ─────────────────────────────────────────────────────────
def test_flag_default_off():
    import config
    assert config.ANALYTICAL_SQL_V2 is False
