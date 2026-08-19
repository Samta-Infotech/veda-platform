"""Unit test for the aggregate-chain audit-edge fix
(planning._resolve_agg_chain + AGG_CHAIN_EXCLUDE_AUDIT).

Deterministic, no DB / no SLM. Proves the grain planner's pre-aggregation BFS does
NOT route a COUNT through an AUDIT edge (created_by_id/updated_by_id) when the flag is
on — the meaningless "who created the asset" hop that produced confidently-wrong
answers for E04/I06/L04. Flag OFF → unchanged (the audit hop is still taken), so the
existing pipeline is byte-identical by default.
"""
import os, sys, json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VC = os.path.join(_ROOT, "veda_core")
sys.path.insert(0, _VC)
from veda.planning import _resolve_agg_chain
_GRAPH = json.load(open(os.path.join(_VC, "data", "veda_relationship_graph.json")))


def _uses_audit_edge(hops):
    """True if any hop is EXACTLY an audit FK edge — matched on (table, column) pairs
    (not loose table/column intersection, which false-positives when the same table
    pair also has a non-audit FK)."""
    for h in hops or []:
        near = (h["near_table"], h.get("near_col"))
        far = (h["far_table"], h.get("far_col"))
        for e in _GRAPH["edges"]:
            if e.get("relationship_type") != "audit":
                continue
            src = (e["source_table"], e.get("source_column"))
            tgt = (e["target_table"], e.get("target_column"))
            if {near, far} == {src, tgt}:
                return True
    return False


def _chain(anchor, tgt, flag):
    import config
    old = getattr(config, "AGG_CHAIN_EXCLUDE_AUDIT", False)
    config.AGG_CHAIN_EXCLUDE_AUDIT = flag
    try:
        return _resolve_agg_chain(anchor, tgt, _GRAPH)
    finally:
        config.AGG_CHAIN_EXCLUDE_AUDIT = old


def test_flag_off_still_traverses_audit_edge():
    # documents the CURRENT (buggy) behavior — default path is unchanged
    hops = _chain("accounts_paymenttransaction", "assets_asset", flag=False)
    assert _uses_audit_edge(hops)                 # created_by_id hop present


def test_flag_on_never_uses_audit_edge():
    hops = _chain("accounts_paymenttransaction", "assets_asset", flag=True)
    assert not _uses_audit_edge(hops)             # audit hop eliminated


def test_flag_default_off():
    import config
    assert config.AGG_CHAIN_EXCLUDE_AUDIT is False
