import traceback
"""Tests for deterministic edge-driven cross-source MULTI (flag-gated).

When a query names the entities of BOTH endpoints of a HIGH-tier cross_source_fk edge between two
in-scope sources, routing overrides to a federated MULTI — fixing the case where retrieval surfaced the
wrong sibling table and the tuned policy mis-routed SINGLE (or the boundary SLM returned an invalid
decision). Uses a fake internal-DB connection so the logic is tested without a live graph.
Run: `python tests/test_cross_source_edge_multi.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
import query.source_coordinator as SC  # noqa: E402
from ingestion import db_abstraction as _dba  # noqa: E402

# One HIGH edge: src4.maintenance ↔ src2.assets_asset (the meaningful cross-source join).
_EDGES = [("4", "maintenance", "2", "assets_asset")]


class _FakeCursor:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, q, params=None): self._rows = _EDGES
    def fetchall(self): return list(self._rows)


class _FakeConn:
    def cursor(self): return _FakeCursor()


def _patch():
    o1, o2 = _dba.get_internal_connection, _dba.release_internal_connection
    _dba.get_internal_connection = lambda: _FakeConn()
    _dba.release_internal_connection = lambda c: None
    def restore():
        _dba.get_internal_connection = o1
        _dba.release_internal_connection = o2
    return restore


def test_tbl_tokens():
    assert SC._tbl_tokens("assets_asset") == {"asset"}
    assert SC._tbl_tokens("vendors") == {"vendor"}
    assert "maintenance" in SC._tbl_tokens("maintenance")


def test_flag_off_is_noop():
    _config.CROSS_SOURCE_EDGE_MULTI_ENABLED = False
    assert SC._edge_multi_pair("which assets have maintenance tickets", (2, 3, 4, 5)) is None


def test_both_entities_named_routes_multi():
    _config.CROSS_SOURCE_EDGE_MULTI_ENABLED = True
    restore = _patch()
    try:
        assert SC._edge_multi_pair("which assets have maintenance tickets", (2, 3, 4, 5)) == ["2", "4"]
        assert SC._edge_multi_pair("how many maintenance tickets are there for each asset",
                                   (2, 3, 4, 5)) == ["2", "4"]
    finally:
        restore(); _config.CROSS_SOURCE_EDGE_MULTI_ENABLED = False


def test_one_entity_named_no_multi():
    # only 'maintenance' named (not 'asset') → single-source, no override.
    _config.CROSS_SOURCE_EDGE_MULTI_ENABLED = True
    restore = _patch()
    try:
        assert SC._edge_multi_pair("how many maintenance tickets are open", (2, 3, 4, 5)) is None
        assert SC._edge_multi_pair("how many properties are there", (2, 3, 4, 5)) is None
    finally:
        restore(); _config.CROSS_SOURCE_EDGE_MULTI_ENABLED = False


def test_source_out_of_scope_no_multi():
    # edge endpoint source 4 not in scope → no override.
    _config.CROSS_SOURCE_EDGE_MULTI_ENABLED = True
    restore = _patch()
    try:
        assert SC._edge_multi_pair("which assets have maintenance tickets", (2, 3)) is None
    finally:
        restore(); _config.CROSS_SOURCE_EDGE_MULTI_ENABLED = False


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
