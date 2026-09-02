from ingestion import db_abstraction as DB
from query import datalake_values as DV
import traceback
"""Tests for datalake source isolation (SOURCE_ISOLATED_RETRIEVAL_ENABLED, flag-gated, default OFF).

The DUP-1/P0 fix: when a query routes SINGLE to a NON-primary datalake source, the deterministic head
runs over a DATALAKE-ONLY semantic model (that source's tables/columns + parquet sample_values) instead
of merging the datalake schema INTO the homzhub sm. This hard-isolates retrieval/planning/value-
grounding to the routed source (no homzhub-table mixing, no shared-value collision).

Offline unit tests (no DB / no live services) — the DB read and parquet sampler are stubbed. They pin:
  - flag OFF  -> merge path runs, homzhub tables PRESENT (byte-identical to prior behaviour);
  - flag ON   -> isolated sm has ONLY the datalake source's tables, homzhub tables ABSENT;
  - sample_values from the parquet sampler land on the isolated columns (value grounding);
  - empty/failed DB read -> None -> caller falls back to merge (safe degrade).

Run: `.venv/bin/python tests/test_source_isolation.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import veda_hybrid as VH  # noqa: E402


# ---- stubs ---------------------------------------------------------------------------------------
class _FakeCur:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCur(self._rows)


def _patch_db(rows):
    """Stub the internal-connection accessor VH imports lazily from ingestion.db_abstraction."""
    orig_get, orig_rel = DB.get_internal_connection, DB.release_internal_connection
    DB.get_internal_connection = lambda: _FakeConn(rows)
    DB.release_internal_connection = lambda c: None
    return lambda: (setattr(DB, "get_internal_connection", orig_get),
                    setattr(DB, "release_internal_connection", orig_rel))


def _patch_sampler(vidx):
    """Stub datalake_values._sample_source so no DuckDB/parquet is touched."""
    orig = DV._sample_source
    DV._sample_source = lambda sid, tenant, limit: vidx
    return lambda: setattr(DV, "_sample_source", orig)


# datalake source_id 5: two tables. homzhub sm carries an unrelated relational table.
_DL_ROWS = [
    ("amenities_catalog", "amenity_name", "DIMENSION"),
    ("amenities_catalog", "monthly_fee", "MEASURE"),
    ("vendors", "city", "DIMENSION"),
]
_HOMZHUB_SM = {
    "version": 1,
    "tables": {"asset": {"table_name": "asset"}},
    "columns": {"asset.city": {"col_name": "city", "table_name": "asset"}},
    "retrieval_documents": {"asset": {"doc": "x"}},
}


# ---- tests ---------------------------------------------------------------------------------------
def test_isolated_sm_has_only_datalake_tables():
    un_db = _patch_db(_DL_ROWS)
    un_s = _patch_sampler({})
    try:
        res = VH._datalake_isolated_sm("5")
        assert res is not None
        sm, cols = res
        assert set(sm["tables"].keys()) == {"amenities_catalog", "vendors"}
        assert "asset" not in sm["tables"]                  # homzhub table isolated OUT
        assert "asset.city" not in sm["columns"]
        assert "amenities_catalog.monthly_fee" in sm["columns"]
        assert sm["retrieval_documents"] == {}              # skeleton, no homzhub docs
        assert set(cols) == set(sm["columns"].keys())
    finally:
        un_s(); un_db()


def test_isolated_sm_grounds_sample_values():
    # token -> [(table, col, semtype, original)]  (the datalake_values inversion contract)
    vidx = {"kochi": [("vendors", "city", "CATEGORY", "Kochi")],
            "mumbai": [("vendors", "city", "CATEGORY", "Mumbai")]}
    un_db = _patch_db(_DL_ROWS)
    un_s = _patch_sampler(vidx)
    try:
        sm, _ = VH._datalake_isolated_sm("5")
        svals = sm["columns"]["vendors.city"]["sample_values"]
        assert "Kochi" in svals and "Mumbai" in svals       # datalake values available to arbiter
        assert sm["columns"]["amenities_catalog.monthly_fee"]["sample_values"] == []
    finally:
        un_s(); un_db()


def test_empty_db_returns_none_for_fallback():
    un_db = _patch_db([])
    un_s = _patch_sampler({})
    try:
        assert VH._datalake_isolated_sm("999") is None      # -> caller uses merge path
    finally:
        un_s(); un_db()


def test_sampler_failure_degrades_to_no_values():
    orig = DV._sample_source

    def boom(*a, **k):
        raise RuntimeError("no parquet")
    DV._sample_source = boom
    un_db = _patch_db(_DL_ROWS)
    try:
        sm, _ = VH._datalake_isolated_sm("5")               # sampler blows up ...
        assert set(sm["tables"].keys()) == {"amenities_catalog", "vendors"}  # ... sm still built
        assert sm["columns"]["vendors.city"]["sample_values"] == []          # just no values
    finally:
        DV._sample_source = orig; un_db()


def test_merge_path_still_includes_homzhub_when_flag_semantics_off():
    """Parity anchor: the OLD merge helper keeps homzhub tables (what runs when the flag is OFF)."""
    un_db = _patch_db(_DL_ROWS)
    try:
        merged, cols = VH._augment_sm_for_datalake(dict(_HOMZHUB_SM),
                                                   list(_HOMZHUB_SM["columns"].keys()), "5")
        assert "asset" in merged["tables"]                  # homzhub PRESENT (mixing, pre-fix)
        assert "vendors" in merged["tables"]                # datalake also present
        assert merged["retrieval_documents"] == _HOMZHUB_SM["retrieval_documents"]
    finally:
        un_db()


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
