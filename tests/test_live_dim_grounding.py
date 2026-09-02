import traceback
"""Tests for fast-path live dimension-value grounding (flag-gated).

When the sampled value-store misses a named filter value, the fast-path probes the entity table's
DIMENSION columns against live data with an EXACT match, grounds a unique column, and disambiguates a
value present in several columns by DOMINANCE (most rows). Uses a fake DB connection so the logic is
tested without a live database. Run: `python tests/test_live_dim_grounding.py`.
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

os.environ["VEDA_SOURCE_SCHEMA"] = "homzhub"
import config as _config  # noqa: E402
import query.fast_path as FP  # noqa: E402
from veda import runtime as _rt  # noqa: E402


class _FakeCursor:
    def __init__(self, data):
        self.data = data            # {col: {value_lower: (raw, count)}}
        self._rows = []

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def execute(self, q, params):
        col = re.search(r'CAST\("(\w+)"', q).group(1)
        tok = params[0]
        hit = self.data.get(col, {}).get(tok)
        if q.strip().upper().startswith("SELECT COUNT"):
            self._rows = [(hit[1],)] if hit else [(0,)]
        elif "DISTINCT" in q:
            self._rows = [(hit[0],)] if hit else []
        else:                       # existence LIMIT 1
            self._rows = [(hit[0],)] if hit else []

    def fetchone(self): return self._rows[0] if self._rows else None
    def fetchall(self): return list(self._rows)


class _FakeConn:
    def __init__(self, data): self.data = data
    def cursor(self): return _FakeCursor(self.data)
    def rollback(self): pass


def _sm_with(table, dim_cols):
    return {"columns": {f"{table}.{c}": {"col_name": c, "table_name": table,
                                         "analytics_role": "DIMENSION"} for c in dim_cols}}


def _patch(data):
    orig = _rt._pg
    _rt._pg = lambda: _FakeConn(data)
    return lambda: setattr(_rt, "_pg", orig)


def test_flag_off_is_noop():
    _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = False
    r = FP._live_ground_dim_value("assets_asset", {"mumbai"}, _sm_with("assets_asset", ["city_name"]))
    assert r is None


def test_unique_column_grounds():
    _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = True
    restore = _patch({"city_name": {"bangalore": ("Bangalore", 17)}})
    try:
        r = FP._live_ground_dim_value("assets_asset", {"bangalore"},
                                      _sm_with("assets_asset", ["city_name", "project_name"]))
        assert r == ("city_name", ["Bangalore"]), r
    finally:
        restore(); _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = False


def test_dominant_column_wins():
    # "Mumbai" in city_name (81) AND project_name (4) → city dominates → grounds to city_name.
    _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = True
    restore = _patch({"city_name": {"mumbai": ("Mumbai", 81)},
                      "project_name": {"mumbai": ("Mumbai", 4)}})
    try:
        r = FP._live_ground_dim_value("assets_asset", {"mumbai"},
                                      _sm_with("assets_asset", ["city_name", "project_name"]))
        assert r == ("city_name", ["Mumbai"]), r
    finally:
        restore(); _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = False


def test_tie_refuses():
    # Equal counts in two columns → genuinely ambiguous → refuse (None).
    _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = True
    restore = _patch({"city_name": {"paris": ("Paris", 10)},
                      "project_name": {"paris": ("Paris", 10)}})
    try:
        r = FP._live_ground_dim_value("assets_asset", {"paris"},
                                      _sm_with("assets_asset", ["city_name", "project_name"]))
        assert r is None, r
    finally:
        restore(); _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = False


def test_absent_value_refuses():
    _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = True
    restore = _patch({"city_name": {"mumbai": ("Mumbai", 81)}})
    try:
        r = FP._live_ground_dim_value("assets_asset", {"nowhereville"},
                                      _sm_with("assets_asset", ["city_name"]))
        assert r is None, r
    finally:
        restore(); _config.FASTPATH_LIVE_DIM_GROUNDING_ENABLED = False


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
