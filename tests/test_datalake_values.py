import importlib
import traceback
"""Tests for query-time datalake value grounding (query/datalake_values.py).

Pure — DuckDB/context are mocked. Run: `python tests/test_datalake_values.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
from query import datalake_values as DV  # noqa: E402

BASE = lambda token: [("assets_asset", "city_name", "CATEGORY", "Mumbai")] if token.lower() == "mumbai" else []
DL_IDX = {"kochi": [("vendors", "city", "CATEGORY", "Kochi")],
          "banda": [("vendors", "city", "CATEGORY", "Banda")]}


def _setup(flag=True, scope_dl=True, idx=None):
    DV.clear_cache()
    _config.DATALAKE_VALUE_GROUNDING_ENABLED = flag
    DV.is_enabled = lambda: flag
    if scope_dl:
        DV._current_scope = lambda: (["2", "4"], "default", {"2": {"source_type": "relational"},
                                                             "4": {"source_type": "datalake"}})
        DV._datalake_source_ids = lambda sids, prof: ["4"]
    else:
        DV._current_scope = lambda: (["2"], "default", {"2": {"source_type": "relational"}})
        DV._datalake_source_ids = lambda sids, prof: []
    DV.datalake_value_index = lambda sids, tenant, prof: (idx if idx is not None else DL_IDX)


def test_flag_off_returns_base_unchanged():
    _setup(flag=False)
    wrapped = DV.augment_lookup(BASE)
    assert wrapped is BASE                       # byte-identical: same function object


def test_no_datalake_in_scope_returns_base():
    _setup(flag=True, scope_dl=False)
    wrapped = DV.augment_lookup(BASE)
    assert wrapped is BASE


def test_empty_index_returns_base():
    _setup(flag=True, scope_dl=True, idx={})
    wrapped = DV.augment_lookup(BASE)
    assert wrapped is BASE                       # nothing sampled → don't wrap


def test_datalake_value_grounds_when_base_empty():
    _setup(flag=True, scope_dl=True)
    wrapped = DV.augment_lookup(BASE)
    assert wrapped is not BASE
    out = wrapped("Kochi")                        # not in base, IS in datalake index
    assert out == [("vendors", "city", "CATEGORY", "Kochi")]


def test_base_wins_over_datalake():
    _setup(flag=True, scope_dl=True)
    wrapped = DV.augment_lookup(BASE)
    out = wrapped("Mumbai")                        # in base → base result, datalake not consulted
    assert out == [("assets_asset", "city_name", "CATEGORY", "Mumbai")]


def test_unknown_token_returns_empty():
    _setup(flag=True, scope_dl=True)
    wrapped = DV.augment_lookup(BASE)
    assert wrapped("nonexistent") == []


def test_datalake_source_ids_filter():
    # the real _datalake_source_ids (restore it) picks only datalake-typed sources
    importlib.reload(DV)
    prof = {"2": {"source_type": "relational"}, "4": {"source_type": "datalake"},
            "5": {"source_type": "datalake"}, "3": {"source_type": "document"}}
    assert sorted(DV._datalake_source_ids(["2", "3", "4", "5"], prof)) == ["4", "5"]


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
