"""Tests for the bounded SEMI_JOIN / FILTER cross-source planner (query/semi_join_planner.py).

Pure — the SLM is injected, no DB/model. Run: `python tests/test_semi_join_planner.py`.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
from query import semi_join_planner as SJ  # noqa: E402

# by_source: vendors (src_4) selected, assets_asset (src_2) selected.
BY_SOURCE = {"4": {"vendors": ["city", "vendor_id"]}, "2": {"assets_asset": ["city_name", "id"]}}
KINDS = {"4": "parquet", "2": "postgres"}
# one grounded hint: vendors.city <-> assets_asset.city_name
HINTS = [{"a_src": "4", "a_tbl": "vendors", "a_col": "city",
          "b_src": "2", "b_tbl": "assets_asset", "b_col": "city_name", "tier": "HIGH"}]


def _slm(obj):
    """A stub slm_call returning a fixed JSON object."""
    return lambda system, user: json.dumps(obj)


def test_build_candidates_directed_and_scoped():
    cands = SJ.build_candidates(BY_SOURCE, HINTS)
    # both directions considered, but only output tables that are SELECTED survive → both are selected
    keys = {(c["output_source_id"], c["output_table"], c["output_col"],
             c["filter_source_id"], c["filter_table"], c["filter_col"]) for c in cands}
    assert ("4", "vendors", "city", "2", "assets_asset", "city_name") in keys
    assert ("2", "assets_asset", "city_name", "4", "vendors", "city") in keys


def test_build_candidates_drops_unselected_output():
    # vendors NOT selected → the vendors-as-output direction is dropped
    bs = {"4": {}, "2": {"assets_asset": ["city_name"]}}
    cands = SJ.build_candidates(bs, HINTS)
    assert all(c["output_table"] != "vendors" for c in cands)


def test_classify_picks_valid_candidate():
    cands = SJ.build_candidates(BY_SOURCE, HINTS)
    # pick candidate 1 (1-based in the prompt)
    idx = SJ.classify("which vendors are in asset cities", cands, slm_call=_slm(
        {"operation": "SEMI_JOIN_FILTER", "candidate": 1}))
    assert idx == 0


def test_classify_not_semi_join_defers():
    cands = SJ.build_candidates(BY_SOURCE, HINTS)
    idx = SJ.classify("total maintenance per city", cands, slm_call=_slm(
        {"operation": "NOT_SEMI_JOIN", "candidate": -1}))
    assert idx is None


def test_classify_out_of_range_rejected():
    cands = SJ.build_candidates(BY_SOURCE, HINTS)
    idx = SJ.classify("q", cands, slm_call=_slm({"operation": "SEMI_JOIN_FILTER", "candidate": 99}))
    assert idx is None                              # candidate-set escape rejected


def test_classify_bad_json_defers():
    cands = SJ.build_candidates(BY_SOURCE, HINTS)
    idx = SJ.classify("q", cands, slm_call=lambda s, u: "not json at all")
    assert idx is None


def test_validate_rejects_invalid_source():
    bad = {"output_source_id": "9", "output_table": "vendors", "output_col": "city",
           "filter_source_id": "2", "filter_table": "assets_asset", "filter_col": "city_name"}
    assert SJ.validate(bad, BY_SOURCE, KINDS) is False      # src 9 not in scope


def test_validate_rejects_unselected_output_table():
    bad = {"output_source_id": "4", "output_table": "ghost", "output_col": "city",
           "filter_source_id": "2", "filter_table": "assets_asset", "filter_col": "city_name"}
    assert SJ.validate(bad, BY_SOURCE, KINDS) is False      # ghost table not selected


def test_validate_rejects_same_source():
    bad = {"output_source_id": "4", "output_table": "vendors", "output_col": "city",
           "filter_source_id": "4", "filter_table": "vendors", "filter_col": "city"}
    assert SJ.validate(bad, BY_SOURCE, KINDS) is False      # a semi-join spans two sources


def test_assemble_sql_deterministic_and_schema_aware(monkeypatch=None):
    cand = SJ.build_candidates(BY_SOURCE, HINTS)[0]  # vendors filtered by assets_asset
    # force a known schema so the test is deterministic regardless of env
    SJ._pg_schema = lambda sid: "homzhub"
    sql = SJ.assemble_sql(cand, BY_SOURCE, KINDS)
    assert "FROM src_4.\"vendors\" AS o" in sql
    assert "IN (SELECT DISTINCT f.\"city_name\" FROM src_2.homzhub.\"assets_asset\" AS f" in sql
    assert sql.count("SELECT") == 2 and "DISTINCT" in sql   # semi-join shape


def test_plan_semi_join_end_to_end_valid():
    SJ._pg_schema = lambda sid: "homzhub"
    res = SJ.plan_semi_join("which vendors are in asset cities", BY_SOURCE, HINTS, KINDS,
                            slm_call=_slm({"operation": "SEMI_JOIN_FILTER", "candidate": 1}))
    assert res is not None
    sql, cand = res
    assert cand["output_table"] == "vendors" and cand["filter_table"] == "assets_asset"
    assert "assets_asset" in sql


def test_plan_semi_join_defers_when_not_semi():
    res = SJ.plan_semi_join("total per city", BY_SOURCE, HINTS, KINDS,
                            slm_call=_slm({"operation": "NOT_SEMI_JOIN", "candidate": -1}))
    assert res is None                               # unsupported → safe defer, no forcing


def test_plan_semi_join_no_hints_defers():
    res = SJ.plan_semi_join("q", BY_SOURCE, [], KINDS, slm_call=_slm(
        {"operation": "SEMI_JOIN_FILTER", "candidate": 1}))
    assert res is None                               # no grounded candidates → defer


def test_flag_off_by_default():
    # default OFF → the module gate is closed (production byte-identical)
    _config.FEDERATED_SEMI_JOIN_STRUCTURED_ENABLED = False
    assert SJ.semi_join_enabled() is False


if __name__ == "__main__":
    import traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
