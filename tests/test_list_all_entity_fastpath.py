import traceback
"""Integration test for the deterministic "list all <entity>" fast-path (flag-gated).

"list all amenities" previously DECLINED (the list-verb token collides with a values-catalog concept →
_single_entity reads a false second entity) → the SLM free-formed a wrong junction table
(assets_assetamenitygroup, reported "5") instead of assets_amenity (32 real amenities). When
FASTPATH_LIST_ALL_ENTITY_ENABLED, the fast-path lists the entity's governed display column. Requires the
live registry (the compiled concepts/metrics files), so run inside the container with CWD=/app/veda_core
(config's CONCEPTS_FILE is a path relative to that dir):
    docker exec -w /app/veda_core -e PYTHONPATH=/app veda-platform-inference-1 \
        python -u /app/tests/test_list_all_entity_fastpath.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from veda_core.context import RequestContext, set_context, set_source_profiles  # noqa: E402
import config as _config  # noqa: E402
import query.fast_path as FP  # noqa: E402
from semantic import registry as reg  # noqa: E402


def _ctx():
    set_context(RequestContext(source_id=2, tenant="default", source_ids=(2,)))
    set_source_profiles({"2": {"source_type": "relational"}})


def test_list_all_amenities_flag_off_declines():
    _ctx()
    _config.FASTPATH_LIST_ALL_ENTITY_ENABLED = False
    assert FP.try_fast_path("list all amenities") is None      # byte-identical: still declines


def test_list_all_amenities_flag_on_lists_correct_table():
    _ctx()
    _config.FASTPATH_LIST_ALL_ENTITY_ENABLED = True
    try:
        fp = FP.try_fast_path("list all amenities")
        assert fp is not None, "expected a fast-path result"
        assert "assets_amenity" in fp.sql and "assetamenitygroup" not in fp.sql   # correct catalog table
        assert fp.route == "dimension.list.entity"
    finally:
        _config.FASTPATH_LIST_ALL_ENTITY_ENABLED = False


def test_filtered_list_not_fired_no_filter_drop():
    # "list properties in Mumbai" carries a real filter value → the residual gate must prevent the
    # entity-only listing (which would silently drop the Mumbai filter). Must NOT emit a filterless list.
    _ctx()
    _config.FASTPATH_LIST_ALL_ENTITY_ENABLED = True
    try:
        fp = FP.try_fast_path("list properties in Mumbai")
        # either declines, or (if some other branch fires) must not be the bare entity-listing route
        assert fp is None or fp.route != "dimension.list.entity"
    finally:
        _config.FASTPATH_LIST_ALL_ENTITY_ENABLED = False


if __name__ == "__main__":
    _ctx()          # activate the (source=2) scope before probing the registry
    reg.load()
    if not reg.is_ready():
        print("SKIP — registry not ready (needs the container's semantic model)")
        sys.exit(0)
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
