import traceback
"""Tests for the uniqueness-intent shape guard (answer-safety, flag-gated).

A "how many UNIQUE/DISTINCT X" query answered by a plain COUNT(*) (no DISTINCT, no GROUP BY) counts
total rows, not distinct values — the summariser then reports that total as the unique count
(silent-wrong). The guard refuses ONLY when the query signals uniqueness AND the SQL de-duplicates
nowhere. Run: `python tests/test_distinct_shape_guard.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
from veda.validation import distinct_shape_ok  # noqa: E402


def test_flag_off_always_true():
    _config.DISTINCT_SHAPE_GUARD_ENABLED = False
    assert distinct_shape_ok("how many unique owners",
                             "SELECT COUNT(*) FROM assets_asset") is True


def test_unique_plain_count_refuses():
    _config.DISTINCT_SHAPE_GUARD_ENABLED = True
    try:
        assert distinct_shape_ok("how many unique owners",
                                 "SELECT COUNT(*) FROM assets_asset") is False
    finally:
        _config.DISTINCT_SHAPE_GUARD_ENABLED = False


def test_count_distinct_ok():
    _config.DISTINCT_SHAPE_GUARD_ENABLED = True
    try:
        assert distinct_shape_ok("how many unique owners",
                                 "SELECT COUNT(DISTINCT owner_id) FROM assets_asset") is True
    finally:
        _config.DISTINCT_SHAPE_GUARD_ENABLED = False


def test_select_distinct_list_ok():
    _config.DISTINCT_SHAPE_GUARD_ENABLED = True
    try:
        assert distinct_shape_ok("list distinct cities",
                                 "SELECT DISTINCT city_name FROM assets_asset") is True
    finally:
        _config.DISTINCT_SHAPE_GUARD_ENABLED = False


def test_group_by_de_dups_ok():
    # A GROUP BY already yields one row per distinct value → legitimate shape.
    _config.DISTINCT_SHAPE_GUARD_ENABLED = True
    try:
        assert distinct_shape_ok("how many different tenants",
                                 "SELECT tenant_id, COUNT(*) FROM leases GROUP BY tenant_id") is True
    finally:
        _config.DISTINCT_SHAPE_GUARD_ENABLED = False


def test_no_uniqueness_intent_untouched():
    # No "unique/distinct/different" in the query → guard stays silent even on a plain COUNT(*).
    _config.DISTINCT_SHAPE_GUARD_ENABLED = True
    try:
        assert distinct_shape_ok("how many properties are there",
                                 "SELECT COUNT(*) FROM assets_asset") is True
    finally:
        _config.DISTINCT_SHAPE_GUARD_ENABLED = False


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
