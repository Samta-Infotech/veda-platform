import traceback
"""Tests for the grouped-intent shape guard (answer-safety, flag-gated).

A "how many X by Y" / "distribution" query answered by a PURE PROJECTION (no GROUP BY / aggregate /
WHERE) can't answer the grouping — the guard refuses so the NL summariser can't fabricate a
distribution. Fires ONLY on a pure projection. Run: `python tests/test_grouped_shape_guard.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
from veda.validation import grouped_shape_ok  # noqa: E402


def test_flag_off_always_true():
    _config.GROUPED_SHAPE_GUARD_ENABLED = False
    assert grouped_shape_ok("how many properties by city",
                            "SELECT city_name, id FROM assets_asset LIMIT 100") is True


def test_grouped_projection_refuses():
    _config.GROUPED_SHAPE_GUARD_ENABLED = True
    try:
        assert grouped_shape_ok("how many properties by city",
                                "SELECT city_name, id FROM assets_asset LIMIT 100") is False
    finally:
        _config.GROUPED_SHAPE_GUARD_ENABLED = False


def test_grouped_with_group_by_ok():
    _config.GROUPED_SHAPE_GUARD_ENABLED = True
    try:
        assert grouped_shape_ok("how many properties by city",
                                "SELECT city_name, COUNT(*) FROM assets_asset GROUP BY city_name") is True
    finally:
        _config.GROUPED_SHAPE_GUARD_ENABLED = False


def test_grouped_with_aggregate_ok():
    _config.GROUPED_SHAPE_GUARD_ENABLED = True
    try:
        assert grouped_shape_ok("distribution of rent by city",
                                "SELECT AVG(rent) FROM leasetransaction") is True
    finally:
        _config.GROUPED_SHAPE_GUARD_ENABLED = False


def test_by_person_filter_not_misguarded():
    # "by <person>" is a filter, not a group — a WHERE means a legitimate shape.
    _config.GROUPED_SHAPE_GUARD_ENABLED = True
    try:
        assert grouped_shape_ok("properties owned by john",
                                "SELECT * FROM assets_asset WHERE owner = 'john'") is True
    finally:
        _config.GROUPED_SHAPE_GUARD_ENABLED = False


def test_non_grouped_query_ok():
    _config.GROUPED_SHAPE_GUARD_ENABLED = True
    try:
        assert grouped_shape_ok("how many properties are there",
                                "SELECT COUNT(*) FROM assets_asset") is True
        assert grouped_shape_ok("list amenities",
                                "SELECT amenity_name FROM catalog") is True
    finally:
        _config.GROUPED_SHAPE_GUARD_ENABLED = False


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
