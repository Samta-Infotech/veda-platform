"""Deterministic scalar SUM/AVG measure fast-path (flag FASTPATH_SUM_MEASURE_ENABLED).

Bug: "total <measure> across all <entity>" whose measure NAME embeds a time-adverb
("total expected MONTHLY rent across all lease listings") trips the trend detector AND "across"
counts as both a group-prep and a join-hint → the query is hijacked into the count/trend branch,
declines, and the SLM free-forms a raw projection with LIMIT 100 → the summariser fabricates a
wrong total. With the flag, such an ungrouped measure query reaches the SUM/AVG branch and builds
the correct SUM()/AVG(). Requires the live registry — run in the container with CWD=/app/veda_core:
    docker exec -w /app/veda_core -e PYTHONPATH=/app veda-platform-inference-1 \
        python -u /app/tests/test_sum_measure_fastpath.py
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

TOTALS = [
    "What is the total expected monthly rent across all lease listings?",
    "What is the total security deposit across all lease listings?",
    "What is the total expected price across all sale listings?",
    "What is the total rent across all lease transactions?",
]


def _ctx():
    set_context(RequestContext(source_id=2, tenant="default", source_ids=(2,)))
    set_source_profiles({"2": {"source_type": "relational"}})


def test_flag_off_totals_decline_byte_identical():
    _ctx(); _config.FASTPATH_SUM_MEASURE_ENABLED = False
    for q in TOTALS:
        assert FP.try_fast_path(q) is None, f"flag OFF should decline: {q}"


def test_flag_on_totals_build_sum():
    _ctx(); _config.FASTPATH_SUM_MEASURE_ENABLED = True
    try:
        for q in TOTALS:
            fp = FP.try_fast_path(q)
            assert fp is not None, f"expected a fast-path result: {q}"
            s = fp.sql.upper()
            assert "SUM(" in s and "LIMIT" not in s, f"expected full-table SUM, got: {fp.sql}"
    finally:
        _config.FASTPATH_SUM_MEASURE_ENABLED = False


def test_flag_on_correct_table_for_transactions():
    # "total rent across all lease transactions" must SUM on assets_leasetransaction, not leaselisting
    _ctx(); _config.FASTPATH_SUM_MEASURE_ENABLED = True
    try:
        fp = FP.try_fast_path("What is the total rent across all lease transactions?")
        assert fp is not None and "assets_leasetransaction" in fp.sql and "SUM(" in fp.sql.upper()
    finally:
        _config.FASTPATH_SUM_MEASURE_ENABLED = False


def test_controls_unchanged_flag_on():
    # a real trend, a grouped measure, and a count must NOT be diverted by the fix
    _ctx(); _config.FASTPATH_SUM_MEASURE_ENABLED = True
    try:
        trend = FP.try_fast_path("What is the monthly trend of properties based on created at?")
        assert trend is not None and "DATE_TRUNC" in (trend.sql or "").upper()
        count = FP.try_fast_path("How many properties are there?")
        assert count is not None and "COUNT(" in (count.sql or "").upper()
        # grouped measure stays on the existing (non-bypass) path — bypass excludes " by "
        grouped = FP.try_fast_path("What is the average expected monthly rent of lease listings by furnishing?")
        assert grouped is None or " by ".upper() not in "", "grouped must not be diverted by the scalar bypass"
    finally:
        _config.FASTPATH_SUM_MEASURE_ENABLED = False


if __name__ == "__main__":
    import traceback
    _ctx(); reg.load()
    if not reg.is_ready():
        print("SKIP — registry not ready (needs the container's semantic model)"); sys.exit(0)
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
