"""Entity→table anchor for tied measure metrics (flag ENTITY_TABLE_ANCHOR_ENABLED).

Bug: "average security deposit" is a metric LABEL on BOTH assets_leaselisting and
assets_leasetransaction. match_metric_labels returns both tied, and the fast-path takes whichever is
first — so "average security deposit of lease TRANSACTIONS" resolves against assets_leaselisting (wrong
table, plausible-but-wrong number). With the flag, tied candidates are re-ranked to prefer the table the
query names. Run in the container with CWD=/app/veda_core:
    docker exec -w /app/veda_core -e PYTHONPATH=/app veda-platform-inference-1 \
        python -u /app/tests/test_entity_table_anchor.py
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


def _table(q):
    fp = FP.try_fast_path(q)
    return None if fp is None else fp.sql


def test_flag_off_keeps_current_order():
    # byte-identical: with the anchor off, the transactions query still resolves to the (wrong) first
    # candidate table — we are NOT changing default behaviour.
    _ctx(); _config.ENTITY_TABLE_ANCHOR_ENABLED = False; _config.FASTPATH_SUM_MEASURE_ENABLED = False
    sql = _table("What is the average security deposit of lease transactions?")
    assert sql is None or "assets_leaselisting" in sql


def test_flag_on_anchors_to_named_table():
    _ctx(); _config.ENTITY_TABLE_ANCHOR_ENABLED = True
    try:
        t = _table("What is the average security deposit of lease transactions?")
        assert t is not None and "assets_leasetransaction" in t and "assets_leaselisting" not in t
        # control: "of lease listings" must still resolve to listings, not transactions
        l = _table("What is the average security deposit of lease listings?")
        assert l is not None and "assets_leaselisting" in l and "assets_leasetransaction" not in l
    finally:
        _config.ENTITY_TABLE_ANCHOR_ENABLED = False


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
