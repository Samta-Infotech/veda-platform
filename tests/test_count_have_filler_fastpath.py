"""Count fast-path: "How many <entity> do we have?" (flag FASTPATH_COUNT_HAVE_FILLER_ENABLED).

Bug: the trailing "have" in "how many projects do we have?" is a join-hint token, so the count
fast-path bows out (treating it as a relationship) → the SLM free-forms a raw projection with
LIMIT 100 → the answer is the LIMIT (100), not the real COUNT. With the flag, a TRAILING "have"
that is the only join-hint token is treated as filler so the count path fires. Run in the container
with CWD=/app/veda_core:
    docker exec -w /app/veda_core -e PYTHONPATH=/app veda-platform-inference-1 \
        python -u /app/tests/test_count_have_filler_fastpath.py
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

DO_WE_HAVE = ["How many projects do we have?", "How many properties do we have?"]


def _ctx():
    set_context(RequestContext(source_id=2, tenant="default", source_ids=(2,)))
    set_source_profiles({"2": {"source_type": "relational"}})


def test_flag_off_declines_byte_identical():
    _ctx(); _config.FASTPATH_COUNT_HAVE_FILLER_ENABLED = False
    for q in DO_WE_HAVE:
        assert FP.try_fast_path(q) is None, f"flag OFF should decline: {q}"


def test_flag_on_fires_count():
    _ctx(); _config.FASTPATH_COUNT_HAVE_FILLER_ENABLED = True
    try:
        for q in DO_WE_HAVE:
            fp = FP.try_fast_path(q)
            assert fp is not None and "COUNT(" in fp.sql.upper(), f"expected COUNT, got: {fp and fp.sql}"
        # right table for projects
        fp = FP.try_fast_path("How many projects do we have?")
        assert "assets_project" in fp.sql
    finally:
        _config.FASTPATH_COUNT_HAVE_FILLER_ENABLED = False


def test_controls_unaffected_flag_on():
    _ctx(); _config.FASTPATH_COUNT_HAVE_FILLER_ENABLED = True
    try:
        # "are there" count unchanged
        c = FP.try_fast_path("How many properties are there?")
        assert c is not None and "COUNT(" in c.sql.upper()
        # a REAL "have"-filter (have followed by a content noun) must NOT fire the bare count path
        f = FP.try_fast_path("list properties that have power backup")
        assert f is None or "COUNT(DISTINCT ID)" not in f.sql.upper().replace('"', '')
    finally:
        _config.FASTPATH_COUNT_HAVE_FILLER_ENABLED = False


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
