"""Tests for metadata-driven MONETARY measure selection in the superlative price fast-path (flag-gated).

The superlative "cheapest / most expensive <entity>" path picks the measure to ORDER BY. Historically it
name-matched business vocabulary (_PRICE_COL_HINTS) over MEASURE columns and silently took the FIRST
match when several existed. With FASTPATH_MONETARY_MEASURE_ENABLED, selection is schema-driven: the sole
`semantic_type == MONETARY` MEASURE column wins; two-or-more monetary measures are ambiguous and the
fast-path DECLINES (no guess); zero monetary measures fall back to the name-hint only in that gap.
Run: `python tests/test_monetary_measure_fastpath.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
import query.fast_path as FP  # noqa: E402


def _col(table, sem, role="MEASURE"):
    return {"table_name": table, "semantic_type": sem, "analytics_role": role}


class _FakeReg:
    """Minimal registry stub: one concept resolving to the given table, no dimensions."""
    def __init__(self, table):
        self._table = table

    def match_concepts(self, qtoks):
        return [({"resolves_to": {"table": self._table},
                 "default_display_columns": [f"{self._table}.name"]}, 1.0)]

    def match_dimensions_in_table(self, table, qtoks, query_l, k=2):
        return []


def _install(columns, table):
    """Patch fast_path._sm and fast_path.reg; return a restore fn."""
    o_sm, o_reg = FP._sm, FP.reg
    FP._sm = lambda: {"columns": columns}
    FP.reg = _FakeReg(table)
    return lambda: (setattr(FP, "_sm", o_sm), setattr(FP, "reg", o_reg))


def _order_by_col(sql):
    """Extract the ORDER BY column name from generated SQL (…ORDER BY a."<col>" DIR…)."""
    import re
    m = re.search(r'ORDER BY\s+\w+\."([^"]+)"', sql or "")
    return m.group(1) if m else None


# ── single MONETARY (flag ON) ─────────────────────────────────────────────────────────────────────

def test_single_monetary_selected():
    cols = {"widgets.name": _col("widgets", "FREE_TEXT", "ATTRIBUTE"),
            "widgets.price": _col("widgets", "MONETARY"),
            "widgets.rating": _col("widgets", "METRIC")}       # MEASURE but NOT monetary
    restore = _install(cols, "widgets")
    _config.FASTPATH_MONETARY_MEASURE_ENABLED = True
    try:
        r = FP._superlative_list("cheapest widgets", "cheapest widgets", {"widget"})
        assert r is not None
        assert _order_by_col(r.sql) == "price"               # monetary measure, not rating
        assert "ASC" in r.sql
    finally:
        restore(); _config.FASTPATH_MONETARY_MEASURE_ENABLED = False


def test_non_monetary_measure_never_selected():
    # only a non-monetary MEASURE exists → no monetary candidate → name-hint gap-fallback finds none.
    cols = {"widgets.name": _col("widgets", "FREE_TEXT", "ATTRIBUTE"),
            "widgets.rating": _col("widgets", "METRIC")}
    restore = _install(cols, "widgets")
    _config.FASTPATH_MONETARY_MEASURE_ENABLED = True
    try:
        r = FP._superlative_list("cheapest widgets", "cheapest widgets", {"widget"})
        assert r is None                                     # rating (non-monetary) must NOT be picked
    finally:
        restore(); _config.FASTPATH_MONETARY_MEASURE_ENABLED = False


# ── multiple MONETARY (flag ON) → decline, never guess ──────────────────────────────────────────

def test_multiple_monetary_declines():
    cols = {"widgets.name": _col("widgets", "FREE_TEXT", "ATTRIBUTE"),
            "widgets.price": _col("widgets", "MONETARY"),
            "widgets.deposit": _col("widgets", "MONETARY")}
    restore = _install(cols, "widgets")
    _config.FASTPATH_MONETARY_MEASURE_ENABLED = True
    try:
        r = FP._superlative_list("cheapest widgets", "cheapest widgets", {"widget"})
        assert r is None                                     # ambiguous → fast-path declines
    finally:
        restore(); _config.FASTPATH_MONETARY_MEASURE_ENABLED = False


# ── zero MONETARY (flag ON) → isolated name-hint fallback ────────────────────────────────────────

def test_zero_monetary_falls_back_to_name_hint():
    # No monetary-typed column, but a MEASURE named 'rent' — the gap-fallback may use the name hint.
    cols = {"widgets.name": _col("widgets", "FREE_TEXT", "ATTRIBUTE"),
            "widgets.rent": _col("widgets", "METRIC")}       # MEASURE, name-hint match, not monetary
    restore = _install(cols, "widgets")
    _config.FASTPATH_MONETARY_MEASURE_ENABLED = True
    try:
        r = FP._superlative_list("cheapest widgets", "cheapest widgets", {"widget"})
        assert r is not None and _order_by_col(r.sql) == "rent"   # gap-fallback name-hint
    finally:
        restore(); _config.FASTPATH_MONETARY_MEASURE_ENABLED = False


# ── flag OFF preserves old name-hint behaviour ──────────────────────────────────────────────────

def test_flag_off_uses_name_hint():
    # Two MEASURE cols; old behaviour name-matches _PRICE_COL_HINTS ('rent') — NOT a decline.
    cols = {"widgets.name": _col("widgets", "FREE_TEXT", "ATTRIBUTE"),
            "widgets.rent": _col("widgets", "MONETARY"),
            "widgets.deposit": _col("widgets", "MONETARY")}
    restore = _install(cols, "widgets")
    _config.FASTPATH_MONETARY_MEASURE_ENABLED = False
    try:
        r = FP._superlative_list("cheapest widgets", "cheapest widgets", {"widget"})
        assert r is not None and _order_by_col(r.sql) == "rent"   # name-hint picks first hint match
    finally:
        restore()


# ── direction (language-level, unchanged) ────────────────────────────────────────────────────────

def test_direction_asc_desc():
    cols = {"widgets.name": _col("widgets", "FREE_TEXT", "ATTRIBUTE"),
            "widgets.price": _col("widgets", "MONETARY")}
    restore = _install(cols, "widgets")
    _config.FASTPATH_MONETARY_MEASURE_ENABLED = True
    try:
        r1 = FP._superlative_list("cheapest widgets", "cheapest widgets", {"widget"})
        r2 = FP._superlative_list("most expensive widgets", "most expensive widgets", {"widget"})
        assert "ASC" in r1.sql and "DESC" in r2.sql
    finally:
        restore(); _config.FASTPATH_MONETARY_MEASURE_ENABLED = False


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
