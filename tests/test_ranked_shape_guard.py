import traceback
"""Tests for the ranking-intent shape guard (answer-safety).

A query naming an explicit count of RANKED rows ("top 5 debit transaction", "latest 10 entries")
answered by an UNORDERED projection returns an arbitrary N that the summariser then presents as
the ranked ones — measured 2026-09-23, where "top 5 debit transaction" named the first five rows
the table happened to return while the real top five were three orders of magnitude larger.

The guard fires ONLY on an explicit count with no ORDER BY and no aggregate, so ranking language
without a count ("first name", "highest amount", "most common") is untouched.
Run: `python tests/test_ranked_shape_guard.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from veda.validation import ranked_shape_ok  # noqa: E402

_UNORDERED = 'SELECT "transaction_type", "user_invoice_id" FROM "accounts_paymenttransaction" WHERE x = 1 LIMIT 5'


# ── the real failure this guard exists for ────────────────────────────────────────────────────
def test_explicit_count_without_order_refuses():
    ok, why = ranked_shape_ok("top 5 debit transaction", _UNORDERED)
    assert ok is False
    assert "no particular order" in why


def test_refusal_names_what_was_asked_for():
    _, why = ranked_shape_ok("top 5 debit transaction", _UNORDERED)
    assert "top 5 by value" in why
    _, why = ranked_shape_ok("latest 10 entries", 'SELECT a FROM t LIMIT 10')
    assert "latest 10 by date" in why
    _, why = ranked_shape_ok("oldest 5 records", 'SELECT a FROM t LIMIT 5')
    assert "earliest 5 by date" in why
    _, why = ranked_shape_ok("bottom 3 assets", 'SELECT a FROM t LIMIT 3')
    assert "bottom 3 by value" in why


def test_temporal_ranking_without_order_refuses():
    assert ranked_shape_ok("latest 10 entries", 'SELECT a FROM t LIMIT 10')[0] is False


# ── legitimate shapes must pass ───────────────────────────────────────────────────────────────
def test_ordered_sql_passes():
    assert ranked_shape_ok(
        "top 5 debit transaction",
        'SELECT a FROM t WHERE x = 1 ORDER BY "paid_amount" DESC LIMIT 5')[0] is True


def test_aggregate_already_expresses_the_extreme():
    assert ranked_shape_ok("top 5 debit transaction", 'SELECT MAX(amount) FROM t')[0] is True


def test_grouped_and_ordered_passes():
    assert ranked_shape_ok(
        "top 5 cities by asset count",
        "SELECT c, COUNT(*) FROM a GROUP BY c ORDER BY 2 DESC LIMIT 5")[0] is True


# ── ranking language WITHOUT a count is not this guard's business ─────────────────────────────
def test_ranking_word_without_count_is_untouched():
    # "first"/"highest"/"most" all carry a ranking word; none names a row count, and a plain
    # projection or an aggregate answers them legitimately.
    for q in ("show first name of users", "highest paid amount", "what are the most common issues"):
        assert ranked_shape_ok(q, 'SELECT a, b FROM t')[0] is True, q


def test_plain_count_query_untouched():
    assert ranked_shape_ok("how many assets are there", "SELECT COUNT(*) FROM a")[0] is True


def test_bare_limit_without_ranking_word_untouched():
    # "list 5 assets" names a count but no ranking — an unordered 5 rows is exactly what was asked.
    assert ranked_shape_ok("list 5 assets", 'SELECT a FROM t LIMIT 5')[0] is True


# ── failure-safety ────────────────────────────────────────────────────────────────────────────
def test_empty_sql_passes():
    assert ranked_shape_ok("top 5 debit transaction", "")[0] is True
    assert ranked_shape_ok("top 5 debit transaction", None)[0] is True


def test_unparseable_sql_does_not_block():
    assert ranked_shape_ok("top 5 debit transaction", "NOT ; VALID (( SQL")[0] is True


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


# ── the GROUP BY hole: an aggregate is not automatically a ranking ────────────────────────────
_AMENITY_SQL = ('SELECT t0."amenity_name" AS amenity, COUNT(DISTINCT t0."amenity_id") AS "count" '
                'FROM "catalog_parquet"."amenities_catalog" t0 GROUP BY t0."amenity_name"')


def test_grouped_count_without_order_refuses():
    """The measured escape: `COUNT(...) … GROUP BY` satisfied a bare `exp.AggFunc` test, so a
    "top 5" question was answered with SEVEN unordered rows. A grouped aggregate returns many
    rows; with no ORDER BY they are in no order."""
    ok, why = ranked_shape_ok("top 5 general ledger entries", _AMENITY_SQL)
    assert ok is False
    assert "no particular order" in why


def test_grouped_count_with_order_still_passes():
    assert ranked_shape_ok(
        "top 5 general ledger entries",
        _AMENITY_SQL + ' ORDER BY "count" DESC LIMIT 5')[0] is True


def test_scalar_aggregate_is_not_a_row_ranking():
    """No GROUP BY → one row. That is a superlative answered directly, not an N-row ranking, so
    this guard leaves it alone."""
    assert ranked_shape_ok("top 5 debit transaction", "SELECT MAX(amount) FROM t")[0] is True
    assert ranked_shape_ok("top 5 debit transaction", "SELECT COUNT(*) FROM t")[0] is True
