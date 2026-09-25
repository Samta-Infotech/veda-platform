"""Tests for query/ranking_parser.py — the shared top-N/ranking extractor.
Run from the repo root: ``pytest tests/test_ranking_parser.py``"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

from query.ranking_parser import parse_ranking


def test_no_ranking_language():
    r = parse_ranking("show all incidents")
    assert r.top_n is None and r.ranked is False


def test_top_n_word_before_number():
    r = parse_ranking("top 10 customers")
    assert r.top_n == 10 and r.ranked and r.basis == "metric" and r.direction == "desc"


def test_first_n_word_before_number():
    r = parse_ranking("first 5 orders")
    assert r.top_n == 5 and r.basis == "temporal" and r.direction == "asc"


def test_latest_n_the_original_bug():
    r = parse_ranking("show the latest 10 ledger entries")
    assert r.top_n == 10
    assert r.ranked is True
    assert r.basis == "temporal"
    assert r.direction == "desc"


def test_last_n_no_time_unit():
    r = parse_ranking("last 20 transactions")
    assert r.top_n == 20 and r.basis == "temporal" and r.direction == "desc"


def test_newest_n():
    r = parse_ranking("newest 7 signups")
    assert r.top_n == 7 and r.basis == "temporal" and r.direction == "desc"


def test_most_recent_phrase():
    r = parse_ranking("the 10 most recent invoices")
    assert r.top_n == 10 and r.basis == "temporal" and r.direction == "desc"


def test_number_before_word():
    r = parse_ranking("10 latest ledger entries")
    assert r.top_n == 10 and r.basis == "temporal" and r.direction == "desc"


def test_oldest_n():
    r = parse_ranking("oldest 3 tickets")
    assert r.top_n == 3 and r.basis == "temporal" and r.direction == "asc"


def test_earliest_n():
    r = parse_ranking("earliest 5 signups")
    assert r.top_n == 5 and r.basis == "temporal" and r.direction == "asc"


def test_bottom_n_metric_asc():
    r = parse_ranking("bottom 5 performers")
    assert r.top_n == 5 and r.basis == "metric" and r.direction == "asc"


def test_lowest_n():
    r = parse_ranking("lowest 3 scores")
    assert r.top_n == 3 and r.basis == "metric" and r.direction == "asc"


def test_highest_n():
    r = parse_ranking("highest 8 earners")
    assert r.top_n == 8 and r.basis == "metric" and r.direction == "desc"


def test_fewest_n():
    r = parse_ranking("fewest 2 complaints")
    assert r.top_n == 2 and r.basis == "metric" and r.direction == "asc"


def test_spelled_out_number_top():
    r = parse_ranking("top five customers")
    assert r.top_n == 5


def test_spelled_out_number_latest():
    r = parse_ranking("latest ten entries")
    assert r.top_n == 10


def test_ranking_word_without_number_still_flags_ranked():
    r = parse_ranking("show the highest paying customers")
    assert r.ranked is True
    assert r.top_n is None
    assert r.basis == "metric" and r.direction == "desc"


def test_ranked_word_alone_no_number_temporal():
    r = parse_ranking("show the latest audit log entries")
    assert r.ranked is True
    assert r.top_n is None
    assert r.basis == "temporal"


def test_plain_query_with_incidental_number_not_a_count():
    # "10" here isn't attached to any ranking word — must not be picked up.
    r = parse_ranking("show orders placed after invoice 10")
    assert r.top_n is None
    assert r.ranked is False


def test_sort_requested_covers_bare_sort_verbs():
    """`sort_requested` is the signal veda/ir_equivalence.py's unrequested-ordering rule uses.
    It must be True for a bare sort verb that names no end and no count — that rule used to
    keep its own narrower word list, which is what this field exists to replace."""
    assert parse_ranking("assets sorted by price").sort_requested is True
    assert parse_ranking("rank vendors").sort_requested is True
    assert parse_ranking("order assets by carpet area").sort_requested is True


def test_sort_requested_is_true_for_every_ranking_word():
    """Every ranking word implies an ordering — including the ones the old duplicated list in
    ir_equivalence omitted (bottom/biggest/greatest/maximum/minimum/fewest/latest/newest/
    oldest/earliest), which is why a correct `ORDER BY amount ASC` for "bottom 3" was refused."""
    for word in ("bottom", "biggest", "greatest", "maximum", "minimum", "fewest",
                 "latest", "newest", "oldest", "earliest", "top", "lowest"):
        spec = parse_ranking(f"{word} 3 general ledger entries")
        assert spec.ranked is True, word
        assert spec.sort_requested is True, word


def test_sort_requested_false_when_nothing_asks_for_order():
    for q in ("list all assets", "how many assets are there", "assets in Pune"):
        assert parse_ranking(q).sort_requested is False, q


# ── price superlatives ────────────────────────────────────────────────────────────────
# Measured 2026-09-24 on the live pipeline: "Show me the 5 cheapest ones" produced
# `SELECT ... FROM assets_salelisting` with NO ORDER BY and NO LIMIT — all 373 rows, with
# the summariser narrating an arbitrary first row as the answer. The identical question
# phrased "5 lowest ones" was correct. Cause: "cheapest" was known to config.py's
# `superlative_min` and to fast_path's _SUPERLATIVE_ASC/_SUP_ASC, but not to THIS module,
# which every SQL-construction path reads.
def test_cheapest_is_a_ranking_word():
    spec = parse_ranking("Show me the 5 cheapest ones.")
    assert spec.ranked is True
    assert spec.top_n == 5
    assert spec.direction == "asc"
    assert spec.basis == "metric"


def test_a_price_superlative_with_no_count_is_still_ranked():
    """"Which one is the cheapest?" names no N — it must still order."""
    spec = parse_ranking("Which one is the cheapest?")
    assert spec.ranked is True and spec.top_n is None and spec.direction == "asc"


def test_the_expensive_end_too():
    # "the costliest one" is top_n=1, not None: "one" is a NUM_WORD, and reading it as a
    # count is right — the question asks for a single row.
    for q, n in (("the 3 most expensive listings", 3), ("2 priciest assets", 2),
                 ("the costliest one", 1), ("highest priced properties", None)):
        spec = parse_ranking(q)
        assert spec.ranked is True, q
        assert spec.direction == "desc", q
        assert spec.top_n == n, q


def test_multi_word_forms_beat_the_bare_word_they_contain():
    """"most expensive" must not be read as the bare "most" (desc) — and, worse,
    "least expensive" must not be read as "least" alone, which happens to agree here but
    would not if the phrase list were ordered the other way."""
    assert parse_ranking("least expensive property").direction == "asc"
    assert parse_ranking("most affordable 4 units").direction == "asc"
    assert parse_ranking("lowest priced 5").direction == "asc"


def test_price_superlatives_ask_for_an_ordering():
    for word in ("cheapest", "priciest", "costliest", "dearest"):
        assert parse_ranking(f"{word} 3 listings").sort_requested is True, word
