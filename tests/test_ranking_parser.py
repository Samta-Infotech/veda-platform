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
