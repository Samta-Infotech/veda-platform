"""FLAG and YEAR qualifiers of a drill-down follow-up (veda_core/query/qualifier_grounding.py).

Measured 2026-09-24: "only the gated ones" and "only the ones built in 2026" came back with
the previous turn's SQL unchanged — the qualifier silently dropped.

Pure: no DB, no SLM. Run: `python -m pytest tests/test_qualifier_grounding.py`
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from query.qualifier_grounding import flag_phrase, ground_flags, ground_year  # noqa: E402

FLAGS = {"is_gated", "all_day_access", "corner_property", "power_backup"}
SM = {"columns": {
    "assets_asset.construction_year": {"aliases": ["construction year", "year built", "year"]},
    "assets_asset.facing": {"aliases": ["facing"]},
    "assets_asset.floor_type_id": {"aliases": ["floor type"]},
}}
SM_TWO_YEARS = {"columns": {
    **SM["columns"],
    "assets_asset.possession_year": {"aliases": ["possession year", "year of possession"]},
}}


def _cols(fs):
    return {(f["column"], f["value"]) for f in fs}


# ── flags ─────────────────────────────────────────────────────────────────────────────
def test_the_measured_case_grounds_the_flag():
    assert _cols(ground_flags("only the gated ones", FLAGS)) == {("is_gated", "true")}


def test_negations_ground_false():
    for msg in ("only the non-gated ones", "only the not gated ones",
                "only the non gated ones", "the ones without power backup"):
        got = _cols(ground_flags(msg, FLAGS))
        assert len(got) == 1 and next(iter(got))[1] == "false", msg


def test_multi_word_flags_and_plurals():
    assert _cols(ground_flags("only the corner properties", FLAGS)) == {("corner_property", "true")}
    assert _cols(ground_flags("just the ones with all day access", FLAGS)) == {
        ("all_day_access", "true")}


def test_a_comparison_is_not_a_filter():
    for msg in ("gated vs non-gated", "how many are gated versus not gated",
                "gated or not"):
        assert ground_flags(msg, FLAGS) == [], msg


def test_a_word_that_merely_contains_the_flag_does_not_match():
    assert ground_flags("only the navigated ones", FLAGS) == []
    assert ground_flags("only the Nagpur ones", FLAGS) == []


def test_flag_phrase():
    assert flag_phrase("is_gated") == ["gated"]
    assert flag_phrase("has_power_backup") == ["power", "backup"]
    assert flag_phrase("corner_property") == ["corner", "property"]


# ── years ─────────────────────────────────────────────────────────────────────────────
def test_the_measured_year_case_grounds_the_one_year_column():
    fs, refusal = ground_year("only the ones built in 2026", "assets_asset", SM)
    assert refusal is None and _cols(fs) == {("construction_year", "2026")}


def test_no_year_no_opinion():
    assert ground_year("only the gated ones", "assets_asset", SM) == ([], None)


def test_two_year_columns_are_disambiguated_by_the_users_words():
    fs, refusal = ground_year("only the ones built in 2020", "assets_asset", SM_TWO_YEARS)
    assert refusal is None and _cols(fs) == {("construction_year", "2020")}
    fs, refusal = ground_year("possession in 2020", "assets_asset", SM_TWO_YEARS)
    assert refusal is None and _cols(fs) == {("possession_year", "2020")}


def test_an_unpinnable_year_is_refused_not_guessed():
    fs, refusal = ground_year("only the 2020 ones", "assets_asset", SM_TWO_YEARS)
    assert fs == [] and refusal and "2020" in refusal
    fs, refusal = ground_year("only the 2020 ones", "worklists_ticket", SM)   # no year column
    assert fs == [] and refusal


def test_two_different_years_are_refused():
    fs, refusal = ground_year("built in 2019 or 2020", "assets_asset", SM)
    assert fs == [] and refusal


# ── the two live findings ─────────────────────────────────────────────────────────────
def test_a_negated_flag_reports_the_word_it_consumed():
    """Live 2026-09-24: is_gated = false was grounded, then the qualifier gate refused on
    the leftover word 'non'. The predicate's value now represents it."""
    (f,) = ground_flags("only the non-gated ones", FLAGS)
    assert f["consumed"] == ["non"]
    (f,) = ground_flags("only the gated ones", FLAGS)
    assert f["consumed"] == []


def test_built_names_the_year_column_but_a_bare_year_does_not():
    """Live 2026-09-24: L1 read "built in 2026" as a created_at range. "built" names the
    construction_year column; "in 2026" / "created in 2026" name none of them."""
    from query.qualifier_grounding import named_year_column
    assert named_year_column("only the ones built in 2026", "assets_asset", SM) == "construction_year"
    assert named_year_column("only the ones from 2026", "assets_asset", SM) is None
    assert named_year_column("only the ones created in 2026", "assets_asset", SM) is None
    assert named_year_column("only the 2026 ones by year", "assets_asset", SM) is None
