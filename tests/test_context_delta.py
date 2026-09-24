"""Tests for deterministic context REPLACE / REMOVE — chatbot/memory/frame.py.

The audit's headline defect: the merge only ever ADDED, so

    Q1  "Show revenue by country for 2025."   → frame filters [Year equals 2025]
    Q2  "What about 2024?"
        → 'What about 2024? (for Revenue (finance_revenue), Year equals 2025)'

reached the engine carrying BOTH years. Every reference that replaces or removes a
remembered fact failed the same way, while every reference that narrows already worked.

The fix keeps the model out of the mutation: it emits {delta_type, delta_field, one
grounded value} and Python performs the dict operation. The absence assertions below are
the point — asserting the new value is present would have passed on the broken code too.

Run: ``pytest tests/test_context_delta.py``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.memory import frame as F  # noqa: E402


def _frame(*filters, **over):
    base = {
        "entity": "finance_revenue", "entity_display": "Revenue",
        "understanding": "Find revenue by country for 2025.",
        "filters": [{"field": f, "operator": "equals", "value": v, "source": "executed_sql"}
                    for f, v in filters],
        "group_by": ["Country"], "measures": ["revenue_amount"], "order_by": [],
        "limit": None, "source_id": 2, "drill_path": [],
    }
    base.update(over)
    return base


def _values(frame, field):
    return [f["value"] for f in frame.get("filters") or [] if f["field"] == field]


def _fields(frame):
    return [f["field"] for f in frame.get("filters") or []]


# ---- REPLACE ---------------------------------------------------------------------

def test_a_new_year_replaces_the_remembered_one():
    """THE audit's headline case."""
    out = F.apply_context_delta(_frame(("Year", "2025")), "replace",
                                field="Year", value="2024", message="What about 2024?")
    assert _values(out, "Year") == ["2024"]
    assert "2025" not in str(out["filters"])          # absence is the assertion


def test_a_new_country_replaces_the_remembered_one():
    out = F.apply_context_delta(_frame(("Country", "India")), "replace",
                                field="Country", value="US", message="What about the US?")
    assert _values(out, "Country") == ["US"]
    assert "India" not in str(out["filters"])


def test_replacement_leaves_every_other_filter_intact():
    """Country and CustomerType are different fields and must coexist — only the
    targeted one moves."""
    frame = _frame(("Year", "2025"), ("Country", "India"), ("CustomerType", "Enterprise"))
    out = F.apply_context_delta(frame, "replace", field="Year", value="2024",
                                message="What about 2024?")
    assert _values(out, "Year") == ["2024"]
    assert _values(out, "Country") == ["India"]
    assert _values(out, "CustomerType") == ["Enterprise"]


def test_a_quarter_replaces_a_year():
    out = F.apply_context_delta(_frame(("Year", "2025")), "replace", field="Year",
                                value="Q1 2025", message="What about Q1 2025?")
    assert _values(out, "Year") == ["Q1 2025"]


def test_field_names_match_loosely_across_the_two_vocabularies():
    """The frame's field name comes from business_explain's field_of(); the model echoes
    the user's wording. Case and separators differ, the concept does not."""
    out = F.apply_context_delta(_frame(("Order Year", "2025")), "replace",
                                field="order_year", value="2024", message="what about 2024")
    assert _values(out, "Order Year") == ["2024"]


# ---- REPLACE · the guards --------------------------------------------------------

def test_a_value_the_user_never_typed_is_refused():
    """Vocabulary gate. Carrying the old context is recoverable; writing a value the
    user never said is not — and the engine still sees the user's own words regardless."""
    frame = _frame(("Year", "2025"))
    out = F.apply_context_delta(frame, "replace", field="Year", value="2099",
                                message="What about 2024?")
    assert _values(out, "Year") == ["2025"]            # unchanged


def test_a_field_the_frame_does_not_hold_is_refused():
    """Binding gate. A field the previous turn never filtered on is not something this
    turn can replace, whatever the classifier claimed."""
    frame = _frame(("Year", "2025"))
    out = F.apply_context_delta(frame, "replace", field="Region", value="EMEA",
                                message="What about EMEA?")
    assert _fields(out) == ["Year"] and _values(out, "Year") == ["2025"]


def test_a_relative_period_drops_the_stale_filter_instead_of_guessing_a_date():
    """"last year" carries no literal to copy. Resolving one here would mean this layer
    computing a date — a derived fact the frame is built never to hold. The stale filter
    goes and the engine's existing L1 temporal parser reads the user's own words."""
    out = F.apply_context_delta(_frame(("Year", "2025"), ("Country", "India")), "replace",
                                field="Year", value="last year",
                                message="What about last year?")
    assert _values(out, "Year") == []
    assert _values(out, "Country") == ["India"]        # nothing else disturbed


# ---- REMOVE ----------------------------------------------------------------------

def test_remove_drops_the_named_filter_and_keeps_the_rest():
    frame = _frame(("Country", "India"), ("CustomerType", "Enterprise"))
    out = F.apply_context_delta(frame, "remove", field="Country", message="Remove India")
    assert _values(out, "Country") == []
    assert _values(out, "CustomerType") == ["Enterprise"]


def test_exclude_phrasing_removes_the_condition_and_preserves_the_other():
    frame = _frame(("Country", "India"), ("CustomerType", "Enterprise"))
    out = F.apply_context_delta(frame, "remove", field="CustomerType",
                                message="Exclude enterprise customers")
    assert _values(out, "CustomerType") == []
    assert _values(out, "Country") == ["India"]


def test_remove_of_an_unheld_field_changes_nothing():
    frame = _frame(("Country", "India"))
    out = F.apply_context_delta(frame, "remove", field="Region", message="remove EMEA")
    assert _fields(out) == ["Country"]


# ---- the additive path must not regress ------------------------------------------

def test_refine_is_untouched_by_this_code():
    """"Only active customers" is an ADD. apply_context_delta only acts on replace and
    remove; every other delta type returns the frame unchanged."""
    frame = _frame(("Country", "India"))
    for delta in ("refine", "new_topic", "drill_down", "drill_up", "compare", "ambiguous"):
        assert F.apply_context_delta(frame, delta, field="Country", value="US",
                                     message="what about the US") is frame


def test_an_empty_or_missing_frame_is_safe():
    assert F.apply_context_delta(None, "replace", field="Year", value="2024",
                                 message="2024") == {}
    assert F.apply_context_delta({}, "remove", field="Year", message="x") == {}
    frameless = {"entity": "x", "filters": []}
    assert F.apply_context_delta(frameless, "replace", field="Year", value="2024",
                                 message="2024") is frameless


def test_the_original_frame_is_never_mutated():
    """The caller keeps the pre-delta frame for logging and for the write path."""
    frame = _frame(("Year", "2025"))
    before = [dict(f) for f in frame["filters"]]
    F.apply_context_delta(frame, "replace", field="Year", value="2024", message="2024")
    assert frame["filters"] == before


# ---- backward compatibility ------------------------------------------------------

def test_a_frame_stored_before_these_fields_existed_still_loads():
    """Old Redis entries carry only entity + filters. Nothing added here is required."""
    legacy = {"entity": "finance_revenue",
              "filters": [{"field": "Year", "operator": "equals", "value": "2025"}]}
    out = F.apply_context_delta(legacy, "replace", field="Year", value="2024",
                                message="What about 2024?")
    assert _values(out, "Year") == ["2024"]
    assert F.render_frame_as_query(legacy, "only in Mumbai", "refine").startswith("only in Mumbai")


# ---- the rendered query stays natural --------------------------------------------

def test_the_resolved_query_carries_no_operation_language():
    """Measured 2026-09-17: appending operation words ("measuring <col>, ranked by
    <col>, top 100") made the engine spend 54s and then ask whether "measuring" was a
    column. Operation metadata belongs in structured state, never in the query text."""
    out = F.apply_context_delta(_frame(("Year", "2025")), "replace", field="Year",
                                value="2024", message="What about 2024?")
    resolved = F.render_frame_as_query(out, "What about 2024?", "replace")
    for banned in ("replace", "REPLACE", "measuring", "ranked by", "delta", "slot", "top 100"):
        assert banned not in resolved, f"{banned!r} leaked into {resolved!r}"
    assert "2024" in resolved and "2025" not in resolved


# --- the removal phrasing must not survive into the engine's query -----------------

def test_remove_renders_the_new_context_not_the_removal_words():
    """Live scenario run, 2026-09-17: the structured removal worked — the frame no
    longer held Country — but the resolved query still read

        'Remove India (for Revenue (finance_revenue), CustomerType equals Enterprise)'

    and the engine parses every word of that, so it was handed the very value it had
    just been told to drop. "Remove India" carries no data content of its own; the
    already-mutated frame IS the new question. Same treatment drill_up has had since
    2026-07, for the same reason."""
    frame = _frame(("Country", "India"), ("CustomerType", "Enterprise"))
    after = F.apply_context_delta(frame, "remove", field="Country", message="Remove India")
    resolved = F.render_frame_as_query(after, "Remove India", "remove")
    assert "India" not in resolved
    assert "Enterprise" in resolved
    assert "Remove" not in resolved


def test_replace_still_quotes_the_users_words_because_they_carry_the_new_value():
    """The contrast: "What about 2024?" DOES carry content — the year. It stays."""
    after = F.apply_context_delta(_frame(("Year", "2025")), "replace", field="Year",
                                  value="2024", message="What about 2024?")
    resolved = F.render_frame_as_query(after, "What about 2024?", "replace")
    assert resolved.startswith("What about 2024?")
    assert "2024" in resolved and "2025" not in resolved


def test_untouched_filters_are_copied_not_shared_with_the_input_frame():
    """Found by an adversarial sweep, 2026-09-17. The returned frame used to hand back
    the caller's OWN dicts for every filter the delta did not touch, so editing either
    frame's filters reached into the other. No single-call test can see it — it only
    shows when both frames are still alive, which is exactly the situation
    context_resolve_node creates (it keeps the pre-delta frame for logging)."""
    frame = _frame(("Year", "2025"), ("Country", "India"))
    out = F.apply_context_delta(frame, "replace", field="Year", value="2024",
                                message="What about 2024?")
    out["filters"][1]["value"] = "TAMPERED"
    assert _values(frame, "Country") == ["India"]

    removed = F.apply_context_delta(frame, "remove", field="Year", message="remove the year")
    removed["filters"][0]["value"] = "TAMPERED"
    assert _values(frame, "Country") == ["India"]


# --- word-level field matching (2026-09-17, independent review) --------------------
#
# The first cut matched field names by raw SUBSTRING. Three ways that deleted a filter
# the user had asked to keep, silently:
#   is_temporal_field("candidate_name") -> True   ("candi-DATE")
#   is_temporal_field("notify_email")   -> True   ("noti-FY")
#   _same_field("city", "capacity")     -> True
# Matching whole WORDS instead is the difference between a working guard and a silent one.

def test_a_time_word_inside_another_word_is_not_temporal():
    for field in ("candidate_name", "notify_email", "holiday_owner", "update_by",
                  "daybreak_owner", "Region"):
        assert not F.is_temporal_field(field), field


def test_real_temporal_fields_are_still_recognised():
    for field in ("Year", "payment_date", "as_of_date", "orderYear", "fiscal_quarter",
                  "created_datetime", "billing_period"):
        assert F.is_temporal_field(field), field


def test_a_replace_on_a_field_that_merely_contains_a_time_word_substitutes_not_deletes():
    """The consequence of the bug: the user asked about Mary and the filter vanished,
    so the engine was asked about everyone, with nothing saying anything was dropped."""
    frame = _frame(("Candidate", "John"))
    out = F.apply_context_delta(frame, "replace", field="Candidate", value="Mary",
                                message="what about Mary?")
    assert _values(out, "Candidate") == ["Mary"]


def test_fields_that_merely_share_letters_are_not_the_same_field():
    for a, b in (("city", "capacity"), ("id", "paid"), ("Country", "CustomerType"),
                 ("age", "package"), ("rent", "current")):
        assert not F._same_field(a, b), f"{a!r} wrongly matched {b!r}"


def test_the_same_field_written_differently_still_matches():
    for a, b in (("order_year", "ORDER  YEAR"), ("orderYear", "order_year"),
                 ("Year", "Fiscal Year"), ("name", "user_name")):
        assert F._same_field(a, b), f"{a!r} should match {b!r}"


def test_removing_a_field_the_frame_does_not_hold_leaves_the_lookalike_alone():
    frame = _frame(("Capacity", "10"))
    out = F.apply_context_delta(frame, "remove", field="City", message="remove city")
    assert _values(out, "Capacity") == ["10"]


# --- an engine-internal alias is not a business entity -----------------------------

def test_a_generated_sql_alias_is_not_recorded_as_the_entity():
    """Live run 2026-09-17: a query whose SQL used a CTE returned table="agg_0"; the
    frame stored it and the NEXT turn went to the engine as
    "what about Pune (for Agg 0s (agg_0))" — a string this layer invented, handed to a
    pipeline that parses every word as data ("Could you clarify if 'agg' is a column
    name?", 135s)."""
    for alias in ("agg_0", "cte_1", "t0", "sub2", "tmp", "derived_3"):
        assert F._looks_like_an_engine_alias(alias), alias
    for real in ("assets_asset", "accounts_paymenttransaction", "customers", "t_shirt_sales"):
        assert not F._looks_like_an_engine_alias(real), real


def test_harvest_refuses_an_alias_rather_than_storing_it():
    harvested = F.harvest_frame({
        "status": "answered", "table": "agg_0", "sql": "WITH agg_0 AS (...) SELECT 1",
        "rows": [[1]], "analytics": {"orderings": [], "limit": None, "query_measures": []},
        "explain": {"data_used": {"datasets": ["Agg 0s"]}, "filters": {"applied": []},
                    "operations": [], "understanding": {"summary": "x"}}})
    assert harvested["entity"] is None


# ---------------------------------------------------------------------------
# A replacement value that names a SET, not a value
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from chatbot.memory.frame import (_names_a_set_not_a_value, apply_context_delta,  # noqa: E402
                                  render_frame_as_query)


@pytest.mark.parametrize("value", [
    "verified ones", "the active ones", "paid records", "open items",
    "matching rows", "these entries", "those results", "one",
])
def test_a_collective_noun_is_recognised(value):
    assert _names_a_set_not_a_value(value) is True, value


@pytest.mark.parametrize("value", [
    "Mumbai", "New York", "2024", "ACTIVE", "Tax Invoice", "PROPERTY_VIEW",
    "10", "", None,
])
def test_a_real_value_is_not(value):
    assert _names_a_set_not_a_value(value) is False, value


def test_a_new_dimension_never_overwrites_an_existing_filter():
    """Measured 2026-09-21 in an 18-turn conversation. Frame held City=Pune and
    Status=ACTIVE; "just the verified ones" introduces a NEW dimension, but the model
    returned replace/Status/"verified ones" — so ACTIVE was overwritten with a value no
    column holds and the user's "active" constraint vanished silently.

    Both existing guards passed it: Status IS in the frame, and the value IS verbatim in
    the message. This is the third guard."""
    frame = {
        "entity": "assets_asset", "entity_display": "Assets",
        "filters": [{"field": "City", "operator": "equals", "value": "Pune",
                     "source": "executed_sql"},
                    {"field": "Status", "operator": "equals", "value": "ACTIVE",
                     "source": "executed_sql"}],
        "group_by": [], "measures": [], "order_by": [], "drill_path": [], "limit": None,
    }
    out = apply_context_delta(frame, "replace", "Status", "verified ones",
                              "just the verified ones")
    assert out is frame, "the declined delta must return the caller's own frame"
    assert [f["value"] for f in out["filters"]] == ["Pune", "ACTIVE"]


def test_the_declined_turn_still_reaches_the_engine_with_everything():
    """Declining costs nothing: the frame carries forward AND the user's words go on to
    the engine, so it sees "verified" alongside Pune and ACTIVE and can resolve all
    three. That is why the guard is safe to make strict."""
    frame = {
        "entity": "assets_asset", "entity_display": "Assets",
        "filters": [{"field": "City", "operator": "equals", "value": "Pune",
                     "source": "executed_sql"},
                    {"field": "Status", "operator": "equals", "value": "ACTIVE",
                     "source": "executed_sql"}],
        "group_by": [], "measures": [], "order_by": [], "drill_path": [], "limit": None,
    }
    out = apply_context_delta(frame, "replace", "Status", "verified ones",
                              "just the verified ones")
    resolved = render_frame_as_query(out, "just the verified ones", "replace")
    assert resolved.startswith("just the verified ones")
    # The frame carries filter VALUES, not "Field equals Value": the field name it
    # holds is the engine's display label ("City"/"Location") and echoing it back made
    # Tier-2 try to ground a column that does not exist, derailing the next turn
    # (measured 2026-09-22, chatbot/memory/frame.py::_describe_frame).
    assert "Pune" in resolved
    assert "ACTIVE" in resolved


def test_an_ordinary_value_swap_is_untouched():
    frame = {
        "entity": "assets_asset", "entity_display": "Assets",
        "filters": [{"field": "City", "operator": "equals", "value": "Pune",
                     "source": "executed_sql"}],
        "group_by": [], "measures": [], "order_by": [], "drill_path": [], "limit": None,
    }
    out = apply_context_delta(frame, "replace", "City", "Mumbai", "what about Mumbai")
    assert [f["value"] for f in out["filters"]] == ["Mumbai"]
