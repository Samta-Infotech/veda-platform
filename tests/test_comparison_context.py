"""Comparison context — two sides, explicitly, so a follow-up cannot collapse them.

Before this existed, `compare` was a label with two call sites (a grounding gate and a
telemetry string). A `compare` turn rendered as an ordinary refinement of the current
frame, so the comparand had nowhere to live and the next turn saw a single context.

Design notes the tests pin:
 · a side is a PROJECTION of a frame, not a second frame — version/tenant/session/
   last_sql/drill_path mean nothing for a comparand and would be two things to keep in step
 · the comparison is SESSION-level, not filed under a source: it can span two sources,
   and filing it under one would let that side silently own the other
 · rendering carries VALUES only, never field names — measured: rendering the engine's
   own column label ("Location equals Mumbai") made it ground a non-existent column and
   answer from an unrelated table

Pure: no SLM, no DB, no Redis.
"""
import pytest

from chatbot.memory.frame import (build_comparison, comparison_is_stale,
                                  detect_comparison_target,
                                  render_comparison_as_query)


def revenue(year="2025", source_id=2):
    return {"entity": "revenue_table", "entity_display": "Revenue", "source_id": source_id,
            "measures": ["revenue"], "group_by": ["month"], "limit": 100,
            "order_by": [{"field": "revenue", "desc": True}],
            "filters": [{"field": "Year", "operator": "equals", "value": year}],
            # bookkeeping a side must NOT inherit
            "version": 7, "tenant": "t", "session_id": "s", "last_sql": "SELECT 1",
            "drill_path": [{"dimension": "Year", "value": year}]}


class TestBuildingAComparison:
    def test_both_sides_are_kept(self):
        c = build_comparison(revenue("2025"), revenue("2024"))
        assert c["primary"]["label"] == "2025"
        assert c["comparison"]["label"] == "2024"

    def test_a_side_carries_what_a_follow_up_needs(self):
        c = build_comparison(revenue("2025"), revenue("2024"))
        side = c["primary"]
        for key in ("entity", "source_id", "filters", "measures", "group_by",
                    "order_by", "limit"):
            assert key in side, key

    def test_a_side_is_not_a_second_frame(self):
        """Composition, not duplication — bookkeeping stays with the frame."""
        side = build_comparison(revenue(), revenue("2024"))["primary"]
        for key in ("version", "tenant", "session_id", "last_sql", "drill_path"):
            assert key not in side, key

    def test_the_source_frames_are_not_mutated(self):
        a, b = revenue("2025"), revenue("2024")
        build_comparison(a, b)
        assert a["filters"][0]["value"] == "2025" and b["filters"][0]["value"] == "2024"


class TestDimensionIsDerived:
    def test_same_source_same_entity_is_a_value_comparison(self):
        assert build_comparison(revenue("2025"), revenue("2024"))["dimension"] == "value"

    def test_different_sources_is_a_source_comparison(self):
        c = build_comparison(revenue(source_id=2), revenue(source_id=3))
        assert c["dimension"] == "source"

    def test_different_entities_is_an_entity_comparison(self):
        other = {**revenue(), "entity": "contracts", "entity_display": "Contracts",
                 "filters": []}
        assert build_comparison(revenue(), other)["dimension"] == "entity"


class TestHalfBuiltComparisonsAreRefused:
    """A comparand that names nothing cannot ground a follow-up, and half a comparison
    is worse than none — the next turn would resolve against an empty side."""

    @pytest.mark.parametrize("bad", [{}, None, {"filters": []}])
    def test_a_side_without_an_entity(self, bad):
        assert build_comparison(revenue(), bad) is None
        assert build_comparison(bad, revenue()) is None


class TestRendering:
    def test_both_sides_reach_the_engine(self):
        c = build_comparison(revenue("2025"), revenue("2024"))
        out = render_comparison_as_query(c, "what about profit")
        assert "2025" in out and "2024" in out

    def test_no_field_name_is_rendered(self):
        """The measured rule: values ground correctly, field names do not — even when
        the field name is right."""
        c = build_comparison(revenue("2025"), revenue("2024"))
        out = render_comparison_as_query(c, "what about profit")
        assert "Year" not in out and "equals" not in out

    def test_the_users_words_survive_verbatim(self):
        c = build_comparison(revenue("2025"), revenue("2024"))
        assert render_comparison_as_query(c, "what about profit").startswith("what about profit")

    def test_no_comparison_leaves_the_message_alone(self):
        assert render_comparison_as_query(None, "what about profit") == "what about profit"
        assert render_comparison_as_query({}, "what about profit") == "what about profit"

    def test_a_side_with_no_label_is_not_rendered_half(self):
        c = build_comparison(revenue("2025"), revenue("2024"))
        c["comparison"]["label"] = ""
        assert render_comparison_as_query(c, "x") == "x"


class TestStaleness:
    def test_an_unrelated_topic_makes_it_stale(self):
        c = build_comparison(revenue("2025"), revenue("2024"))
        assert comparison_is_stale(c, {"entity": "assets_asset"})

    def test_a_turn_on_either_side_is_not_stale(self):
        c = build_comparison(revenue("2025"), revenue("2024"))
        assert not comparison_is_stale(c, {"entity": "revenue_table"})

    def test_a_turn_with_no_entity_decides_nothing(self):
        """A refused turn writes no frame; it must not drop the comparison either."""
        c = build_comparison(revenue("2025"), revenue("2024"))
        assert not comparison_is_stale(c, {})
        assert not comparison_is_stale(c, None)

    def test_no_comparison_is_never_stale(self):
        assert not comparison_is_stale(None, {"entity": "x"})


class TestComparandExtraction:
    """The model's `compare` label is reliable; the comparand VALUE it extracts is not.

    Measured live, 2026-09-22: "compare that with Mumbai" classified as `compare` on
    every attempt, and the model's own slot_candidates came back EMPTY every time — so
    nothing populated delta_value, and rendering fell through to sending the raw message
    (word "compare" included) to the engine, which asked whether "compare" was a column
    name. The turn was refused, and refused turns never reach memory_write_node, so no
    comparison was ever built — despite the classifier doing its one job correctly.

    Same fix shape as detect_shape_delta / detect_measure_addition: detection (is this a
    compare?) stays the model's; extraction (which value?) becomes deterministic Python,
    verbatim from the message.
    """

    @pytest.mark.parametrize("message,expected", [
        ("compare that with Mumbai", "Mumbai"),
        ("compare with 2024", "2024"),
        ("compare 2025 vs 2024", "2024"),
        ("Compare Source A with Source B", "Source B"),
        ("now compare that with 2023", "2023"),
        ("compare profit against revenue", "revenue"),
    ])
    def test_the_comparand_is_extracted_verbatim(self, message, expected):
        assert detect_comparison_target(message) == expected

    def test_a_bare_compare_with_no_target_extracts_nothing(self):
        """Nothing to bind to — this must not invent a value."""
        assert detect_comparison_target("compare") is None
        assert detect_comparison_target("") is None

    def test_the_extracted_text_is_a_literal_substring(self):
        """Never a paraphrase or an invented word — the whole reason this exists rather
        than trusting the model's own extraction."""
        message = "compare that with Mumbai"
        assert detect_comparison_target(message) in message
