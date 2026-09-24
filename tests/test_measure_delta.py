"""Measure-level delta: "revenue" → "also include profit".

The closed delta set (prompts/delta_types.py) has no "add", and the model is never asked
for one — adding a MEASURE is detected deterministically from the user's own words and
grounded against the table's real measure columns, the same discipline detect_shape_delta
uses for "make it top 10".

Why grounding matters here specifically: the failure this layer has already caused once
was inventing a column name and sending it to the engine ("Location equals Mumbai", a
display label for a column that does not exist). A measure the user names but the table
does not offer must therefore change nothing.

SCOPE: this updates MEMORY, not the request. Measures are not rendered into the resolved
query (frame.py::_describe_frame), so the engine learns about "profit" from the user's
verbatim message; the frame gains the knowledge that the question now has two measures.

Pure: no SLM, no DB, no Redis.
"""
import pytest

from chatbot.memory.frame import add_measure, detect_measure_addition


def frame(**kw):
    f = {"entity": "revenue_table", "entity_display": "Revenue",
         "route": "deterministic",
         "measures": ["revenue"],
         "available_measures": ["revenue", "profit", "carpet_area", "total_views"],
         "group_by": ["month"],
         "order_by": [{"field": "revenue", "desc": True}],
         "limit": 100,
         "filters": [{"field": "Location", "operator": "equals", "value": "pune"}]}
    f.update(kw)
    return f


class TestAddMeasure:
    @pytest.mark.parametrize("message,expected", [
        ("also include profit", "profit"),
        ("add profit", "profit"),
        ("include profit", "profit"),
        ("revenue along with profit", "profit"),
        ("profit too", "profit"),
        ("show me the carpet area as well", "carpet_area"),
        ("total views too", "total_views"),
    ])
    def test_a_named_available_measure_is_added(self, message, expected):
        assert detect_measure_addition(frame(), message) == expected

    def test_the_measure_list_is_extended(self):
        out = add_measure(frame(), "profit")
        assert out["measures"] == ["revenue", "profit"]

    def test_nothing_else_is_touched(self):
        """The whole point of a deterministic mutation: one field moves."""
        before = frame()
        out = add_measure(before, "profit")
        for slot in ("group_by", "order_by", "limit", "filters", "entity"):
            assert out[slot] == before[slot], slot


class TestGroundingRefusals:
    def test_a_measure_the_table_does_not_offer_is_refused(self):
        """"margin" is not a column here. Inventing one is the failure mode that sent a
        non-existent column name to the engine and derailed a conversation."""
        assert detect_measure_addition(frame(), "also include margin") is None

    def test_an_already_measured_column_adds_nothing(self):
        assert detect_measure_addition(frame(), "also include revenue") is None

    def test_a_frame_with_no_measure_vocabulary_refuses(self):
        """No evidence to ground against — carry on unchanged rather than guess."""
        assert detect_measure_addition(frame(available_measures=[]), "also include profit") is None

    def test_no_frame_refuses(self):
        assert detect_measure_addition(None, "also include profit") is None
        assert detect_measure_addition({}, "also include profit") is None


class TestNotAnAddition:
    """Messages that name no available measure add nothing, whatever else they are."""

    @pytest.mark.parametrize("message", [
        "only the ones in Pune",      # a filter
        "make it top 10",             # a shape change
        "go back",                    # navigation
        "how many assets are there",  # a new question
        "",
    ])
    def test_these_add_no_measure(self, message):
        assert detect_measure_addition(frame(), message) is None


class TestAddVersusReplaceLivesInTheCaller:
    """The add-vs-replace line is drawn by delta_type, NOT by this function.

    An earlier version required the message to match an English addition phrase
    ("also include|add|along with|as well|too") before it would look at all. That was
    redundant and fragile at once: `refine` is DEFINED in the prompt as addition ("ADDS
    a filter/grouping, keeps everything in the frame"), and "what about profit" is
    classified `replace`, so it never reaches this function — while the word list needed
    a new phrasing forever ("as well" was missed on the first pass, and only caught
    because a test happened to include it).

    Detection is the model's job; GROUNDING is this function's. So the function
    deliberately does not second-guess the label.
    """

    def test_the_function_itself_does_not_judge_intent(self):
        """It grounds the named measure and leaves add-vs-replace to the caller."""
        assert detect_measure_addition(frame(), "what about profit") == "profit"

    def test_the_caller_gates_on_refine(self):
        """context_resolve_node calls this only for delta_type == "refine" — and
        "what about profit" is classified `replace`, so it never arrives.

        Pinned as source, because the guard is one line in another module and a future
        edit that widens it back to `ambiguous` (a failed classify) would silently
        reintroduce measure writes on turns the model could not read at all."""
        import inspect

        from chatbot import nodes
        src = inspect.getsource(nodes.context_resolve_node)
        assert 'if delta_type == "refine":' in src
        assert "detect_measure_addition" in src


class TestAddMeasureIsSafe:
    def test_a_duplicate_returns_the_same_object(self):
        """Identity is the caller's "did anything change" signal — the same rule
        apply_context_delta follows."""
        f = frame(measures=["revenue", "profit"])
        assert add_measure(f, "profit") is f

    def test_case_differences_are_not_duplicates_added_twice(self):
        f = frame(measures=["Revenue"])
        assert add_measure(f, "revenue") is f

    def test_an_empty_column_is_a_no_op(self):
        f = frame()
        assert add_measure(f, "") is f

    def test_the_original_frame_is_never_mutated(self):
        f = frame()
        add_measure(f, "profit")
        assert f["measures"] == ["revenue"]
