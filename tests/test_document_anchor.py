"""A document follow-up must stay in the document the conversation is already in.

This corpus is 92% one file — the employee handbook holds 165 of 179 chunks — so an
unanchored follow-up drifts there on sheer volume. Measured before the fix: after an
answer from maintenance_policy.docx, "and what about response times" came back from the
HANDBOOK, about absence time tracking, and the frame silently followed it there.

These tests pin the four decisions that make the anchor safe rather than sticky:
a new topic is never trapped, contentless navigation triggers are never glued to a
document name, a self-anchoring question is not annotated twice, and the SQL path is
untouched.
"""
import pytest

from chatbot.nodes import (_is_bare_referential, _RESET_RE, classify_node,
                           recall_node)
from chatbot.memory.frame import (harvest_frame, is_document_frame, newly_added_filter,
                                  render_frame_as_query, update_drill_level,
                                  stabilise_document_entity)


def doc_frame(entity="maintenance_policy.docx"):
    return {"entity": entity, "entity_display": entity, "route": "rag", "filters": []}


def sql_frame():
    return {"entity": "assets_asset", "entity_display": "Assets", "route": "deterministic",
            "filters": [{"field": "City", "operator": "equals", "value": "Pune"}]}


class TestIsDocumentFrame:
    @pytest.mark.parametrize("route", ["rag", "doc", "document", "RAG", "Document"])
    def test_retrieval_routes_are_document_frames(self, route):
        assert is_document_frame({"entity": "x", "route": route})

    @pytest.mark.parametrize("route", ["deterministic", "sql", "", None])
    def test_other_routes_are_not(self, route):
        assert not is_document_frame({"entity": "x", "route": route})

    def test_empty_frame_is_not(self):
        assert not is_document_frame(None) and not is_document_frame({})


class TestAnchoring:
    @pytest.mark.parametrize("delta", ["refine", "replace", "drill_down", "compare"])
    def test_continuations_name_the_document(self, delta):
        out = render_frame_as_query(doc_frame(), "and what about response times", delta)
        assert out == "and what about response times (in maintenance_policy.docx)"

    @pytest.mark.parametrize("delta", ["new_topic", "ambiguous"])
    def test_a_new_question_is_never_trapped(self, delta):
        """The whole risk of anchoring: asking something else and being answered from
        the previous document anyway. classify_node calls a self-contained question
        "answer", which arrives here as referential=False."""
        assert render_frame_as_query(doc_frame(), "what is the dress code", delta) == \
            "what is the dress code"


class TestReferentialSignal:
    """On a document frame the anchor is driven by classify_node's action label, not by
    delta_type.

    Measured over 6 live document turns: delta_type came back "new_topic" or
    "ambiguous" on every single one — continuations and new questions alike — because
    the delta block reasons about filters and slots and a document frame has neither.
    The first version of this fix keyed on delta_type and therefore did nothing at all
    live, while passing its unit tests.
    """

    @pytest.mark.parametrize("delta", ["new_topic", "ambiguous"])
    def test_a_follow_up_anchors_despite_an_uninformative_delta(self, delta):
        out = render_frame_as_query(doc_frame(), "and what about response times", delta,
                                    referential=True)
        assert out == "and what about response times (in maintenance_policy.docx)"

    def test_referential_does_not_override_a_contentless_trigger(self):
        assert render_frame_as_query(doc_frame(), "go back", "drill_up",
                                     referential=True) == "go back"

    def test_referential_does_not_override_shape(self):
        assert render_frame_as_query(doc_frame(), "make it top 10", "new_topic",
                                     shape_delta=True, referential=True) == \
            "make it top 10"

    def test_referential_still_respects_a_self_anchoring_question(self):
        out = render_frame_as_query(doc_frame(), "what does the maintenance policy say "
                                    "about response times", "new_topic", referential=True)
        assert "(in" not in out

    def test_referential_changes_nothing_on_a_sql_frame(self):
        """delta_type remains authoritative wherever it is actually informative."""
        assert render_frame_as_query(sql_frame(), "what is the dress code", "new_topic",
                                     referential=True) == "what is the dress code"
        assert render_frame_as_query(sql_frame(), "what about that", "ambiguous",
                                     referential=True) == "what about that"

    @pytest.mark.parametrize("delta", ["drill_up", "remove"])
    def test_contentless_triggers_are_left_alone(self, delta):
        """"go back" names no subject, so "go back (in maintenance_policy.docx)" is not
        a question — and a document frame has no drill stack to pop anyway."""
        assert render_frame_as_query(doc_frame(), "go back", delta) == "go back"

    def test_shape_ops_are_left_alone(self):
        assert render_frame_as_query(doc_frame(), "make it top 10", "refine",
                                     shape_delta=True) == "make it top 10"

    def test_frame_without_an_entity_cannot_anchor(self):
        assert render_frame_as_query({"entity": "", "route": "rag"}, "and response times",
                                     "refine") == "and response times"


class TestAlreadyNamed:
    def test_prose_mention_of_the_file_counts(self):
        """The frame holds a FILE NAME, the user writes prose — so the check is on
        content words, not substrings, or it would never fire."""
        out = render_frame_as_query(doc_frame(), "in the maintenance policy, what about "
                                    "response times", "refine")
        assert out == "in the maintenance policy, what about response times"

    def test_extension_does_not_have_to_be_typed(self):
        out = render_frame_as_query(doc_frame(), "what does the maintenance policy say",
                                    "refine")
        assert "(in" not in out

    def test_a_partial_mention_still_anchors(self):
        """"policy" alone is not this document — several documents are policies."""
        out = render_frame_as_query(doc_frame(), "what does the policy say", "refine")
        assert out.endswith("(in maintenance_policy.docx)")

    def test_dated_handbook_name_is_recognised(self):
        frame = doc_frame("Samta-Employee Handbook April_2026.pdf")
        out = render_frame_as_query(frame, "in the samta employee handbook april 2026, "
                                    "is it paid", "refine")
        assert "(in" not in out

    def test_dated_handbook_name_is_appended_when_absent(self):
        """Measured 2026-09-22: appending the dated name is safe on the retrieval path —
        no date filter appears, and the answer improved from "the context does not
        specify" to a cited "Yes, it is paid"."""
        frame = doc_frame("Samta-Employee Handbook April_2026.pdf")
        out = render_frame_as_query(frame, "is it paid", "refine")
        assert out == "is it paid (in Samta-Employee Handbook April_2026.pdf)"


class TestSqlPathUnchanged:
    def test_sql_follow_up_still_uses_for_not_in(self):
        assert render_frame_as_query(sql_frame(), "only the active ones", "refine") == \
            "only the active ones (for Assets (assets_asset), Pune)"

    def test_sql_drill_up_still_returns_the_restated_frame(self):
        assert render_frame_as_query(sql_frame(), "go back", "drill_up") == \
            "Assets (assets_asset), Pune"


class TestDocumentEntityStability:
    """A follow-up that stayed in its document must not move the frame to another one.

    Measured 2026-09-22: "and what about response times" was answered citing
    maintenance_policy.docx, but the engine listed the employee handbook FIRST, and the
    frame takes datasets[0] — so the conversation silently relocated to the handbook
    and every later turn anchored to the wrong document.
    """

    def harvested(self, entity, datasets):
        return {"entity": entity, "entity_display": entity, "route": "rag",
                "datasets": datasets, "filters": []}

    def test_previous_document_is_kept_when_the_answer_used_it(self):
        out = stabilise_document_entity(
            doc_frame("maintenance policy"),
            self.harvested("Samta-Employee Handbook April 2026",
                           ["Samta-Employee Handbook April_2026.pdf", "maintenance_policy.docx"]),
            referential=True)
        assert out["entity"] == "maintenance policy"

    def test_it_moves_when_the_answer_did_not_use_the_old_document(self):
        """Evidence-only: the old document is never carried over an answer that did not
        read it."""
        out = stabilise_document_entity(
            doc_frame("maintenance policy"),
            self.harvested("Samta-Employee Handbook April 2026",
                           ["Samta-Employee Handbook April_2026.pdf"]),
            referential=True)
        assert out["entity"] == "Samta-Employee Handbook April 2026"

    def test_a_new_question_is_not_held_in_the_old_document(self):
        """The handbook is cited constantly — it holds 92% of the corpus — so without
        this gate a new question would be pinned by a coincidental citation."""
        out = stabilise_document_entity(
            doc_frame("maintenance policy"),
            self.harvested("Samta-Employee Handbook April 2026",
                           ["Samta-Employee Handbook April_2026.pdf", "maintenance_policy.docx"]),
            referential=False)
        assert out["entity"] == "Samta-Employee Handbook April 2026"

    def test_sql_answers_are_untouched(self):
        sql = {"entity": "assets_asset", "route": "deterministic", "datasets": []}
        assert stabilise_document_entity(sql_frame(), sql, referential=True) is sql

    def test_a_sql_frame_followed_by_a_document_answer_is_untouched(self):
        h = self.harvested("maintenance policy", ["maintenance_policy.docx"])
        assert stabilise_document_entity(sql_frame(), h, referential=True) is h

    def test_no_previous_frame_is_untouched(self):
        h = self.harvested("maintenance policy", ["maintenance_policy.docx"])
        assert stabilise_document_entity(None, h, referential=True) is h

    def test_same_document_needs_no_correction(self):
        h = self.harvested("maintenance policy", ["maintenance_policy.docx"])
        assert stabilise_document_entity(doc_frame("maintenance policy"), h,
                                         referential=True) is h


class TestBareReferentialBackstop:
    """A referential message with no frame must never reach the engine.

    This guard used to rest on the model: "more details" was classified smalltalk, so
    it never arrived at the backstop. When the supervisor's HARD RULE was corrected on
    2026-09-22 to stop calling information requests smalltalk, "more details" came
    through as a followup with no frame — ungrounded text on its way to the engine,
    which is the exact failure this backstop exists to stop (observed once as a
    confident answer built from an unrelated table, complete with a real name and
    email).
    """

    @pytest.mark.parametrize("message", [
        "what about that", "the other one", "more details", "more info",
        "tell me more", "that one", "same as before", "it",
    ])
    def test_ungrounded_referential_is_caught(self, message):
        assert _is_bare_referential(message)

    @pytest.mark.parametrize("message", [
        "how many assets are there", "count of assets by city", "list 5 assets",
        "what is the dress code", "how long is the probation period",
        "this maintenance policy for asset 21", "show me more rows",
    ])
    def test_a_real_question_is_not(self, message):
        """The cost of a false positive here is a silently dropped question, so the
        self-contained cases matter more than the referential ones."""
        assert not _is_bare_referential(message)

    def test_a_data_hint_always_wins(self):
        """_DATA_QUESTION_HINTS is checked first: a message naming data is never bare,
        however short or referential it looks."""
        assert not _is_bare_referential("count of those")
        assert not _is_bare_referential("how many more")


class TestHybridDocumentFrames:
    """The hybrid head answers from documents but calls itself "hybrid".

    Measured 2026-09-22: the employee handbook is answered by that head, "hybrid" is
    not a retrieval route name, and so `is_document_frame` said False for it — every
    document behaviour (anchoring, the shape and drill-up refusals) was silently
    skipped for an entire live conversation. The route name says which head ran;
    `entity_is_document` says what the answer was built from, which is what this layer
    actually needs.
    """

    def test_a_hybrid_answer_from_documents_is_a_document_frame(self):
        assert is_document_frame({"entity": "Samta-Employee Handbook April 2026",
                                  "route": "hybrid", "entity_is_document": True})

    def test_a_hybrid_answer_from_a_table_is_not(self):
        assert not is_document_frame({"entity": "assets_asset", "route": "hybrid",
                                      "entity_is_document": False})

    def test_the_recorded_fact_beats_the_route_name(self):
        """Either direction — the fact is the authority once present."""
        assert is_document_frame({"entity": "x", "route": "deterministic",
                                  "entity_is_document": True})
        assert not is_document_frame({"entity": "x", "route": "rag",
                                      "entity_is_document": False})

    @pytest.mark.parametrize("route,expected", [("rag", True), ("document", True),
                                                ("deterministic", False),
                                                ("hybrid", False)])
    def test_frames_written_before_the_fact_existed_fall_back_to_the_route(self, route,
                                                                           expected):
        """Redis holds frames for 7 days, so older ones carry no such key."""
        assert is_document_frame({"entity": "x", "route": route}) is expected

    def test_a_hybrid_document_follow_up_anchors(self):
        frame = {"entity": "Samta-Employee Handbook April 2026", "route": "hybrid",
                 "entity_is_document": True, "filters": []}
        assert render_frame_as_query(frame, "is it carried forward", "new_topic",
                                     referential=True) == \
            "is it carried forward (in Samta-Employee Handbook April 2026)"


class TestHarvestRecordsDocumentness:
    def base(self, **kw):
        out = {"status": "answered", "explain": {"data_used": {"datasets": []}},
               "rows": []}
        out.update(kw)
        return out

    def test_a_document_answer_records_true(self):
        h = harvest_frame(self.base(
            explain={"data_used": {"datasets": ["maintenance_policy.docx"]}}))
        assert h["entity_is_document"] is True
        assert h["datasets"] == ["maintenance_policy.docx"]

    def test_a_table_answer_records_false(self):
        h = harvest_frame(self.base(
            table="assets_asset",
            explain={"data_used": {"datasets": ["Assets"]}}))
        assert h["entity_is_document"] is False

    def test_neither_table_nor_datasets_records_false(self):
        assert harvest_frame(self.base())["entity_is_document"] is False


class TestPresentationOnADocumentAnswer:
    """"as a pie chart" asked of a document answer must be settled from memory.

    The represent fast path requires the previous result to carry rows, and a retrieval
    answer never does — so this fell through to the model and then the engine, and
    measured 2026-09-22 it came back with the generic "I'm here for questions about
    your data", which does not tell the user why no chart appeared. Same family as the
    shape and drill_up refusals already handled on this path.
    """

    def state(self, frame, message="as a pie chart", rows=None):
        return {"message": message, "history": [{"role": "user", "content": "q"}],
                "frame": frame, "last_result": {"rows": rows} if rows else {},
                "session_id": "s", "tenant": "default", "source_id": 3,
                "drill_stack": [], "episodic": []}

    def doc(self):
        return {"entity": "Samta-Employee Handbook April 2026",
                "entity_display": "Samta-Employee Handbook April 2026",
                "route": "hybrid", "entity_is_document": True, "filters": []}

    def test_it_never_reaches_the_engine(self):
        out = classify_node(self.state(self.doc()), None)
        assert out["action"] == "recall"
        assert out["recall_kind"] == "presentation_not_applicable"

    def test_the_reply_names_the_document_and_says_why(self):
        state = self.state(self.doc())
        state["recall_kind"] = "presentation_not_applicable"
        out = recall_node(state)
        reply = out["reply_text"]
        assert "Samta-Employee Handbook April 2026" in reply
        assert "nothing to chart" in reply.lower()

    def test_a_sql_answer_with_rows_still_charts(self):
        """The fast path above owns that case and must keep it."""
        sql = {"entity": "assets_asset", "route": "deterministic",
               "entity_is_document": False, "filters": []}
        out = classify_node(self.state(sql, rows=[{"a": 1}]), None)
        assert out.get("recall_kind") != "presentation_not_applicable"

    def test_a_document_answer_that_somehow_has_rows_is_left_alone(self):
        out = classify_node(self.state(self.doc(), rows=[{"a": 1}]), None)
        assert out.get("recall_kind") != "presentation_not_applicable"


class TestReplacementDoesNotDeepenTheDrillPath:
    """Swapping a filter's value is not a further narrowing.

    Measured 2026-09-22 on a live chain: "how many assets" -> "only the ones in Pune" ->
    "what about Mumbai" left a drill stack of depth 2 for what the user experienced as
    ONE narrowing, because the stack keyed on (field, operator, value) and Mumbai looked
    like a brand-new filter. "go back" then returned to Pune — the value the user had
    just replaced — instead of to all assets.
    """

    PREV = {"filters": [{"field": "Location", "operator": "equals", "value": "pune"}]}

    def test_the_same_filter_again_adds_nothing(self):
        assert newly_added_filter(self.PREV, {"filters": list(self.PREV["filters"])}) is None

    def test_a_replacement_adds_no_level(self):
        harvested = {"filters": [{"field": "Location", "operator": "equals",
                                  "value": "mumbai"}]}
        assert newly_added_filter(self.PREV, harvested) is None

    def test_a_genuinely_new_field_still_adds_one(self):
        """The drill stack must still record real narrowings, or "go back" has nothing
        to pop — the bug this detection was written for in the first place."""
        harvested = {"filters": [{"field": "Location", "operator": "equals", "value": "pune"},
                                 {"field": "Status", "operator": "equals", "value": "ACTIVE"}]}
        got = newly_added_filter(self.PREV, harvested)
        assert got is not None and got["field"] == "Status"

    def test_field_names_are_compared_loosely(self):
        """"Location" and "location_name" are the same concept — a replacement, not a
        new level."""
        harvested = {"filters": [{"field": "location name", "operator": "equals",
                                  "value": "mumbai"}]}
        assert newly_added_filter(self.PREV, harvested) is None

    def test_the_existing_level_is_re_pointed(self):
        stack = [{"dimension": "Location", "value": "pune"}]
        assert update_drill_level(stack, "Location", "mumbai") == \
            [{"dimension": "Location", "value": "mumbai"}]

    def test_other_levels_are_untouched(self):
        stack = [{"dimension": "Location", "value": "pune"},
                 {"dimension": "Status", "value": "ACTIVE"}]
        out = update_drill_level(stack, "Location", "mumbai")
        assert out[1] == {"dimension": "Status", "value": "ACTIVE"}
        assert len(out) == 2

    def test_an_unknown_field_changes_nothing(self):
        stack = [{"dimension": "Location", "value": "pune"}]
        assert update_drill_level(stack, "Colour", "red") == stack
        assert update_drill_level(stack, None, "red") == stack


class TestGoBackToTheRoot:
    """Popping the LAST drill level must return the user to their original question.

    Measured 2026-09-22 across four live scenarios: the frame's bare restatement
    ("Assets (assets_asset)") was sent instead, and the engine refused it every time —
    "I couldn't apply the condition you asked for" — so drilling once and saying "go
    back" always failed. Alternatives were measured, not chosen: "all <entity>" and
    "show <entity>" were also refused, and "list <entity>" ANSWERED but ran a different
    question entirely (DISTINCT project_name), which is worse than a refusal because it
    looks like success.
    """

    def root(self, **kw):
        f = {"entity": "assets_asset", "entity_display": "Assets",
             "route": "deterministic", "filters": [],
             "base_query": "how many assets are there"}
        f.update(kw)
        return f

    def test_the_users_own_question_is_replayed(self):
        assert render_frame_as_query(self.root(), "go back", "drill_up") == \
            "how many assets are there"

    def test_a_remaining_level_still_restates_the_frame(self):
        """Only the ROOT case changes — a pop that leaves a level behind keeps
        describing the narrowed frame, as before."""
        frame = self.root(filters=[{"field": "Location", "operator": "equals",
                                    "value": "pune"}])
        assert render_frame_as_query(frame, "go back", "drill_up") == \
            "Assets (assets_asset), pune"

    @pytest.mark.parametrize("base", ["", None, "   "])
    def test_without_a_recorded_base_the_old_behaviour_stands(self, base):
        """Frames written before this field existed (Redis holds them for 7 days) must
        not crash or invent a question."""
        assert render_frame_as_query(self.root(base_query=base), "go back",
                                     "drill_up") == "Assets (assets_asset)"

    def test_the_base_is_not_used_for_other_deltas(self):
        """A refinement is not a return to the root."""
        out = render_frame_as_query(self.root(), "only the active ones", "refine")
        assert out == "only the active ones (for Assets (assets_asset))"

    def test_remove_is_unaffected(self):
        """"remove" shares the contentless-trigger branch but is not a stack pop."""
        frame = self.root(filters=[{"field": "Location", "operator": "equals",
                                    "value": "pune"}])
        assert render_frame_as_query(frame, "remove the city filter", "remove") == \
            "Assets (assets_asset), pune"


class TestResetIsComposedNotEnumerated:
    """A reset request is a DISCARD VERB plus, optionally, the conversation's own state.

    The previous pattern listed whole phrases, and 7 of the suite's 12 fell through it —
    "clear everything", "wipe memory", "forget it all", "forget what i said", "start
    again", "clear", "new conversation" — so the user believed the context was gone
    while the next answer silently carried it. A longer list of phrases only moves that
    boundary, so the rule composes two small closed sets instead.

    What keeps it safe is the anchored whole-message match plus the object set: every
    object names the conversation itself, never data. No data question is only a discard
    verb and a conversation noun.
    """

    @pytest.mark.parametrize("message", [
        "start over", "start again", "reset", "clear", "clear everything",
        "forget everything", "forget it all", "forget what i said",
        "let's start fresh", "new conversation", "clear the context", "wipe memory",
    ])
    def test_every_phrasing_the_suite_carries(self, message):
        assert _RESET_RE.match(message)

    @pytest.mark.parametrize("message", [
        "wipe the context", "clear this chat", "forget the history",
        "erase everything", "drop the context", "discard this conversation",
        "reset my session", "forget what i asked", "clear it", "begin afresh",
    ])
    def test_combinations_nobody_wrote_down(self, message):
        """The point of composing: these were never enumerated anywhere."""
        assert _RESET_RE.match(message)

    @pytest.mark.parametrize("message", [
        "clear the table", "clear all dues", "clear dues by city",
        "clear the top 5 by amount", "reset the filter to Pune",
        "drop the amount column", "forget about Pune and show Mumbai",
        "new topic for the report", "start over the lease analysis",
        "how many assets are there", "what did i say earlier", "show me what i asked",
        "wipe down report",
    ])
    def test_a_real_question_never_resets(self, message):
        """A false positive here silently destroys the user's context mid-analysis —
        the expensive direction, and why the match stays anchored to the whole message."""
        assert not _RESET_RE.match(message)
