"""Result reference memory — "the second one", "its price", "the 7th one" (out of range).

Before this, the only record of a result was `last_result`: the whole engine result in the
checkpoint, unbounded, with no notion of which column identifies a row. "the second one"
had nothing to resolve against and ran as a fresh question.

Pure except the engine-side `result_key` tests, which need sqlglot (skipped without it).
Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_result_reference.py -q`
"""
import os
import sys
import uuid

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chatbot.memory import reference as R  # noqa: E402

# ── the shapes the engine hands over ──────────────────────────────────────────────────
_ROWS = {"status": "answered", "table": "assets_salelisting",
         "cols": ["listing_id", "expected_price", "project_name"],
         "rows": [[101, 4000, "A"], [102, 5000, "B"], [103, 9000, "C"]],
         "result_key": {"kind": "rows", "table": "assets_salelisting",
                        "column": "id", "result_column": "listing_id"}}
_GROUPS = {"status": "answered", "table": "assets_asset",
           "cols": ["facing", "corner_property", "assets_asset_count"],
           "rows": [["EAST", True, 40], ["NORTH", False, 12]],
           "result_key": {"kind": "groups", "table": "assets_asset",
                          "columns": [{"column": "facing", "result_column": "facing"},
                                      {"column": "corner_property",
                                       "result_column": "corner_property"}]}}
_FRAME_SL = {"entity": "assets_salelisting", "source_id": 2}
_FRAME_AS = {"entity": "assets_asset", "source_id": 2}


# ── ordinal grammar ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("msg,pos", [
    ("show the second one", 2), ("what about the 3rd one", 3), ("the first one", 1),
    ("the last one", -1), ("details of number 2", 2), ("open #4", 4),
    ("the TENTH one", 10), ("and the 12th?", 12),
])
def test_positions(msg, pos):
    assert R.parse_ordinal(msg) == pos


@pytest.mark.parametrize("msg", [
    "show the first 10 payments",       # a count follows: a ranking, not a position
    "last 5 ledger entries",
    "how many properties are there",
    "properties created in 2024",       # a bare year is not an ordinal
])
def test_not_positions(msg):
    assert R.parse_ordinal(msg) is None


# ── building ──────────────────────────────────────────────────────────────────────────
def test_rows_keep_their_ids_in_display_order():
    ref = R.build_reference(_ROWS, 2)
    assert ref["kind"] == "rows" and ref["items"] == ["101", "102", "103"]
    assert ref["key_column"] == "id" and ref["entity"] == "assets_salelisting"
    assert ref["complete"] is True and ref["source_id"] == 2


def test_groups_keep_their_group_values():
    ref = R.build_reference(_GROUPS, 2)
    assert ref["items"] == [{"facing": "EAST", "corner_property": "True"},
                            {"facing": "NORTH", "corner_property": "False"}]


def test_values_are_selectors_only_and_never_reach_the_engine():
    """Changed deliberately 2026-09-26 (was test_values_are_never_stored): naming and
    ranking references ("details of One & Only House", "the cheapest one") need the shown
    labels/values to decide WHICH row. They are used for that alone — what goes to the
    engine is still only the row's key, and the row is re-queried under the current
    turn's authorisation."""
    ref = R.build_reference(_ROWS, 2)
    assert ref["values"] == {"expected_price": [4000.0, 5000.0, 9000.0]}
    kind, filters, _ = R.resolve_reference(ref, "which one is the cheapest",
                                           frame=_FRAME_SL, referential=True)
    assert kind == "filters"
    assert [(f["column"], f["value"]) for f in filters] == [("id", "101")]
    assert "values" not in R.build_reference(_GROUPS, 2)     # groups: identity only


def test_bounded():
    big = dict(_ROWS, rows=[[i, 1, "x"] for i in range(500)])
    ref = R.build_reference(big, 2)
    assert len(ref["items"]) == R._MAX_ITEMS and ref["complete"] is False


@pytest.mark.parametrize("er", [
    {**_ROWS, "result_key": {}},                                  # engine stated no key
    {**_ROWS, "result_key": {"kind": "rows", "column": "id", "result_column": "missing"}},
    {**_ROWS, "rows": []},
    {**_ROWS, "rows": [[None, 1, "x"]]},                          # a row with no identity
])
def test_not_referable_is_none_not_a_guess(er):
    assert R.build_reference(er, 2) is None


def test_positional_and_keyed_rows_agree():
    keyed = dict(_ROWS, rows=[dict(zip(_ROWS["cols"], r)) for r in _ROWS["rows"]])
    assert R.build_reference(keyed, 2)["items"] == R.build_reference(_ROWS, 2)["items"]


# ── resolving ─────────────────────────────────────────────────────────────────────────
def test_second_one_is_the_second_row():
    kind, filters, terms = R.resolve_reference(R.build_reference(_ROWS, 2),
                                               "show the second one",
                                               frame=_FRAME_SL, referential=True)
    assert kind == "filters" and terms == ["second", "one"]
    assert filters == [{"field": "id", "column": "id", "operator": "equals",
                        "value": "102", "source": "result_reference"}]


def test_last_one():
    _, f, _t = R.resolve_reference(R.build_reference(_ROWS, 2), "the last one",
                                   frame=_FRAME_SL, referential=True)
    assert f[0]["value"] == "103"


def test_a_group_row_becomes_its_group_filters():
    _, f, _t = R.resolve_reference(R.build_reference(_GROUPS, 2), "the first one",
                                   frame=_FRAME_AS, referential=True)
    assert {(x["column"], x["value"]) for x in f} == {("facing", "EAST"),
                                                      ("corner_property", "True")}


def test_out_of_range_is_refused_honestly():
    kind, why = R.resolve_reference(R.build_reference(_ROWS, 2), "show the 7th one",
                                    frame=_FRAME_SL, referential=True)
    assert kind == "refuse" and "3 rows" in why and "row 7" in why


def test_past_the_bound_on_an_incomplete_result_says_so():
    big = dict(_ROWS, rows=[[i, 1, "x"] for i in range(500)])
    kind, why = R.resolve_reference(R.build_reference(big, 2), "number 120",
                                    frame=_FRAME_SL, referential=True)
    assert kind == "refuse" and "first 50" in why


def test_a_pronoun_about_a_one_row_answer_is_that_row():
    one = dict(_ROWS, rows=[[101, 4000, "A"]])
    _, f, terms = R.resolve_reference(R.build_reference(one, 2), "what is its location?",
                                      frame=_FRAME_SL, referential=True)
    assert f[0]["value"] == "101" and terms == []      # "its" is grammar, not a pointer


def test_a_pronoun_about_several_rows_is_not_guessed():
    assert R.resolve_reference(R.build_reference(_ROWS, 2), "what is its location?",
                               frame=_FRAME_SL, referential=True) is None


def test_not_referential_never_touches_the_reference():
    assert R.resolve_reference(R.build_reference(dict(_ROWS, rows=[[1, 1, "a"]]), 2),
                               "what is its location?", frame=_FRAME_SL,
                               referential=False) is None


def test_a_stale_reference_is_ignored():
    """The frame moved to another entity or source: the reference no longer applies."""
    ref = R.build_reference(_ROWS, 2)
    assert R.resolve_reference(ref, "the second one", frame=_FRAME_AS, referential=True) is None
    assert R.resolve_reference(ref, "the second one", frame={**_FRAME_SL, "source_id": 3},
                               referential=True) is None
    assert R.resolve_reference(None, "the second one", frame=_FRAME_SL,
                               referential=True) is None


def test_a_null_group_value_is_refused_not_matched():
    g = dict(_GROUPS, rows=[[None, True, 3]])
    kind, _ = R.resolve_reference(R.build_reference(g, 2), "the first one",
                                  frame=_FRAME_AS, referential=True)
    assert kind == "refuse"


# ── pointer grammar ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("msg,expected", [
    ("show the second one", (2, ["second", "one"])),
    ("then the 2nd one", (2, ["2nd", "one"])),
    ("the 3rd row", (3, ["3rd", "row"])),
    ("number 2", (2, ["2", "number"])),
    ("#4", (4, ["4"])),
    ("and the last?", (-1, ["last"])),
])
def test_pointers(msg, expected):
    assert R.result_pointer(msg) == expected


@pytest.mark.parametrize("msg", [
    "the first transaction of 2024",     # names its own subject
    "the second listing",                # may name its own subject — not a bare pointer
    "show the first 10 payments",        # a ranking
    "how many properties are there",
])
def test_not_pointers(msg):
    assert R.result_pointer(msg) is None


# ── the contract: pointer words travel, and only real ones ───────────────────────────
def test_resolved_terms_travel_and_are_held_to_the_message():
    from chatbot.memory.context import ConversationContext
    p = ConversationContext.from_frame(_FRAME_SL, "show the second one", operation="refine",
                                       resolved_terms=["second", "one", "invented"]).to_payload()
    assert p["resolved_terms"] == ["second", "one"]


def test_no_resolved_terms_is_no_new_key():
    from chatbot.memory.context import ConversationContext
    assert "resolved_terms" not in ConversationContext.from_frame(_FRAME_SL, "m").to_payload()


def test_inference_boundary_holds_resolved_terms_to_the_message():
    from inference.routes.hybrid import _validated_conversation_context as V
    out = V({"conversation_context": {"user_message": "show the second one",
                                      "resolved_terms": ["second", "one", "price",
                                                         "a b", "x" * 40, 3]}})
    assert out["resolved_terms"] == ["second", "one"]


# ── refusals never reach the engine ───────────────────────────────────────────────────
def test_a_pointer_with_no_frame_is_refused_not_passed_through():
    from chatbot import nodes as N
    out = N.context_resolve_node({"message": "show the 9th one", "history": [{"role": "user",
                                  "content": "x"}], "frame": {}, "action": "answer"}, {})
    assert out["engine_result"]["route"] == "reference"
    assert "no earlier list" in out["engine_result"]["refuse_reason"]


def test_a_first_turn_pointer_is_routed_to_the_refusal():
    from chatbot.graph import _route_after_classify
    assert _route_after_classify({"action": "answer", "message": "show the 2nd one",
                                  "history": []}) == "context_resolve"
    assert _route_after_classify({"action": "answer", "message": "how many assets",
                                  "history": []}) == "call_engine"


def test_a_pointer_at_a_stale_reference_is_refused():
    """Frame moved on to another entity: its rows are not what "the 2nd one" means."""
    from chatbot import nodes as N
    out = N.context_resolve_node({
        "message": "the 2nd one", "history": [{"role": "user", "content": "x"}],
        "frame": {"entity": "assets_asset", "source_id": 2, "filters": []},
        "result_reference": R.build_reference(_ROWS, 2),     # rows of assets_salelisting
        "action": "answer", "delta_type": "new_topic"}, {})
    assert out["engine_result"]["route"] == "reference"


def test_a_pointer_resolves_even_when_the_classifier_said_new_topic():
    from chatbot import nodes as N
    out = N.context_resolve_node({
        "message": "then the 2nd one", "history": [{"role": "user", "content": "x"}],
        "frame": {"entity": "assets_salelisting", "source_id": 2, "filters": [],
                  "route": "deterministic"},
        "result_reference": R.build_reference(_ROWS, 2),
        "action": "answer", "delta_type": "new_topic"}, {})
    ctx = out["conversation_context"]
    assert out["delta_type"] == "refine"
    assert {"column": "id", "operator": "equals", "value": "102"} in ctx["filters"]
    assert ctx["resolved_terms"] == ["2nd", "one"]
    assert ctx["user_message"] == "then the 2nd one"          # the user's words, unchanged


# ── through the real nodes ────────────────────────────────────────────────────────────
class TestNodes:
    def setup_method(self):
        from chatbot.memory.store import MemoryStore
        self.MS = MemoryStore
        self._orig = {n: getattr(MemoryStore, n) for n in
                      ("write_frame", "write_stack", "push_episodic_turn",
                       "write_comparison", "write_reference")}
        self.refs = []
        for n in self._orig:
            setattr(MemoryStore, n, staticmethod(lambda *a, **k: None))
        MemoryStore.write_reference = staticmethod(
            lambda t, s, ref, source_id=None: self.refs.append(ref))

    def teardown_method(self):
        for n, fn in self._orig.items():
            setattr(self.MS, n, fn)

    def test_an_answered_turn_writes_its_reference(self):
        from chatbot import nodes as N
        er = dict(_ROWS, sql="SELECT 1", analytics={"orderings": [], "limit": None,
                                                    "query_measures": []})
        er["explain"] = dict(data_used={"datasets": ["Sale Listings"]},
                             filters={"applied": []}, operations=[],
                             understanding={"summary": "s"})
        out = N.memory_write_node({"status": "answered", "tenant": "t",
                                   "session_id": str(uuid.uuid4()), "message": "5 cheapest",
                                   "frame": {}, "drill_stack": [], "delta_type": "new_topic",
                                   "source_id": 2, "engine_result": er})
        assert self.refs and self.refs[-1]["items"] == ["101", "102", "103"]
        assert out["result_reference"]["items"] == ["101", "102", "103"]

    def test_an_unreferable_answer_clears_the_old_reference(self):
        from chatbot import nodes as N
        er = {"status": "answered", "table": "assets_asset", "cols": ["n"], "rows": [[5]],
              "sql": "SELECT COUNT(*)", "analytics": {"orderings": [], "limit": None,
                                                       "query_measures": []},
              "explain": {"data_used": {"datasets": ["Assets"]}, "filters": {"applied": []},
                          "operations": [], "understanding": {"summary": "s"}}}
        N.memory_write_node({"status": "answered", "tenant": "t",
                             "session_id": str(uuid.uuid4()), "message": "how many",
                             "frame": {}, "drill_stack": [], "delta_type": "new_topic",
                             "source_id": 2, "engine_result": er})
        assert self.refs[-1] is None


def test_the_refusal_route_skips_the_engine():
    from chatbot.graph import _route_after_resolve
    assert _route_after_resolve({"engine_result": {"route": "reference"}}) == "ask_clarification"
    assert _route_after_resolve({"engine_result": {}}) == "call_engine"


def test_state_declares_the_reference():
    """LangGraph silently drops an undeclared key — the trap that once disabled memory."""
    from chatbot.state import ChatState
    assert "result_reference" in ChatState.__annotations__


# ── engine side: which result column identifies a row ─────────────────────────────────
sqlglot = pytest.importorskip("sqlglot")


def _rk(sql, anchor, cols=("assets_salelisting.id", "assets_asset.id")):
    sys.path.insert(0, os.path.join(ROOT, "veda_core"))
    from veda.business_explain import result_key
    return result_key(sql, anchor, {"columns": {c: {} for c in cols}})


def test_engine_names_an_aliased_key():
    assert _rk('SELECT "t0"."id" AS "listing_id", "t0"."expected_price" FROM '
               '"assets_salelisting" AS "t0" ORDER BY 2 LIMIT 5', "assets_salelisting") == \
        {"kind": "rows", "table": "assets_salelisting", "column": "id",
         "result_column": "listing_id"}


def test_engine_names_group_columns():
    assert _rk('SELECT "facing", COUNT(*) AS n FROM "assets_asset" GROUP BY "facing"',
               "assets_asset")["kind"] == "groups"


@pytest.mark.parametrize("sql,anchor", [
    ('SELECT COUNT(*) FROM "assets_asset"', "assets_asset"),
    ('SELECT "price" FROM "assets_salelisting"', "assets_salelisting"),
    ('WITH c AS (SELECT id FROM assets_asset) SELECT id FROM c', "assets_asset"),
    ('SELECT "t1"."id" FROM "assets_salelisting" AS "t0" JOIN "assets_asset" AS "t1" '
     'ON "t1"."id" = "t0"."asset_id"', "assets_salelisting"),
    ('SELECT "facing" FROM "assets_asset" GROUP BY "facing", "city"', "assets_asset"),
])
def test_engine_says_nothing_when_rows_are_not_referable(sql, anchor):
    assert _rk(sql, anchor) == {}


# ── a row picked from an earlier answer stays referable ──────────────────────────────
_PINNED = {"status": "answered", "table": "accounts_generalledger",
           "cols": ["label", "amount"], "rows": [["Electricity", 512.0]],
           "result_key": {"kind": "rows", "table": "accounts_generalledger",
                          "column": "id", "pinned": "1092"}}
_FRAME_GL = {"entity": "accounts_generalledger", "source_id": 2}


def test_a_pinned_single_row_is_referable():
    ref = R.build_reference(_PINNED, 2)
    assert ref["items"] == ["1092"] and ref["key_column"] == "id" and ref["complete"]


def test_a_pinned_key_with_several_rows_is_not_referable():
    assert R.build_reference(dict(_PINNED, rows=[["a", 1], ["b", 2]]), 2) is None


def test_its_after_one_record_is_evidence_of_that_record():
    ref = R.build_reference(_PINNED, 2)
    assert R.points_at_the_one_row(ref, "what is its amount?", _FRAME_GL)
    kind, f, _t = R.resolve_reference(ref, "what is its amount?", frame=_FRAME_GL,
                                      referential=True)
    assert f[0]["value"] == "1092"


@pytest.mark.parametrize("msg", [
    "is it possible to see all vendors?",   # bare "it" is not a pointer
    "show that city",                        # a determiner, not a pronoun
    "how many assets are there",
])
def test_other_pronoun_shapes_are_not_evidence(msg):
    assert not R.points_at_the_one_row(R.build_reference(_PINNED, 2), msg, _FRAME_GL)


def test_its_is_not_evidence_when_several_rows_are_on_screen():
    assert not R.points_at_the_one_row(R.build_reference(_ROWS, 2), "what is its price?",
                                       _FRAME_SL)


def test_its_is_not_evidence_for_another_entity():
    assert not R.points_at_the_one_row(R.build_reference(_PINNED, 2), "what is its price?",
                                       _FRAME_SL)


def test_engine_states_a_pinned_key():
    sys.path.insert(0, os.path.join(ROOT, "veda_core"))
    from veda.business_explain import result_key
    rk = result_key('SELECT "label", "amount" FROM "accounts_generalledger" '
                    'WHERE LOWER(CAST("id" AS TEXT)) = %s ORDER BY "created_at" DESC',
                    "accounts_generalledger", {"columns": {"accounts_generalledger.id": {}}},
                    params=["1092"])
    assert rk == {"kind": "rows", "table": "accounts_generalledger", "column": "id",
                  "pinned": "1092"}


# ── the engine does not re-read pointer words ─────────────────────────────────────────
def test_ranking_does_not_read_a_resolved_pointer():
    """"the first one" resolved to a row must not also mean "earliest, LIMIT 1"."""
    sys.path.insert(0, os.path.join(ROOT, "veda_core"))
    from veda_core.context import set_conversation_context
    from query.ranking_parser import parse_ranking
    assert parse_ranking("the first one").top_n == 1          # unchanged without context
    tok = set_conversation_context({"user_message": "the first one",
                                    "resolved_terms": ["first", "one"]})
    try:
        spec = parse_ranking("the first one")
        assert spec.top_n is None and spec.ranked is False
        # ...and a real ranking word the user ALSO typed still counts
        assert parse_ranking("the first one, top 3").top_n == 3
    finally:
        set_conversation_context(None)



def test_the_row_key_never_rides_in_the_client_facing_explain():
    """explain is streamed to the client verbatim; row identities are memory plumbing."""
    sys.path.insert(0, os.path.join(ROOT, "veda_core"))
    from veda.business_explain import build_explain
    p = build_explain(sql='SELECT "id", "label" FROM "accounts_generalledger" LIMIT 5',
                      table="accounts_generalledger",
                      sm={"columns": {"accounts_generalledger.id": {}}})
    assert "result_key" not in p


def test_the_engine_finds_the_anchor_itself():
    sys.path.insert(0, os.path.join(ROOT, "veda_core"))
    from veda.business_explain import result_key
    assert result_key('SELECT "id" FROM "accounts_generalledger"', None,
                      {"columns": {"accounts_generalledger.id": {}}})["table"] == \
        "accounts_generalledger"


# ── a one-row result is NOT a pointer for every follow-up ────────────────────────────
@pytest.mark.parametrize("msg", ["go back", "only the FULL ones", "what about Pune?",
                                 "show it as a table"])
def test_a_one_row_result_does_not_pin_non_possessive_follow_ups(msg):
    """Measured 2026-09-25 (demo chains C5, C7): after a level that returned ONE row,
    "go back" was pinned to that row and its group values were added as filters."""
    one = R.build_reference(dict(_GROUPS, rows=[["EAST", True, 40]]), 2)
    assert R.resolve_reference(one, msg, frame=_FRAME_AS, referential=True) is None


def test_go_back_after_a_one_row_level_is_never_resolved_in_the_node():
    from chatbot import nodes as N
    one = R.build_reference(dict(_GROUPS, rows=[["EAST", True, 40]]), 2)
    out = N.context_resolve_node({
        "message": "go back", "history": [{"role": "user", "content": "x"}],
        "frame": {"entity": "assets_asset", "source_id": 2, "route": "deterministic",
                  "filters": [{"field": "Location", "column": "city_name", "value": "pune"}],
                  "base_query": "distribution by facing"},
        "drill_stack": [{"dimension": "Location", "column": "city_name", "value": "pune"}],
        "result_reference": one, "action": "followup", "delta_type": "drill_up"}, {})
    cols = [f.get("column") for f in (out.get("conversation_context") or {}).get("filters") or []]
    assert "corner_property" not in cols and "facing" not in cols


def test_a_pointer_at_rows_that_are_not_pickable_does_not_deny_the_list():
    """Edge E16: after a 2-row list with no row key, "the second one" was told
    "there's no earlier list" — the user could see the list. Say what is actually true."""
    from chatbot import nodes as N
    out = N.context_resolve_node({
        "message": "the second one", "history": [{"role": "user", "content": "x"}],
        "frame": {"entity": "accounts_paymenttransaction", "source_id": 2, "filters": []},
        "result_reference": None, "last_result": {"rows": [["CREDIT"], ["DEBIT"]]},
        "action": "followup", "delta_type": "ambiguous"}, {})
    why = out["engine_result"]["refuse_reason"]
    assert "no earlier list" not in why and "single row" in why


def test_a_distinct_list_is_referable_by_its_values():
    """Edge E16: a DISTINCT list (CREDIT, DEBIT) had no row key, so "the second one" could
    not point at DEBIT. Each DISTINCT row is identified by its values, like a group."""
    sys.path.insert(0, os.path.join(ROOT, "veda_core"))
    from veda.business_explain import result_key
    rk = result_key('SELECT DISTINCT "transaction_type" FROM "accounts_paymenttransaction" '
                    'WHERE NOT "transaction_type" IS NULL', None, {"columns": {}})
    assert rk == {"kind": "groups", "table": "accounts_paymenttransaction",
                  "columns": [{"column": "transaction_type",
                               "result_column": "transaction_type"}]}
    er = {"status": "answered", "table": "accounts_paymenttransaction",
          "cols": ["transaction_type"], "rows": [["CREDIT"], ["DEBIT"]], "result_key": rk}
    _, f, _t = R.resolve_reference(R.build_reference(er, 2), "the second one",
                                   frame={"entity": "accounts_paymenttransaction",
                                          "source_id": 2}, referential=True)
    assert f == [{"field": "transaction_type", "column": "transaction_type",
                  "operator": "equals", "value": "DEBIT", "source": "result_reference"}]
