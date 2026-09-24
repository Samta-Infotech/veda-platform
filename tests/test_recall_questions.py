"""Tests for questions about the CONVERSATION itself — chatbot/nodes.py::recall_node.

"What was my last query", "what SQL did you run", "which table did you use", "how many
rows did that return". Every one is already answered by the QueryFrame the memory layer
stores after each successful turn, so none needs the engine, SQL or a model call.
Routed normally they went to the engine, which tried to find a TABLE for "what sql did
you run" — ~30s, then a refusal, or an answer pulled from an unrelated table.

In an analytics product this class matters more than ordinary chit-chat: someone asking
"which table did that come from" is deciding whether to trust a number.

Run: ``pytest tests/test_recall_questions.py``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import nodes as N  # noqa: E402

_RECALL = {
    "what was my last query": "query", "what we have in my last query": "query",
    "what did we have in my last query": "query", "what did i ask": "query",
    "what did i ask before": "query", "what was the previous question": "query",
    "what sql did you run": "sql", "show me the sql": "sql",
    "what query did you run": "sql",
    "which table did you use": "table", "what table did you use": "table",
    "which tables did you use": "table",
    "what filters were applied": "filters", "which filters did you apply": "filters",
    "how many rows did that return": "rows", "how many results": "rows",
    "what did you do": "trail", "how did you get that": "trail",
    "where did that come from": "trail", "why did you choose that table": "trail",
    "why that table": "trail",
}

# Real data questions that read almost identically. The object noun is what separates
# them — "my last QUERY" is recall, "my last PAYMENT" is for the engine.
_MUST_REACH_ENGINE = [
    "what was my last payment", "what was my last transaction",
    "how many rows are in assets", "how many results per project",
    "which table has payment data", "show me the sql server logs",
    "what filters do tenants have", "what did you do to the rent",
    "where did the tenant come from", "why is revenue down",
    "show me the cheapest properties", "only the ones in Nagpur",
    "hi", "as a pie chart", "go back",
]

_FRAME = {
    "entity": "assets_salelisting", "entity_display": "Sale Listings",
    "understanding": "Find the 100 cheapest sale listings.",
    "filters": [{"field": "Status", "operator": "equals", "value": "For Sale"}],
    "measures": [], "order_by": [{"field": "expected_price", "desc": False}],
    "limit": 100, "last_row_count": 100, "source_id": 2,
    "last_sql": "SELECT id, expected_price FROM assets_salelisting "
                "ORDER BY expected_price ASC LIMIT 100",
}
_PROFILES = {"2": {"name": "homzhub", "source_type": "relational"}}


def _recall(message, kind, frame=None):
    return N.recall_node({"message": message, "frame": frame or _FRAME,
                          "recall_kind": kind, "source_profiles": _PROFILES})


# ---- detection ------------------------------------------------------------------

def test_recall_phrasings_are_recognized():
    for message, expected in _RECALL.items():
        assert N._recall_kind(message) == expected, message


def test_real_data_questions_are_not_mistaken_for_recall():
    for message in _MUST_REACH_ENGINE:
        assert N._recall_kind(message) is None, message


# ---- routing --------------------------------------------------------------------

def _state(message, frame=None, **over):
    state = {"message": message, "history": [{"role": "user", "content": "q"}],
             "frame": frame if frame is not None else _FRAME,
             "engine_result": {"status": "answered", "rows": [[1]], "cols": ["a"]}}
    state.update(over)
    return state


def test_classify_routes_recall_without_any_model_call():
    def _boom(*a, **k):
        raise AssertionError("a recall question reached the model")

    original = N.call_slm
    N.call_slm = _boom
    try:
        out = N.classify_node(_state("what sql did you run"), {})
    finally:
        N.call_slm = original
    assert out["action"] == "recall" and out["recall_kind"] == "sql"


def test_without_a_frame_the_answer_is_the_absence_not_a_fabrication():
    """Without a prior answered turn there is no "last query" to describe. The recall
    path still owns the turn — it answers "nothing has been run yet", which is true and
    instant — rather than falling through to the engine, which has no answer for it at
    all (measured: 62s, then "Could you clarify if 'did' is a column name")."""
    out = N.classify_node(_state("what sql did you run", frame={}), {})
    assert out["action"] == "recall" and out["recall_kind"] == "nothing"


def test_the_previous_result_is_CLEARED_rather_than_re_emitted():
    """recall_node answers from the FRAME alone and never reads engine_result — but
    apps/chat/services.py renders a markdown table and charts from whatever the result
    carries. Letting the previous turn's result survive made "what SQL did you run"
    re-emit that whole table and its charts alongside the one-line answer. Found by an
    independent review, 2026-09-17; an earlier version of this test asserted the
    opposite because a stale comment claimed the trail answer needed it."""
    out = N.classify_node(_state("why that table"), {})
    assert out["engine_result"] == {}
    assert out["sql"] is None and out["rows"] is None and out["status"] is None


# ---- the answers ----------------------------------------------------------------

def test_last_query_quotes_the_engines_own_understanding():
    assert "Find the 100 cheapest sale listings." in _recall("q", "query")["reply_text"]


def test_sql_is_quoted_verbatim():
    reply = _recall("q", "sql")["reply_text"]
    assert _FRAME["last_sql"] in reply and "```sql" in reply


def test_table_answer_names_the_source_not_a_bare_id():
    reply = _recall("q", "table")["reply_text"]
    assert "Sale Listings (assets_salelisting)" in reply and "homzhub" in reply
    assert "source 2" not in reply


def test_filters_are_listed_and_absence_is_stated_plainly():
    assert "Status equals For Sale" in _recall("q", "filters")["reply_text"]
    no_filters = _recall("q", "filters", frame={**_FRAME, "filters": []})["reply_text"]
    assert "No filters" in no_filters


def test_row_count_comes_from_the_executed_result():
    assert "100 row(s)" in _recall("q", "rows")["reply_text"]


def test_the_trail_answer_covers_table_filters_ranking_and_count():
    reply = _recall("q", "trail")["reply_text"]
    for fragment in ("assets_salelisting", "Status equals For Sale",
                     "expected_price", "lowest first", "100 rows", "100 row(s)"):
        assert fragment in reply, fragment


def test_missing_facts_are_admitted_never_invented():
    """A frame without SQL/row count must say so rather than produce a plausible one."""
    empty = {"entity": "assets_salelisting", "entity_display": "Sale Listings"}
    assert "don't have the SQL" in _recall("q", "sql", frame=empty)["reply_text"]
    assert "don't have a row count" in _recall("q", "rows", frame=empty)["reply_text"]


def test_the_turn_is_recorded_in_history_exactly_once():
    out = _recall("what sql did you run", "sql")
    assert [t["role"] for t in out["history"]] == ["user", "assistant"]


# ---- graph wiring ---------------------------------------------------------------

def test_the_graph_routes_recall_away_from_the_engine():
    from chatbot.graph import _route_after_classify
    assert _route_after_classify({"action": "recall", "history": [{}]}) == "recall"


def test_recall_does_not_advance_the_query_frame():
    """Nothing was executed, so there is no new evidence to record."""
    import inspect

    from chatbot import graph as G
    assert 'g.add_edge("recall", END)' in inspect.getsource(G.build_graph)


def test_a_recall_with_nothing_to_recall_is_answered_instantly():
    """After a "start over" (or any session with no answered turn yet) there is no
    frame. Live test 2026-09-17: this fell through to the engine, which spent 62s and
    came back "Could you clarify if 'did' is a column name or a value to filter on?".
    "Nothing has been run yet" is both true and instant."""
    out = N.classify_node({"message": "what sql did you run",
                           "history": [{"role": "user", "content": "start over"}],
                           "frame": {}, "engine_result": {}}, {})
    assert out["action"] == "recall" and out["recall_kind"] == "nothing"


def test_that_answer_states_the_absence_rather_than_inventing_one():
    out = N.recall_node({"message": "what sql did you run", "recall_kind": "nothing",
                         "frame": {}, "source_profiles": {}})
    assert "don't have an earlier question" in out["reply_text"]
    assert "SELECT" not in out["reply_text"].upper()


def test_the_very_first_message_of_a_session_is_also_answered_honestly():
    """This used to be excluded on the theory that a first message might be a real
    question. Measured 2026-09-17: as the first message of a session it went to the
    engine for 101s and came back "Could you clarify if 'did' is a column name or a
    value to filter on?". There is no reading of "what SQL did you run" that a SQL
    engine can answer — "nothing has been run yet" is both true and instant."""
    N.call_slm = lambda *a, **k: '{"action": "answer", "delta_type": "new_topic"}'
    out = N.classify_node({"message": "what sql did you run", "history": [],
                           "frame": {}, "engine_result": {}}, {})
    assert out["action"] == "recall" and out["recall_kind"] == "nothing"


# ---------------------------------------------------------------------------
# Document conversations — where there is no SQL because none was run
# ---------------------------------------------------------------------------

_DOC_FRAME = {
    "entity": "Samta-Employee Handbook April 2026",
    "entity_display": "Samta-Employee Handbook April 2026",
    "understanding": "Answered from Samta-Employee Handbook April 2026 using 5 passages.",
    "route": "rag", "filters": [], "group_by": [], "measures": [], "order_by": [],
    "drill_path": [], "last_sql": None, "last_row_count": None, "source_id": 3,
    "last_status": "answered",
}


def _doc_recall(kind, frame):
    """Deliberately NOT named _recall — this file already has one with a different
    signature, and shadowing it silently broke twelve existing tests."""
    state = {"message": "x", "history": [], "frame": frame, "drill_stack": [],
             "episodic": [], "recall_kind": kind, "engine_result": {},
             "last_result": {}, "tenant": "t", "session_id": "s"}
    return N.recall_node(state)["reply_text"]


def test_recall_no_longer_claims_nothing_has_been_run():
    """Measured 2026-09-21 against the real docs_contracts source: after TWO answered
    document turns, "what sql did you run" replied "nothing has been run in this
    conversation yet". The cause was upstream — a retrieval result carries no `status`
    key, so harvest_frame refused it and no frame was ever written — but the visible
    failure was recall lying about the conversation."""
    reply = _doc_recall("sql", _DOC_FRAME)
    assert "nothing has been run" not in reply.lower()
    assert "didn't run any SQL" in reply
    assert "Samta-Employee Handbook April 2026" in reply


def test_recall_calls_a_document_a_document():
    reply = _doc_recall("table", _DOC_FRAME)
    assert "a document, not a table" in reply
    assert "Samta-Employee Handbook April 2026" in reply


def test_a_sql_frame_is_unaffected():
    sql_frame = {**_DOC_FRAME, "route": "deterministic", "entity": "assets_asset",
                 "entity_display": "Assets",
                 "last_sql": "SELECT count(*) FROM assets_asset"}
    assert "```sql" in _doc_recall("sql", sql_frame)
    assert "a document, not a table" not in _doc_recall("table", sql_frame)


def test_a_sql_frame_with_no_sql_keeps_its_old_wording():
    """The "came back without one" line is still right for a SQL route that genuinely
    produced no statement — it is only wrong for retrieval, where none was expected."""
    odd = {**_DOC_FRAME, "route": "tier2", "entity": "assets_asset", "last_sql": None}
    assert "came back without one" in _doc_recall("sql", odd)


def test_go_back_with_nothing_to_go_back_to_never_reaches_the_engine():
    """Measured 2026-09-21 on the document source: "go back" cost 27 SECONDS and came
    back "The provided context does not contain the answer to the question 'go back'" —
    the retrieval pipeline was asked to find a navigation word in a PDF. A document
    conversation never builds a drill stack at all, so that was the permanent state
    there, not an edge case."""
    def must_not_be_called(*a, **k):
        raise AssertionError("an empty-stack drill-up must not reach the model")

    original = N.call_slm
    N.call_slm = must_not_be_called
    try:
        for message in ("go back", "zoom out", "undo that", "previous", "back up"):
            state = {"message": message, "history": [], "frame": dict(_DOC_FRAME),
                     "drill_stack": [], "episodic": [], "last_result": {},
                     "engine_result": {}, "pending_clarification": {},
                     "memory_reset": False, "tenant": "t", "session_id": "s"}
            got = N.classify_node(state, {})
            assert got["action"] == "recall", message
            assert got["recall_kind"] == "drill_up_empty", message
    finally:
        N.call_slm = original


def test_the_empty_drill_up_reply_says_what_is_actually_true():
    reply = _doc_recall("drill_up_empty", _DOC_FRAME)
    assert "nothing to go back to" in reply


def test_a_real_drill_up_is_untouched():
    """With a level to pop, the old deterministic path still owns it."""
    state = {"message": "go back", "history": [],
             "frame": {"entity": "assets_asset", "entity_display": "Assets",
                       "filters": [], "route": "deterministic"},
             "drill_stack": [{"dimension": "City", "value": "Pune"}],
             "episodic": [], "last_result": {}, "engine_result": {},
             "pending_clarification": {}, "memory_reset": False,
             "tenant": "t", "session_id": "s"}
    got = N.classify_node(state, {})
    assert got["action"] == "followup"
    assert got["delta_type"] == "drill_up"


def test_a_document_name_is_not_printed_twice():
    """A document frame's entity IS its display name; printing both gave
    "Samta-Employee Handbook April 2026 (Samta-Employee Handbook April 2026)"."""
    reply = _doc_recall("table", _DOC_FRAME)
    assert reply.count("Samta-Employee Handbook April 2026") == 1


def test_a_sql_frame_still_shows_both_names():
    sql_frame = {**_DOC_FRAME, "route": "deterministic", "entity": "assets_asset",
                 "entity_display": "Assets"}
    assert "Assets (assets_asset)" in _doc_recall("table", sql_frame)


def test_document_recall_quotes_the_user_not_the_answer():
    """Measured 2026-09-21: "what did i ask earlier" on a document conversation replied
    "Your last question was: Answered from Samta-Employee Handbook April 2026 using 5
    relevant passages" — a description of the ANSWER, offered as the question.

    `understanding` means different things per head: a restatement of the question on
    SQL, a description of the process on retrieval. The user's own words are kept
    verbatim in the episodic buffer, so quote those."""
    state = {"message": "x", "history": [], "frame": dict(_DOC_FRAME),
             "drill_stack": [], "recall_kind": "query", "engine_result": {},
             "last_result": {}, "tenant": "t", "session_id": "s",
             "episodic": [{"role": "user", "content": "what is the notice period"},
                          {"role": "assistant", "content": "answered: 90 days"},
                          {"role": "user", "content": "what about for a full time employee"},
                          {"role": "assistant", "content": "answered: ..."}]}
    reply = N.recall_node(state)["reply_text"]
    assert reply == "You asked: what about for a full time employee"
    assert "relevant passages" not in reply


def test_document_recall_falls_back_when_the_buffer_is_empty():
    state = {"message": "x", "history": [], "frame": dict(_DOC_FRAME),
             "drill_stack": [], "recall_kind": "query", "engine_result": {},
             "last_result": {}, "tenant": "t", "session_id": "s", "episodic": []}
    reply = N.recall_node(state)["reply_text"]
    assert "Samta-Employee Handbook April 2026" in reply


def test_sql_recall_still_uses_the_engines_restatement():
    """On SQL, `understanding` IS a restatement of the question and is better than the
    raw words — unchanged."""
    sql_frame = {**_DOC_FRAME, "route": "deterministic", "entity": "assets_asset",
                 "entity_display": "Assets",
                 "understanding": "Find the 100 cheapest sale listings."}
    state = {"message": "x", "history": [], "frame": sql_frame, "drill_stack": [],
             "recall_kind": "query", "engine_result": {}, "last_result": {},
             "tenant": "t", "session_id": "s",
             "episodic": [{"role": "user", "content": "raw words"}]}
    assert "Find the 100 cheapest sale listings." in N.recall_node(state)["reply_text"]
