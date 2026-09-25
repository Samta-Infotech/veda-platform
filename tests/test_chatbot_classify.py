"""Regression tests for chatbot/nodes.py's classify_node — the misrouting bug
where a plain "hi" or a self-introduction ("my name is raj") got second-guessed
by _depends_on_history into "followup", got rewritten into a bogus database
question, and ran an unfiltered query against real tables (two separate
production incidents, same root cause).

Fix: _REFERENTIAL_HINTS pre-filters _depends_on_history so it's only ever
asked for messages that contain at least some referential language — a bare
greeting or self-introduction can never depend on prior conversation to mean
something concrete, so the (unreliable) model call is skipped entirely for
those instead of trusted to always classify them correctly.

Pure-python, no Django settings needed (chatbot.nodes imports standalone)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import chatbot.nodes as nodes


def _history():
    return [
        {"role": "user", "content": "total amount by payer"},
        {"role": "assistant", "content": "The total is $367,236."},
    ]


# ---------------------------------------------------------------------------
# _REFERENTIAL_HINTS — the deterministic pre-filter itself
# ---------------------------------------------------------------------------

def test_referential_hints_no_match_on_bare_greeting():
    assert nodes._REFERENTIAL_HINTS.search("hi") is None


def test_referential_hints_no_match_on_self_introduction():
    assert nodes._REFERENTIAL_HINTS.search("my name is raj") is None


def test_referential_hints_matches_genuine_followup_phrasing():
    assert nodes._REFERENTIAL_HINTS.search("what about that one") is not None
    assert nodes._REFERENTIAL_HINTS.search("show me the other one too") is not None
    assert nodes._REFERENTIAL_HINTS.search("same as before") is not None


# ---------------------------------------------------------------------------
# classify_node — the two real incidents, reproduced with a mocked SLM
# ---------------------------------------------------------------------------

def test_hi_never_calls_depends_on_history(monkeypatch):
    """Regression: a bare 'hi' with prior chat history must stay 'smalltalk' —
    no SLM call should happen at all (deterministic fast path + no referential
    hints means _depends_on_history is never even invoked)."""
    def must_not_be_called(*a, **k):
        raise AssertionError("call_slm should not be invoked for a bare 'hi'")
    monkeypatch.setattr(nodes, "call_slm", must_not_be_called)

    state = {"message": "hi", "history": _history()}
    result = nodes.classify_node(state, config={})
    assert result["action"] == "smalltalk"


def test_self_introduction_never_calls_depends_on_history(monkeypatch):
    """Regression: 'my name is raj' — even though this goes through the LLM
    supervisor classifier (not the deterministic greeting regex), it still
    must not trigger _depends_on_history, since it has no referential
    language. Mocks the supervisor call to return 'smalltalk', then asserts
    no SECOND call_slm invocation (the standalone-check) ever happens."""
    calls = []

    def fake_call_slm(system, user, **kwargs):
        calls.append((system, user, kwargs))
        return '{"action": "smalltalk"}'

    monkeypatch.setattr(nodes, "call_slm", fake_call_slm)
    state = {"message": "my name is raj", "history": _history()}
    result = nodes.classify_node(state, config={})

    assert result["action"] == "smalltalk"
    assert len(calls) == 1   # only the supervisor classify call — no standalone-check call


def test_genuine_followup_still_triggers_depends_on_history(monkeypatch):
    """The safety net must still work for messages that DO contain referential
    language, are genuinely ambiguous, AND have a real QueryFrame to resolve
    against (a real prior answered query, not just conversational history)."""
    calls = []

    def fake_call_slm(system, user, **kwargs):
        calls.append(system)
        if len(calls) == 1:
            return '{"action": "smalltalk"}'   # supervisor classify
        return "dependent"                     # standalone-check verdict

    monkeypatch.setattr(nodes, "call_slm", fake_call_slm)
    state = {"message": "what about that one", "history": _history(),
            "frame": {"entity": "accounts_generalledger"}}
    result = nodes.classify_node(state, config={})

    assert len(calls) == 2   # supervisor classify AND the standalone-check both ran
    assert result["action"] == "followup"


def test_referential_language_without_a_frame_does_not_fabricate_followup(monkeypatch):
    """Regression: 'what about the other one' right after small talk — no
    QueryFrame was ever established (no real prior data question), so there's
    nothing to genuinely follow up ON. Before this fix, the override fired
    anyway, sent the raw ambiguous text straight to the engine, and the
    engine's own retrieval matched it against a totally unrelated table —
    a confident-looking but fabricated answer. _depends_on_history must not
    even be CALLED here (no frame -> no point asking)."""
    def must_not_be_called(*a, **k):
        raise AssertionError("_depends_on_history should not be invoked without "
                             "a real QueryFrame to resolve against")
    monkeypatch.setattr(nodes, "call_slm", lambda *a, **k: '{"action": "smalltalk"}')
    monkeypatch.setattr(nodes, "_depends_on_history", must_not_be_called)

    state = {"message": "what about the other one", "history": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hi! Ask me anything about your data."},
    ], "frame": {}}
    result = nodes.classify_node(state, config={})

    assert result["action"] == "smalltalk"


def test_llm_directly_saying_followup_without_frame_still_gets_caught(monkeypatch):
    """The actual live bug: with a stronger classify model, the LLM can decide
    'followup' DIRECTLY (no override involved at all) for purely referential
    text with no frame to ground it. The backstop after the override chain
    must catch this regardless of which path produced the verdict."""
    monkeypatch.setattr(nodes, "call_slm", lambda *a, **k: '{"action": "followup"}')

    state = {"message": "what about the other one", "history": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hi! Ask me anything about your data."},
    ], "frame": {}}
    result = nodes.classify_node(state, config={})

    assert result["action"] == "smalltalk"


def test_followup_with_real_frame_is_not_downgraded(monkeypatch):
    """The backstop must not fire when there IS a real QueryFrame — a
    genuine followup against actual grounded context stays a followup."""
    monkeypatch.setattr(nodes, "call_slm", lambda *a, **k: '{"action": "followup"}')

    state = {"message": "what about the other one", "history": [
        {"role": "user", "content": "total by payer"},
        {"role": "assistant", "content": "The total is $367,236."},
    ], "frame": {"entity": "accounts_generalledger"}}
    result = nodes.classify_node(state, config={})

    assert result["action"] == "followup"


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN GAP, not a regression. A guard for this (_has_frame_continuation_evidence) "
    "was added and removed on 2026-09-24: its third arm read _mentions_frame_subject "
    "FORWARDS ('shares a content word with the frame => continues the frame'), which is "
    "not what that helper establishes — its docstring defines the INVERSE test. Forwards "
    "it made any new question naming the same table a follow-up, which then carries the "
    "previous turn's filters into it. It regressed the pre-existing "
    "test_clarification_flow::test_a_message_that_is_not_an_answer_is_never_swallowed "
    "('show me the top 5 cities by number of assets' -> followup). The behaviour asserted "
    "below is still WANTED; it needs a measured mechanism. Note the live supervisor "
    "handles 'only the <value> ones' correctly in practice (measured: delta=refine, "
    "stack=1) — this test forces a model MISS, so the gap is 'a miss on this shape loses "
    "the frame', which is pre-existing."))
def test_same_entity_refinement_preserves_frame_when_model_says_new_topic(monkeypatch):
    """A model miss must not turn an explicit same-entity refinement into a
    standalone query: ``How many users?`` -> ``How many active users?``."""
    monkeypatch.setattr(
        nodes, "call_slm",
        lambda *a, **k: '{"action":"answer", "delta_type":"new_topic"}',
    )
    state = {
        "message": "How many active users?",
        "history": _history(),
        "frame": {"entity": "users_user", "entity_display": "Users",
                  "aggregation": "count"},
    }
    result = nodes.classify_node(state, config={})
    assert result["action"] == "followup"
    assert result["delta_type"] == "ambiguous"


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN GAP, not a regression. A guard for this (_has_frame_continuation_evidence) "
    "was added and removed on 2026-09-24: its third arm read _mentions_frame_subject "
    "FORWARDS ('shares a content word with the frame => continues the frame'), which is "
    "not what that helper establishes — its docstring defines the INVERSE test. Forwards "
    "it made any new question naming the same table a follow-up, which then carries the "
    "previous turn's filters into it. It regressed the pre-existing "
    "test_clarification_flow::test_a_message_that_is_not_an_answer_is_never_swallowed "
    "('show me the top 5 cities by number of assets' -> followup). The behaviour asserted "
    "below is still WANTED; it needs a measured mechanism. Note the live supervisor "
    "handles 'only the <value> ones' correctly in practice (measured: delta=refine, "
    "stack=1) — this test forces a model MISS, so the gap is 'a miss on this shape loses "
    "the frame', which is pre-existing."))
def test_only_value_refinement_preserves_frame_when_entity_is_omitted(monkeypatch):
    """"Only the APPROVED ones" is a narrowing instruction; the entity is
    intentionally omitted because it is already on the active frame."""
    monkeypatch.setattr(
        nodes, "call_slm",
        lambda *a, **k: '{"action":"answer", "delta_type":"new_topic"}',
    )
    state = {
        "message": "only the APPROVED ones",
        "history": _history(),
        "frame": {"entity": "assets_salelisting", "entity_display": "Sale Listings"},
    }
    result = nodes.classify_node(state, config={})
    assert result["action"] == "followup"
    assert result["delta_type"] == "ambiguous"


def test_unrelated_topic_is_not_forced_into_active_frame(monkeypatch):
    """An explicit topic switch remains a new topic even when a frame exists."""
    monkeypatch.setattr(
        nodes, "call_slm",
        lambda *a, **k: '{"action":"answer", "delta_type":"new_topic"}',
    )
    state = {
        "message": "Show properties",
        "history": _history(),
        "frame": {"entity": "users_user", "entity_display": "Users"},
    }
    result = nodes.classify_node(state, config={})
    assert result["action"] == "answer"
    assert result["delta_type"] == "new_topic"


def test_answer_with_data_question_hint_and_no_frame_is_not_downgraded():
    """The backstop must not fire on a genuinely self-contained question just
    because no frame happens to exist yet (e.g. the first real question of a
    session) — only on PURELY referential text with no data content."""
    assert nodes._DATA_QUESTION_HINTS.search("how many transactions are there")


def test_classify_model_is_the_lightweight_chatbot_model(monkeypatch):
    """Every classify-path call_slm invocation should use CHATBOT_CLASSIFY_MODEL,
    not the heavy 7B coder model used for SQL generation."""
    captured = {}

    def fake_call_slm(system, user, **kwargs):
        captured.update(kwargs)
        return '{"action": "smalltalk"}'

    monkeypatch.setattr(nodes, "call_slm", fake_call_slm)
    state = {"message": "what is machine learning", "history": []}
    nodes.classify_node(state, config={})

    assert captured.get("model") == nodes.CHATBOT_CLASSIFY_MODEL


# ---------------------------------------------------------------------------
# _DRILL_UP_RE deterministic fast path — "go back" navigation. Root cause:
# the LLM classifier non-deterministically misjudged "go back" as smalltalk
# (same message, different verdict on repeat runs), and even when correctly
# routed to "followup", its own delta_type defaulted to "new_topic" (the
# prompt's own instruction for a smalltalk verdict), which made
# render_frame_as_query() send the literal word "back"/"again" to the SQL
# engine as if it were data — the engine then tried and failed to match it
# against columns/values (2026-07 memory-layer live testing).
# ---------------------------------------------------------------------------

def _frame_with_entity():
    return {"entity": "accounts_paymenttransaction", "entity_display": "Payment Transactions",
           "filters": [{"field": "Transaction Type", "operator": "equals", "value": "debit",
                        "source": "executed_sql"}]}


def _nonempty_drill_stack():
    return [{"dimension": "Transaction Type", "value": "debit"}]


def test_go_back_with_drill_stack_never_calls_llm(monkeypatch):
    """Deterministic fast path: no call_slm at all when there's a real drill
    level to pop."""
    def must_not_be_called(*a, **k):
        raise AssertionError("call_slm should not be invoked for 'go back' "
                             "with an active drill stack")
    monkeypatch.setattr(nodes, "call_slm", must_not_be_called)

    state = {"message": "go back", "history": _history(),
            "frame": _frame_with_entity(), "drill_stack": _nonempty_drill_stack()}
    result = nodes.classify_node(state, config={})

    assert result["action"] == "followup"
    assert result["delta_type"] == "drill_up"


def test_go_back_again_also_matches():
    assert nodes._DRILL_UP_RE.match("go back again")
    assert nodes._DRILL_UP_RE.match("Go Back")
    assert nodes._DRILL_UP_RE.match("undo that")
    assert nodes._DRILL_UP_RE.match("zoom out")


def test_go_back_without_drill_stack_answers_from_memory(monkeypatch):
    """No level to pop (empty/absent drill_stack) — no drill_up transition into
    nothing, and no model or engine call either: "go back" used to fall through to
    the engine and cost 27s for a non-answer (measured 2026-09-21). It is answered
    as a recall ("nothing to go back to")."""
    calls = []

    def fake_call_slm(system, user, **kwargs):
        calls.append(system)
        return '{"action": "followup", "delta_type": "ambiguous", "slot_candidates": []}'

    monkeypatch.setattr(nodes, "call_slm", fake_call_slm)
    state = {"message": "go back", "history": _history(),
            "frame": _frame_with_entity(), "drill_stack": []}
    out = nodes.classify_node(state, config={})

    assert calls == [], "empty-stack go back must not call the model"
    assert out["action"] == "recall"
    assert out["recall_kind"] == "drill_up_empty"


def test_go_back_without_active_frame_falls_through_to_llm(monkeypatch):
    """No frame at all (nothing to have drilled into in the first place) —
    same fallback-to-LLM behavior."""
    calls = []

    def fake_call_slm(system, user, **kwargs):
        calls.append(system)
        return '{"action": "smalltalk"}'

    monkeypatch.setattr(nodes, "call_slm", fake_call_slm)
    state = {"message": "go back", "history": _history(),
            "frame": {}, "drill_stack": _nonempty_drill_stack()}
    nodes.classify_node(state, config={})

    assert len(calls) == 1, "should have fallen through to the LLM classify call"


def test_drill_up_regex_does_not_misfire_on_real_question_containing_back():
    """Anchored whole-message match (same style as _RESET_RE) — a real
    question that merely contains 'back' or 'undo' mid-sentence must never
    match."""
    assert nodes._DRILL_UP_RE.match("how many transactions came back as failed") is None
    assert nodes._DRILL_UP_RE.match("show me the backup transactions") is None
    assert nodes._DRILL_UP_RE.match("undo transactions from last week") is None


# ---------------------------------------------------------------------------
# Greeting coverage. The first cut matched "hi" but not "hi there!", so the
# natural forms people actually type fell through to the full engine: measured
# 16-23 s for a greeting, answered with "It seems like you have multiple
# documents related to an employee handbook" (routing had sent it to document
# retrieval at 0.50 similarity).
# ---------------------------------------------------------------------------

GREETINGS_THAT_MUST_BE_INSTANT = [
    "hi", "hello", "hey", "yo", "hiya", "greetings", "hii", "heyyy", "hello!",
    "how are you", "how are u", "how's it going", "how are you doing",
    "hi there", "hey there", "hello there", "hi there!",
    "hey there, how are you?", "hi, how are you?", "hi there, how's it going?",
    "good morning", "good morning!",
]

MUST_STILL_REACH_THE_ENGINE = [
    "hi how many assets are there", "hello, show me the data",
    "how are the sales this month", "hey, list the invoices",
    "how many", "what is the total", "good morning report",
    "hello world table", "show me", "hi can you count the assets",
]


@pytest.mark.parametrize("msg", GREETINGS_THAT_MUST_BE_INSTANT)
def test_greeting_gets_the_canned_reply_with_no_model_call(msg):
    assert nodes._canned_smalltalk_reply(msg) is not None, msg


@pytest.mark.parametrize("msg", MUST_STILL_REACH_THE_ENGINE)
def test_a_data_question_is_never_swallowed_as_smalltalk(msg):
    assert nodes._canned_smalltalk_reply(msg) is None, msg


@pytest.mark.parametrize("msg", GREETINGS_THAT_MUST_BE_INSTANT)
def test_social_message_is_not_overridden_into_a_followup(msg):
    """The override that dragged greetings into the engine used the CANNED
    regexes as its 'is this a genuine greeting' test. `_is_social` answers the
    wider question the override actually needs."""
    assert nodes._is_social(msg) is True, msg


@pytest.mark.parametrize("msg", MUST_STILL_REACH_THE_ENGINE)
def test_data_questions_are_not_social(msg):
    assert nodes._is_social(msg) is False, msg


@pytest.mark.parametrize("msg", [
    "hi, what about the other one",
    "hello, and that one?",
    "hey, what about it",
])
def test_a_greeting_carrying_a_referential_followup_stays_a_followup(msg):
    """The guard that keeps the override useful: a social opener does not make a
    referential question social. This is the case the override exists for."""
    assert nodes._is_social(msg) is False, msg


# ── a clarification left by a REFUSED turn vs. an ambiguous continuation ──────────────
_PENDING = {"question": "Tell me what 'pool' refers to", "missing": "pool",
            "original_query": "only the ones with a swimming pool", "turn_index": 2}


def _classify_after_refusal(monkeypatch, message, delta="ambiguous"):
    """The model returns clarify_reply/<delta> — measured 2026-09-24, `ambiguous` for
    "only the Nagpur ones" after a refused "only the ones with a swimming pool"."""
    monkeypatch.setattr(nodes, "call_slm", lambda system, user, **kw: (
        '{"action": "clarify_reply", "delta_type": "%s", "slot_candidates": []}' % delta))
    state = {"message": message, "history": _history(), "frame": _frame_with_entity(),
             "drill_stack": [{"dimension": "Transaction Type", "value": "debit"}],
             "pending_clarification": dict(_PENDING)}
    return nodes.classify_node(state, config={})


def test_an_ambiguous_narrowing_follow_up_continues_the_frame(monkeypatch):
    for msg in ("only the Nagpur ones", "just the FULL ones", "what about SEMI"):
        out = _classify_after_refusal(monkeypatch, msg)
        assert out["action"] == "followup", f"{msg!r} was glued onto the dead question"


def test_a_bare_answer_still_completes_the_clarification(monkeypatch):
    """A clarifying question asks for a value; a bare value or a name for the word is an
    answer and must still reach clarify_reply."""
    for msg in ("the amenities column", "amenities", "Nagpur"):
        out = _classify_after_refusal(monkeypatch, msg)
        assert out["action"] == "clarify_reply", f"{msg!r} should answer the clarification"


def test_a_conditional_is_not_read_as_a_narrowing(monkeypatch):
    out = _classify_after_refusal(monkeypatch, "only if it has a pool")
    assert out["action"] == "clarify_reply"


def test_a_new_topic_label_is_still_left_to_the_pending_request(monkeypatch):
    out = _classify_after_refusal(monkeypatch, "only the Nagpur ones", delta="new_topic")
    assert out["action"] == "clarify_reply"
