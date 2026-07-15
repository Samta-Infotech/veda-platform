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


def test_answer_with_data_question_hint_and_no_frame_is_not_downgraded():
    """The backstop must not fire on a genuinely self-contained question just
    because no frame happens to exist yet (e.g. the first real question of a
    session) — only on PURELY referential text with no data content."""
    import chatbot.nodes as nodes
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
    import chatbot.nodes as nodes
    assert nodes._DRILL_UP_RE.match("go back again")
    assert nodes._DRILL_UP_RE.match("Go Back")
    assert nodes._DRILL_UP_RE.match("undo that")
    assert nodes._DRILL_UP_RE.match("zoom out")


def test_go_back_without_drill_stack_falls_through_to_llm(monkeypatch):
    """No level to pop (empty/absent drill_stack) — the deterministic path
    must NOT fire; falls through to the normal LLM classification instead of
    forcing a drill_up transition into nothing."""
    calls = []

    def fake_call_slm(system, user, **kwargs):
        calls.append(system)
        return '{"action": "followup", "delta_type": "ambiguous", "slot_candidates": []}'

    monkeypatch.setattr(nodes, "call_slm", fake_call_slm)
    state = {"message": "go back", "history": _history(),
            "frame": _frame_with_entity(), "drill_stack": []}
    nodes.classify_node(state, config={})

    assert len(calls) == 1, "should have fallen through to the LLM classify call"


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
    import chatbot.nodes as nodes
    assert nodes._DRILL_UP_RE.match("how many transactions came back as failed") is None
    assert nodes._DRILL_UP_RE.match("show me the backup transactions") is None
    assert nodes._DRILL_UP_RE.match("undo transactions from last week") is None
