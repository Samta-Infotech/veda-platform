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
    language and are genuinely ambiguous."""
    calls = []

    def fake_call_slm(system, user, **kwargs):
        calls.append(system)
        if len(calls) == 1:
            return '{"action": "smalltalk"}'   # supervisor classify
        return "dependent"                     # standalone-check verdict

    monkeypatch.setattr(nodes, "call_slm", fake_call_slm)
    state = {"message": "what about that one", "history": _history()}
    result = nodes.classify_node(state, config={})

    assert len(calls) == 2   # supervisor classify AND the standalone-check both ran
    assert result["action"] == "followup"


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
