"""A real question is not smalltalk just because a frame is present.

Measured 2026-09-25 (api container, the real classifier): with a properties frame,
"What is the probation period for new recruits?", "what is the dress code?" and "how long
is the probation period" were classified smalltalk 9/9 and answered "I'm here for
questions about your data". The same prompt without the frame, same history: answer 12/12
for those questions, smalltalk 27/27 for acknowledgements.

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_frame_free_smalltalk.py -q`
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chatbot.nodes as N  # noqa: E402

FRAME = {"entity": "assets_asset", "entity_display": "Assets", "source_id": 2,
         "route": "deterministic", "filters": [], "group_by": ["facing"]}
HISTORY = [{"role": "user", "content": "What is the distribution of properties by facing?"},
           {"role": "assistant", "content": "EAST is the most common facing."}]


def _slm(frame_verdict, frame_free_verdict, calls):
    """The frame-bearing prompt carries the frame JSON; the frame-free one does not."""
    def fake(system, user, **k):
        calls.append(k.get("purpose"))
        if k.get("purpose") == "classify_frame_free":
            assert "assets_asset" not in system and "Assets" not in system
            return frame_free_verdict
        if k.get("purpose") in ("standalone_check", "carryover_check"):
            return "standalone"
        return frame_verdict
    return fake


def _classify(monkeypatch, message, frame_verdict, frame_free_verdict, frame=FRAME):
    calls = []
    monkeypatch.setattr(N, "call_slm", _slm(frame_verdict, frame_free_verdict, calls))
    out = N.classify_node({"message": message, "history": HISTORY, "frame": frame}, config={})
    return out, calls


SMALLTALK = '{"action": "smalltalk", "delta_type": "new_topic"}'
ANSWER = '{"action": "answer", "reason": "asks for a rule"}'


@pytest.mark.parametrize("msg", ["What is the probation period for new recruits?",
                                 "how long is the probation period"])
def test_the_measured_questions_reach_the_engine(monkeypatch, msg):
    out, calls = _classify(monkeypatch, msg, SMALLTALK, ANSWER)
    assert out["action"] == "answer" and out["delta_type"] == "new_topic"
    assert "classify_frame_free" in calls


def test_an_acknowledgement_stays_smalltalk(monkeypatch):
    out, _ = _classify(monkeypatch, "makes sense", SMALLTALK, SMALLTALK)
    assert out["action"] == "smalltalk"


@pytest.mark.parametrize("second", [None, "", "not json", '{"action": "bogus"}',
                                    '{"action": "followup"}', '{"action": "clarify_reply"}'])
def test_anything_but_a_clear_answer_keeps_the_verdict(monkeypatch, second):
    out, _ = _classify(monkeypatch, "what is the dress code?", SMALLTALK, second)
    assert out["action"] == "smalltalk"


def test_a_greeting_never_pays_for_a_second_call(monkeypatch):
    out, calls = _classify(monkeypatch, "hi there, good evening friend", SMALLTALK, ANSWER)
    assert "classify_frame_free" not in calls


def test_no_frame_no_second_opinion(monkeypatch):
    """Without a frame the first call already IS the frame-free reading."""
    out, calls = _classify(monkeypatch, "what is the dress code?", SMALLTALK, ANSWER, frame={})
    assert "classify_frame_free" not in calls


def test_other_verdicts_are_never_second_guessed(monkeypatch):
    """clarify_reply is NOT covered: measured, the frame-free reading also says "answer" for
    real follow-ups ("what is its amount?", "sorry, I meant Pune"), so it cannot separate
    them from a new question there."""
    out, calls = _classify(monkeypatch, "sorry, I meant Pune",
                           '{"action": "clarify_reply", "delta_type": "replace"}', ANSWER)
    assert "classify_frame_free" not in calls


# ── clarify_reply with nothing pending goes through the ungrounded-text backstops ──────
@pytest.mark.parametrize("msg", ["Help me understand this", "explain this"])
def test_a_relabelled_clarify_reply_pointing_at_nothing_never_reaches_the_engine(monkeypatch, msg):
    """Measured 2026-09-26: the model said clarify_reply, the relabel to followup ran after
    the backstops, and the message reached the engine (107s)."""
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action": "clarify_reply"}'
                        if k.get("purpose") == "classify" else "standalone")
    out = N.classify_node({"message": msg, "history": HISTORY[:0] + [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi!"}],
        "frame": {}, "message_mentions_data": False}, config={})
    assert out["action"] in ("smalltalk", "no_match")


def test_a_relabelled_clarify_reply_with_a_frame_is_still_a_followup(monkeypatch):
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action": "clarify_reply"}'
                        if k.get("purpose") == "classify" else "standalone")
    out = N.classify_node({"message": "sorry, I meant Pune", "history": HISTORY,
                           "frame": FRAME}, config={})
    assert out["action"] == "followup"
