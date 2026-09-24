"""The answer must survive a completely broken explainability subsystem.

    No Explainability, Thinking, narration, timing, audit-enrichment or optional
    SLM operation may delay, reorder, replace, prevent, or break a terminal
    answer event.

These tests are the proof. Each one breaks a different observability component and
asserts that the `content` event — the answer — is still delivered.

WHY THIS FILE EXISTS. The property was previously held only by defensive coding
inside the engine; the api tier had an UNGUARDED block that built the four-step
model before the answer was yielded, so a step-model bug propagated out of the
generator and the view turned it into an SSE `error`. An answer the engine had
already produced correctly was destroyed by a progress-bar failure.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_django():
    """App registry only — no DB. `config` (the package) must win over
    `veda_core/config.py`, the same name collision the rest of the suite avoids."""
    import config
    from django.conf import settings
    assert hasattr(config, "__path__"), "the config/ PACKAGE must win on sys.path"
    if not settings.configured:
        settings.configure(
            INSTALLED_APPS=["django.contrib.contenttypes", "django.contrib.auth",
                            "apps.core", "apps.chat"],
            DATABASES={}, USE_TZ=True, SECRET_KEY="test")
        import django
        django.setup()


_setup_django()
from apps.chat.services import ConversationQueryService          # noqa: E402
import apps.chat.thinking_steps as ts                            # noqa: E402
from apps.chat.thinking_context import ThinkingContext           # noqa: E402
from apps.chat.turn_events import TurnEventAccumulator           # noqa: E402

ANSWER = "There are 7,814 assets."


def _service(**over):
    """A service instance with only what `_build_reply_events` touches — no DB,
    no HTTP, no __init__."""
    svc = object.__new__(ConversationQueryService)
    svc._steps = ts.ThinkingStepTracker()
    svc._steps.consume({"phase": "supervisor_classify", "message": "x"})
    svc._step_ctx = ThinkingContext()
    svc.last_action = ""
    svc.last_audit = {}
    for k, v in over.items():
        setattr(svc, k, v)
    return svc


def _response():
    return {"reply_text": ANSWER,          # the summary block is built from THIS
            "engine_result": {"ok": True, "status": "answered",
                              "answer": ANSWER, "rows": [{"n": 7814}],
                              "cols": ["n"], "explain": {"version": "1.0"}},
            "_turn_latency_ms": 1234}


def _drain(svc, response=None):
    return list(svc._build_reply_events(response or _response()))


def _content(events):
    """Every content block, in order — the answer as the reader receives it.

    Compared against the baseline rather than pattern-matched: the property under
    test is that breaking observability leaves the answer IDENTICAL, not merely
    present.
    """
    return [e["data"] for e in events if e["event"] == "content"]


def _answer_of(events):
    for d in _content(events):
        if ANSWER in ((d or {}).get("content") or ""):
            return ANSWER
    return None


BASELINE = None


def _baseline():
    global BASELINE
    if BASELINE is None:
        BASELINE = _content(_drain(_service()))
    return BASELINE


def _kinds(events):
    return [e["event"] for e in events]


# ─────────────────────────────────────────────── baseline
def test_baseline_the_answer_is_delivered():
    ev = _drain(_service())
    assert _answer_of(ev) == ANSWER
    assert "explainability" in _kinds(ev)


# ─────────────────────────────────────────────── D. thinking-step failure
def test_D_step_aggregation_failure_does_not_cost_the_answer(monkeypatch):
    svc = _service()
    monkeypatch.setattr(svc._steps, "finish",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ev = _drain(svc)
    assert _answer_of(ev) == ANSWER, "a progress-bar failure must not destroy the answer"


def test_D2_payload_assembly_failure_does_not_cost_the_answer(monkeypatch):
    svc = _service()
    monkeypatch.setattr(svc._steps, "as_payload",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad shape")))
    assert _answer_of(_drain(svc)) == ANSWER


def test_D3_detail_building_failure_does_not_cost_the_answer(monkeypatch):
    svc = _service()
    monkeypatch.setattr(ThinkingContext, "details",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("rows")))
    assert _content(_drain(svc)) == _baseline()


def test_D4_a_broken_step_model_degrades_thinking_not_the_turn(monkeypatch):
    """The contract: thinking degrades, the answer survives."""
    svc = _service()
    monkeypatch.setattr(svc._steps, "finish",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ev = _drain(svc)
    assert _answer_of(ev) == ANSWER
    frames = [e for e in ev if e["event"] == "thinking"]
    assert not any((f["data"] or {}).get("steps") for f in frames), (
        "the step model failed, so no step model should be claimed")


# ─────────────────────────────────────────────── C. explainability exception
def test_C_explainability_absorption_failure_does_not_cost_the_answer(monkeypatch):
    svc = _service()
    monkeypatch.setattr(ThinkingContext, "absorb_explain",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    assert _content(_drain(svc)) == _baseline()


def test_C2_a_malformed_explain_payload_does_not_cost_the_answer():
    base = _baseline()
    r = _response()
    r["engine_result"]["explain"] = ["not", "a", "dict"]
    ev = _drain(_service(), r)
    assert _content(ev) == base, "a malformed explain must not change the answer"
    assert not [e for e in ev if e["event"] == "error"]


# ─────────────────────────────────────── G. answer produced, explainability fails
def test_G_an_already_produced_answer_survives_a_later_explainability_failure(monkeypatch):
    """The most important case. The engine has answered; everything downstream of
    the answer is observability and must not be able to take it back."""
    svc = _service()
    ev = []
    gen = svc._build_reply_events(_response())
    for e in gen:
        ev.append(e)
        if e["event"] == "content":
            # From here on, make every remaining observability step explode.
            monkeypatch.setattr(svc._steps, "as_payload",
                                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert _answer_of(ev) == ANSWER
    assert _kinds(ev).index("content") < len(ev)


def test_G2_the_answer_is_yielded_before_the_explainability_event():
    k = _kinds(_drain(_service()))
    assert k.index("content") < k.index("explainability"), (
        "explainability is downstream of the answer, never upstream of it")


# ─────────────────────────────────────────────── E. persistence failure
def test_E_metadata_failure_leaves_the_content_blocks_intact(monkeypatch):
    """`metadata()` is what we save ABOUT a turn, not the turn. The view catches it;
    this pins that the answer blocks themselves are unaffected."""
    acc = TurnEventAccumulator()
    acc.consume("content", {"type": "markdown", "content": ANSWER, "is_summary": True})
    monkeypatch.setattr(type(acc), "metadata",
                        lambda self: (_ for _ in ()).throw(RuntimeError("json fail")))
    with pytest.raises(RuntimeError):
        acc.metadata()
    assert acc.content_blocks[0]["content"] == ANSWER
    assert acc.summary_text == ANSWER


def test_E2_the_view_guards_metadata_on_both_paths():
    """Both the JSON and the streaming path must survive it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "apps", "chat", "views.py"), encoding="utf-8").read()
    assert src.count("turn metadata failed") == 2, (
        "the JSON path and the streaming path each need the guard")
