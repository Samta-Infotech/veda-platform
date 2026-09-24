"""Refusal polishing must never delay or break a refusal.

SEPARATE FILE ON PURPOSE. These import `veda_core`, whose `config` module collides
with the Django `config` PACKAGE — the two cannot be loaded in one process, which
is why the api-tier isolation tests live in `tests/test_answer_isolation.py`.
"""
import os
import sys
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, "veda_core"))


def test_F_refusal_polish_is_off_by_default():
    """Read from disk: `config` in this process is the DJANGO package, and the
    engine's `veda_core/config.py` cannot be imported under the same name. The
    previous value was a hardcoded `True` sitting under a comment that claimed
    "(default OFF)" — the point of this test is that the two now agree."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "veda_core", "config.py"), encoding="utf-8").read()
    line = [l for l in src.splitlines() if l.startswith("FEEDBACK_LLM_POLISH")]
    assert line == ['FEEDBACK_LLM_POLISH = _os.environ.get("FEEDBACK_LLM_POLISH", "0") == "1"'], line


def test_F2_polish_failure_keeps_the_deterministic_refusal(monkeypatch):
    """A broken SLM must cost the WORDING, never the refusal."""
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "veda_core"))
    from veda import feedback as fb
    import slm as _slm
    monkeypatch.setattr(_slm, "call_slm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("slm down")))
    assert fb._polish({"why": "w", "what_needed": "n", "suggestions": []}) is None


def test_F3_polish_is_bounded_by_a_wall_clock_deadline(monkeypatch):
    """`urlopen`'s timeout is applied AFTER getaddrinfo, so a DNS stall sits outside
    it — which is how a call nominally capped at 6s was measured at 16,028 ms."""
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "veda_core"))
    from veda import feedback as fb
    import slm as _slm

    def _hang(*a, **k):
        time.sleep(30)                      # simulates a DNS stall
        return "never"

    monkeypatch.setattr(_slm, "call_slm", _hang)
    monkeypatch.setattr(fb, "POLISH_DEADLINE_S", 0.4)
    t0 = time.time()
    out = fb._polish({"why": "w", "what_needed": "n", "suggestions": []})
    el = time.time() - t0
    assert out is None
    assert el < 3.0, f"the deadline did not bound the call: took {el:.1f}s"


# ═══════════════════════════════════════════ A/B. narration never blocks
def test_A_narrator_failure_leaves_the_caller_untouched(monkeypatch):
    """A narration is a sentence swap. If the SLM is down the deterministic
    sentence already on screen is the answer — nothing else may change."""
    from veda import narrator as nar
    import slm as _slm
    monkeypatch.setattr(_slm, "call_slm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("slm down")))
    got = []
    h = nar.start({"intent": "count"}, "analyzing", lambda s, t: got.append((s, t)))
    h.cancel()
    assert got == []


def test_B_narration_start_returns_immediately_even_when_the_slm_hangs(monkeypatch):
    """The whole design rests on this: `start()` must not wait for the model."""
    from veda import narrator as nar
    import slm as _slm

    def _hang(*a, **k):
        time.sleep(30)
        return "never"

    monkeypatch.setattr(_slm, "call_slm", _hang)
    t0 = time.time()
    h = nar.start({"intent": "count"}, "analyzing", lambda s, t: None)
    started = time.time() - t0
    t1 = time.time()
    h.cancel()
    cancelled = time.time() - t1
    assert started < 0.5, f"start() blocked for {started:.1f}s"
    assert cancelled < 0.5, f"cancel() joined the thread: {cancelled:.1f}s"


def test_B2_a_narration_that_lands_after_cancel_is_discarded(monkeypatch):
    """A late narration must not reach a stream the turn has already closed."""
    from veda import narrator as nar
    import slm as _slm

    def _slow(*a, **k):
        time.sleep(0.6)
        return "Counting the records."

    monkeypatch.setattr(_slm, "call_slm", _slow)
    got = []
    h = nar.start({"intent": "count"}, "analyzing", lambda s, t: got.append(t))
    h.cancel()
    time.sleep(1.0)
    assert got == [], "a cancelled narration was delivered anyway"


# ═══════════════════════════════════════════ C. engine-side explainability
def test_C_engine_explainability_failure_does_not_break_the_result(monkeypatch):
    """`_backfill_missing_explain` runs at the front door's single exit, BEFORE the
    result is returned. If it raises, the caller must still get its result."""
    import veda_hybrid as VH
    from query.multi_result import MultiResult, SubResult, STATUS_OK
    import veda.business_explain as be
    monkeypatch.setattr(be, "build_explain",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("explain boom")))
    payload = {"answer": "42", "rows": [{"n": 42}]}
    res = MultiResult(items=[SubResult(sub_query="q", status=STATUS_OK,
                                       route="deterministic", result=payload)])
    VH._backfill_missing_explain(res)          # must not raise
    assert res.items[0].result["answer"] == "42"


def test_C2_every_front_door_post_processor_swallows_its_own_failure():
    """Each helper at the single exit owns a guard, so one broken projection cannot
    take the turn down with it."""
    import inspect
    import veda_hybrid as VH
    for name in ("_mark_empty_results", "_backfill_missing_explain",
                 "_sync_reported_row_count", "_reconcile_access_check",
                 "_refresh_persisted_timeline"):
        src = inspect.getsource(getattr(VH, name))
        assert "except Exception" in src, f"{name} has no failure guard"


def test_C3_safe_projection_survives_a_trace_that_makes_no_sense():
    """Projection is observability: a malformed trace costs the blocks, not the turn."""
    from veda import safe_projection as sp

    class _Broken:
        enabled = True

        @property
        def sections(self):
            raise RuntimeError("trace exploded")

    t = _Broken()
    assert sp.build_warnings(t) == []
    assert sp.build_limitations(t) == []
    assert sp.build_data_sources(t) == []
    assert sp.build_timeline(t) == []
    assert isinstance(sp.build_explain_extension(t, trace_id="x"), dict)
