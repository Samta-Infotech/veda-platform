"""Summary prompt shaping (2026-07-17):
  #3 few-shot style exemplar — pins the 2-3 sentence business format/tone.
  #4 result-shape-aware guidance — a ranking reads like a ranking, a trend like a
     trend, etc., keyed by veda/result_analyzer.py's RESULT_SHAPES.
Verified by capturing the prompt handed to the SLM (mocked). No network / DB / SLM.
Run: ``pytest tests/test_summary_prompt.py``"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


def _capture_prompt(monkeypatch, **kwargs):
    import slm as slm_mod
    import query.result_explainer as re_mod
    seen = {}

    def _fake(prompt, **kw):
        seen["prompt"] = prompt
        return "North leads with $1.2M, ahead of the rest."   # grounded-enough placeholder

    monkeypatch.setattr(slm_mod, "call_slm", _fake)
    # Disable the numeric guard here — we're asserting on the PROMPT, not the answer,
    # and the placeholder answer's numbers aren't in this tiny fixture.
    monkeypatch.setattr(re_mod, "NL_SUMMARY_NUMERIC_GUARD", False, raising=False)
    re_mod.run_nl_answer("q", ["region", "revenue"],
                         [{"region": "North", "revenue": 1200}, {"region": "West", "revenue": 900}],
                         **kwargs)
    return seen["prompt"]


def test_style_exemplar_always_in_prompt(monkeypatch):
    prompt = _capture_prompt(monkeypatch)
    assert "Style example" in prompt
    assert "Top 3 regions by revenue" in prompt          # the exemplar Q
    assert "never reuse its numbers" in prompt            # anti-leak instruction


def test_exemplar_is_currency_neutral(monkeypatch):
    """Gap fix: the few-shot exemplar must not anchor the model to a currency the
    data doesn't use — no $ / ₹ in the exemplar, and an explicit 'use the data's
    own currency' instruction."""
    from query.result_explainer import _STYLE_EXEMPLAR
    assert "$" not in _STYLE_EXEMPLAR and "₹" not in _STYLE_EXEMPLAR
    prompt = _capture_prompt(monkeypatch)
    assert "currency/units/dates exactly as shown" in prompt
    assert "never introduce a currency symbol or unit that isn't in the data" in prompt


def test_no_forced_insight_when_no_pattern(monkeypatch):
    """Gap fix: when there's no notable pattern, the prompt must tell the model to
    stop after the direct answer instead of inventing an insight to fill 2-3
    sentences."""
    prompt = _capture_prompt(monkeypatch)                 # no patterns passed
    # brief mode (non-analytical shape): short, direct, no forced insight
    assert "1-2 sentences" in prompt
    assert "do NOT invent an insight" in prompt
    assert "otherwise stop" in prompt


def test_ranking_shape_guidance(monkeypatch):
    prompt = _capture_prompt(monkeypatch, result_shape="RANKING")
    assert "This is a ranking" in prompt
    assert "leads" in prompt


def test_trend_shape_guidance(monkeypatch):
    prompt = _capture_prompt(monkeypatch, result_shape="TREND")
    assert "This is a time trend" in prompt


def test_grouped_shape_guidance(monkeypatch):
    prompt = _capture_prompt(monkeypatch, result_shape="GROUPED")
    assert "broken down by category" in prompt


def test_unknown_shape_adds_no_guidance_line(monkeypatch):
    from query.result_explainer import _SHAPE_GUIDANCE
    prompt = _capture_prompt(monkeypatch, result_shape=None)
    # none of the shape-specific lines should appear when shape is unknown/None
    assert not any(line in prompt for line in _SHAPE_GUIDANCE.values())


def test_metrics_and_findings_in_prompt(monkeypatch):
    prompt = _capture_prompt(monkeypatch, patterns=["North leads by a wide margin"])
    assert "metrics" in prompt                            # #1 aggregates present
    assert "Verified findings already computed" in prompt  # patterns woven-in instruction
    assert "North leads by a wide margin" in prompt
