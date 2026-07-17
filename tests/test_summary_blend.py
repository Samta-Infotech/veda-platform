"""Natural pattern-blend for the final summary (2026-07-17).

query/result_explainer.py::blend_patterns folds the deterministic detected
patterns into the answer as ONE natural clause (replacing the old mechanical
"Analysis: …" suffix), and only when a summary SLM did not already weave them.
Pure-python, no SLM / network / DB. Run: ``pytest tests/test_summary_blend.py``"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


def test_blend_two_patterns_reads_naturally():
    from query.result_explainer import blend_patterns
    out = blend_patterns("Total is 3.2L across 5 payments",
                         ["DEBIT dominates (4 of 5)", "top value is 40% above the median"])
    assert out == ("Total is 3.2L across 5 payments — DEBIT dominates (4 of 5), "
                   "and top value is 40% above the median.")
    assert "Analysis:" not in out          # no mechanical suffix
    assert ";" not in out                   # no semicolon list


def test_blend_single_pattern():
    from query.result_explainer import blend_patterns
    out = blend_patterns("Revenue is 1.2M.", ["revenue grew 18% over the period"])
    assert out == "Revenue is 1.2M — revenue grew 18% over the period."


def test_blend_no_patterns_is_passthrough():
    from query.result_explainer import blend_patterns
    assert blend_patterns("Just the answer.", []) == "Just the answer."
    assert blend_patterns("Just the answer.", None) == "Just the answer."


def test_blend_empty_answer_leads_with_finding():
    from query.result_explainer import blend_patterns
    out = blend_patterns("", ["debit dominates the top 5"])
    assert out == "Debit dominates the top 5."   # capitalized, standalone sentence


def test_blend_caps_at_two_patterns():
    from query.result_explainer import blend_patterns
    out = blend_patterns("X.", ["a is high", "b is low", "c is medium"])
    assert "a is high" in out and "b is low" in out
    assert "c is medium" not in out             # top-2 only


def test_nl_answer_result_reports_slm_used_flag():
    """The deterministic fallback (no SLM reachable) must report slm_used=False and
    blend the patterns itself, so the caller does not re-blend."""
    import query.result_explainer as re_mod
    # Force the SLM call to fail so we hit the deterministic fallback branch.
    import slm as slm_mod

    def _boom(*a, **k):
        raise RuntimeError("no slm in test")

    orig = slm_mod.call_slm
    slm_mod.call_slm = _boom
    try:
        res = re_mod.run_nl_answer("top payments", ["name", "amount"],
                                   [{"name": "A", "amount": 100}, {"name": "B", "amount": 50}],
                                   patterns=["A leads by a wide margin"])
    finally:
        slm_mod.call_slm = orig
    assert res.slm_used is False
    assert "A leads by a wide margin" in res.answer     # fallback blended it in
    assert "Analysis:" not in res.answer
