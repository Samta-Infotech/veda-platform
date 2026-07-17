"""Summary anti-hallucination levers (2026-07-17):
  #1 _numeric_aggregates — exact totals/extremes handed to the SLM so it never
     has to compute (and thus invent) figures.
  #2 _answer_numbers_grounded — rejects a summary that states a number absent from
     the precomputed facts/metrics/patterns, so run_nl_answer falls back to the
     deterministic blend instead of shipping a confident wrong number.
Pure-python, no SLM / network / DB. Run: ``pytest tests/test_summary_numeric_guard.py``"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


# ---------------------------------------------------------------------------
# #1 — precomputed aggregates
# ---------------------------------------------------------------------------

def test_numeric_aggregates_computes_exact_totals():
    from query.result_explainer import _numeric_aggregates
    rows = [{"name": "A", "amount": 100}, {"name": "B", "amount": 50},
            {"name": "C", "amount": 30}]
    m = _numeric_aggregates(["name", "amount"], rows)
    assert "amount" in m and "name" not in m           # only numeric columns
    a = m["amount"]
    assert a["sum"] == 180 and a["min"] == 30 and a["max"] == 100
    assert a["count"] == 3 and a["mean"] == 60 and a["median"] == 50


def test_extract_facts_includes_metrics():
    from query.result_explainer import _extract_facts
    facts = _extract_facts(["name", "amount"],
                           [{"name": "A", "amount": 100}, {"name": "B", "amount": 50}])
    assert "metrics" in facts and facts["metrics"]["amount"]["sum"] == 150


# ---------------------------------------------------------------------------
# #2 — number extraction + grounding check
# ---------------------------------------------------------------------------

def test_parse_numbers_handles_currency_commas_magnitude_percent():
    from query.result_explainer import _parse_numbers_from_text
    got = _parse_numbers_from_text("₹98,400 total, ₹3.2L overall, up 40% and 2 crore more")
    assert 98400.0 in got
    assert 320000.0 in got          # 3.2L
    assert 40.0 in got              # 40%
    assert 20000000.0 in got        # 2 crore


def test_grounded_when_all_numbers_present():
    from query.result_explainer import _answer_numbers_grounded
    facts = {"row_count": 5, "metrics": {"amount": {"sum": 320000, "max": 98400, "median": 60000}}}
    patterns = ["top value is 40% above the median"]
    ans = ("Rahul's ₹98,400 payment leads the 5, which total ₹3.2L — about 40% above "
           "the median.")
    assert _answer_numbers_grounded(ans, facts, patterns) is True


def test_ungrounded_number_is_rejected():
    from query.result_explainer import _answer_numbers_grounded
    facts = {"row_count": 5, "metrics": {"amount": {"sum": 320000, "max": 98400}}}
    # 750000 appears nowhere in facts/patterns — a hallucinated total.
    ans = "The five payments total ₹7,50,000 overall."
    assert _answer_numbers_grounded(ans, facts, []) is False


def test_small_counts_are_always_allowed():
    from query.result_explainer import _answer_numbers_grounded
    facts = {"row_count": 5, "metrics": {"amount": {"sum": 320000}}}
    # "4 of the 5" — small counts <= row_count, fair game even if not in metrics.
    assert _answer_numbers_grounded("DEBIT accounts for 4 of the 5 payments.", facts, []) is True


def test_rounding_within_tolerance_is_grounded():
    from query.result_explainer import _answer_numbers_grounded
    facts = {"row_count": 5, "metrics": {"amount": {"max": 98400}}}
    # "nearly ₹1,00,000" ~ rounding of 98,400 (within 2%).
    assert _answer_numbers_grounded("The largest is nearly ₹98,000.", facts, []) is True


def test_guard_triggers_fallback_in_run_nl_answer(monkeypatch):
    """When the SLM returns a hallucinated number, run_nl_answer must discard it
    and use the deterministic blended answer (slm_used=False)."""
    import query.result_explainer as re_mod
    import slm as slm_mod

    # SLM returns a summary with a fabricated total (₹9,99,999 not in the data).
    monkeypatch.setattr(slm_mod, "call_slm",
                        lambda *a, **k: "The payments total ₹9,99,999 across the board.")
    monkeypatch.setattr(re_mod, "NL_SUMMARY_NUMERIC_GUARD", True, raising=False)

    res = re_mod.run_nl_answer("total payments", ["name", "amount"],
                               [{"name": "A", "amount": 100}, {"name": "B", "amount": 50}],
                               patterns=["A is the largest payer"])
    assert res.slm_used is False                     # guard forced the fallback
    assert "9,99,999" not in res.answer              # hallucinated figure discarded
    assert "A is the largest payer" in res.answer    # deterministic blend used instead


# ---------------------------------------------------------------------------
# Currency-strip backstop + anti-inference prompt line (2026-07-17)
# ---------------------------------------------------------------------------

def test_strip_invented_currency_removes_symbol_absent_from_data():
    from query.result_explainer import _strip_invented_currency
    facts = {"row_count": 3, "metrics": {"amount": {"sum": 1991700}}}   # no currency symbol
    out = _strip_invented_currency("Total is $1,991,700 across 3 payments.", facts)
    assert "$" not in out and "1,991,700" in out


def test_strip_keeps_currency_present_in_data():
    from query.result_explainer import _strip_invented_currency
    facts = {"sample_rows": [{"price": "$100"}]}   # data genuinely carries $
    out = _strip_invented_currency("The item costs $100.", facts)
    assert "$100" in out                                   # not stripped — it's in the data


def test_anti_inference_line_in_prompt(monkeypatch):
    import slm, query.result_explainer as re_mod
    seen = {}
    monkeypatch.setattr(slm, "call_slm", lambda p, **k: seen.setdefault("p", p) or "ok.")
    monkeypatch.setattr(re_mod, "NL_SUMMARY_NUMERIC_GUARD", False, raising=False)
    re_mod.run_nl_answer("q", ["a", "b"], [{"a": 1, "b": 2}])
    assert "Do NOT infer causes" in seen["p"]
    assert "what a blank/empty column implies" in seen["p"]


def test_strip_keeps_non_ascii_currency_present_in_data():
    """Regression: json.dumps default escapes non-ASCII, which used to strip a
    genuine ₹/€/£ from the data. ensure_ascii=False keeps the presence check honest."""
    from query.result_explainer import _strip_invented_currency
    assert _strip_invented_currency("Total is ₹500.", {"sample_rows": [{"amt": "₹500"}]}) == "Total is ₹500."
    assert _strip_invented_currency("Sum is €90.", {"sample_rows": [{"v": "€90"}]}) == "Sum is €90."
    # a wrong symbol the data does NOT carry is still stripped
    assert "$" not in _strip_invented_currency("Largest is $500.", {"sample_rows": [{"amt": "₹500"}]})
