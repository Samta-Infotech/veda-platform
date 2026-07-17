"""Tests for query/result_explainer.py (Result Explanation Layer) — pure-python,
no DB, no network. Run from the repo root:
``pytest tests/test_result_explainer.py``"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


# ---------------------------------------------------------------------------
# Deterministic shapes — no SLM call
# ---------------------------------------------------------------------------

def test_empty_result():
    from query.result_explainer import template_answer
    assert template_answer("how many users", ["count"], []) == "No results found."


def test_count_scalar():
    from query.result_explainer import template_answer
    ans = template_answer("how many users", ["count"], [{"count": 42}])
    assert ans == "The count is 42."


def test_sum_scalar():
    from query.result_explainer import template_answer
    ans = template_answer("total revenue", ["total_amount"], [{"total_amount": 15000.5}])
    assert "15,000.5" in ans and "total amount" in ans


def test_avg_scalar():
    from query.result_explainer import template_answer
    ans = template_answer("average score", ["avg_score"], [{"avg_score": 3.14159}])
    assert ans.startswith("The avg score is")


def test_min_max_scalar():
    from query.result_explainer import template_answer
    ans_min = template_answer("lowest price", ["min_price"], [{"min_price": 10}])
    ans_max = template_answer("highest price", ["max_price"], [{"max_price": 999}])
    assert "10" in ans_min and "999" in ans_max


def test_single_row_multi_column():
    from query.result_explainer import template_answer
    ans = template_answer("show me user 5", ["name", "age"],
                          [{"name": "Alice", "age": 30}])
    assert ans == "Result: name Alice, age 30."


def test_single_row_non_aggregate_single_column():
    from query.result_explainer import template_answer
    ans = template_answer("what is the status", ["status"], [{"status": "active"}])
    assert ans == "status: active"


def test_ranking_multi_row_defers_to_slm():
    from query.result_explainer import template_answer
    multi = template_answer("top 5 customers", ["name", "amount"],
                            [{"name": "a", "amount": 1}, {"name": "b", "amount": 2}])
    assert multi is None


def test_deterministic_fallback_answer_empty():
    from query.result_explainer import deterministic_fallback_answer
    assert deterministic_fallback_answer("q", [], []) == "No results found."


def test_deterministic_fallback_answer_with_rows():
    from query.result_explainer import deterministic_fallback_answer
    ans = deterministic_fallback_answer("q", ["a", "b"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert ans.startswith("Returned 2 row(s).")
    assert "a=1" in ans and "b=2" in ans


# ---------------------------------------------------------------------------
# SLM path — multi-row narrative results
# ---------------------------------------------------------------------------

def _multi_row_args():
    columns = ["name", "amount"]
    rows = [{"name": "a", "amount": 1}, {"name": "b", "amount": 2},
            {"name": "c", "amount": 3}]
    return columns, rows


def test_slm_path_uses_small_model(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured.update(kwargs)
        captured["prompt"] = prompt
        return "Three customers were found, with amounts ranging from 1 to 3."

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    columns, rows = _multi_row_args()
    result = re_mod.run_nl_answer("list top customers", columns, rows)

    assert result.answer.startswith("Three customers")
    assert captured["model"] == re_mod.NL_SUMMARY_MODEL
    assert captured["model"] != "qwen2.5-coder:7b"   # never the heavy coder model
    assert captured["endpoint"] == "generate"


def test_slm_timeout_falls_back_to_deterministic(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    def fake_call_slm(prompt, **kwargs):
        raise RuntimeError("SLM unreachable: timed out")

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    columns, rows = _multi_row_args()
    result = re_mod.run_nl_answer("list top customers", columns, rows)

    assert result.answer.startswith("Returned 3 row(s).")


def test_slm_invalid_empty_response_falls_back(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda prompt, **kwargs: "   ")
    columns, rows = _multi_row_args()
    result = re_mod.run_nl_answer("list top customers", columns, rows)

    assert result.answer.startswith("Returned 3 row(s).")


def test_semantic_metadata_enriches_prompt(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "Summary."

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    columns, rows = _multi_row_args()
    semantic_model = {"columns": {
        "customers.amount": {"analytics_role": "METRIC",
                             "business_definition": "Total amount paid by the customer."},
    }}
    re_mod.run_nl_answer("list top customers", columns, rows,
                        table="customers", semantic_model=semantic_model)

    assert "Column meanings" in captured["prompt"]
    assert "Total amount paid" in captured["prompt"]


# ---------------------------------------------------------------------------
# run_nl_answer now phrases EVERY non-empty result via the SLM (including
# "simple" scalar/single-row shapes) — but only ever sends the precomputed
# facts payload, never the raw rows, so cost stays flat regardless of size.
# ---------------------------------------------------------------------------

def test_empty_result_never_calls_slm(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    def must_not_be_called(prompt, **kwargs):
        raise AssertionError("SLM should not be called for an empty result")

    monkeypatch.setattr(slm, "call_slm", must_not_be_called)
    result = re_mod.run_nl_answer("how many users", ["count"], [])
    assert result.answer == "No results found."


def test_scalar_result_now_goes_through_slm(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "There are 137 open incidents."

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    result = re_mod.run_nl_answer("how many open incidents", ["count"], [{"count": 137}])

    assert result.answer == "There are 137 open incidents."   # SLM phrasing, not the canned template
    assert "Extracted data" in captured["prompt"]
    assert '"row_count": 1' in captured["prompt"]
    assert '"count": 137' in captured["prompt"]


def test_scalar_slm_failure_falls_back_to_template_not_generic_count(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda prompt, **kwargs: (_ for _ in ()).throw(
        RuntimeError("SLM unreachable")))
    result = re_mod.run_nl_answer("how many open incidents", ["count"], [{"count": 137}])

    # falls back to the nice deterministic phrasing, not the generic "Returned 1 row(s)"
    assert result.answer == "The count is 137."


def test_rank_column_hint_reaches_the_prompt(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "The largest entries are led by row 684 with amount 9000."

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    columns = ["id", "label", "amount"]
    rows = [{"id": 684, "label": "hello", "amount": 9000},
            {"id": 734, "label": "kj", "amount": 8000}]
    re_mod.run_nl_answer("top 2 largest ledger entries", columns, rows, rank_column="amount")

    assert '"ranked_by": "amount"' in captured["prompt"]
    assert "amount" in captured["prompt"].lower()


def test_rank_column_included_even_if_outside_top_six_columns():
    from query.result_explainer import _extract_facts
    columns = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "amount"]
    rows = [{"c1": 1, "c2": 2, "c3": 3, "c4": 4, "c5": 5, "c6": 6, "c7": 7, "amount": 999}]
    facts = _extract_facts(columns, rows, rank_column="amount")
    assert facts["fields"].get("amount") == 999
    assert facts["ranked_by"] == "amount"


def test_facts_payload_stays_compact_for_large_results(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "Summary of a large result."

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    columns = ["id", "name"]
    rows = [{"id": i, "name": f"item{i}"} for i in range(500)]
    re_mod.run_nl_answer("list all items", columns, rows)

    prompt = captured["prompt"]
    assert '"row_count": 500' in prompt
    # only a handful of sample rows are ever embedded, never all 500
    assert prompt.count('"id":') <= re_mod._FACTS_SAMPLE_ROWS
    assert "item499" not in prompt   # the tail of a 500-row result is never dumped verbatim


# ---------------------------------------------------------------------------
# Backward-compat shim — query/nl_answer.py still works unchanged
# ---------------------------------------------------------------------------

def test_nl_answer_shim_reexports():
    from query.nl_answer import template_answer, deterministic_fallback_answer, run_nl_answer
    from query import result_explainer
    assert template_answer is result_explainer.template_answer
    assert deterministic_fallback_answer is result_explainer.deterministic_fallback_answer
    assert run_nl_answer is result_explainer.run_nl_answer


# ---------------------------------------------------------------------------
# NoSQL answer contract fix (was: computed then discarded)
# ---------------------------------------------------------------------------

def test_queryresult_carries_answer_field():
    from connectors.base import QueryResult
    res = QueryResult(source_id="1", source_type="nosql", rows=[], row_count=0,
                      columns=[], sql_or_query="{}", duration_ms=1.0, truncated=False,
                      error=None)
    assert res.answer is None   # backward-compatible default
    res.answer = "Found 3 matching documents."
    assert res.answer == "Found 3 matching documents."


# ---------------------------------------------------------------------------
# Insight Engine (Phase 4) — one combined SLM call for summary + insights +
# visualization suggestion + follow-ups. Replaces run_nl_answer's own SLM
# call for the same query — never both (see veda/pipeline.py's _done()).
# ---------------------------------------------------------------------------

def _ctx(question="who are the top spenders", sm=None):
    from veda.result_analyzer import analyze_result
    sql = ('SELECT payer_name, SUM(amount) AS total FROM ledger '
           'GROUP BY payer_name ORDER BY total DESC LIMIT 5')
    rows = [{"payer_name": "Alice", "total": 500}, {"payer_name": "Bob", "total": 300},
            {"payer_name": "Carl", "total": 200}]
    return analyze_result(question, sql, ["payer_name", "total"], rows, sm=sm, table="ledger")


def test_insight_engine_empty_result_never_calls_slm(monkeypatch):
    import slm
    from veda.result_analyzer import analyze_result
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("SLM should not be called for an empty result")))
    ctx = analyze_result("nothing here", "SELECT * FROM ledger WHERE 1=0", ["id"], [],
                        sm=None, table="ledger")
    result = re_mod.run_insight_engine(ctx)
    assert result.answer == "No results found."
    assert result.insights == [] and result.visualization is None


def test_insight_engine_uses_small_model_and_json_format(monkeypatch):
    import json
    import slm
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured.update(kwargs)
        captured["prompt"] = prompt
        return json.dumps({
            "summary": "Alice spent the most at 500.",
            "insights": ["Alice's total is over 60% more than Bob's."],
            "visualization": {"type": "bar", "x_axis": "payer_name", "y_axis": "total",
                             "reason": "category vs numeric"},
            "follow_up_questions": ["Who are the bottom payers by total?",
                                    "Show customer churn by campaign"],
        })

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    import config
    monkeypatch.setattr(config, "INSIGHT_FOLLOW_UPS_ENABLED", True, raising=False)
    result = re_mod.run_insight_engine(_ctx())

    assert result.answer == "Alice spent the most at 500."
    assert result.insights == ["Alice's total is over 60% more than Bob's."]
    assert result.visualization["type"] == "bar"
    # groundedness gate (validate_follow_up_questions): the follow-up naming a
    # real column survives; the one inventing business concepts is dropped.
    assert result.follow_up_questions == ["Who are the bottom payers by total?"]
    assert captured["json_format"] is True
    assert captured["model"] == re_mod.NL_SUMMARY_MODEL
    assert captured["model"] != "qwen2.5-coder:7b"


def test_insight_engine_falls_back_on_slm_failure(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("SLM down")))
    result = re_mod.run_insight_engine(_ctx())

    assert result.answer.startswith("Returned 3 row(s).")
    assert result.insights == [] and result.visualization is None and result.follow_up_questions == []


def test_insight_engine_falls_back_on_malformed_json(monkeypatch):
    import slm
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: "not json at all")
    result = re_mod.run_insight_engine(_ctx())

    assert result.answer.startswith("Returned 3 row(s).")


def test_insight_engine_drops_visualization_with_unknown_column(monkeypatch):
    import json
    import slm
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: json.dumps({
        "summary": "ok", "insights": [], "follow_up_questions": [],
        "visualization": {"type": "bar", "x_axis": "does_not_exist", "y_axis": "total"},
    }))
    result = re_mod.run_insight_engine(_ctx())
    assert result.visualization is None


def test_insight_engine_drops_visualization_for_non_multi_row(monkeypatch):
    import json
    import slm
    from veda.result_analyzer import analyze_result
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: json.dumps({
        "summary": "137 open incidents.", "insights": [], "follow_up_questions": [],
        "visualization": {"type": "bar", "x_axis": "count", "y_axis": "count"},
    }))
    ctx = analyze_result("how many open incidents", "SELECT COUNT(*) AS count FROM incidents",
                        ["count"], [{"count": 137}], sm=None, table="incidents")
    result = re_mod.run_insight_engine(ctx)
    assert result.visualization is None   # no chart for a scalar/count result


def test_insight_engine_uses_semantic_metadata_in_prompt(monkeypatch):
    import json
    import slm
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured["prompt"] = prompt
        return json.dumps({"summary": "ok", "insights": [], "follow_up_questions": []})

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    sm = {"columns": {"ledger.total": {"analytics_role": "METRIC",
                                       "business_definition": "Total amount paid."}}}
    re_mod.run_insight_engine(_ctx(sm=sm))
    assert "Column meanings" in captured["prompt"]
    assert "Total amount paid" in captured["prompt"]


# ---------------------------------------------------------------------------
# Final Polish: prompt grounds insights in precomputed statistics (not just
# raw sample rows), and pushes for analytical (not descriptive) summaries.
# ---------------------------------------------------------------------------

def test_insight_engine_prompt_includes_precomputed_statistics(monkeypatch):
    import json
    import slm
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured["prompt"] = prompt
        return json.dumps({"summary": "ok", "insights": [], "follow_up_questions": []})

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    re_mod.run_insight_engine(_ctx())
    assert "Statistics" in captured["prompt"]
    assert "range 200-500" in captured["prompt"]   # from the fixture's total column (200/300/500)


def test_insight_engine_prompt_discourages_descriptive_summaries(monkeypatch):
    import json
    import slm
    from query import result_explainer as re_mod

    captured = {}
    monkeypatch.setattr(slm, "call_slm", lambda p, **k: (captured.setdefault("prompt", p),
                                                         json.dumps({"summary": "ok", "insights": [],
                                                                    "follow_up_questions": []}))[1])
    re_mod.run_insight_engine(_ctx())
    assert "analytical" in captured["prompt"].lower()
    assert "Never restate LIMIT/COUNT/SQL mechanics" in captured["prompt"]


def test_insight_engine_prompt_excludes_identifier_columns_from_stats(monkeypatch):
    import json
    import slm
    from veda.result_analyzer import analyze_result
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured["prompt"] = prompt
        return json.dumps({"summary": "ok", "insights": [], "follow_up_questions": []})

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    sql = 'SELECT id, amount FROM ledger ORDER BY amount DESC LIMIT 5'
    ctx = analyze_result("top ledger entries", sql, ["id", "amount"],
                        [{"id": 1, "amount": 500}, {"id": 2, "amount": 300},
                         {"id": 3, "amount": 200}], sm=None, table="ledger")
    re_mod.run_insight_engine(ctx)
    assert "- id (identifier)" not in captured["prompt"]
    assert "- amount (measure)" in captured["prompt"]


# ---------------------------------------------------------------------------
# Phase 2 gap-fill: confidence synthesized from ctx.confidence_inputs (never
# invented, never the SLM's own self-report), and result_shape hint reaching
# the prompt so insights are shape-aware (ranking/trend/distribution/etc).
# ---------------------------------------------------------------------------

def test_insight_engine_confidence_from_context_weakest_link(monkeypatch):
    import json
    import slm
    from veda.result_analyzer import analyze_result
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: json.dumps(
        {"summary": "ok", "insights": [], "follow_up_questions": []}))
    sql = ('SELECT payer_name, SUM(amount) AS total FROM ledger '
           'GROUP BY payer_name ORDER BY total DESC LIMIT 5')
    ctx = analyze_result("q", sql, ["payer_name", "total"],
                        [{"payer_name": "A", "total": 1}, {"payer_name": "B", "total": 2}],
                        sm=None, table="ledger",
                        confidence_inputs={"anchor": 0.9, "join": 0.4})
    result = re_mod.run_insight_engine(ctx)
    assert result.confidence == 0.4   # weakest link, not an average


def test_insight_engine_confidence_defaults_to_one_when_no_inputs(monkeypatch):
    import json
    import slm
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: json.dumps(
        {"summary": "ok", "insights": [], "follow_up_questions": []}))
    result = re_mod.run_insight_engine(_ctx())   # no confidence_inputs supplied
    assert result.confidence == 1.0


def test_insight_engine_confidence_present_even_on_slm_failure(monkeypatch):
    import slm
    from veda.result_analyzer import analyze_result
    from query import result_explainer as re_mod

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("down")))
    ctx = analyze_result("q", "SELECT id FROM t", ["id"], [{"id": 1}, {"id": 2}],
                        sm=None, table="t", confidence_inputs={"anchor": 0.7})
    result = re_mod.run_insight_engine(ctx)
    assert result.confidence == 0.7   # confidence is ctx-derived, independent of SLM outcome


def test_insight_engine_prompt_includes_result_shape_hint(monkeypatch):
    import json
    import slm
    from veda.result_analyzer import analyze_result
    from query import result_explainer as re_mod

    captured = {}

    def fake_call_slm(prompt, **kwargs):
        captured["prompt"] = prompt
        return json.dumps({"summary": "ok", "insights": [], "follow_up_questions": []})

    monkeypatch.setattr(slm, "call_slm", fake_call_slm)
    sql = 'SELECT id, amount FROM ledger ORDER BY amount DESC LIMIT 10'
    ctx = analyze_result("top 10", sql, ["id", "amount"],
                        [{"id": 1, "amount": 5}, {"id": 2, "amount": 3}], sm=None, table="ledger")
    assert ctx.result_shape == "RANKING"
    re_mod.run_insight_engine(ctx)
    assert "RANKING" in captured["prompt"]
