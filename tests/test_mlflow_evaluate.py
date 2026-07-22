"""Model-free tests for the golden-eval harness (mlflow_observability/evaluate.py).

No SLM, no engine, no MLflow server — the query runner is INJECTED as a stub, so
scoring + aggregation logic is verified deterministically. The live runner
(inference_run_query_fn) is exercised only in a real eval run.

Run from repo root: ``pytest tests/test_mlflow_evaluate.py``
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "mlflow_observability"))


# ── load_golden ────────────────────────────────────────────────────────────────
def test_load_golden_skips_blank_and_bad_lines(tmp_path):
    from evaluate import load_golden
    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps({"id": "g1", "query": "how many x", "gold_tables": ["t_x"]}) + "\n"
        "\n"                                          # blank
        "# a comment\n"                               # comment
        "{not json}\n"                                # malformed → skipped
        + json.dumps({"query": "q2", "gold_tables": []}) + "\n"
        + json.dumps({"gold_tables": ["t"]}) + "\n",  # no query → skipped
        encoding="utf-8")
    cases = load_golden(str(p))
    assert [c.get("query") for c in cases] == ["how many x", "q2"]


# ── tables_used ─────────────────────────────────────────────────────────────────
def test_tables_used_from_sql_and_primary_table():
    from evaluate import tables_used
    er = {"table": "assets_asset",
          "sql": 'SELECT a.x FROM assets_asset a JOIN assets_project b ON a.pid=b.id'}
    assert tables_used(er) == {"assets_asset", "assets_project"}


# ── score_case ───────────────────────────────────────────────────────────────────
def test_score_case_pass_when_answered_and_table_hit():
    from evaluate import score_case
    case = {"id": "g1", "query": "q", "gold_tables": ["accounts_paymenttransaction"]}
    er = {"status": "answered", "table": "accounts_paymenttransaction",
          "sql": "SELECT COUNT(*) FROM accounts_paymenttransaction",
          "usage": {"total_tokens": 120}, "latency_ms": 2100}
    r = score_case(case, er)
    assert r.answered and r.table_hit and r.passed
    assert r.total_tokens == 120 and r.latency_ms == 2100


def test_score_case_fail_when_wrong_table():
    from evaluate import score_case
    case = {"query": "q", "gold_tables": ["accounts_paymenttransaction"]}
    er = {"status": "answered", "table": "assets_asset",
          "sql": "SELECT COUNT(*) FROM assets_asset"}
    r = score_case(case, er)
    assert r.answered and not r.table_hit and not r.passed


def test_score_case_refused_is_not_pass():
    from evaluate import score_case
    r = score_case({"query": "q", "gold_tables": ["t"]},
                   {"status": "no_table"})
    assert not r.answered and not r.passed


def test_score_case_no_gold_tables_passes_on_answered():
    from evaluate import score_case
    r = score_case({"query": "q", "gold_tables": []},
                   {"status": "answered", "sql": "SELECT 1"})
    assert r.table_hit and r.passed          # nothing to check → answered is enough


# ── aggregate ────────────────────────────────────────────────────────────────────
def test_aggregate_metrics():
    from evaluate import score_case, aggregate
    cases_results = [
        score_case({"query": "q1", "gold_tables": ["t1"]},
                   {"status": "answered", "table": "t1", "sql": "SELECT * FROM t1",
                    "usage": {"total_tokens": 100}, "latency_ms": 1000}),
        score_case({"query": "q2", "gold_tables": ["t2"]},
                   {"status": "answered", "table": "t9", "sql": "SELECT * FROM t9",
                    "usage": {"total_tokens": 200}, "latency_ms": 3000}),   # wrong table
        score_case({"query": "q3", "gold_tables": ["t3"]},
                   {"status": "no_table"}),                                  # refused
    ]
    rep = aggregate(cases_results)
    m = rep.metrics
    assert m["n_cases"] == 3
    assert m["answered_rate"] == round(2 / 3, 4)
    assert m["refused_rate"] == round(1 / 3, 4)
    assert m["pass_rate"] == round(1 / 3, 4)          # only q1 passes
    assert m["table_hit_rate"] == round(1 / 3, 4)     # 1 of 3 gold-table cases hit
    assert m["latency_ms_mean"] == 2000.0
    assert m["latency_ms_p95"] == 3000
    assert m["tokens_total"] == 300.0


# ── evaluate_golden (injected stub runner — no SLM) ──────────────────────────────
def test_evaluate_golden_with_stub_runner():
    from evaluate import evaluate_golden
    cases = [{"query": "how many payments", "gold_tables": ["accounts_paymenttransaction"]},
             {"query": "list assets", "gold_tables": ["assets_asset"]}]

    def stub(query):   # deterministic fake engine — NO SLM
        table = "accounts_paymenttransaction" if "payment" in query else "assets_asset"
        return {"status": "answered", "table": table,
                "sql": f"SELECT * FROM {table}", "usage": {"total_tokens": 50},
                "latency_ms": 1500}

    rep = evaluate_golden(cases, stub)
    assert rep.n_cases == 2 and rep.metrics["pass_rate"] == 1.0


def test_evaluate_golden_runner_error_is_isolated():
    from evaluate import evaluate_golden
    cases = [{"query": "boom", "gold_tables": ["t"]},
             {"query": "ok", "gold_tables": []}]

    def flaky(query):
        if query == "boom":
            raise RuntimeError("engine down")
        return {"status": "answered", "sql": "SELECT 1"}

    rep = evaluate_golden(cases, flaky)          # must NOT abort on the bad case
    assert rep.n_cases == 2
    statuses = {c.query: c.status for c in rep.cases}
    assert statuses["boom"].startswith("harness_error")
    assert rep.metrics["answered_rate"] == 0.5   # only the "ok" case answered


def test_empty_aggregate_safe():
    from evaluate import aggregate
    rep = aggregate([])
    assert rep.n_cases == 0 and rep.metrics["pass_rate"] == 0.0
