"""Regression test for veda/pipeline.py's verified-cache-hit branch: the
resolved `table` field (which flows into chatbot/memory/frame.py's
harvest_frame() as QueryFrame["entity"]) must be a real table name, never the
literal placeholder string "(cached)" it used to be. Pure-python: the DB
(execute_sql) and cache backend (verified_cache_lookup) are monkeypatched;
qualifier_completeness/value_grounding run for real (no DB call for a
WHERE-less query — value_grounding only opens a connection when it finds a
string-literal predicate to check, and this query has none)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


def test_cache_hit_table_is_real_name_not_placeholder(monkeypatch):
    import veda.pipeline as pipeline

    monkeypatch.setattr(pipeline, "verified_cache_lookup",
                        lambda query: ("SELECT COUNT(*) FROM ledger", 0.97))
    monkeypatch.setattr(pipeline, "execute_sql", lambda sql, params: (["count"], [(42,)], None))
    monkeypatch.setattr(pipeline, "save_verified_query", lambda *a, **k: None)

    sm = {"tables": {"ledger": {"primary_entity": "A ledger entry."}}, "columns": {}}
    all_cols = ["ledger.id", "ledger.amount"]

    result = pipeline.run_query("how many ledger entries are there", sm, all_cols,
                                return_result=True)

    assert result["status"] == "answered", result
    assert result["table"] == "ledger"
    assert result["table"] != "(cached)"


def test_cache_hit_table_multi_table_picks_deterministically_not_placeholder(monkeypatch):
    """A cached query spanning more than one table: still never the placeholder
    string — picks a real table name deterministically (first alphabetically)."""
    import veda.pipeline as pipeline

    # A subquery (not a JOIN) references two tables without tripping the
    # separate join-FK-backing gate (ast_readonly_parameterized_fanout) —
    # irrelevant to the bug under test here.
    monkeypatch.setattr(
        pipeline, "verified_cache_lookup",
        lambda query: ("SELECT COUNT(*) FROM ledger WHERE payer_id IN (SELECT id FROM payer)", 0.95),
    )
    monkeypatch.setattr(pipeline, "execute_sql", lambda sql, params: (["count"], [(7,)], None))
    monkeypatch.setattr(pipeline, "save_verified_query", lambda *a, **k: None)

    sm = {"tables": {"ledger": {"primary_entity": "A ledger entry."},
                     "payer": {"primary_entity": "A payer."}},
         "columns": {}}
    all_cols = ["ledger.id", "ledger.payer_id", "payer.id"]

    result = pipeline.run_query("how many ledger entries are there", sm, all_cols,
                                return_result=True)

    assert result["status"] == "answered", result
    assert result["table"] != "(cached)"
    assert result["table"] in ("ledger", "payer")
