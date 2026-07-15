"""Regression coverage for a production bug found via direct inspection of
the live verified-query cache (substrate_verifiedquerycache): the entry for

    "Which of our properties on the market are priced above 10,000?"

held SQL for a COMPLETELY different question (a "properties in the UAE"
country filter, no price predicate at all) — yet verified_cache_lookup()'s
0.85 cosine-similarity threshold and the pre-existing FASTPATH_EVIDENCE_GUARD
(table-level only: "does this query have ANY typed evidence for the cached
SQL's table?") both let it through, because the cached SQL's table
(assets_asset) genuinely IS the right table — the guard just can't see that
the WHERE clause answers a different question on that same table.

The fix (veda/pipeline.py, right after the existing evidence-guard block for
`cached_sql`) reuses qualifier_completeness() — the SAME gate already applied
to freshly-generated SQL later in this function — to re-validate a cache hit
against the CURRENT query text before serving it. A cache hit whose SQL drops
a qualifier the user actually named is demoted (cached_sql = None), falling
through to real retrieval/planning exactly like a table-evidence miss already
does.

Pure-python: no DB, no LLM, no embedding model. verified_cache_lookup() and
the schema-linking calls are monkeypatched out (same dependency-level pattern
as tests/test_execution_state_reuse.py and tests/test_refusal_explain.py) so
this proves the WIRING in run_query(), not qualifier_completeness() itself
(which has its own tests in tests/test_business_explain.py's neighbors /
veda/validation.py's own suite).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

# The exact corrupted (query, sql) pair found live in substrate_verifiedquerycache.
_BAD_QUERY = "Which of our properties on the market are priced above 10,000?"
_BAD_SQL = ('SELECT "id", "digital_id", "project_name" FROM "assets_asset" '
            'WHERE "country_id" IN (SELECT "id" FROM "generics_country" '
            "WHERE lower(\"iso3_code\"::text) = lower('ARE')) LIMIT 100")


class _FakeEngine:
    def retrieve(self, **kwargs):
        return []


def _quiet_pipeline_deps(monkeypatch, cached=None):
    """Same planner/rerank/expand silencing as test_refusal_explain.py, plus
    stubbing verified_cache_lookup() to return `cached` (sql, similarity).
    The pre-existing table-evidence guard is turned OFF here so each test
    below isolates the NEW qualifier gate specifically — that older guard
    already has no coverage for this bug (see module docstring) and is
    exercised on its own in other tests."""
    import config
    import veda.pipeline as pipeline

    monkeypatch.setattr(pipeline, "verified_cache_lookup", lambda q: cached or (None, 0.0))
    monkeypatch.setattr(pipeline, "get_engine", lambda sm: _FakeEngine())
    monkeypatch.setattr(config, "FASTPATH_EVIDENCE_GUARD", False)
    monkeypatch.setattr(config, "GRAPH_EXPAND_ENABLED", False)
    monkeypatch.setattr(config, "PRIMARY_RERANK_ENABLED", False)
    monkeypatch.setattr(config, "FAST_PATH_ENABLED", False)
    monkeypatch.setattr(config, "SUPERLATIVE_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "GROUPED_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "RATIO_PLAN_ENABLED", False)
    return pipeline


def test_cache_hit_with_dropped_qualifier_is_demoted(monkeypatch):
    """The exact production bug: a cache hit whose SQL doesn't cover a
    qualifier the user named ('above [10,000]') must be discarded, falling
    through to real retrieval — proven here by forcing that fallback path to
    a deterministic 'no_table' (select_primary_table/vet_primary stubbed to
    None) rather than letting the corrupted cached SQL answer the query."""
    pipeline = _quiet_pipeline_deps(monkeypatch, cached=(_BAD_SQL, 0.9))
    monkeypatch.setattr(pipeline, "select_primary_table", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "vet_primary", lambda *a, **k: None)

    res = pipeline.run_query(_BAD_QUERY, sm={}, all_cols=[], return_result=True)

    # If the cache hit had NOT been demoted, execution would have gone down
    # the `elif cached_sql:` lane instead (never touching select_primary_table/
    # vet_primary at all) — "no_table" is only reachable via the real
    # retrieval branch, so this status is unambiguous proof of demotion.
    assert res["status"] == "no_table"


def test_legitimate_cache_hit_is_not_falsely_demoted(monkeypatch):
    """Regression guard: the new gate must not demote a cache hit whose SQL
    DOES cover the query's qualifiers — proven by making the real-retrieval
    fallback functions raise if reached at all; a passing cache hit must
    short-circuit before ever calling them."""
    def _must_not_be_called(*a, **k):
        raise AssertionError("legitimate cache hit was demoted — false positive")

    legit_sql = "SELECT COUNT(*) FROM accounts_paymenttransaction LIMIT 100"
    pipeline = _quiet_pipeline_deps(monkeypatch, cached=(legit_sql, 0.97))
    monkeypatch.setattr(pipeline, "select_primary_table", _must_not_be_called)
    monkeypatch.setattr(pipeline, "vet_primary", _must_not_be_called)
    # The cache lane still runs SQL through validation + execution like any
    # other answer — stub execute_sql so this stays DB-free (a legitimate
    # cache hit reaching real execution is the whole point being proven, but
    # a live database is not part of this test's contract).
    monkeypatch.setattr(pipeline, "execute_sql",
                        lambda sql, params=None: (["count"], [(137,)], None))

    res = pipeline.run_query("how many transactions are there", sm={}, all_cols=[],
                             return_result=True)

    assert res["status"] == "answered"
    # Not a byte-identical comparison: validate_and_parameterize's AST rewrite
    # legitimately re-quotes identifiers (accounts_paymenttransaction ->
    # "accounts_paymenttransaction") even for an already-valid cached query —
    # the content, not the exact text, proves the cached SQL survived intact.
    assert "COUNT(*)" in res["sql"]
    assert "accounts_paymenttransaction" in res["sql"]
