"""Wiring-fix pass D.1-D.3 (2026-09-26) — docs/QUERY_PIPELINE_FLOW.md §0, §5, §10.7.

Covers:
  * scripts/_flags.py: --expect mismatch aborts with a readable message, a match
    doesn't, and effective_flags() reads config.py's ATTRIBUTES (not os.environ).
  * veda/execution.py::_execute_duckdb: the DuckDB/parquet path no longer caps a
    result at a bare 20 rows — it now shares EXECUTION_RESULT_LIMIT with the
    psycopg2 path (built with duckdb itself, no fixture DB needed).
  * veda/pipeline.py::_run_query: the WHO-distinct tail and both analytical-list SQL
    builders no longer fall back to a hardcoded 'LIMIT 100' when the user named no
    row count (source-level regression guard — the branches need a full semantic
    model + FK graph to exercise directly, which no lightweight fixture here provides).
  * veda/generation.py::generate_join_sql: the join-fill SLM call's timeout comes
    from config.SLM_JOIN_TIMEOUT_SECS, not a bare hardcoded 120.

Run (per the wiring-fix pass's own instructions):
    docker compose run --rm -T --entrypoint sh inference -c \
        "cd /app && python -m pytest tests/test_integration_d.py -q -p no:cacheprovider"
"""
import inspect
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
SCRIPTS = os.path.abspath(os.path.join(ROOT, "scripts"))
sys.path.insert(0, CORE)
sys.path.insert(0, SCRIPTS)

import pytest  # noqa: E402

import _flags  # noqa: E402 (scripts/_flags.py)
import config  # noqa: E402 (veda_core/config.py)
import veda.execution as execution  # noqa: E402
import veda.generation as generation  # noqa: E402
import veda.pipeline as pipeline  # noqa: E402


# ── scripts/_flags.py ─────────────────────────────────────────────────────────────────

def test_effective_flags_reads_config_attributes_not_environ(monkeypatch):
    """effective_flags() must reflect config.py's ATTRIBUTE (the value the engine
    actually reads), not raw os.environ — the whole point of this module is that the
    two can disagree (docs/QUERY_PIPELINE_FLOW.md §0)."""
    monkeypatch.setenv("FRAME_PATH_ENABLED", "1")            # os.environ says ON
    monkeypatch.setattr(config, "FRAME_PATH_ENABLED", False)  # config.py's value says OFF
    flags = _flags.effective_flags(cfg=config)
    assert flags["FRAME_PATH_ENABLED"] is False


def test_check_expect_match_does_not_raise():
    flags = {"FRAME_PATH_ENABLED": True, "AGENT_JUDGE_MODE": "enforce"}
    _flags.check_expect(flags, {"FRAME_PATH_ENABLED": "1", "AGENT_JUDGE_MODE": "enforce"})


def test_check_expect_mismatch_raises_with_readable_message():
    flags = {"FRAME_PATH_ENABLED": False, "SLM_MODEL_NAME": "qwen2.5-coder:7b"}
    with pytest.raises(_flags.FlagExpectationError) as exc:
        _flags.check_expect(flags, {"FRAME_PATH_ENABLED": "1"})
    msg = str(exc.value)
    assert "FRAME_PATH_ENABLED" in msg
    assert "expected '1'" in msg
    assert "got False" in msg


def test_enforce_expect_mismatch_sys_exits(capsys):
    flags = {"FRAME_PATH_ENABLED": False}
    with pytest.raises(SystemExit) as exc:
        _flags.enforce_expect(flags, ["FRAME_PATH_ENABLED=1"])
    assert "FRAME_PATH_ENABLED" in str(exc.value)


def test_enforce_expect_match_does_not_exit():
    flags = {"FRAME_PATH_ENABLED": True}
    _flags.enforce_expect(flags, ["FRAME_PATH_ENABLED=1"])  # must not raise/exit


def test_parse_expect_rejects_malformed_pair():
    with pytest.raises(_flags.FlagExpectationError):
        _flags.parse_expect(["NOT_A_KEY_VALUE_PAIR"])


# ── veda/execution.py::_execute_duckdb (§10.7 row cap) ──────────────────────────────────

def test_duckdb_path_returns_more_than_20_rows_for_a_25_row_parquet(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    pq_path = tmp_path / "t.parquet"
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t AS SELECT i AS id FROM range(25) AS s(i)")
        con.execute(f"COPY t TO '{pq_path}' (FORMAT PARQUET)")
    finally:
        con.close()

    class _FakeSurface:
        tables = {"t": str(pq_path)}

    cols, rows, err = execution._execute_duckdb(
        "SELECT * FROM t", None, {"fake-source": _FakeSurface()})
    assert err is None
    assert cols == ["id"]
    # The old code hardcoded fetchmany(20) — 25 rows in, at most 20 would come back.
    assert len(rows) == 25, f"expected all 25 rows, got {len(rows)} (old cap was 20)"


def test_duckdb_fetch_cap_is_execution_result_limit(monkeypatch, tmp_path):
    """Same cap as the psycopg2 path, and it must come from config — not a second,
    independently-chosen literal that happens to also be > 20."""
    duckdb = pytest.importorskip("duckdb")
    monkeypatch.setattr(config, "EXECUTION_RESULT_LIMIT", 3)
    pq_path = tmp_path / "t.parquet"
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t AS SELECT i AS id FROM range(25) AS s(i)")
        con.execute(f"COPY t TO '{pq_path}' (FORMAT PARQUET)")
    finally:
        con.close()

    class _FakeSurface:
        tables = {"t": str(pq_path)}

    _, rows, err = execution._execute_duckdb(
        "SELECT * FROM t", None, {"fake-source": _FakeSurface()})
    assert err is None
    assert len(rows) == 3, "expected the fetch to respect config.EXECUTION_RESULT_LIMIT"


# ── veda/pipeline.py::_run_query (§10.7 hardcoded LIMIT 100 tails) ─────────────────────

def _code_only(src: str) -> str:
    """Source with '#'-comment-only lines dropped, so a docstring/comment that
    NAMES the old bug (e.g. this file's own '# Was a hardcoded ...LIMIT 100...'
    notes) doesn't false-positive a check for the literal in live code."""
    return "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))


def test_run_query_has_no_hardcoded_limit_100_fallback():
    """Source-level regression guard: the WHO-distinct tail (`_limit_only_tail`) and
    both analytical-list SQL builders used to default to 'LIMIT 100' when the user
    named no row count. Exercising those branches directly needs a full semantic
    model + FK graph fixture this test suite doesn't have; asserting the literal is
    gone (and the correct conditional is in place) is the lightweight equivalent."""
    src = _code_only(inspect.getsource(pipeline._run_query))
    assert "LIMIT 100" not in src, "a hardcoded 'LIMIT 100' string is back in _run_query"
    assert "else 100" not in src, "a hardcoded LIMIT-100 fallback is back in _run_query"
    assert "top_n or 100" not in src, "the stale 'LIMIT {top_n or 100}' print is back"
    # The WHO-distinct tail + both analytical-list builders + the ranked_temporal_only
    # print all gate their LIMIT on _rank.top_n being explicitly requested.
    assert src.count("_rank.top_n is not None") >= 4, (
        "expected >=4 uses of the '_rank.top_n is not None' no-LIMIT-unless-requested "
        "guard (WHO-distinct tail, 2 analytical builders, ranked_temporal_only print)")


def test_rank_order_limit_sql_emits_no_limit_when_not_requested():
    """The sibling helper (`_rank_order_limit_sql`) already implemented this policy —
    confirms it still holds and anchors what the WHO-distinct tail is matching."""
    from query.ranking_parser import RankingSpec
    rank = RankingSpec(top_n=None, ranked=False, direction="desc", basis=None,
                       sort_requested=False, subject=None)
    tail = pipeline._rank_order_limit_sql(rank, "some_table", {"tables": {}}, None)
    assert "LIMIT" not in tail.upper()


def test_rank_order_limit_sql_honors_an_explicit_count():
    """basis=None + ranked=True with no resolvable sort column deliberately drops the
    LIMIT too (rank_order_limit_sql's docstring: an unresolved ORDER BY makes 'LIMIT N'
    the worst answer, since N arbitrary rows would impersonate a real ranking). A count
    with no ranking language at all ('give me 5 of them') is the case that keeps it."""
    from query.ranking_parser import RankingSpec
    rank = RankingSpec(top_n=5, ranked=False, direction="desc", basis=None,
                       sort_requested=False, subject=None)
    tail = pipeline._rank_order_limit_sql(rank, "some_table", {"tables": {}}, None)
    assert "LIMIT 5" in tail


# ── veda/execution.py print + generation.py join timeout ───────────────────────────────

def test_run_query_execute_print_reads_config_limit():
    """The '[L7] Execute ... fetch <=20' print used to hardcode 20 regardless of the
    real cap; it must now read config.EXECUTION_RESULT_LIMIT."""
    src = _code_only(inspect.getsource(pipeline._run_query))
    assert "fetch ≤{_exec_limit_print}" in src, (
        "the L7 Execute print no longer reads EXECUTION_RESULT_LIMIT")
    assert "fetch ≤20" not in src, "the L7 Execute print still hardcodes 20"


def test_generate_join_sql_timeout_reads_config(monkeypatch):
    """generate_join_sql's SLM call must use config.SLM_JOIN_TIMEOUT_SECS, not a bare
    hardcoded 120 — proven by changing the config value and checking the call_slm
    kwarg actually moves with it."""
    captured = {}

    def _fake_call_slm(user, system=None, purpose=None, **kw):
        captured.update(kw)
        return "SELECT 1"

    monkeypatch.setattr(generation, "call_slm", _fake_call_slm)
    monkeypatch.setattr(generation, "SLM_JOIN_TIMEOUT_SECS", 321)

    sm = {"columns": {}, "domain_synonyms": {}}
    generation.generate_join_sql(
        query="how many rows",
        skeleton='FROM "a" t0',
        alias_map={"t0": "a"},
        sm=sm,
        tf=None,
        results=None,
    )
    assert captured.get("timeout") == 321


def test_slm_join_timeout_secs_is_config_default_120_and_env_overridable(monkeypatch):
    assert config.SLM_JOIN_TIMEOUT_SECS == config._env_int("SLM_JOIN_TIMEOUT_SECS", 120)
    monkeypatch.setenv("SLM_JOIN_TIMEOUT_SECS", "77")
    assert config._env_int("SLM_JOIN_TIMEOUT_SECS", 120) == 77
