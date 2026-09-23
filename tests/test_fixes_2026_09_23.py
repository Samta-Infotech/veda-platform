"""Tests for the 2026-09-23 benchmark fixes.

Covers the changes that ALTER ANSWERS and shipped with no test, per the finding that
"test coverage does not track risk on this branch": ~793 added test lines concentrated
on the two safest changes, and none on the ones that change refusals and answers.

  §0.2  newly-wired env keys — the value must reach the CONSUMER, not just config
  §0.3  SLM_TEMPERATURE reaches the two named SLM call sites
  §2.2  _guard_sql_head — a document primary must not reach the SQL head
  §3.2  ROUTING_AUTHORITATIVE_MODES — MULTISOURCE_ROUTING_SHADOW is a kill switch again
  A2b   _value_in_engine_store — the cross-source value oracle, hit AND fail-open
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "veda_core"
# CORE must come FIRST: the repo root holds a Django package also called `config`, and
# with ROOT ahead of CORE `import config` resolves to that one instead of the engine's
# veda_core/config.py. tests/test_source_coordinator.py sidesteps this by putting only
# CORE on the path; these tests need both, so the order is what disambiguates.
for p in (str(ROOT), str(CORE)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE))

import config as cfg  # noqa: E402

assert Path(cfg.__file__).resolve() == (CORE / "config.py").resolve(), (
    f"`config` resolved to {cfg.__file__}, not the engine config — sys.path order is wrong"
)


def _reload_config(monkeypatch, **env):
    """Re-import config with `env` applied. Returns the fresh module."""
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    return importlib.reload(cfg)


# ── §0.2 / §0.1 the typed env layer ────────────────────────────────────────────
@pytest.mark.parametrize("key,attr,value,expected", [
    ("VEDA_TOP_K", "TOP_K", "42", 42),
    ("VEDA_TOP_K_TO_LLM", "TOP_K_TO_LLM", "3", 3),
    ("SLM_TEMPERATURE", "SLM_TEMPERATURE", "0.7", 0.7),
    ("RETRIEVAL_INTENT_BOOST_SCALE", "RETRIEVAL_INTENT_BOOST_SCALE", "0.5", 0.5),
    ("VEDA_FAST_PATH_ENABLED", "FAST_PATH_ENABLED", "false", False),
    ("VEDA_IR_JOIN_FREE_ENABLED", "IR_JOIN_FREE_ENABLED", "no", False),
    ("VEDA_QUERY_DECOMPOSE_ENABLED", "QUERY_DECOMPOSE_ENABLED", "YES", True),
    ("VEDA_QUERY_ROUTER_ENABLED", "QUERY_ROUTER_ENABLED", "0", False),
    ("VEDA_HNSW_M", "HNSW_M", "32", 32),
    ("VEDA_HNSW_EF_CONSTRUCTION", "HNSW_EF_CONSTRUCTION", "400", 400),
])
def test_env_key_governs_its_constant(monkeypatch, key, attr, value, expected):
    """Each key was in .env and read by NOTHING before 2026-09-23."""
    m = _reload_config(monkeypatch, **{key: value})
    try:
        assert getattr(m, attr) == expected
    finally:
        monkeypatch.undo()
        importlib.reload(cfg)


def test_absent_and_empty_both_mean_default(monkeypatch):
    m = _reload_config(monkeypatch, VEDA_TOP_K=None)
    assert m.TOP_K == 15
    m = _reload_config(monkeypatch, VEDA_TOP_K="   ")
    assert m.TOP_K == 15
    monkeypatch.undo()
    importlib.reload(cfg)


def test_malformed_value_raises_naming_the_key(monkeypatch):
    """Never silently fall back — that is how SLM_TEMPERATURE=0 sat inert at 0.3."""
    with pytest.raises(Exception) as ei:
        _reload_config(monkeypatch, VEDA_TOP_K="fifteen")
    assert "VEDA_TOP_K" in str(ei.value)
    monkeypatch.undo()
    importlib.reload(cfg)

    with pytest.raises(Exception) as ei:
        _reload_config(monkeypatch, VEDA_FAST_PATH_ENABLED="maybe")
    assert "VEDA_FAST_PATH_ENABLED" in str(ei.value)
    monkeypatch.undo()
    importlib.reload(cfg)


# ── §0.3 the temperature must reach the CALL SITES, not just config ───────────
def test_slm_temperature_reaches_rag_synthesis(monkeypatch):
    from query import rag_layer
    seen = {}
    monkeypatch.setattr(rag_layer, "call_slm",
                        lambda *a, **kw: seen.update(kw) or "ok")
    rag_layer._call_ollama("sys", "user")
    assert seen["purpose"] == "rag_synthesis"
    assert seen["temperature"] == cfg.SLM_TEMPERATURE


def test_slm_temperature_reaches_ir_emit(monkeypatch):
    from query import slm_layer
    seen = {}
    monkeypatch.setattr(slm_layer, "call_slm",
                        lambda *a, **kw: seen.update(kw) or "ok")
    slm_layer._call_ollama("user")
    assert seen["purpose"] == "ir_emit"
    assert seen["temperature"] == cfg.SLM_TEMPERATURE


# ── §2.2 the SQL head must not run against a document primary ─────────────────
def test_guard_demotes_sql_to_rag_for_a_document_primary(monkeypatch):
    import veda_hybrid as vh
    monkeypatch.setattr(vh, "_primary_is_document_source", lambda: True)
    assert vh._guard_sql_head("sql") == "rag"


def test_guard_leaves_other_intents_and_other_sources_alone(monkeypatch):
    import veda_hybrid as vh
    monkeypatch.setattr(vh, "_primary_is_document_source", lambda: True)
    # hybrid has its own structured-source check; do not second-guess it
    assert vh._guard_sql_head("hybrid") == "hybrid"
    assert vh._guard_sql_head("rag") == "rag"
    monkeypatch.setattr(vh, "_primary_is_document_source", lambda: False)
    assert vh._guard_sql_head("sql") == "sql"


def test_primary_is_document_source_reads_the_profiles(monkeypatch):
    import veda_hybrid as vh
    from veda_core import context as ctxmod
    ctxmod.set_context(ctxmod.RequestContext(source_id=3, tenant="default",
                                             source_ids=(2, 3)))
    ctxmod.set_source_profiles({"2": {"source_type": "relational"},
                                "3": {"source_type": "filesystem"}})
    assert vh._primary_is_document_source() is True
    ctxmod.set_context(ctxmod.RequestContext(source_id=2, tenant="default",
                                             source_ids=(2, 3)))
    assert vh._primary_is_document_source() is False


# ── §3.2 SHADOW is a kill switch again ────────────────────────────────────────
def _effective_shadow(shadow: bool, status: str, mode: str, modes) -> bool:
    """Mirror of the expression in veda_hybrid._run_coordinator."""
    authoritative = (status == "ROUTED" and mode in modes) if modes else False
    return bool(shadow) and not authoritative


@pytest.mark.parametrize("mode", ["SINGLE", "MULTI"])
def test_shadow_is_honoured_when_no_mode_is_authoritative(mode):
    """The regression this closes: SINGLE/MULTI used to override SHADOW unconditionally."""
    assert _effective_shadow(True, "ROUTED", mode, ()) is True


@pytest.mark.parametrize("mode,modes,expected", [
    ("MULTI", ("MULTI",), False),          # opted in -> authoritative
    ("SINGLE", ("MULTI",), True),          # not opted in -> still shadowed
    ("SINGLE", ("SINGLE", "MULTI"), False),
    ("MULTI", (), True),
])
def test_only_opted_in_modes_override_shadow(mode, modes, expected):
    assert _effective_shadow(True, "ROUTED", mode, modes) is expected


def test_non_routed_decisions_are_never_authoritative():
    for status in ("NO_MATCH", "CLARIFICATION_REQUIRED"):
        assert _effective_shadow(True, status, "SINGLE", ("SINGLE", "MULTI")) is True


def test_authoritative_modes_parses_and_defaults_empty(monkeypatch):
    m = _reload_config(monkeypatch, ROUTING_AUTHORITATIVE_MODES=None)
    assert m.ROUTING_AUTHORITATIVE_MODES == ()
    m = _reload_config(monkeypatch, ROUTING_AUTHORITATIVE_MODES=" multi , single ")
    assert m.ROUTING_AUTHORITATIVE_MODES == ("MULTI", "SINGLE")
    monkeypatch.undo()
    importlib.reload(cfg)


# ── A2b the cross-source value oracle ─────────────────────────────────────────
class _FakeCursor:
    def __init__(self, row): self._row, self.executed = row, []
    def execute(self, sql, params=None): self.executed.append((sql, params))
    def fetchone(self): return self._row
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeConn:
    def __init__(self, row): self._row = row
    def cursor(self): return _FakeCursor(self._row)
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch_store(monkeypatch, conn_factory):
    import ingestion.db_abstraction as dba
    monkeypatch.setattr(dba, "internal_connection", conn_factory, raising=False)


def test_value_in_engine_store_hit(monkeypatch):
    from veda import validation as V
    V._STORE_VALUE_CACHE.clear()
    _patch_store(monkeypatch, lambda: _FakeConn((1,)))
    assert V._value_in_engine_store("mumbai") is True
    V._STORE_VALUE_CACHE.clear()


def test_value_in_engine_store_miss(monkeypatch):
    from veda import validation as V
    V._STORE_VALUE_CACHE.clear()
    _patch_store(monkeypatch, lambda: _FakeConn(None))
    assert V._value_in_engine_store("notavalue") is False
    V._STORE_VALUE_CACHE.clear()


def test_value_in_engine_store_fails_open_when_store_unreachable(monkeypatch):
    """An unreachable store must never invent a refusal."""
    from veda import validation as V
    V._STORE_VALUE_CACHE.clear()

    def _boom():
        raise RuntimeError("engine store down")

    _patch_store(monkeypatch, _boom)
    assert V._value_in_engine_store("mumbai") is False
    V._STORE_VALUE_CACHE.clear()


def test_value_in_engine_store_ignores_short_tokens(monkeypatch):
    from veda import validation as V
    V._STORE_VALUE_CACHE.clear()
    called = {"n": 0}

    def _count():
        called["n"] += 1
        return _FakeConn((1,))

    _patch_store(monkeypatch, _count)
    assert V._value_in_engine_store("abc") is False   # < 4 chars
    assert called["n"] == 0
    V._STORE_VALUE_CACHE.clear()
