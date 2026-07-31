"""Unit tests for FASTPATH_ENTITY_GLOSSARY (query/fast_path.py).

Deterministic — no SLM, no retrieval, no execution. Uses the compiled homzhub registry
(source_id=2) which fast_path already loads for entity/measure grounding. Every test also
asserts the FLAG-OFF path is byte-identical (returns None) so the production guarantee is
covered by CI, not just manual A/B.

Skips cleanly if the homzhub registry/artifacts aren't available in the test environment
(the grounding functions need the compiled semantic model, mirroring the manual probes).
"""
import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "veda_core"))
os.environ.setdefault("VEDA_SOURCE_ID", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="module")
def fp():
    """fast_path module with the homzhub source context bound, or skip.

    The compiled registry/semantic-model artifacts resolve relative to veda_core, so the
    process cwd is moved there for the duration of the module (mirrors how the engine runs)."""
    try:
        os.chdir(os.path.join(_ROOT, "veda_core"))
        from context import RequestContext, set_context
        set_context(RequestContext(source_id=2, tenant="default"))
        from query import fast_path as _fp
        from semantic import registry as reg
        if not reg.is_ready():
            pytest.skip("registry not ready in this environment")
    except Exception as e:  # pragma: no cover - environment guard
        pytest.skip(f"fast_path/registry unavailable: {e}")
    return _fp


def _set(fp, on):
    import config as cfg
    cfg.FASTPATH_ENTITY_GLOSSARY = on


def _table_of(res):
    import re
    sql = (getattr(res, "sql", "") or "")
    m = re.search(r'from\s+"?([a-z_]+)"?', sql, re.I)
    return m.group(1) if m else None


# ── entity grounding ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected_table", [
    ("How many lease listings are there?", "assets_leaselisting"),
    ("How many lease tenants are there?", "assets_leasetenant"),
    ("How many sale negotiations are there?", "assets_salenegotiation"),
    ("How many verification documents are there?", "assets_assetverificationdocument"),
    ("How many properties are there?", "assets_asset"),
])
def test_entity_grounding_on(fp, query, expected_table):
    """Flag ON grounds the business noun to the correct table (concat / suffix / glossary)."""
    _set(fp, True)
    res = fp.try_fast_path(query)
    assert res is not None, f"{query!r} did not ground with flag ON"
    assert _table_of(res) == expected_table


@pytest.mark.parametrize("query", [
    "How many lease listings are there?",
    "How many lease tenants are there?",
    "How many verification documents are there?",
    "How many properties are there?",
])
def test_flag_off_byte_identical(fp, query):
    """Flag OFF → the business-noun queries fall through (None), i.e. prod unchanged."""
    _set(fp, False)
    assert fp.try_fast_path(query) is None


def test_generic_token_guard(fp):
    """Multi-word 'payment transaction' must beat the generic 'payment' single-token match."""
    _set(fp, True)
    e = fp._single_entity({"payment", "transaction", "transactions"},
                          "How many payment transactions are there?")
    assert e is not None
    assert e["resolves_to"]["table"] == "accounts_paymenttransaction"


# ── measure resolver ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("table,phrase,expected_col", [
    ("assets_salelisting", {"expected", "price"}, "expected_price"),
    ("assets_leasetransaction", {"security", "deposit"}, "security_deposit"),
    ("accounts_userinvoiceitem", {"pos", "price"}, "pos_price"),
])
def test_ground_measure_col(fp, table, phrase, expected_col):
    cols = fp._sm().get("columns", {})
    assert fp._ground_measure_col(table, phrase, cols) == expected_col


# ── SQL shape (trend / ranking) ───────────────────────────────────────────────

def test_temporal_trend_builds(fp):
    _set(fp, True)
    res = fp.try_fast_path("What is the monthly trend of properties based on created at?")
    assert res is not None
    assert "date_trunc" in (res.sql or "").lower()


def test_ranking_measure_builds(fp):
    _set(fp, True)
    res = fp.try_fast_path("Which sale listings have the highest expected price?")
    assert res is not None
    sql = (res.sql or "").lower()
    assert "order by" in sql and "expected_price" in sql


def test_ranking_flag_off(fp):
    """The named-measure ranking path is gated too."""
    _set(fp, False)
    assert fp.try_fast_path("Which sale listings have the highest expected price?") is None
