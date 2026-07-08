"""Pure-python tests for the layered-ingestion contracts, the SLM seam, and the
Track-4 precompute helpers — no DB, no models, no network. Run from the repo
root: ``pytest tests/test_layers_and_precompute.py`` (veda_core is added to
sys.path below, matching the engine's cwd convention)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


# ---------------------------------------------------------------------------
# contracts.SourceContext — env injection round-trip (§3.1)
# ---------------------------------------------------------------------------

def test_source_context_from_env_json(monkeypatch):
    monkeypatch.setenv("VEDA_SOURCE_JSON", json.dumps({
        "id": "42", "type": "relational", "engine": "postgresql",
        "host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p",
        "exclude_tables": ["client_tbl"],
    }))
    monkeypatch.setenv("VEDA_TENANT", "acme")
    import importlib
    import config
    importlib.reload(config)
    from ingestion.contracts import SourceContext
    ctx = SourceContext.from_env()
    assert ctx.source_id == "42"
    assert ctx.tenant == "acme"
    assert ctx.type == "relational"
    assert "client_tbl" in ctx.exclude_tables            # client row exclusion
    assert "django_migrations" in ctx.exclude_tables     # framework-noise merge


def test_get_source_hard_fails_without_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VEDA_SOURCE"):
            monkeypatch.delenv(k, raising=False)
    import importlib
    import config
    importlib.reload(config)
    with pytest.raises(RuntimeError):
        config.get_source()


# ---------------------------------------------------------------------------
# SLM seam (§10, review Finding 1) — backend selection, no network
# ---------------------------------------------------------------------------

def test_slm_backend_selection(monkeypatch):
    from slm import _call_slm
    monkeypatch.setenv("SLM_BACKEND", "vllm")
    _call_slm.reset_backend()
    assert _call_slm.get_backend().name == "vllm"
    monkeypatch.setenv("SLM_BACKEND", "ollama")
    _call_slm.reset_backend()
    assert _call_slm.get_backend().name == "ollama"
    _call_slm.reset_backend()


def test_slm_unreachable_raises_runtimeerror(monkeypatch):
    from slm import _call_slm
    monkeypatch.setenv("SLM_BACKEND", "ollama")
    monkeypatch.setenv("SLM_OLLAMA_BASE_URL", "http://127.0.0.1:1")  # nothing listens
    _call_slm.reset_backend()
    with pytest.raises(RuntimeError):
        _call_slm.call_slm("ping", purpose="test", timeout=1)
    _call_slm.reset_backend()


# ---------------------------------------------------------------------------
# Q-7 — deterministic NL templates (canonical shapes only)
# ---------------------------------------------------------------------------

def test_nl_template_shapes():
    from query.nl_answer import template_answer
    assert template_answer("how many users", ["count"], []) == "No results found."
    one = template_answer("how many users", ["count"], [{"count": 42}])
    assert one is not None and "42" in one
    # multi-row → None (SLM keeps narrative results)
    multi = template_answer("list users", ["name"], [{"name": "a"}, {"name": "b"}])
    assert multi is None

