"""Tests for query.agents — per-source-type execution agents (routing Phase 2.5).

Delegates are monkeypatched so no DB/model is needed. Run: `pytest tests/test_source_agents.py`.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

import query.agents as A  # noqa: E402


# ── registry ──────────────────────────────────────────────────────────────────
def test_registry_resolves_by_kind():
    assert isinstance(A.resolve_agent("relational"), A.DatabaseAgent)
    assert isinstance(A.resolve_agent("datalake"), A.DataLakeAgent)
    assert isinstance(A.resolve_agent("document"), A.FileSystemAgent)
    assert isinstance(A.resolve_agent("nosql"), A.NoSqlAgent)


def test_unknown_kind_returns_none():
    assert A.resolve_agent("graphdb") is None      # never silently guess an agent
    assert A.resolve_agent("") is None


# ── DB/datalake agent (deterministic SQL) ──────────────────────────────────────
def test_db_agent_ok(monkeypatch):
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": True, "cols": ["c"], "rows": [[1]], "sql": "SELECT 1", "table": "t", "explain": {}})
    r = A.DatabaseAgent().execute("q", source_id="5", sm={}, cols=[])
    assert r.status == A.STATUS_OK and r.engine == "deterministic_sql" and r.data["rows"] == [[1]]


def test_db_agent_clarify_is_refused_not_failed(monkeypatch):
    monkeypatch.setattr(A, "_sql_delegate", lambda q, sm, cols, on_event=None: {
        "ok": False, "status": "clarify", "answer": "which region?"})
    r = A.DatabaseAgent().execute("q", source_id="5", sm={}, cols=[])
    assert r.status == A.STATUS_REFUSED and "region" in r.reason


def test_db_agent_missing_sm_is_failed_not_crash():
    r = A.DatabaseAgent().execute("q", source_id="5")
    assert r.status == A.STATUS_FAILED and "semantic model" in r.error


def test_datalake_uses_same_sql_path(monkeypatch):
    seen = {}
    def deleg(q, sm, cols, on_event=None):
        seen["ran"] = True
        return {"ok": True, "cols": [], "rows": []}
    monkeypatch.setattr(A, "_sql_delegate", deleg)
    r = A.DataLakeAgent().execute("q", source_id="7", sm={}, cols=[])
    assert seen.get("ran") and r.status == A.STATUS_OK and r.source_type == "datalake"


# ── document agent (RAG) ───────────────────────────────────────────────────────
class _FakeRag:
    def __init__(self):
        self.answer, self.citations, self.error = "per policy...", ["SLA.pdf"], None


def test_filesystem_agent_ok(monkeypatch):
    monkeypatch.setattr(A, "_rag_delegate", lambda q, sids, on_event=None: _FakeRag())
    r = A.FileSystemAgent().execute("q", source_ids=["9"], source_id="9")
    assert r.status == A.STATUS_OK and r.data["citations"] == ["SLA.pdf"]


def test_agent_exception_is_guarded(monkeypatch):
    def boom(q, sids, on_event=None):
        raise RuntimeError("boom")
    monkeypatch.setattr(A, "_rag_delegate", boom)
    r = A.FileSystemAgent().execute("q", source_id="9")
    assert r.status == A.STATUS_FAILED and "boom" in r.error


if __name__ == "__main__":
    # minimal runner without pytest (provides a stub monkeypatch)
    import traceback

    class _MP:
        def __init__(self): self._undo = []
        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for obj, name, val in reversed(self._undo):
                setattr(obj, name, val)

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        mp = _MP()
        try:
            fn(mp) if fn.__code__.co_argcount else fn()
            print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
        finally:
            mp.undo()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
