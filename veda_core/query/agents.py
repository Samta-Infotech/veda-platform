"""query/agents.py — per-source-type execution agents (multi-source routing, Phase 2.5/2.6).

One thin agent per source KIND (relational / datalake / document / nosql). An agent's ONLY job is
``execute()`` — run the source's *existing* specialized pipeline and normalize its output into a
common ``AgentResult``. Agents do NOT do routing, retrieval scoring, or SQL/RAG logic themselves —
they wrap what already exists (see docs/multisource_routing/MEMORY.md):

    DatabaseAgent / DataLakeAgent → veda.pipeline.run_query   (Tier-1/Tier-2; execution.py routes
                                                               datalake SQL to DuckDB natively)
    FileSystemAgent               → query.rag_layer.run_rag_layer
    NoSqlAgent                    → veda_hybrid._run_nosql

The coordinator (Phase 3) resolves ``source_kind → AgentClass`` via ``resolve_agent`` and calls
``execute``. Evidence reuse (Phase 2.6): ``execute`` accepts pre-retrieved ``evidence`` and passes
the reusable parts through where the underlying pipeline's signature supports it, so the selected
source does not re-run retrieval it already has.

Delegates are module-level and lazily import their pipeline, so this module is import-cheap and
its delegates are monkeypatchable in tests (no DB/model needed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

STATUS_OK = "ok"
STATUS_REFUSED = "refused"
STATUS_FAILED = "failed"


@dataclass
class AgentResult:
    source_id: str
    source_type: str            # source KIND: relational | datalake | document | nosql
    status: str                 # ok | refused | failed
    engine: str = ""            # which underlying pipeline ran (deterministic_sql | rag | nosql)
    data: dict = field(default_factory=dict)   # normalized: cols/rows/answer/citations/sql/table
    provenance: list = field(default_factory=list)
    error: Optional[str] = None
    reason: str = ""            # refusal/clarify reason when not ok


# ── delegates (lazy import; monkeypatch these in tests) ───────────────────────────────────────
def _sql_delegate(query, sm, cols, on_event=None):
    from veda.pipeline import run_query
    return run_query(query, sm, cols, return_result=True, on_event=on_event)


def _rag_delegate(query, source_ids, on_event=None):
    from query.rag_layer import run_rag_layer
    return run_rag_layer(query, source_ids=source_ids, on_event=on_event)


def _nosql_delegate(query, source_ids, on_event=None):
    from veda_hybrid import _run_nosql
    return _run_nosql(query, source_ids, on_event=on_event)


# ── normalization: existing pipeline outputs → AgentResult ────────────────────────────────────
def _from_sql_dict(d, source_id, source_type) -> AgentResult:
    """veda.pipeline.run_query dict → AgentResult."""
    if not isinstance(d, dict):
        return AgentResult(source_id, source_type, STATUS_FAILED, engine="deterministic_sql",
                           error="unexpected run_query return shape")
    status_str = d.get("status")
    if d.get("ok"):
        return AgentResult(
            source_id, source_type, STATUS_OK, engine="deterministic_sql",
            data={k: d.get(k) for k in ("cols", "rows", "answer", "sql", "table", "explain")},
        )
    # clarify is a terminal, understood outcome (not a failure) — surface as refused-with-reason.
    if status_str in ("clarify", "refuse", "tier2_rejected"):
        return AgentResult(source_id, source_type, STATUS_REFUSED, engine="deterministic_sql",
                           reason=d.get("answer") or d.get("reason") or status_str or "refused")
    return AgentResult(source_id, source_type, STATUS_FAILED, engine="deterministic_sql",
                       error=d.get("error") or status_str or "sql failed")


def _from_result_obj(obj, source_id, source_type, engine) -> AgentResult:
    """RAG/NoSQL result object (has .answer/.citations/.error) → AgentResult."""
    err = getattr(obj, "error", None)
    if err:
        return AgentResult(source_id, source_type, STATUS_FAILED, engine=engine, error=str(err))
    return AgentResult(
        source_id, source_type, STATUS_OK, engine=engine,
        data={
            "answer": getattr(obj, "answer", ""),
            "citations": list(getattr(obj, "citations", []) or []),
            "cols": list(getattr(obj, "cols", []) or []),
            "rows": list(getattr(obj, "rows", []) or []),
        },
    )


# ── agents ────────────────────────────────────────────────────────────────────────────────────
class BaseSourceAgent:
    source_type = ""

    def execute(self, query, *, source_id="", source_ids=None, sm=None, cols=None,
                evidence=None, execution_context=None, on_event=None) -> AgentResult:
        raise NotImplementedError

    def _guard(self, fn, source_id) -> AgentResult:
        """Run fn(), turning any exception into a failed AgentResult — a source that errors must
        never crash the coordinator; it becomes one failed result among possibly several."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return AgentResult(source_id or "", self.source_type, STATUS_FAILED,
                               error=f"{type(e).__name__}: {e}")


class _SqlAgent(BaseSourceAgent):
    """Shared by relational + datalake — both answer via the deterministic SQL head (run_query);
    execution.py routes a datalake source's SQL to DuckDB, so no separate code path is needed."""

    def execute(self, query, *, source_id="", source_ids=None, sm=None, cols=None,
                evidence=None, execution_context=None, on_event=None) -> AgentResult:
        if sm is None or cols is None:
            # The coordinator loads the (merged) semantic model + columns for the selected scope.
            return AgentResult(source_id, self.source_type, STATUS_FAILED,
                               engine="deterministic_sql",
                               error="semantic model (sm, cols) not provided by coordinator")
        # Evidence reuse (P2/G5): `execution_context` optionally carries the routing artifacts (query
        # embedding, per-source evidence). The deterministic SQL head (run_query) owns its own,
        # correctness-critical retrieval + firewall, so it is NOT forced to consume routing evidence —
        # it safely falls back to its normal retrieval. The context is threaded so a future run_query
        # that accepts a compatible seed can reuse it without any interface change here. Correctness
        # over reuse: this agent never hands the SQL head evidence its firewall didn't produce.
        return self._guard(
            lambda: _from_sql_dict(_sql_delegate(query, sm, cols, on_event), source_id,
                                   self.source_type),
            source_id)


class DatabaseAgent(_SqlAgent):
    source_type = "relational"


class DataLakeAgent(_SqlAgent):
    source_type = "datalake"


class FileSystemAgent(BaseSourceAgent):
    source_type = "document"

    def execute(self, query, *, source_id="", source_ids=None, sm=None, cols=None,
                evidence=None, execution_context=None, on_event=None) -> AgentResult:
        sids = source_ids if source_ids is not None else ([source_id] if source_id else None)
        return self._guard(
            lambda: _from_result_obj(_rag_delegate(query, sids, on_event), source_id,
                                     self.source_type, "rag"),
            source_id)


class NoSqlAgent(BaseSourceAgent):
    source_type = "nosql"

    def execute(self, query, *, source_id="", source_ids=None, sm=None, cols=None,
                evidence=None, execution_context=None, on_event=None) -> AgentResult:
        sids = source_ids if source_ids is not None else ([source_id] if source_id else None)
        return self._guard(
            lambda: _from_result_obj(_nosql_delegate(query, sids, on_event), source_id,
                                     self.source_type, "nosql"),
            source_id)


# ── registry ────────────────────────────────────────────────────────────────────────────────
# Keyed by the engine source-KIND (apps.sources.Source.source_kind()), NOT the dialect — a new
# same-kind dialect needs no new agent, mirroring _DIALECT_TO_ENGINE.
_AGENT_BY_KIND = {
    "relational": DatabaseAgent,
    "datalake": DataLakeAgent,
    "document": FileSystemAgent,
    "nosql": NoSqlAgent,
}


def resolve_agent(source_kind: str) -> Optional[BaseSourceAgent]:
    """Return a fresh agent for a source kind, or None if the kind has no agent (caller decides
    how to handle an unroutable source — never silently guess)."""
    cls = _AGENT_BY_KIND.get((source_kind or "").lower())
    return cls() if cls is not None else None


def supported_kinds() -> List[str]:
    return list(_AGENT_BY_KIND.keys())
