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
    # Observability (traceability Part 23). Set by query/reliability.execute_reliably
    # when it actually re-ran this agent, so the per-source execution record can
    # report a retry instead of the count being inferred from the `reason` string.
    # 0 on every non-retried path, so nothing changes for an existing consumer.
    retry_count: int = 0


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
        # `status` is part of the payload contract, not decoration: chatbot/memory/frame.py::
        # harvest_frame refuses to harvest anything whose payload does not say
        # status == "answered" (it documents itself as receiving "pipeline.py's _done() payload
        # forwarded verbatim"). Dropping it here made that check fail for EVERY answer that came
        # through a source agent, so memory_write_node's harvest returned None, no QueryFrame or
        # DrillStack was ever persisted, and every follow-up was treated as a new topic — which
        # is why drill-down never accumulated. state["status"] was right all along (nodes.py::
        # _extract_engine_result derives it from the SubResult); only the payload was missing it.
        return AgentResult(
            source_id, source_type, STATUS_OK, engine="deterministic_sql",
            # `analytics` and `ir` (M4, 2026-09-21) are the same class of omission this
            # whitelist already made once with `status` (see the note above). The chat
            # tier's IR stack is built from exactly these two — the engine's QueryIR and
            # the single post-execution analytics pass — so dropping them here left every
            # stack entry empty: no dimensions, no measures, no top_values, and therefore
            # no deterministic delta could ever resolve, silently sending every follow-up
            # back to the SLM classifier. Anything the memory layer needs has to be named
            # here; this list is the actual payload contract.
            data={k: d.get(k) for k in ("cols", "rows", "answer", "sql", "table", "explain",
                                        "status", "analytics", "ir", "business_intent")},
        )
    # clarify is a terminal, understood outcome (not a failure) — surface as refused-with-reason.
    if status_str in ("clarify", "refuse", "tier2_rejected"):
        # Carry the pipeline's OWN explanation through, not just `reason`. run_query builds a
        # purpose-written clarifying question for every non-answered status (veda/feedback.py::
        # explain_failure, e.g. "Is 'Security' a column name or a value you want to filter on?"),
        # but this mapping used to keep only `reason` — which falls back to the bare status string
        # "clarify" when the dict has no `answer`. Everything downstream then had nothing to show
        # and the user got the contentless "Could you clarify what you're asking about?" instead
        # of being told what was actually unclear. Only the small diagnostic fields are copied
        # (no cols/rows — a refusal has no result set to render).
        return AgentResult(source_id, source_type, STATUS_REFUSED, engine="deterministic_sql",
                           data={k: d.get(k) for k in ("feedback", "answer", "msg", "status",
                                                       "missing", "sql") if d.get(k) is not None},
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
            # `status` for the same reason the SQL branch above carries it, and it was
            # missed here: chatbot/memory/frame.py::harvest_frame refuses to harvest any
            # payload that does not say status == "answered". Without it NO document or
            # NoSQL answer could ever be harvested, so a document source had no session
            # memory at all — every follow-up on it was treated as a new topic and
            # re-routed from scratch (measured on session script s3: inherited 0/9).
            "status": "answered",
            "source_id": source_id,
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
