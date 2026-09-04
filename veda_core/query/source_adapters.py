"""query/source_adapters.py — Source Adapter interface (Phase A2, foundation only).

Phase A2 of the Source Capability & Execution Adapter Foundation
(docs/architecture/VEDA_SOURCE_CAPABILITY_ADAPTER_AUDIT.md §5/§6). Per the audit's evidence:
``agents.py::_AGENT_BY_KIND`` is already the query-time adapter-dispatch pattern in embryo — a
kind-keyed map to a class whose only real job is ``execute()``. ``SourceAdapter`` here is a THIN,
TYPED, CALL-THROUGH wrapper over that existing dispatch — it adds a name and a capability lookup,
nothing else. It does NOT re-implement execution, does NOT decompose into
discover_metadata/ground_query/plan stages (the audit found no evidence current agents expose
those as separable steps — inventing them now would be structure the implementation doesn't have),
and does NOT touch fast_path, SQL generation, semantic retrieval, or the alignment layer.

Like source_capabilities.py (Phase A1), this module has NO call sites anywhere else in the
codebase yet. ``resolve_adapter()`` exists so a later phase (A3) can be introduced by adding ONE
new call site behind a flag, without this module itself changing. Importing it changes nothing.
"""
from __future__ import annotations

from typing import List, Optional

from query.agents import AgentResult, BaseSourceAgent, resolve_agent, supported_kinds as _agent_kinds
from query.execution_request import ExecutionRequest
from query.source_capabilities import SourceCapabilities, capabilities_for


class SourceAdapter:
    """Wraps an existing ``BaseSourceAgent`` instance. ``execute()`` is a verbatim pass-through —
    same arguments, same return type, same underlying pipeline call. The only new surface is
    ``get_capabilities()``, which is a pure lookup (Phase A1) keyed on the wrapped agent's own
    ``source_type`` — never a second, independent kind derivation."""

    def __init__(self, agent: BaseSourceAgent):
        self._agent = agent

    @property
    def source_kind(self) -> str:
        return self._agent.source_type

    def get_capabilities(self) -> SourceCapabilities:
        return capabilities_for(self.source_kind)

    def execute(self, query, *, source_id: str = "", source_ids: Optional[list] = None,
                sm=None, cols=None, evidence=None, execution_context=None,
                on_event=None) -> AgentResult:
        """Call-through to the wrapped agent's execute() — identical signature, identical
        behavior, identical AgentResult. No new logic lives here."""
        return self._agent.execute(
            query, source_id=source_id, source_ids=source_ids, sm=sm, cols=cols,
            evidence=evidence, execution_context=execution_context, on_event=on_event,
        )

    def execute_request(self, request: ExecutionRequest, *, evidence=None) -> AgentResult:
        """Phase B1: unpack an ExecutionRequest and delegate to execute() verbatim — no new logic,
        no new behavior. ``evidence`` stays a separate kwarg (not a request field) because it is
        routing-stage retrieval evidence, not part of the canonical request per the audit's §5
        field list; a future phase can fold it in if evidence shows it belongs there."""
        return self.execute(
            request.query, source_id=request.source_id, source_ids=request.source_ids,
            sm=request.sm, cols=request.cols, evidence=evidence,
            execution_context=request.execution_context, on_event=request.on_event,
        )


def resolve_adapter(source_kind: str) -> Optional[SourceAdapter]:
    """Return a SourceAdapter wrapping a fresh agent for this kind, or None if the kind has no
    agent — mirrors resolve_agent()'s own contract exactly (never silently guess)."""
    agent = resolve_agent(source_kind)
    return SourceAdapter(agent) if agent is not None else None


def supported_kinds() -> List[str]:
    return _agent_kinds()
