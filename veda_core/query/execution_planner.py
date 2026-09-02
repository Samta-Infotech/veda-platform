"""query/execution_planner.py — turn a RoutingDecision into an ExecutionPlan (routing Phase 4.1).

MULTI does NOT always mean the same thing. Two strategies, chosen deterministically from WHY the
route is MULTI:

    RELATIONSHIP_EDGE MULTI  → strategy "federated"   → one cross-source SQL via run_federated
                                                         (the sources genuinely join on a
                                                          cross_source_fk key).
    SLM-resolved MULTI (no edge) → strategy "independent" → run each source's own pipeline and
                                                         merge the answers (result_orchestrator).

Execution mode is PARALLEL for both (steps are independent). DEPENDENT (staged A→B) is explicitly
DEFERRED — the interface exists but the planner never emits it yet (see docs/multisource_routing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from query.routing_contracts import (
    RoutingDecision, MODE_SINGLE, MODE_MULTI, RC_RELATIONSHIP_EDGE,
)

MODE_PARALLEL = "PARALLEL"
MODE_DEPENDENT = "DEPENDENT"     # deferred — never emitted yet

STRATEGY_SINGLE = "single"
STRATEGY_FEDERATED = "federated"
STRATEGY_INDEPENDENT = "independent"


@dataclass
class ExecutionStep:
    source_id: str
    source_type: str = ""
    depends_on: List[str] = field(default_factory=list)
    required: bool = True         # optional sources may fail without failing the whole query


@dataclass
class ExecutionPlan:
    mode: str                      # PARALLEL | DEPENDENT
    strategy: str                  # single | federated | independent
    steps: List[ExecutionStep] = field(default_factory=list)
    reason: str = ""


def plan_execution(decision: RoutingDecision) -> ExecutionPlan:
    """Turn a RoutingDecision into an ExecutionPlan """
    by_id = {c.source_id: c for c in decision.candidate_sources}

    def _step(sid: str) -> ExecutionStep:
        c = by_id.get(sid)
        return ExecutionStep(source_id=sid, source_type=(c.source_type if c else ""), required=True)

    if decision.mode == MODE_SINGLE:
        sid = decision.source_ids[0]
        return ExecutionPlan(mode=MODE_PARALLEL, strategy=STRATEGY_SINGLE, steps=[_step(sid)],
                             reason="Single selected source.")

    if decision.mode == MODE_MULTI:
        steps = [_step(s) for s in decision.source_ids]
        if decision.reason_code == RC_RELATIONSHIP_EDGE:
            return ExecutionPlan(mode=MODE_PARALLEL, strategy=STRATEGY_FEDERATED, steps=steps,
                                 reason="Sources join on a cross-source relationship → federated SQL.")
        return ExecutionPlan(mode=MODE_PARALLEL, strategy=STRATEGY_INDEPENDENT, steps=steps,
                             reason="Multiple relevant sources with no join → run independently and merge.")

    # NO_MATCH / CLARIFY / NONE — nothing to execute.
    return ExecutionPlan(mode=MODE_PARALLEL, strategy=STRATEGY_SINGLE, steps=[],
                         reason="No executable route.")
