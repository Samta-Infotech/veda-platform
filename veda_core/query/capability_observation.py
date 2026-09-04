"""query/capability_observation.py — Capability Planning Shadow Observation (Phase C1).

Phase C1 (docs/architecture/VEDA_PHASE_C_CAPABILITY_PLANNING_AUDIT.md §6/§10). OBSERVE-ONLY:
compares QueryRequirements (query_requirements.py) against each candidate's SourceCapabilities
(Phase A's source_capabilities.py) and logs the result. This module NEVER filters, reorders,
removes, or mutates the candidate list it is given — every function here returns new objects or
None; nothing here has a side effect on its inputs.

CAPABILITY-MODEL GAP, discovered while writing this (not invented, not assumed — see
`SourceCapability`'s actual definition in source_capabilities.py): there is NO temporal
capability in the Phase A model (only STRUCTURED_QUERY / SCHEMA_DISCOVERY / DOCUMENT_RETRIEVAL /
AGGREGATION / FILTERING / JOINING / FEDERATION). `QueryRequirements.requires_temporal` is
therefore captured and logged for future evidence-gathering, but deliberately NOT compared
against any capability here — inventing a mapping to a capability that doesn't exist would be
exactly the kind of unjustified assumption this phase is required to avoid. Only
`requires_aggregation` is checked (against `SourceCapability.AGGREGATION`), because that mapping
is the one the capability model actually supports today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from query.query_requirements import QueryRequirements
from query.source_capabilities import SourceCapabilities, SourceCapability, capabilities_for


@dataclass(frozen=True)
class CandidateCapabilityObservation:
    source_id: str
    requirements: QueryRequirements
    capabilities: SourceCapabilities
    compatible: bool
    incompatibilities: List[str] = field(default_factory=list)


def observe_candidate_capabilities(candidate, requirements: QueryRequirements
                                    ) -> CandidateCapabilityObservation:
    """Pure comparison — takes a CandidateSource (duck-typed: only .source_id/.source_type are
    read) and a QueryRequirements, returns a new CandidateCapabilityObservation. Never touches
    `candidate` itself."""
    caps = capabilities_for(getattr(candidate, "source_type", "") or "")
    incompatibilities: List[str] = []
    if requirements.requires_aggregation and not caps.has(SourceCapability.AGGREGATION):
        incompatibilities.append("requires_aggregation but source lacks AGGREGATION capability")
    # requires_temporal: NOT checked — see module docstring (no TEMPORAL capability exists yet).
    return CandidateCapabilityObservation(
        source_id=getattr(candidate, "source_id", ""),
        requirements=requirements,
        capabilities=caps,
        compatible=not incompatibilities,
        incompatibilities=incompatibilities,
    )


def _shadow_enabled() -> bool:
    try:
        import config as _cfg
        return bool(getattr(_cfg, "CAPABILITY_PLANNING_SHADOW_ENABLED", False))
    except Exception:
        return False


def run_capability_planning_shadow(query: str, candidates) -> None:
    """Phase C1 entry point, called from source_coordinator.py::plan_route() right after
    build_candidates(). OBSERVE-ONLY: derives QueryRequirements, compares against every
    candidate's capabilities, logs the result. Returns None always. Does NOT return, filter, or
    reorder `candidates` — callers must (and do) keep using the exact same list object they
    already had. Swallows every exception: a shadow-observation failure must never affect
    routing, even when the flag is on."""
    if not _shadow_enabled():
        return
    try:
        from query.query_requirements import derive_requirements
        requirements = derive_requirements(query)
        observations = [observe_candidate_capabilities(c, requirements) for c in candidates]
        _log_shadow_observation(query, requirements, observations)
    except Exception:
        pass


def _log_shadow_observation(query: str, requirements: QueryRequirements,
                             observations: List[CandidateCapabilityObservation]) -> None:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info(
        "capability_planning_shadow query=%r requirements=%s observations=%s",
        query, requirements,
        [{"source_id": o.source_id, "compatible": o.compatible,
          "incompatibilities": o.incompatibilities} for o in observations],
    )
