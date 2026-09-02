"""query/routing_contracts.py — typed routing contracts (multi-source routing, Phase 3.3).

The structured outputs of the routing layer. Kept in their own module so the policy, coordinator,
validator, SLM, and trace all share ONE definition and there is never a flat-enum-only decision
(the brief's B11 requirement).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# RoutingDecision.status
STATUS_ROUTED = "ROUTED"
STATUS_NO_MATCH = "NO_MATCH"
STATUS_CLARIFY = "CLARIFICATION_REQUIRED"
STATUS_FAILED = "FAILED"

# RoutingDecision.mode
MODE_SINGLE = "SINGLE"
MODE_MULTI = "MULTI"
MODE_NONE = "NONE"

# decision_method
METHOD_DETERMINISTIC = "deterministic"
METHOD_SLM = "slm"
METHOD_CLARIFY = "clarification"

# reason_code (coarse, stable, for slicing "why" in traces/metrics)
RC_NO_EVIDENCE = "NO_EVIDENCE"
RC_SINGLE_CANDIDATE = "SINGLE_CANDIDATE"
RC_RELATIONSHIP_EDGE = "RELATIONSHIP_EDGE"           # cross_source_fk join → MULTI
RC_CANONICAL_SELECTED = "CANONICAL_SELECTED"         # same-domain dup resolved by is_canonical
RC_AMBIGUOUS = "AMBIGUOUS_SOURCE_SELECTION"
RC_SLM_RESOLVED = "SLM_RESOLVED"
RC_INVALID_SLM = "INVALID_SLM_DECISION"


@dataclass
class CandidateSource:
    source_id: str
    source_type: str = ""              # engine kind: relational|datalake|document|nosql
    presence_tier: str = "NONE"        # STRONG | WEAK | NONE (from source_evidence)
    top_score: float = 0.0             # the source's top relevance cosine (for co-leader checks)
    # Item-prior only (query ↔ this source's item/dataset summaries). Kept SEPARATE from top_score
    # (which is the max over item/column/chunk) so the Required-Source Escalation can tell a secondary
    # that is semantically ABOUT the query (item-prior > 0) from one that merely shares a column word.
    top_item_score: float = 0.0
    is_canonical: bool = False
    domain_tags: List[str] = field(default_factory=list)
    evidence_summary: dict = field(default_factory=dict)   # SourceEvidence.summary()


@dataclass
class RoutingDecision:
    status: str                                    # ROUTED | NO_MATCH | CLARIFICATION_REQUIRED | FAILED
    mode: str = MODE_NONE                           # SINGLE | MULTI | NONE
    source_ids: List[str] = field(default_factory=list)
    candidate_sources: List[CandidateSource] = field(default_factory=list)
    evidence_summary: List[dict] = field(default_factory=list)
    decision_method: str = METHOD_DETERMINISTIC
    reason_code: str = ""
    reason: str = ""
    relationship_basis: Optional[dict] = None       # e.g. {"edge": "cross_source_fk", "pairs": [...]}
    canonical_basis: Optional[dict] = None          # e.g. {"domain": "finance", "source_id": "5"}
    validation_status: str = ""                     # set by the routing validator (Phase 3.5)
    query_id: str = ""
    trace_id: str = ""


@dataclass
class ExecutionContext:
    """Request-scoped context threaded from routing into source-agent execution (P2, G5). Carries the
    routing decision plus the artifacts already computed during routing (the query embedding, the
    per-source evidence) so a pipeline that CAN reuse them safely does, and any pipeline that cannot
    simply ignores it and retrieves normally. A stable, optional interface — never forced into a
    pipeline. Correctness over reuse: an agent reuses only what is safe for its engine."""
    query: str = ""
    query_id: str = ""
    trace_id: str = ""
    selected_source_ids: List[str] = field(default_factory=list)
    routing_decision: Optional["RoutingDecision"] = None
    query_embedding: object = None          # BGE-M3 vector computed once during routing
    routing_evidence: dict = field(default_factory=dict)   # {source_id: {columns, chunks, items}}


@dataclass
class ClarificationRequired:
    reason_code: str
    candidate_sources: List[CandidateSource] = field(default_factory=list)
    question: str = ""
    query_id: str = ""
    trace_id: str = ""

    def to_decision(self) -> RoutingDecision:
        return RoutingDecision(
            status=STATUS_CLARIFY, mode=MODE_NONE,
            candidate_sources=list(self.candidate_sources),
            decision_method=METHOD_CLARIFY, reason_code=self.reason_code, reason=self.question,
            query_id=self.query_id, trace_id=self.trace_id)
