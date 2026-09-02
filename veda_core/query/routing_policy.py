"""query/routing_policy.py — deterministic-first source routing policy (multi-source routing, Phase 3.2).

Pure decision function over candidate sources + relationship edges. No DB, no SLM, no keywords —
every branch is a computed fact (presence tier, cross_source_fk edge, is_canonical). The coordinator
supplies the candidates and the edge set; this module only decides.

Policy (docs/multisource_routing/PLAN.md §8):

    candidates = sources whose tier ∈ {STRONG, WEAK}   (STRONG preferred; WEAK only if no STRONG)
    0 candidates                         → NO_MATCH
    1 candidate                          → SINGLE
    ≥2 candidates:
        a cross_source_fk edge connects ≥2 of them   → MULTI  (genuine join)
        else same domain + exactly one is_canonical  → SINGLE (canonical tie-break)
        else                                          → AMBIGUOUS  (→ SLM / clarify, upstream)
"""
from __future__ import annotations

from typing import Iterable, List, Set, Tuple

from query.routing_contracts import (
    CandidateSource, RoutingDecision,
    STATUS_ROUTED, STATUS_NO_MATCH,
    MODE_SINGLE, MODE_MULTI, MODE_NONE,
    METHOD_DETERMINISTIC,
    RC_NO_EVIDENCE, RC_SINGLE_CANDIDATE, RC_RELATIONSHIP_EDGE, RC_CANONICAL_SELECTED, RC_AMBIGUOUS,
)

TIER_RANK = {"NONE": 0, "WEAK": 1, "STRONG": 2}

# Edge-MULTI co-leader epsilon: connected sources must be within this cosine of the top to be treated
# as genuine co-leaders (else the top one wins SINGLE). Tighter than the dominance gap on purpose.
try:
    import config as _cfg
    _EDGE_MULTI_EPS = float(getattr(_cfg, "ROUTING_EDGE_MULTI_EPS", 0.015))
except Exception:
    _EDGE_MULTI_EPS = 0.015


def _kind(c: CandidateSource) -> str:
    """Coarse evidence kind for cross-type-fair pooling. Column-cosine (tabular) and chunk-cosine
    (document) live in different score distributions, so pooling must not let one kind's STRONG
    exclude another kind's WEAK (that would silently drop a doc source whenever any table matches)."""
    t = (c.source_type or "").lower()
    if t in ("document", "filesystem", "s3_docs"):
        return "document"
    if t == "nosql":
        return "nosql"
    return "tabular"   # relational + datalake share the column-cosine distribution


def _pool(candidates: List[CandidateSource]) -> List[CandidateSource]:
    """Per-KIND STRONG-preferred pooling, unioned across kinds (routing tuning, benchmark finding).

    Within one kind (comparable scores) STRONG excludes WEAK. ACROSS kinds we never cross-exclude —
    a STRONG table never drops a WEAK document, because their cosines aren't comparable. NONE never
    routes."""
    by_kind: dict = {}
    for c in candidates:
        if c.presence_tier in ("STRONG", "WEAK"):
            by_kind.setdefault(_kind(c), []).append(c)
    pool: List[CandidateSource] = []
    for cs in by_kind.values():
        strong = [c for c in cs if c.presence_tier == "STRONG"]
        pool.extend(strong if strong else cs)
    return pool


def _edge_connected_subset(pool: List[CandidateSource],
                           edge_pairs: Set[frozenset]) -> List[CandidateSource]:
    """Pool members that are joined to at least one OTHER pool member by a cross_source_fk edge.
    ``edge_pairs`` is a set of ``frozenset({sid_a, sid_b})``."""
    ids = {c.source_id for c in pool}
    connected_ids = set()
    for pair in edge_pairs:
        inter = pair & ids
        if len(inter) >= 2:               # both endpoints are candidates
            connected_ids |= inter
    return [c for c in pool if c.source_id in connected_ids]


def _shared_domain(pool: List[CandidateSource]) -> Iterable[str]:
    """Domain tag(s) common to EVERY pool member (strict same-domain). Empty if any member is
    untagged or they don't all overlap."""
    tag_sets = [set(c.domain_tags or []) for c in pool]
    if not tag_sets or any(not s for s in tag_sets):
        return set()
    common = tag_sets[0]
    for s in tag_sets[1:]:
        common &= s
    return common


def _decision(mode, sources, reason_code, reason, **extra) -> RoutingDecision:
    return RoutingDecision(
        status=STATUS_ROUTED, mode=mode,
        source_ids=[c.source_id for c in sources],
        candidate_sources=list(sources),
        evidence_summary=[c.evidence_summary for c in sources if c.evidence_summary],
        decision_method=METHOD_DETERMINISTIC, reason_code=reason_code, reason=reason, **extra)


def decide(candidates: List[CandidateSource],
           edge_pairs: Set[frozenset] = None) -> Tuple[RoutingDecision, bool]:
    """Return (decision, ambiguous). When ambiguous is True the decision is a provisional
    AMBIGUOUS marker (status ROUTED, mode NONE) that the coordinator escalates to the bounded SLM
    or a clarification — the policy itself never guesses.
    """
    edge_pairs = edge_pairs or set()
    pool = _pool(candidates)

    if not pool:
        return RoutingDecision(
            status=STATUS_NO_MATCH, mode=MODE_NONE, candidate_sources=list(candidates),
            decision_method=METHOD_DETERMINISTIC, reason_code=RC_NO_EVIDENCE,
            reason="No source had sufficient evidence for this query."), False

    if len(pool) == 1:
        return _decision(MODE_SINGLE, pool, RC_SINGLE_CANDIDATE,
                         "Exactly one source has relevant evidence."), False

    # Anti-over-MULTI (benchmark finding): if exactly ONE source is STRONG while the rest are only
    # WEAK, that one source clearly dominates — route SINGLE to it rather than dragging spuriously-
    # matched weak sources into a MULTI. (A genuine MULTI needs ≥2 sources that are each STRONG, or,
    # when nothing is STRONG at all, the WEAK pool falls through to the edge/canonical logic below.)
    strong = [c for c in pool if c.presence_tier == "STRONG"]
    if len(strong) == 1:
        return _decision(MODE_SINGLE, strong, RC_SINGLE_CANDIDATE,
                         "One source is clearly strongest; the rest are weak matches."), False

    # ≥2 candidates — deterministic disambiguation before any SLM. A relationship-edge MULTI is only
    # driven by sources that are BOTH genuinely relevant (STRONG); when there is no STRONG at all the
    # whole (WEAK) pool is eligible, preserving the "weak-only still routes" behaviour.
    edge_scope = strong if len(strong) >= 2 else pool
    connected = _edge_connected_subset(edge_scope, edge_pairs)
    # Edge-MULTI must be driven by genuine CO-LEADERS, not a clear winner plus an edge-connected
    # runner-up that merely sits within the (looser) dominance gap. A big source (many items) almost
    # always has some item within the gap of any query's winner AND a discovered cross_source_fk edge
    # to it — that dragged clear-SINGLE queries into MULTI (benchmark: "vendor details"→[4,2]). So
    # require the connected sources to be near-TIED at the top (within a tight epsilon); otherwise the
    # top source alone wins (SINGLE). Falls back to old behaviour when scores are unavailable (0.0).
    if len(connected) >= 2:
        best = max(c.top_score for c in connected)
        coleaders = [c for c in connected if c.top_score >= best - _EDGE_MULTI_EPS] \
            if best > 0 else connected
        if len(coleaders) >= 2:
            pairs = [sorted(p) for p in edge_pairs
                     if len(p & {c.source_id for c in coleaders}) >= 2]
            if pairs:
                return _decision(MODE_MULTI, coleaders, RC_RELATIONSHIP_EDGE,
                                 "Sources are joined by a discovered cross-source relationship.",
                                 relationship_basis={"edge": "cross_source_fk", "pairs": pairs}), False
        # a clear top source among edge-connected candidates → route SINGLE to it
        top = max(connected, key=lambda c: c.top_score)
        return _decision(MODE_SINGLE, [top], RC_SINGLE_CANDIDATE,
                         "One edge-connected source is clearly strongest."), False

    common = set(_shared_domain(pool))
    canon = [c for c in pool if c.is_canonical]
    if common and len(canon) == 1:
        dec = _decision(MODE_SINGLE, canon, RC_CANONICAL_SELECTED,
                        "Same-domain sources with no join; the canonical source was selected.",
                        canonical_basis={"domain": sorted(common)[0], "source_id": canon[0].source_id})
        return dec, False

    # Genuinely unclear — hand off (SLM / clarify). Provisional marker, not a route.
    return RoutingDecision(
        status=STATUS_ROUTED, mode=MODE_NONE, source_ids=[c.source_id for c in pool],
        candidate_sources=list(pool),
        evidence_summary=[c.evidence_summary for c in pool if c.evidence_summary],
        decision_method=METHOD_DETERMINISTIC, reason_code=RC_AMBIGUOUS,
        reason="Multiple unrelated sources are relevant; needs disambiguation."), True
