"""query/capability_filter.py — Phase C2: narrow aggregation capability filtering.

STRICT SCOPE, per the real-query evidence in
docs/architecture/VEDA_PHASE_C1_UNBLOCKED_BENCHMARK.md (🟡 LIMITED GO — aggregation only, not a
general capability-filtering go): this module filters candidates on EXACTLY ONE rule — when
QueryRequirements.requires_aggregation is True, remove any candidate lacking
SourceCapability.AGGREGATION. No temporal filtering, no join/federation filtering, no strategy
hinting, no detector changes. Adding any of those would exceed what the C1 benchmark evidence
actually supports.

Reuses C1's own pure functions verbatim — `query_requirements.derive_requirements()` and
`capability_observation.observe_candidate_capabilities()` — so there is exactly one place
"does this candidate satisfy this requirement" is computed; this module only decides what to DO
with that answer (keep or remove), it does not recompute it.
"""
from __future__ import annotations

from typing import List

from query.capability_observation import observe_candidate_capabilities
from query.query_requirements import derive_requirements
from utils.logger import get_logger


def _filtering_enabled() -> bool:
    try:
        import config as _cfg
        return bool(getattr(_cfg, "CAPABILITY_FILTERING_ENABLED", False))
    except Exception:
        return False


def filter_candidates_by_aggregation_capability(query: str, candidates: List) -> List:
    """Phase C2 entry point, called from source_coordinator.py::plan_route() right after the C1
    shadow observation. Returns the SAME candidate list object whenever nothing needs removing
    (flag off, non-aggregation query, nothing incompatible found, or the all-incompatible safety
    fallback) — a new, filtered list is only constructed when a genuine removal happens. Never
    mutates `candidates` in place (no in-place removal, ever). Order is preserved by construction
    (single forward pass). Never returns an empty list: if every candidate would be removed, the
    original list is returned unchanged and the reason is logged, not silently dropped."""
    if not _filtering_enabled():
        return candidates
    try:
        requirements = derive_requirements(query)
        if not requirements.requires_aggregation:
            return candidates

        kept, removed = [], []
        for c in candidates:
            obs = observe_candidate_capabilities(c, requirements)
            (kept if obs.compatible else removed).append(c)

        if not removed:
            return candidates                     # nothing incompatible — no-op, same object
        if not kept:
            _log_fallback(query, requirements, candidates)
            return candidates                     # safety fallback — never return [] silently
        _log_filtered(query, requirements, kept, removed)
        return kept
    except Exception:
        return candidates                         # any failure here must never affect routing


def _log_filtered(query, requirements, kept, removed) -> None:
    logger = get_logger(__name__)
    logger.info(
        "capability_filtering_applied query=%r requirements=%s kept=%s removed=%s",
        query, requirements,
        [c.source_id for c in kept], [c.source_id for c in removed],
    )


def _log_fallback(query, requirements, candidates) -> None:
    logger = get_logger(__name__)
    logger.warning(
        "capability_filtering_fallback query=%r requirements=%s all_candidates_incompatible=%s "
        "— retaining original candidates rather than returning an empty list",
        query, requirements, [c.source_id for c in candidates],
    )
