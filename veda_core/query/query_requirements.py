"""query/query_requirements.py — QueryRequirements (Phase C1, shadow-mode observation only).

Phase C1 of the Capability-Based Planning work
(docs/architecture/VEDA_PHASE_C_CAPABILITY_PLANNING_AUDIT.md §5/§6/§10). Derives a MINIMAL,
schema-independent requirements signal from raw NL text — no semantic model, no `sm`/`cols`, no
SLM call. Every field here is backed by an EXISTING, already-shipped deterministic classifier;
nothing is a new heuristic invented for this module.

REUSE-FIRST AUDIT (done before writing this file, per the Phase C1 task's strict rule):

- `veda/planning.py::aggregate_mode(query)` was inspected and found to represent a DIFFERENT,
  narrower concept — per-anchor CHILD-grouped counting with an optional threshold (e.g.
  "counterparties with more than one annotation"), not a general "does this query want an
  aggregate, and which kind" signal. Using it here would misrepresent what it actually detects.
- The correct, general-purpose, already-shipped signal is `query/fast_path.py`'s own trigger
  tuples — `_COUNT_TRIGGERS`, `_COUNT_WORDS`, `_SUM_VERBS`, `_AVG_VERBS` — pure string-literal
  matching against lowercased query text, genuinely schema-independent (confirmed by reading
  `fast_path.py::_count_intent()`, which uses these same tuples this same way). This module
  imports those exact constants rather than re-deriving new keyword lists, so there is exactly
  ONE source of truth for "what counts as a count/sum/avg trigger phrase."
- `query/temporal_parser.py::run_temporal_parser(query)` already returns a schema-independent
  `TemporalParserResult.temporal_filter` (None when no temporal expression is found) — used
  directly, unmodified.
- `requires_join`/multi-entity was investigated and explicitly OMITTED: no existing
  deterministic, schema-independent signal was found for it (join detection everywhere in the
  codebase — `join_planner.py`, the cross_source_fk edge graph — requires a loaded schema or
  relationship graph, which by definition isn't available at this pre-routing point). Adding a
  keyword-based join heuristic here would be exactly the "weak heuristic" the task forbids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QueryRequirements:
    """Minimal, schema-independent facts about what a query needs. Every field is derived from
    an existing deterministic classifier — see the module docstring for the reuse audit."""

    requires_aggregation: bool = False
    aggregate_type: Optional[str] = None       # "count" | "sum" | "avg", when deterministically known
    requires_temporal: bool = False
    temporal_requirement: Optional[str] = None  # first matched raw temporal expression, if any


def derive_requirements(query: str) -> QueryRequirements:
    """Derive QueryRequirements from raw NL text ONLY. Never loads a semantic model, never calls
    an SLM, never raises on malformed input — a query this can't classify simply yields the
    all-False default (never guesses True)."""
    if not query or not query.strip():
        return QueryRequirements()

    ql = f" {query.lower()} "
    qtoks = set(query.lower().split())

    from query.fast_path import _COUNT_TRIGGERS, _COUNT_WORDS, _SUM_VERBS, _AVG_VERBS

    aggregate_type: Optional[str] = None
    if any(t in ql for t in _COUNT_TRIGGERS) or bool(_COUNT_WORDS & qtoks):
        aggregate_type = "count"
    elif any(t in ql for t in _SUM_VERBS):
        aggregate_type = "sum"
    elif any(t in ql for t in _AVG_VERBS):
        aggregate_type = "avg"

    from query.temporal_parser import run_temporal_parser
    temporal_result = run_temporal_parser(query)
    requires_temporal = temporal_result.temporal_filter is not None
    temporal_requirement = (
        temporal_result.raw_expressions[0] if temporal_result.raw_expressions else None
    )

    return QueryRequirements(
        requires_aggregation=aggregate_type is not None,
        aggregate_type=aggregate_type,
        requires_temporal=requires_temporal,
        temporal_requirement=temporal_requirement,
    )
