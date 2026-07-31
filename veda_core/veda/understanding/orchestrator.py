"""veda.understanding.orchestrator — the one public entry point.

    understand_query(query, sm, graph=None, junctions=None, retrieval_scores=None)
        → GroundedIntent | Refusal | None

Wires: flag-gate → EXTRACT (LLM) → GROUND (deterministic firewall) → trace.
Returns None to mean "DEGRADE — use the existing pipeline unchanged" (flag off, SLM
down, low confidence, or nothing extractable). A Refusal means the firewall blocked a
guess (ungroundable / impossible / ambiguous) — the caller should surface it, not fall
through to SQL. A GroundedIntent is fully validated and safe to plan from.

Everything is best-effort and exception-safe: understanding must NEVER crash the query
path (the whole point is to be strictly additive under a flag).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from veda.understanding.schema import RawIntent, GroundedIntent, Refusal
from veda.understanding.extractor import extract
from veda.understanding.grounding import ground


def _entity_catalog(sm) -> List[str]:
    """Business-entity concept list handed to the LLM (humanized table names)."""
    tabs = sm.get("tables", {}) if isinstance(sm.get("tables"), dict) else {}
    out = set()
    for t, meta in tabs.items():
        out.add((meta or {}).get("business_name") or (meta or {}).get("display_name") or t)
    return sorted(out)


def _trace(section: str, **kw) -> None:
    try:
        from veda.explain import current_trace
        current_trace().set(section, **kw)
    except Exception:
        pass


def understand_query(query: str, sm, graph=None, junctions=None,
                     retrieval_scores: Optional[Dict[str, float]] = None
                     ) -> Optional[Union[GroundedIntent, Refusal]]:
    # ── flag gate (default OFF → prod byte-identical) ──────────────────────────
    try:
        from config import (QUERY_UNDERSTANDING_ENABLED as _on,
                            QUERY_UNDERSTANDING_MIN_CONFIDENCE as _minc)
    except Exception:
        _on, _minc = False, 0.5
    if not _on:
        return None

    try:
        if graph is None:
            from veda.planning import get_graph
            graph = get_graph()
        if junctions is None:
            from veda.planning import _junction_tables
            junctions = _junction_tables(graph, sm)

        raw = extract(query, _entity_catalog(sm))
        if raw is None:
            _trace("understanding", enabled=True, status="degrade",
                   reason="extract_failed_or_lowconf")
            return None                      # degrade → existing pipeline
        _trace("understanding", enabled=True, status="extracted",
               intent=raw.intent, grain=raw.grain, measure=raw.measure,
               entities=raw.entities, confidence=raw.confidence)

        result = ground(raw, sm, graph, junctions, retrieval_scores, min_confidence=_minc)
        if result is None:
            _trace("understanding", status="degrade", reason="grounding_degrade")
            return None
        if isinstance(result, Refusal):
            _trace("understanding", status="refuse", refuse_reason=result.reason,
                   unresolved=result.unresolved)
            return result
        _trace("understanding", status="grounded", anchor=result.anchor,
               secondaries=result.secondaries,
               measure=(result.measure.kind if result.measure else None),
               confidence=result.confidence)
        return result
    except Exception as e:
        # understanding must never break the query path — degrade on any error
        _trace("understanding", status="degrade", reason=f"exception:{type(e).__name__}")
        return None
