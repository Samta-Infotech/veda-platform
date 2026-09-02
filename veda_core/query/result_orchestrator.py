"""query/result_orchestrator.py — merge independent multi-source results (routing Phase 4.3/4.4).

When a MULTI route runs sources INDEPENDENTLY (no join relationship), their ``AgentResult``s must be
combined under an EXPLICIT policy — "preserve provenance" alone is not a merge. Policies:

    APPEND             — independent answers presented together (default; different facts).
    CANONICAL_PRIORITY — same scalar metric, sources disagree, exactly one is canonical → it wins,
                         the other is retained as provenance.
    CONFLICT_DETECTED  — same scalar metric, sources disagree, no canonical → surfaced, NOT blended.

Conservative by design: conflict detection only fires for directly-comparable SINGLE-SCALAR numeric
results (one row, one numeric cell). Anything richer is APPENDed with full provenance rather than
risk a wrong reconciliation. Never blends conflicting values silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

POLICY_APPEND = "APPEND"
POLICY_CANONICAL_PRIORITY = "CANONICAL_PRIORITY"
POLICY_CONFLICT_DETECTED = "CONFLICT_DETECTED"


@dataclass
class MergeResult:
    policy: str
    source_ids: List[str] = field(default_factory=list)
    parts: List[dict] = field(default_factory=list)     # per-source normalized {source_id, answer/data}
    provenance: List[dict] = field(default_factory=list)
    conflict: Optional[dict] = None                     # {metric?, values:[{source_id,value}...]}
    winner_source_id: str = ""                          # set for CANONICAL_PRIORITY
    needs_clarification: bool = False                   # True on unresolved conflict


def _scalar(res) -> Optional[float]:
    """The single numeric value of a one-row/one-numeric-cell SQL result, else None (not comparable)."""
    data = getattr(res, "data", {}) or {}
    rows = data.get("rows") or []
    if len(rows) != 1:
        return None
    row = rows[0]
    cells = list(row.values()) if isinstance(row, dict) else list(row)
    nums = [c for c in cells if isinstance(c, (int, float)) and not isinstance(c, bool)]
    if len(nums) == 1:
        return float(nums[0])
    return None


def _part(res) -> dict:
    data = getattr(res, "data", {}) or {}
    return {"source_id": res.source_id, "source_type": res.source_type,
            "answer": data.get("answer", ""), "rows": data.get("rows", []),
            "cols": data.get("cols", []), "sql": data.get("sql")}


def merge_results(agent_results, *, canonical_ids=None) -> MergeResult:
    """Merge independent per-source ``AgentResult``s. Only OK results participate; failed/refused
    ones are dropped from the merge (partial-failure handling lives upstream in Phase 5).
    ``canonical_ids``: set of source_ids marked canonical, used to break a scalar conflict.
    """
    canonical_ids = set(canonical_ids or [])
    ok = [r for r in (agent_results or []) if getattr(r, "status", "") == "ok"]
    sids = [r.source_id for r in ok]
    parts = [_part(r) for r in ok]
    prov = [{"source_id": r.source_id, "source_type": r.source_type,
             "engine": getattr(r, "engine", "")} for r in ok]

    # Conflict check: exactly two OK results, both single-scalar, values differ.
    if len(ok) == 2:
        a, b = _scalar(ok[0]), _scalar(ok[1])
        if a is not None and b is not None and a != b:
            conflict = {"values": [{"source_id": ok[0].source_id, "value": a},
                                   {"source_id": ok[1].source_id, "value": b}]}
            canon = [r for r in ok if r.source_id in canonical_ids]
            if len(canon) == 1:
                return MergeResult(POLICY_CANONICAL_PRIORITY, sids, parts, prov,
                                   conflict=conflict, winner_source_id=canon[0].source_id)
            return MergeResult(POLICY_CONFLICT_DETECTED, sids, parts, prov,
                               conflict=conflict, needs_clarification=True)

    return MergeResult(POLICY_APPEND, sids, parts, prov)
