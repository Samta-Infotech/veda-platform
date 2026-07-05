"""Phase 4 / L4 — Consensus Engine (ARCHITECTURE_HYBRID.md §2 L4).

Compares INTENT (IR), not SQL. Reconciliation is field-weighted, not equality, because the
two engines are reliable on different fields:

    anchor          → deterministic authority   | disagreement = HARD STOP (clarify)
    filters         → LLM, grounded reconcile    | union of both
    projections     → LLM                        | union (over-projecting is safe)
    aggregations    → deterministic authority
    group_by/order  → deterministic authority

Output: merged IR + agreement flag + confidence + provenance. A semantic-drift on the
anchor never silently resolves — it surfaces for clarification (refuse-over-guess).

Pure logic. No DB, no Ollama.
"""
from typing import Any, Dict, Optional

from veda.ir_validator import empty_ir_v2, validate_ir_v2


def _dedup(seq):
    return list(dict.fromkeys(seq))


def _filter_key(f):
    return (f.get("field"), f.get("op"), str(f.get("value")))


def reconcile(ir_det: Optional[Dict[str, Any]],
              ir_llm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconcile a deterministic IR and an LLM IR into one. Returns:
        {ir, agreement, anchor_conflict, confidence, provenance, reason}
    anchor_conflict=True means the engines disagree on the SUBJECT → caller must clarify."""
    # Single-engine cases — pass through (no consensus possible, lower confidence).
    if ir_det and not ir_llm:
        return {"ir": ir_det, "agreement": False, "anchor_conflict": False,
                "confidence": ir_det.get("confidence", 0.0), "provenance": "deterministic",
                "reason": "only deterministic IR available"}
    if ir_llm and not ir_det:
        return {"ir": ir_llm, "agreement": False, "anchor_conflict": False,
                "confidence": ir_llm.get("confidence", 0.0), "provenance": "llm",
                "reason": "only llm IR available"}
    if not ir_det and not ir_llm:
        return {"ir": None, "agreement": False, "anchor_conflict": False,
                "confidence": 0.0, "provenance": "none", "reason": "no IR from either engine"}

    a_det = (ir_det.get("anchor") or "").lower()
    a_llm = (ir_llm.get("anchor") or "").lower()

    # ── anchor: deterministic authority; disagreement is a HARD STOP ───────────
    if a_det and a_llm and a_det != a_llm:
        return {
            "ir": None, "agreement": False, "anchor_conflict": True,
            "confidence": 0.0, "provenance": "conflict",
            "reason": f"subject drift: deterministic='{a_det}' vs llm='{a_llm}' — clarify",
        }
    anchor = a_det or a_llm

    merged = empty_ir_v2()
    merged["anchor"] = ir_det.get("anchor") or ir_llm.get("anchor")

    # ── projections: union (over-projecting is safe; missing a column isn't) ───
    merged["projections"] = _dedup(list(ir_det.get("projections", []))
                                    + list(ir_llm.get("projections", [])))

    # ── filters: union by (field, op, value) — grounding happens downstream ────
    seen, filters = set(), []
    for f in list(ir_det.get("filters", [])) + list(ir_llm.get("filters", [])):
        k = _filter_key(f)
        if k not in seen:
            seen.add(k)
            filters.append(f)
    merged["filters"] = filters

    # ── aggregations / grouping / ordering: deterministic authority ────────────
    merged["aggregations"] = list(ir_det.get("aggregations") or ir_llm.get("aggregations") or [])
    merged["group_by"] = list(ir_det.get("group_by") or ir_llm.get("group_by") or [])
    merged["order_by"] = list(ir_det.get("order_by") or ir_llm.get("order_by") or [])
    merged["limit"] = ir_det.get("limit") if ir_det.get("limit") is not None else ir_llm.get("limit")
    merged["temporal"] = ir_det.get("temporal") or ir_llm.get("temporal")

    # agreement = anchors match AND filter sets match AND aggregation intent matches
    same_filters = ({_filter_key(f) for f in ir_det.get("filters", [])}
                    == {_filter_key(f) for f in ir_llm.get("filters", [])})
    same_aggs = ([a.get("func") for a in ir_det.get("aggregations", [])]
                 == [a.get("func") for a in ir_llm.get("aggregations", [])])
    agreement = bool(a_det and a_llm and a_det == a_llm and same_filters and same_aggs)

    # confidence: consensus boosts; partial agreement keeps the lower of the two.
    c_det = float(ir_det.get("confidence", 0.0) or 0.0)
    c_llm = float(ir_llm.get("confidence", 0.0) or 0.0)
    confidence = round(min(1.0, max(c_det, c_llm) + 0.1), 3) if agreement else round(min(c_det, c_llm), 3)

    merged["confidence"] = confidence
    merged["provenance"] = "consensus" if agreement else "merged"

    return {
        "ir": merged, "agreement": agreement, "anchor_conflict": False,
        "confidence": confidence, "provenance": merged["provenance"],
        "reason": "engines agree" if agreement else "merged (partial agreement)",
        "errors": validate_ir_v2(merged),
    }
