"""veda/canonical_intent_shadow.py — Canonical-QueryIntent SHADOW measurement (Phase 1 spike).

OBSERVE-ONLY. Compares the fast-path QueryIntent that was PRESERVED-on-decline
(query/fast_path.get_preserved_intent) against the referents of the SQL the downstream pipeline
eventually generated (business_explain.extract_sql_facts), field by field, and LOGS a structured record.
It NEVER influences SQL, planning, routing, execution, or the response — its sole purpose is to measure
how reliable the discarded QueryIntent is versus the downstream SLM SQL, to inform (not decide) whether
QueryIntent can become the canonical IR. Gated by CANONICAL_INTENT_SHADOW_ENABLED (default OFF).

Per-field agreement values: MATCH | MISMATCH | NOT_APPLICABLE | UNKNOWN. Filters on a
QUALIFIER_GROUNDING_DECLINE are reported UNKNOWN (the intent's filter was the ungrounded reason for the
decline — a mismatch there is expected, not evidence of intent unreliability).
"""
from __future__ import annotations

import json
import re

MATCH, MISMATCH, NA, UNKNOWN = "MATCH", "MISMATCH", "NOT_APPLICABLE", "UNKNOWN"


def _enabled() -> bool:
    try:
        from config import CANONICAL_INTENT_SHADOW_ENABLED
        return bool(CANONICAL_INTENT_SHADOW_ENABLED)
    except Exception:
        return False


def _facts(sql):
    try:
        from veda.business_explain import extract_sql_facts
        return extract_sql_facts(sql) or {}
    except Exception:
        return {}


# ── field extractors ────────────────────────────────────────────────────────────────────────────
def _intent_aggregation(intent):
    """The aggregate the QueryIntent implies: count → COUNT, measure → the SUM/AVG/MAX/MIN in its
    select_expr, else None (no single aggregate — list/filter/ratio/trend/compare)."""
    qt = getattr(intent, "query_type", "") or ""
    if qt == "count":
        return "COUNT"
    if qt in ("measure",):
        expr = (getattr(intent, "select_expr", "") or "").upper()
        m = re.match(r"\s*(SUM|AVG|MAX|MIN|COUNT)\s*\(", expr)
        return m.group(1) if m else None
    return None


def _agg(intent, facts):
    intent_agg = _intent_aggregation(intent)
    sql_funcs = {str(f).upper() for f, _c in facts.get("aggregations", [])}
    if intent_agg is None:
        return NA if not sql_funcs else UNKNOWN
    if not sql_funcs:
        return MISMATCH
    return MATCH if intent_agg in sql_funcs else MISMATCH


def _entity(intent, facts):
    subj = (getattr(intent, "subject_table", "") or "").lower()
    ents = {str(e).lower() for e in facts.get("entities", [])}
    if not subj:
        return UNKNOWN
    if not ents:
        return UNKNOWN
    return MATCH if subj in ents else MISMATCH


def _dimension(intent, facts):
    gcols = {c for c in (getattr(intent, "group_col", None), getattr(intent, "group_col2", None)) if c}
    sql_groups = set(facts.get("groupings", []))
    if not gcols and not sql_groups:
        return NA
    if not gcols or not sql_groups:
        return MISMATCH
    return MATCH if gcols & sql_groups else MISMATCH


def _temporal(intent, sql, facts, sm):
    wants = bool(getattr(intent, "time_bucket", None))
    try:
        from veda.intent_sql_alignment import _sql_has_temporal_bucket
        has = _sql_has_temporal_bucket(sql, sm, facts)
    except Exception:
        return UNKNOWN
    if not wants and not has:
        return NA
    if wants and has:
        return MATCH
    return MISMATCH


def _filters(intent, facts, reason):
    if reason == "QUALIFIER_GROUNDING_DECLINE":
        return UNKNOWN                                   # intent's filter was the ungrounded decline reason
    icols = {getattr(f, "column", None) for f in (getattr(intent, "filters", []) or [])}
    icols.discard(None)
    scols = {c for (c, _op, _v) in facts.get("filters", [])}
    if not icols and not scols:
        return NA
    if not icols or not scols:
        return MISMATCH
    return MATCH if icols & scols else MISMATCH


def shadow_agreement(intent, sql, sm, reason=""):
    """Field-level agreement dict between the preserved QueryIntent and the SQL's actual referents."""
    facts = _facts(sql)
    return {
        "aggregation": _agg(intent, facts),
        "entity": _entity(intent, facts),
        "dimension": _dimension(intent, facts),
        "temporal": _temporal(intent, sql, facts, sm),
        "filters": _filters(intent, facts, reason),
    }


def record_shadow(query, sql, sm):
    """Build + LOG the structured shadow record for THIS request's preserved fast-path intent vs the
    final SQL. No-op when the flag is off, no intent was preserved, or there is no SQL. Fully guarded —
    a measurement failure can never affect the query."""
    if not _enabled() or not sql:
        return None
    try:
        from query.fast_path import get_preserved_intent
        preserved = get_preserved_intent()
        if not preserved or not preserved.get("intent"):
            return None
        intent, reason = preserved["intent"], preserved.get("reason", "")
        facts = _facts(sql)
        rec = {
            "query": query,
            "fastpath_decline_reason": reason,
            "intent": {
                "subject_table": getattr(intent, "subject_table", None),
                "query_type": getattr(intent, "query_type", None),
                "aggregation": _intent_aggregation(intent),
                "group_column": getattr(intent, "group_col", None),
                "time_bucket": getattr(intent, "time_bucket", None),
                "filters": [getattr(f, "column", None) for f in (getattr(intent, "filters", []) or [])],
            },
            "sql_facts": {
                "tables": facts.get("entities", []),
                "aggregations": [list(a) for a in facts.get("aggregations", [])],
                "group_by": facts.get("groupings", []),
                "filters": [c for (c, _o, _v) in facts.get("filters", [])],
            },
            "agreement": shadow_agreement(intent, sql, sm, reason),
        }
        try:
            from utils.logger import get_logger
            get_logger(__name__).info("CANONICAL_INTENT_SHADOW %s", json.dumps(rec, default=str))
        except Exception:
            print("CANONICAL_INTENT_SHADOW " + json.dumps(rec, default=str), flush=True)
        return rec
    except Exception:
        return None
