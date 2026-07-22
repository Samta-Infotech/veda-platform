# veda/semantic_validation.py
# VEDA — SHARED analytical-semantics validation for generated SQL.
#
# One deterministic, model-free check reused by every path that produces
# analytical SQL/plans (Tier-1 LLM branch, Tier-2, LangGraph). It answers "does
# the SQL still mean what the question asked?" using GENERIC SQL/analytical
# invariants + the onboarded SEMANTIC MODEL — never table/column/business-entity
# names, never an `_id`/`_name` suffix rule (dimension vs identifier is decided by
# `semantic_type`, display columns by the one governed resolver
# generation._resolve_display_column).
#
# It DETECTS and REPORTS; it never rewrites SQL. Callers decide what to do with a
# finding (record to the trace in advisory mode, or — behind a flag — route to the
# existing refuse/clarify/repair machinery). Zero new LLM/SLM calls.
#
# Findings are dicts: {"code", "severity" ("error"|"warning"), "detail",
# "column" (optional)}. `error` = the SQL provably lost requested semantics;
# `warning` = a semantic-quality issue (e.g. a technical id used as a user-facing
# dimension when a display column exists).
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# generic tokens by which a user EXPLICITLY asks for a technical key/code — when
# present, an identifier column is what they want, so identifier-dimension and
# id-drop findings are suppressed. Language layer, not schema vocabulary.
_EXPLICIT_ID_RE = re.compile(
    r"\b(ids?|codes?|keys?|numbers?|references?|identifiers?|uuids?|guids?)\b", re.I)


def user_requested_identifier(query: str) -> bool:
    """True when the question explicitly asks for a technical id/code/key/number —
    in which case an identifier column is the correct, requested output and must be
    preserved (not swapped for a display label)."""
    return bool(_EXPLICIT_ID_RE.search(query or ""))


def _owning_tables(sm: dict, entities: List[str], col: str) -> List[str]:
    cols = (sm or {}).get("columns", {})
    return [t for t in entities if f"{t}.{col}" in cols]


def _semantic_type(sm: dict, table: str, col: str) -> Optional[str]:
    return ((sm or {}).get("columns", {}).get(f"{table}.{col}", {}) or {}).get("semantic_type")


def _display_column(table: str, sm: dict) -> Optional[str]:
    try:
        from veda.generation import _resolve_display_column
        return _resolve_display_column(table, sm)
    except Exception:
        return None


def _fk_connected(entities: List[str], graph: Optional[dict]) -> bool:
    """True if every table in `entities` is reachable from the others over the
    undirected FK/relationship graph — i.e. every JOIN is grounded in a known
    relationship. None graph → treated as connected (skip the check)."""
    if not graph or len(entities) < 2:
        return True
    adj: Dict[str, set] = {}
    for e in graph.get("edges", []) or []:
        s, t = e.get("source_table"), e.get("target_table")
        if s and t:
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)
    ents = set(entities)
    start = next(iter(ents))
    seen, stack = {start}, [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, ()):  # walk the whole graph, not only ents
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return ents <= seen


def validate_analytical_semantics(
    query: str,
    sql: str,
    sm: Optional[dict] = None,
    *,
    graph: Optional[dict] = None,
    facts: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return a list of semantic findings for `sql` given the `query` intent and the
    semantic model. Empty list = no issues detected. Pure/deterministic.

    `facts`: pre-parsed extract_sql_facts output (aggregations/groupings/orderings/
    limit/entities). Passed in when the caller already parsed the SQL, else parsed
    here. `graph`: FK/relationship graph {"edges":[{source_table,target_table}]} for
    the join-grounding check; omit to skip it."""
    findings: List[Dict[str, Any]] = []
    if not sql:
        return findings
    try:
        from veda.planning import aggregate_operator, grouped_mode
    except Exception:
        return findings
    if facts is None:
        try:
            from veda.business_explain import extract_sql_facts
            facts = extract_sql_facts(sql)
        except Exception:
            return findings

    aggregations = facts.get("aggregations") or []
    groupings = facts.get("groupings") or []
    entities = facts.get("entities") or []
    agg_funcs = {str(f).upper() for f, _ in aggregations}

    requested_op = aggregate_operator(query)
    is_grouped = grouped_mode(query) is not None

    # 1. OPERATOR PRESERVED — the requested aggregate must survive into SQL.
    if requested_op:
        if agg_funcs and requested_op not in agg_funcs:
            findings.append({
                "code": "operator_mismatch", "severity": "error",
                "detail": (f"query requests {requested_op} but SQL aggregates with "
                           f"{', '.join(sorted(agg_funcs))}")})
        elif not agg_funcs and is_grouped:
            # a grouped aggregate question that produced no aggregate at all
            findings.append({
                "code": "operator_dropped", "severity": "error",
                "detail": f"query requests a {requested_op} aggregate but SQL has none"})

    # 2. GROUP BY PRESENT — a grouped aggregate must actually GROUP BY.
    if is_grouped and agg_funcs and not groupings:
        findings.append({
            "code": "missing_group_by", "severity": "error",
            "detail": "grouped-aggregate question but SQL has no GROUP BY"})

    # 3. USER-FACING DIMENSION NOT AN UNNECESSARY IDENTIFIER — a GROUP BY column
    #    that is a technical IDENTIFIER (by semantic_type, not by name) when its
    #    table has a governed display column is a semantic-quality issue — UNLESS the
    #    user explicitly asked for the id/code/key.
    if groupings and not user_requested_identifier(query):
        for g in groupings:
            for t in _owning_tables(sm, entities, g):
                if _semantic_type(sm, t, g) == "IDENTIFIER":
                    disp = _display_column(t, sm)
                    if disp and disp != g and _semantic_type(sm, t, disp) in (
                            "CATEGORY", "CATEGORICAL", "FREE_TEXT"):
                        findings.append({
                            "code": "identifier_dimension", "severity": "warning",
                            "column": f"{t}.{g}",
                            "detail": (f"grouped by identifier {t}.{g}; a display column "
                                       f"{t}.{disp} exists for this entity")})
                    break

    # 4. JOINS GROUNDED — every table in a multi-table query must be connected over
    #    known relationships (no cartesian/invented join).
    if not _fk_connected(entities, graph):
        findings.append({
            "code": "ungrounded_join", "severity": "error",
            "detail": (f"tables {sorted(entities)} are not all connected by known "
                       f"relationships")})

    return findings


def has_errors(findings: List[Dict[str, Any]]) -> bool:
    return any(f.get("severity") == "error" for f in findings)
