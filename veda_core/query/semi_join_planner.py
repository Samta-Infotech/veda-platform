"""query/semi_join_planner.py — bounded SEMI_JOIN / FILTER_BY_OTHER_SOURCE cross-source strategy.

The ONE thing the SLM decides here is a CHOICE among pre-built, already-validated candidate directed
key-pairs (or "not a semi-join"). It never emits a source id, table, column, key, or SQL — every
candidate is assembled by CODE from the grounded `cross_source_fk` join hints, so the model cannot
invent schema. Deterministic code then assembles the semi-join SQL and hands it to the EXISTING
federated executor (compose_federated → FederatedExecutor).

Semantics (the only operation implemented here):
    SELECT DISTINCT <output rows>
    FROM   <output_source.output_table>
    WHERE  <output_key> IN (SELECT DISTINCT <filter_key> FROM <filter_source.filter_table>)

Flag: FEDERATED_SEMI_JOIN_STRUCTURED_ENABLED (default OFF → this module is never entered).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from query.federated_executor import catalog_name
from slm._call_slm import call_slm

OP_SEMI_JOIN = "SEMI_JOIN_FILTER"


def _pg_schema(source_id: str) -> str:
    """The postgres SCHEMA a relational source's tables live under (DuckDB's postgres ATTACH exposes
    them at <catalog>.<schema>.<table>). Delegates to the shared, per-source resolver
    (cross_source_composer.resolve_pg_schema) — discovered from the source's own DB when
    FEDERATED_SCHEMA_DISCOVERY_ENABLED, else 'public' (byte-identical). One source of truth, keyed per
    source_id, so a multi-relational-source system resolves each correctly."""
    try:
        from query.cross_source_composer import resolve_pg_schema
        return resolve_pg_schema(source_id)
    except Exception:
        return "public"


def _qualified(source_id: str, table: str, kinds: dict) -> str:
    """Fully-qualified DuckDB name, SCHEMA-AWARE for postgres. Parquet views: <cat>."table".
    Postgres: <cat>.<schema>."table" with the source's real schema."""
    cat = catalog_name(source_id)
    if kinds.get(str(source_id)) == "postgres":
        return f'{cat}.{_pg_schema(source_id)}."{table}"'
    return f'{cat}."{table}"'


def semi_join_enabled() -> bool:
    try:
        from config import FEDERATED_SEMI_JOIN_STRUCTURED_ENABLED
        return bool(FEDERATED_SEMI_JOIN_STRUCTURED_ENABLED)
    except Exception:
        return False


def build_candidates(by_source: Dict[str, Dict[str, list]], hints: List[dict]) -> List[dict]:
    """Directed (output, filter) candidates from the grounded join hints. Each hint {a,b} yields TWO
    directions (a filtered by b, and b filtered by a); we keep only directions whose OUTPUT table is
    actually a selected/retrieved table for its source (so we return rows the query is about). Every
    field is copied from the hint — nothing is invented."""
    def _selected(sid: str, tbl: str) -> bool:
        return tbl in (by_source.get(str(sid), {}) or {})

    seen = set()
    out: List[dict] = []
    for h in hints or []:
        a = (str(h.get("a_src")), h.get("a_tbl"), h.get("a_col"))
        b = (str(h.get("b_src")), h.get("b_tbl"), h.get("b_col"))
        if not all(a) or not all(b):
            continue
        for (o_src, o_tbl, o_col), (f_src, f_tbl, f_col) in ((a, b), (b, a)):
            if o_src == f_src:
                continue                       # a semi-join spans two DIFFERENT sources
            if not _selected(o_src, o_tbl):
                continue                       # output must be a table the query surfaced
            key = (o_src, o_tbl, o_col, f_src, f_tbl, f_col)
            if key in seen:
                continue
            seen.add(key)
            out.append({"output_source_id": o_src, "output_table": o_tbl, "output_col": o_col,
                        "filter_source_id": f_src, "filter_table": f_tbl, "filter_col": f_col,
                        "tier": h.get("tier", "MEDIUM")})
    return out


_SYSTEM = (
    "You classify whether a data question is a SEMI-JOIN FILTER: return rows from ONE source, keeping "
    "only those whose key value ALSO appears in ANOTHER source (e.g. 'which vendors are in cities where "
    "we have assets' = vendors whose city is among asset cities). You are given NUMBERED candidate "
    "(output, filter) key-pairs. Reply with STRICT JSON only: "
    "{\"operation\": \"SEMI_JOIN_FILTER\" | \"NOT_SEMI_JOIN\", \"candidate\": <the number, or -1>}. "
    "Choose a candidate ONLY if the question means 'list/which <output> that are also present in / "
    "match / are where <filter>'. If the question AGGREGATES (per / average / total / count by / "
    "broken down), or COMPARES totals, or is about a DOCUMENT/policy, reply NOT_SEMI_JOIN. When unsure, "
    "reply NOT_SEMI_JOIN — a wrong pick is worse than deferring."
)


def _build_user(query: str, candidates: List[dict]) -> str:
    lines = [f"QUESTION: {query}", "", "CANDIDATES (pick one number, or -1 for NOT_SEMI_JOIN):"]
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}: output = rows of '{c['output_table']}' (source {c['output_source_id']}), "
            f"kept only where {c['output_table']}.{c['output_col']} appears in "
            f"{c['filter_table']}.{c['filter_col']} (source {c['filter_source_id']})")
    lines.append("")
    lines.append("Reply with STRICT JSON only.")
    return "\n".join(lines)


def _parse(raw: str) -> Optional[dict]:
    if raw is None:
        return None
    s = str(raw).strip()
    if "```" in s and s.count("```") >= 2:
        s = s.split("```")[1].lstrip("json").strip()
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        s = s[a:b + 1]
    try:
        return json.loads(s)
    except Exception:
        return None


def classify(query: str, candidates: List[dict], slm_call=None) -> Optional[int]:
    """Return the chosen candidate INDEX (0-based) or None. The SLM output is validated: operation must
    be SEMI_JOIN_FILTER and the candidate number must be within the supplied range — an out-of-range or
    NOT_SEMI_JOIN answer returns None (safe defer), never a guess."""
    if not candidates:
        return None
    slm_call = slm_call or (lambda system, user: call_slm(user, system=system,
                                                          purpose="semi_join_classify",
                                                          temperature=0.0, json_format=True))
    parsed = _parse(slm_call(_SYSTEM, _build_user(query, candidates)))
    if not isinstance(parsed, dict):
        return None
    if parsed.get("operation") != OP_SEMI_JOIN:
        return None
    n = parsed.get("candidate")
    try:
        n = int(n)
    except Exception:
        return None
    if n < 1 or n > len(candidates):        # 1-based in the prompt; -1/0/oob → defer
        return None
    return n - 1


def validate(cand: dict, by_source: Dict[str, Dict[str, list]], kinds: Dict[str, str]) -> bool:
    """Deterministic post-parse validation. The candidate was BUILT from grounded hints, but re-check
    every part is still in scope/schema before we assemble SQL (defense in depth)."""
    o_src, f_src = cand["output_source_id"], cand["filter_source_id"]
    if o_src == f_src:
        return False
    if o_src not in by_source or f_src not in by_source:
        return False                                        # both sources in the routed scope
    if cand["output_table"] not in by_source[o_src]:
        return False                                        # output table was selected
    if o_src not in kinds or f_src not in kinds:
        return False                                        # both resolvable to a surface kind
    if not (cand["output_col"] and cand["filter_col"] and cand["filter_table"]):
        return False
    return True


def assemble_sql(cand: dict, by_source: Dict[str, Dict[str, list]], kinds: Dict[str, str]) -> str:
    """Deterministic SEMI_JOIN SQL. No SLM. Output columns are the selected columns for the output
    table (the key column guaranteed present), else '*'. Source-qualified via the existing helper."""
    o_q = _qualified(cand["output_source_id"], cand["output_table"], kinds)
    f_q = _qualified(cand["filter_source_id"], cand["filter_table"], kinds)
    sel_cols = list(by_source.get(cand["output_source_id"], {}).get(cand["output_table"], []) or [])
    if cand["output_col"] not in sel_cols:
        sel_cols = sel_cols + [cand["output_col"]] if sel_cols else []
    proj = ", ".join(f'o."{c}"' for c in sel_cols) if sel_cols else "o.*"
    return (f'SELECT DISTINCT {proj} FROM {o_q} AS o '
            f'WHERE o."{cand["output_col"]}" IN '
            f'(SELECT DISTINCT f."{cand["filter_col"]}" FROM {f_q} AS f '
            f'WHERE f."{cand["filter_col"]}" IS NOT NULL)')


def plan_semi_join(query: str, by_source: Dict[str, Dict[str, list]], hints: List[dict],
                   kinds: Dict[str, str], slm_call=None) -> Optional[Tuple[str, dict]]:
    """Orchestrate: build validated candidates → bounded SLM picks one (or defers) → re-validate →
    assemble SQL. Returns (sql, chosen_candidate) or None (defer to the existing federated path)."""
    candidates = build_candidates(by_source, hints)
    if not candidates:
        return None
    idx = classify(query, candidates, slm_call=slm_call)
    if idx is None:
        return None
    cand = candidates[idx]
    if not validate(cand, by_source, kinds):
        return None
    return assemble_sql(cand, by_source, kinds), cand
