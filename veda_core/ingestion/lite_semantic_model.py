# =============================================================================
# ingestion/lite_semantic_model.py — deterministic semantic model for tabular/doc sources
#
# The heavy relational lane runs semantic_layer_v2 (Qwen understanding) to produce a rich
# semantic model. Tabular/doc sources have no LLM understanding pass, so without this they
# had NO per-source model and the query tier fell back to the global (homzhub) model — a
# source_ids:[tabular] query then validated against homzhub and refused its own columns.
#
# build_lite_sm() derives a structurally-compatible model (tables / columns /
# retrieval_documents) straight from the column schema, so the query tier can retrieve,
# validate and (for tabular) execute against the source's OWN columns. Shared by the
# ingestion dispatch (durable, per run) and scripts/backfill_semantic_model.py (one-off).
# =============================================================================
from __future__ import annotations

import re
from typing import Dict, List

_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_NUMERIC_DTYPES = ("integer", "bigint", "numeric", "double", "smallint", "real", "money", "float")


def _tokens(*names: str) -> List[str]:
    out: List[str] = []
    for n in names:
        for t in _TOKEN_RE.split((n or "").lower()):
            if t and len(t) >= 2 and t not in out:
                out.append(t)
    return out


def _agg_for(sem_type: str, data_type: str) -> List[str]:
    st = (sem_type or "").upper()
    if st in ("METRIC", "MEASURE") or (data_type or "").lower() in _NUMERIC_DTYPES:
        return ["SUM", "AVG", "MIN", "MAX", "COUNT"]
    return ["COUNT", "GROUP_BY"]


def build_lite_sm(columns: List[dict]) -> dict:
    """Deterministic semantic model from column rows. Each column dict needs:
    table_name, col_name, semantic_type, data_type, is_pk, is_fk. Output mirrors the
    semantic_layer_v2 keys (version/tables/columns/retrieval_documents/…)."""
    tables: Dict[str, dict] = {}
    cols: Dict[str, dict] = {}
    retrieval_documents: Dict[str, str] = {}
    by_table: Dict[str, list] = {}
    for c in columns:
        by_table.setdefault(c["table_name"], []).append(c)

    for tname, tcols in by_table.items():
        measures = [c["col_name"] for c in tcols
                    if (c.get("data_type") or "").lower() in _NUMERIC_DTYPES
                    and not c.get("is_pk") and not c.get("is_fk")]
        tables[tname] = {
            "table_name": tname,
            "business_purpose": f"Records from the {tname} source table.",
            "primary_entity": f"A single {tname} row.",
            "table_type": "TRANSACTION" if measures else "REFERENCE",
            "candidate_temporal_columns": [c["col_name"] for c in tcols
                                           if "date" in c["col_name"].lower() or "time" in c["col_name"].lower()],
            "candidate_measure_columns": measures,
        }
        for c in tcols:
            key = f"{tname}.{c['col_name']}"
            role = "IDENTIFIER" if (c.get("is_pk") or c.get("is_fk") or c.get("semantic_type") == "IDENTIFIER") else \
                   ("METRIC" if c["col_name"] in measures else "DIMENSION")
            aliases = _tokens(c["col_name"])
            cols[key] = {
                "col_name": c["col_name"], "table_name": tname,
                "semantic_type": c.get("semantic_type") or "FREE_TEXT", "analytics_role": role,
                "business_definition": f"{c['col_name'].replace('_', ' ')} of {tname}.",
                "aliases": aliases, "allowed_aggregations": _agg_for(c.get("semantic_type"), c.get("data_type")),
                "confidence": 0.6, "business_role": role.title(), "column_domain": tname,
                "sql_usage": "", "contains_pii": False, "sample_values": [],
                "importance_class": "NORMAL", "data_type": c.get("data_type") or "",
            }
            retrieval_documents[key] = (
                f"COLUMN: {c['col_name']} | ROLE: {role} | DEFINITION: "
                f"{c['col_name'].replace('_', ' ')} of {tname} | TERMS: {', '.join(aliases)}")
    return {"version": "2.0-lite", "tables": tables, "columns": cols,
            "retrieval_documents": retrieval_documents, "domain_synonyms": {}, "concept_graph": {}}


def columns_from_typed(typed_columns) -> List[dict]:
    """Normalize reg/inference typed-column objects to build_lite_sm's dict shape."""
    out: List[dict] = []
    for tc in typed_columns or []:
        out.append({
            "table_name": getattr(tc, "table_name", "") or "",
            "col_name": getattr(tc, "col_name", "") or "",
            "semantic_type": getattr(tc, "semantic_type", "") or "",
            "data_type": getattr(tc, "data_type", "") or "",
            "is_pk": bool(getattr(tc, "is_pk", False)),
            "is_fk": bool(getattr(tc, "is_fk", False)),
        })
    return out
