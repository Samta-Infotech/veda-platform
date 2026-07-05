# =============================================================================
# ingestion/schema_unifier.py
# VEDA — Schema Unifier (Phase 5)
#
# Responsibility:
#   Converts any connector's schema output into the legacy dict format that
#   schema_scanner.py's run_schema_scanner() expects as its `raw_schema` arg.
#
# Why this exists:
#   schema_scanner.py was written for the relational source's dict format.
#   New connectors (datalake, nosql) return typed dataclasses (RawSchema,
#   List[NoSQLCollection]).  This adapter converts those to the same dict so
#   schema_scanner.py — and everything downstream — works unchanged.
#
# Entry points:
#   raw_schema_to_dict(raw_schema)              — RawSchema dataclass → dict
#   nosql_collections_to_dict(collections, …)  — NoSQL schema → dict
#
# Output dict format (consumed by schema_scanner.run_schema_scanner):
#   {
#       "tables": [
#           {
#               "table_id":   str,
#               "table_name": str,
#               "row_count":  int,
#               "columns": [
#                   {
#                       "col_id":          str,
#                       "col_name":        str,
#                       "data_type":       str,
#                       "is_pk":           bool,
#                       "is_fk":           bool,
#                       "fk_ref_table":    Optional[str],
#                       "fk_ref_col":      Optional[str],
#                       "fk_ref_table_id": Optional[str],
#                       "nullable":        bool,
#                       "cardinality":     Optional[int],
#                   }
#               ]
#           }
#       ],
#       "name_to_id":       {table_name: table_id},
#       "excluded_columns": [],
#       "stats":            {total_tables, total_columns, total_fk_edges, excluded_count},
#   }
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
from typing import List

from connectors.base import NoSQLCollection, RawSchema


# =============================================================================
# RawSchema → dict
# =============================================================================

def raw_schema_to_dict(raw_schema: RawSchema) -> dict:
    """
    Converts a RawSchema dataclass (from datalake and relational connectors)
    to the legacy dict format expected by schema_scanner.run_schema_scanner().
    """
    tables:     list = []
    name_to_id: dict = {}

    for rt in raw_schema.tables:
        name_to_id[rt.table_name] = rt.table_id
        cols = []
        for rc in rt.columns:
            cols.append({
                "col_id":          rc.col_id,
                "col_name":        rc.col_name,
                "data_type":       rc.data_type,
                "is_pk":           rc.is_pk,
                "is_fk":           rc.is_fk,
                "fk_ref_table":    rc.fk_ref_table,
                "fk_ref_col":      rc.fk_ref_col,
                "fk_ref_table_id": rc.fk_ref_table_id,
                "nullable":        rc.nullable,
                "cardinality":     rc.cardinality,
            })
        tables.append({
            "table_id":   rt.table_id,
            "table_name": rt.table_name,
            "row_count":  rt.row_count,
            "columns":    cols,
        })

    stats = dict(raw_schema.stats)
    stats.setdefault("excluded_count", 0)

    return {
        "tables":           tables,
        "name_to_id":       name_to_id,
        "excluded_columns": [],
        "stats":            stats,
    }


# =============================================================================
# NoSQL collections → dict
# =============================================================================

def nosql_collections_to_dict(
    collections: List[NoSQLCollection],
    source_id:   str,
    engine:      str,
) -> dict:
    """
    Converts a list of NoSQLCollection objects (from nosql connectors) to the
    legacy dict format expected by schema_scanner.run_schema_scanner().

    Each collection becomes a table; each inferred field becomes a column.
    """
    tables:     list = []
    name_to_id: dict = {}
    total_cols       = 0

    for col in collections:
        table_id = col.collection_id or str(uuid.uuid4())
        name_to_id[col.collection_name] = table_id
        cols = []

        for field in col.inferred_fields:
            fname      = field["name"]
            name_lower = fname.lower()
            is_pk      = (
                name_lower in ("id", "_id")
                or name_lower == f"{col.collection_name.lower()}_id"
            )
            cols.append({
                "col_id":          str(uuid.uuid4()),
                "col_name":        fname,
                "data_type":       field.get("type", "varchar"),
                "is_pk":           is_pk,
                "is_fk":           False,
                "fk_ref_table":    None,
                "fk_ref_col":      None,
                "fk_ref_table_id": None,
                "nullable":        field.get("nullable", True),
                "cardinality":     None,
            })

        tables.append({
            "table_id":   table_id,
            "table_name": col.collection_name,
            "row_count":  col.doc_count,
            "columns":    cols,
        })
        total_cols += len(cols)

    return {
        "tables":           tables,
        "name_to_id":       name_to_id,
        "excluded_columns": [],
        "stats": {
            "total_tables":   len(tables),
            "total_columns":  total_cols,
            "total_fk_edges": 0,
            "excluded_count": 0,
        },
    }
