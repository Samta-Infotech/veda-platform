# =============================================================================
# ingestion/schema_scanner.py
# VEDA POC — Step 1: Schema Scanner
#
# Responsibility:
#   - Accepts the raw schema dict from get_simulated_schema()
#   - Validates structure and completeness
#   - Normalises every column into a flat ScanResult
#   - Enforces sensitive column exclusion (second gate after simulate_schema.py)
#   - Produces a clean, typed output consumed by semantic_type_inference.py
#
# In production: replace get_simulated_schema() with a real DB connector.
# This file's output contract (ScanResult) never changes.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass, field
from typing import List, Optional
from schema.simulate_schema import get_simulated_schema
from config import SENSITIVE_PATTERNS
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class ScannedColumn:
    """
    Normalised representation of a single column after scanning.
    This is the unit of data passed to all downstream ingestion steps.
    """
    col_id:           str             # stable UUID
    col_name:         str             # raw column name from schema
    table_id:         str             # UUID of parent table
    table_name:       str             # name of parent table
    data_type:        str             # integer | varchar | numeric | timestamp | boolean
    is_pk:            bool
    is_fk:            bool
    fk_ref_table_id:  Optional[str]   # UUID of referenced table (None if not FK)
    fk_ref_table:     Optional[str]   # name of referenced table (None if not FK)
    fk_ref_col:       Optional[str]   # name of referenced column (None if not FK)
    nullable:         bool
    cardinality:      Optional[int]   # None = not sampled


@dataclass
class ScannedTable:
    """
    Normalised representation of a single table after scanning.
    """
    table_id:    str
    table_name:  str
    row_count:   int
    columns:     List[ScannedColumn] = field(default_factory=list)


@dataclass
class ScanResult:
    """
    Top-level output of the schema scanner.
    Consumed directly by semantic_type_inference.py.
    """
    tables:            List[ScannedTable]
    all_columns:       List[ScannedColumn]   # flat list across all tables
    fk_edges:          List[dict]            # list of {from_col_id, to_col_id, from_table_id, to_table_id}
    excluded_columns:  List[str]             # "table.col" strings removed at this stage
    stats: dict = field(default_factory=dict)


# =============================================================================
# Validation helpers
# =============================================================================

VALID_DATA_TYPES = {"integer", "varchar", "numeric", "timestamp", "boolean", "timestamptz", "date", "uuid", "character", "bigint", "text", "double", "jsonb", "inet", "smallint", "time"} 


def _validate_column(raw_col: dict, table_name: str) -> List[str]:
    """
    Returns a list of validation error strings for a column.
    Empty list means the column is valid.
    """
    errors = []

    if not raw_col.get("col_id"):
        errors.append(f"[{table_name}] column missing col_id")

    if not raw_col.get("col_name"):
        errors.append(f"[{table_name}] column missing col_name")

    dtype = raw_col.get("data_type", "").lower()
    if dtype not in VALID_DATA_TYPES:
        errors.append(
            f"[{table_name}.{raw_col.get('col_name')}] "
            f"unknown data_type '{dtype}'"
        )

    if raw_col.get("is_fk") and not raw_col.get("fk_ref_table"):
        errors.append(
            f"[{table_name}.{raw_col.get('col_name')}] "
            f"is_fk=True but fk_ref_table is missing"
        )

    return errors


def _validate_table(raw_table: dict) -> List[str]:
    """
    Returns a list of validation error strings for a table.
    """
    errors = []

    if not raw_table.get("table_id"):
        errors.append(f"table '{raw_table.get('table_name')}' missing table_id")

    if not raw_table.get("table_name"):
        errors.append("table missing table_name")

    if not isinstance(raw_table.get("columns"), list) or len(raw_table["columns"]) == 0:
        errors.append(f"table '{raw_table.get('table_name')}' has no columns")

    pk_cols = [c for c in raw_table.get("columns", []) if c.get("is_pk")]
    # Missing PK is a warning, not an error — some tables are legitimately PK-less
    if len(pk_cols) > 1:
        errors.append(
            f"table '{raw_table.get('table_name')}' has multiple PKs: "
            + ", ".join(c["col_name"] for c in pk_cols)
        )

    return errors


# =============================================================================
# Sensitive column filter — second gate
# simulate_schema.py already excluded these, but we re-check defensively
# =============================================================================

def _is_sensitive(col_name: str) -> bool:
    col_lower = col_name.lower()
    return any(pattern in col_lower for pattern in SENSITIVE_PATTERNS)


# =============================================================================
# Core scanner
# =============================================================================

def _normalise_column(raw_col: dict, table_id: str, table_name: str) -> ScannedColumn:
    """Converts a raw column dict into a typed ScannedColumn."""
    return ScannedColumn(
        col_id          = raw_col["col_id"],
        col_name        = raw_col["col_name"],
        table_id        = table_id,
        table_name      = table_name,
        data_type       = raw_col["data_type"].lower(),
        is_pk           = raw_col.get("is_pk", False),
        is_fk           = raw_col.get("is_fk", False),
        fk_ref_table_id = raw_col.get("fk_ref_table_id", None),
        fk_ref_table    = raw_col.get("fk_ref_table", None),
        fk_ref_col      = raw_col.get("fk_ref_col", None),
        nullable        = raw_col.get("nullable", True),
        cardinality     = raw_col.get("cardinality", None),
    )


def _extract_fk_edges(all_columns: List[ScannedColumn]) -> List[dict]:
    """
    Builds the FK edge list from the flat column list.
    Each edge: { from_col_id, from_table_id, to_table_id, to_col_name }
    Used by reg_builder.py to construct graph edges.
    """
    edges = []
    # Build a quick lookup: (table_name, col_name) -> col_id
    col_lookup = {
        (c.table_name, c.col_name): c.col_id
        for c in all_columns
    }

    for col in all_columns:
        if col.is_fk and col.fk_ref_table and col.fk_ref_col:
            to_col_id = col_lookup.get((col.fk_ref_table, col.fk_ref_col))
            edges.append({
                "from_col_id":   col.col_id,
                "from_table_id": col.table_id,
                "to_col_id":     to_col_id,        # None if ref not found
                "to_table_id":   col.fk_ref_table_id,
                "from_col_name": col.col_name,
                "to_col_name":   col.fk_ref_col,
                "from_table":    col.table_name,
                "to_table":      col.fk_ref_table,
            })

    return edges


# =============================================================================
# Public entry point
# =============================================================================

def run_schema_scanner(raw_schema: dict = None, verbose: bool = False) -> ScanResult:
    """
    Main entry point for Step 1.

    Parameters
    ----------
    raw_schema : dict, optional
        Output of get_simulated_schema(). If None, calls it internally.
        Pass explicitly in tests to inject a custom schema.
    verbose : bool
        Print progress to stdout if True.

    Returns
    -------
    ScanResult
        Normalised, validated, sensitive-column-free schema ready for Step 2.

    Raises
    ------
    ValueError
        If validation errors are found. Lists all errors before raising.
    """
    if raw_schema is None:
        raw_schema = get_simulated_schema()

    logger.debug("Starting schema scan: %d raw tables", raw_schema["stats"]["total_tables"])

    if verbose:
        print("[SchemaScanner] Starting scan...")
        print(f"  Raw tables      : {raw_schema['stats']['total_tables']}")
        print(f"  Raw columns     : {raw_schema['stats']['total_columns']}")
        print(f"  Pre-excluded    : {raw_schema['stats']['excluded_count']} sensitive cols")

    # ------------------------------------------------------------------
    # Phase 1 — validate all tables and columns
    # ------------------------------------------------------------------
    all_errors = []
    for raw_table in raw_schema["tables"]:
        all_errors.extend(_validate_table(raw_table))
        for raw_col in raw_table["columns"]:
            all_errors.extend(_validate_column(raw_col, raw_table["table_name"]))

    # Collect no-PK warnings separately — they don't block the pipeline
    all_warnings = []
    for raw_table in raw_schema["tables"]:
        pk_cols = [c for c in raw_table.get("columns", []) if c.get("is_pk")]
        if not pk_cols:
            all_warnings.append(f"table '{raw_table.get('table_name')}' has no primary key (PK bridge injection skipped)")

    if all_errors:
        error_msg = "\n".join(f"  - {e}" for e in all_errors)
        raise ValueError(f"[SchemaScanner] Validation failed:\n{error_msg}")

    if verbose:
        print(f"  Validation      : PASSED")
        for w in all_warnings:
            print(f"  ⚠  {w}")

    # ------------------------------------------------------------------
    # Phase 2 — normalise and apply second sensitive-column gate
    # ------------------------------------------------------------------
    scanned_tables   = []
    all_columns_flat = []
    second_gate_excluded = []

    for raw_table in raw_schema["tables"]:
        table_id   = raw_table["table_id"]
        table_name = raw_table["table_name"]
        row_count  = raw_table.get("row_count", 0)

        safe_columns = []
        for raw_col in raw_table["columns"]:
            if _is_sensitive(raw_col["col_name"]):
                second_gate_excluded.append(f"{table_name}.{raw_col['col_name']}")
                continue

            scanned_col = _normalise_column(raw_col, table_id, table_name)
            safe_columns.append(scanned_col)
            all_columns_flat.append(scanned_col)

        scanned_table = ScannedTable(
            table_id   = table_id,
            table_name = table_name,
            row_count  = row_count,
            columns    = safe_columns,
        )
        scanned_tables.append(scanned_table)

    # ------------------------------------------------------------------
    # Phase 3 — extract FK edges from flat column list
    # ------------------------------------------------------------------
    fk_edges = _extract_fk_edges(all_columns_flat)

    # ------------------------------------------------------------------
    # Phase 4 — combine excluded lists and build stats
    # ------------------------------------------------------------------
    all_excluded = raw_schema.get("excluded_columns", []) + second_gate_excluded

    stats = {
        "total_tables":         len(scanned_tables),
        "total_columns":        len(all_columns_flat),
        "total_fk_edges":       len(fk_edges),
        "pk_columns":           sum(1 for c in all_columns_flat if c.is_pk),
        "fk_columns":           sum(1 for c in all_columns_flat if c.is_fk),
        "nullable_columns":     sum(1 for c in all_columns_flat if c.nullable),
        "excluded_total":       len(all_excluded),
        "second_gate_excluded": len(second_gate_excluded),
    }

    if verbose:
        print(f"  Tables scanned  : {stats['total_tables']}")
        print(f"  Columns retained: {stats['total_columns']}")
        print(f"  FK edges        : {stats['total_fk_edges']}")
        print(f"  Excluded (total): {stats['excluded_total']}")
        if second_gate_excluded:
            print(f"  ⚠ Second gate excluded: {second_gate_excluded}")
        print("[SchemaScanner] Done.\n")

    logger.info(
        "Schema scan complete: %d tables, %d columns, %d FK edges, %d excluded",
        stats["total_tables"], stats["total_columns"],
        stats["total_fk_edges"], stats["excluded_total"],
    )
    if all_warnings:
        for w in all_warnings:
            logger.debug("Schema warning: %s", w)

    return ScanResult(
        tables           = scanned_tables,
        all_columns      = all_columns_flat,
        fk_edges         = fk_edges,
        excluded_columns = all_excluded,
        stats            = stats,
    )


# =============================================================================
# Smoke test — python ingestion/schema_scanner.py
# =============================================================================

if __name__ == "__main__":
    result = run_schema_scanner(verbose=True)

    print("=" * 60)
    print("VEDA POC — Schema Scanner Output")
    print("=" * 60)
    print(f"  Tables          : {result.stats['total_tables']}")
    print(f"  Columns         : {result.stats['total_columns']}")
    print(f"  FK edges        : {result.stats['total_fk_edges']}")
    print(f"  PK columns      : {result.stats['pk_columns']}")
    print(f"  FK columns      : {result.stats['fk_columns']}")
    print(f"  Nullable cols   : {result.stats['nullable_columns']}")
    print(f"  Excluded total  : {result.stats['excluded_total']}")
    print()

    print("FK edges detected:")
    for edge in result.fk_edges:
        print(
            f"  {edge['from_table']}.{edge['from_col_name']}"
            f"  →  {edge['to_table']}.{edge['to_col_name']}"
        )
    print()

    print("Excluded sensitive columns:")
    for exc in result.excluded_columns:
        print(f"  ✗  {exc}")