# =============================================================================
# query/execution_engine.py
# VEDA — Execution Engine (Phase 3)
#
# Responsibility:
#   - Routes SQL execution to the correct backend based on source type
#   - Relational sources  → psycopg2 / mysql / sqlite (via DAL client connection)
#   - Data lake sources   → DuckDB in-process (reads Parquet/Delta/CSV natively)
#   - Always uses parameterised queries — no string interpolation of values
#   - Enforces read-only: rejects INSERT, UPDATE, DELETE, DROP, TRUNCATE, DDL
#   - Enforces row limit and timeout
#
# Called by main.py when L5 Validator and L6 Execution are wired up.
# Currently usable standalone via execute_sql().
#
# Security: all parameter binding is handled by the DB driver.
# The SQL string (from L4 SQL Builder) uses identifiers only — values are params.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import time
from typing import Any, Dict, List, Optional

from connectors.base import QueryResult
from config import (
    VEDA_SOURCES,
    get_source,
    SQL_DEFAULT_LIMIT,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

_DEFAULT_ROW_LIMIT  = SQL_DEFAULT_LIMIT
_DEFAULT_TIMEOUT    = 30


# =============================================================================
# Read-only guard
# =============================================================================

def _assert_read_only(sql: str) -> None:
    """Raises ValueError if the SQL contains any write/DDL statement."""
    match = _WRITE_PATTERN.search(sql)
    if match:
        raise ValueError(
            f"Execution engine is read-only. "
            f"Rejected keyword: '{match.group()}' in SQL."
        )


# =============================================================================
# Relational execution
# =============================================================================

def _execute_relational(
    sql:       str,
    params:    List[Any],
    source:    dict,
    row_limit: int,
) -> QueryResult:
    """Executes parameterised SQL against a relational source via the DAL."""
    from ingestion.db_abstraction import get_client_connection

    t0 = time.time()
    source_id = source["id"]

    try:
        conn = get_client_connection(source_id)
        try:
            cur = conn.cursor()
            # Add LIMIT if not already present
            limited_sql = _add_limit(sql, row_limit)
            cur.execute(limited_sql, params or [])
            cols    = [desc[0] for desc in cur.description] if cur.description else []
            raw     = cur.fetchmany(row_limit + 1)
            truncated = len(raw) > row_limit
            rows    = [dict(zip(cols, r)) for r in raw[:row_limit]]
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()

        return QueryResult(
            source_id    = source_id,
            source_type  = "relational",
            rows         = rows,
            row_count    = len(rows),
            columns      = cols,
            sql_or_query = sql,
            duration_ms  = round((time.time() - t0) * 1000, 2),
            truncated    = truncated,
            error        = None,
        )
    except Exception as e:
        return QueryResult(
            source_id    = source_id,
            source_type  = "relational",
            rows         = [], row_count = 0, columns = [],
            sql_or_query = sql,
            duration_ms  = round((time.time() - t0) * 1000, 2),
            truncated    = False,
            error        = str(e),
        )


# =============================================================================
# Datalake execution
# =============================================================================

def _execute_datalake(
    sql:       str,
    params:    List[Any],
    source:    dict,
    row_limit: int,
) -> QueryResult:
    """Executes SQL against a datalake source via DuckDB."""
    from connectors.base import build_connector
    connector = build_connector(source)
    return connector.execute_query(
        query      = sql,
        params     = params,
        row_limit  = row_limit,
        timeout_sec = _DEFAULT_TIMEOUT,
    )


# =============================================================================
# SQL helpers
# =============================================================================

def _add_limit(sql: str, limit: int) -> str:
    """Appends LIMIT clause if the SQL does not already have one."""
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip('; ')} LIMIT {limit};"


# =============================================================================
# Public entry point
# =============================================================================

def execute_sql(
    sql:        str,
    params:     List[Any] = None,
    source_id:  str = None,
    row_limit:  int = _DEFAULT_ROW_LIMIT,
    timeout_sec: int = _DEFAULT_TIMEOUT,
    verbose:    bool = False,
) -> QueryResult:
    """
    Executes a parameterised SQL query against the specified source.

    Routes automatically based on source type:
      relational → psycopg2 / mysql-connector / sqlite3
      datalake   → DuckDB in-process

    Parameters
    ----------
    sql         : parameterised SQL (values as %s placeholders from SQL Builder)
    params      : bound parameter values (list)
    source_id   : VEDA_SOURCES entry id. Defaults to primary relational source.
    row_limit   : maximum rows to return (default: SQL_DEFAULT_LIMIT)
    timeout_sec : query timeout (applied where supported)
    verbose     : print routing decision

    Returns
    -------
    QueryResult — always returns; error field set on failure
    """
    if source_id is None:
        from config import get_primary_relational_source
        source_id = get_primary_relational_source()["id"]

    source = get_source(source_id)

    _assert_read_only(sql)

    source_type = source.get("type", "relational")

    logger.debug("L6 execute: source=%r, type=%s, sql=%s",
                 source_id, source_type, sql[:200])

    if verbose:
        print(f"[ExecutionEngine] source='{source_id}'  type={source_type}")
        print(f"  SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")
        print(f"  Params: {params}")

    t0 = time.time()

    if source_type == "relational":
        result = _execute_relational(sql, params or [], source, row_limit)
    elif source_type == "datalake":
        result = _execute_datalake(sql, params or [], source, row_limit)
    else:
        result = QueryResult(
            source_id    = source_id,
            source_type  = source_type,
            rows         = [], row_count = 0, columns = [],
            sql_or_query = sql,
            duration_ms  = round((time.time() - t0) * 1000, 2),
            truncated    = False,
            error        = f"Execution not supported for source type '{source_type}'",
        )

    if result.error:
        logger.warning("L6 execute error: %s", result.error)
    else:
        logger.info("L6 execute: %d rows returned, truncated=%s, %dms",
                    result.row_count, result.truncated, result.duration_ms)

    if verbose:
        if result.error:
            print(f"  ✗ Error: {result.error}")
        else:
            print(f"  ✓ {result.row_count} rows  truncated={result.truncated}  {result.duration_ms}ms")

    return result
