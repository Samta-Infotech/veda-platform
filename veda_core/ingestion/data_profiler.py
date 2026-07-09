# =============================================================================
# ingestion/data_profiler.py
# VEDA Final Architecture — Stage 1: Data Profiling
#
# Purpose:
#   Extract statistical profiles for every column in the database schema.
#   Used by L2 semantic layer for retrieval document enrichment.
#
# Output:
#   veda_profiling.json = {
#     "table.column": {
#       "null_percentage": float,
#       "distinct_count": int,
#       "min": scalar (numeric only),
#       "max": scalar (numeric only),
#       "avg": float (numeric only),
#       "top_values": List[str]
#     }
#   }
#
# Constraints:
#   - All statistics computed via SQL (no loading into memory)
#   - Sampling used for large tables
#   - Fast: <5 minutes for typical schemas (100-300 columns)
# =============================================================================

import sys
import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psycopg2
import psycopg2.extras
import psycopg2.sql
from config import (
    get_primary_relational_source,
    PROFILING_ENABLED,
    PROFILING_NULL_SAMPLE_SIZE,
    PROFILING_DISTINCT_LIMIT,
    PROFILING_TOP_VALUES_LIMIT,
    PROFILING_FILE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnProfile:
    """Profile statistics for a single column."""
    table_name: str
    col_name: str
    col_type: str
    null_percentage: float
    distinct_count: int
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_value: Optional[float] = None
    top_values: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "null_percentage": round(self.null_percentage, 2),
            "distinct_count": self.distinct_count,
            "min": str(self.min_value) if self.min_value is not None else None,
            "max": str(self.max_value) if self.max_value is not None else None,
            "avg": round(float(self.avg_value), 2) if self.avg_value is not None else None,
            "top_values": self.top_values or [],
        }


def _get_db_connection():
    """Get psycopg2 connection to primary relational source."""
    source = get_primary_relational_source()
    kw = dict(
        host=source["host"],
        port=source["port"],
        database=source["dbname"],
        user=source["user"],
        password=source["password"],
        connect_timeout=source.get("connect_timeout", 10),
    )
    if source.get("sslmode"):
        kw["sslmode"] = source["sslmode"]
    conn = psycopg2.connect(**kw)
    # Profiling queries below are unqualified ("table_name" with no schema prefix) — make
    # the configured schema authoritative for this connection instead of depending on the
    # role's server-side search_path, else a non-default schema gets silently profiled
    # against the wrong (usually empty/public) namespace.
    schema = source.get("schema")
    if schema:
        with conn.cursor() as cur:
            cur.execute(psycopg2.sql.SQL("SET search_path TO {}, public").format(
                psycopg2.sql.Identifier(schema)))
        conn.commit()
    return conn


def profile_column(
    conn,
    table_name: str,
    col_name: str,
    col_type: str,
) -> ColumnProfile:
    """
    Compute profile statistics for a single column.

    Args:
        conn: psycopg2 connection
        table_name: table name
        col_name: column name
        col_type: PostgreSQL data type (e.g., "integer", "varchar", "timestamp")

    Returns:
        ColumnProfile with all statistics
    """
    cursor = conn.cursor()

    # Total row count
    total_count_sql = f'SELECT COUNT(*) FROM "{table_name}"'
    cursor.execute(total_count_sql)
    total_rows = cursor.fetchone()[0]

    if total_rows == 0:
        logger.warning(f"{table_name}.{col_name}: empty table, skipping profile")
        return ColumnProfile(
            table_name=table_name,
            col_name=col_name,
            col_type=col_type,
            null_percentage=100.0,
            distinct_count=0,
            top_values=[],
        )

    # Null count
    null_count_sql = f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" IS NULL'
    cursor.execute(null_count_sql)
    null_count = cursor.fetchone()[0]
    null_percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0.0

    # Distinct count (with sample for large tables)
    sample_size = min(PROFILING_NULL_SAMPLE_SIZE, total_rows)
    distinct_sql = f"""
    SELECT COUNT(DISTINCT "{col_name}")
    FROM (
        SELECT "{col_name}" FROM "{table_name}"
        WHERE "{col_name}" IS NOT NULL
        LIMIT {sample_size}
    ) AS sample
    """
    try:
        cursor.execute(distinct_sql)
        distinct_count = cursor.fetchone()[0] or 0
    except Exception as e:
        logger.warning(f"{table_name}.{col_name}: could not compute distinct count: {e}")
        distinct_count = 0

    # Min / Max / Avg (numeric columns only)
    min_val, max_val, avg_val = None, None, None
    is_numeric = col_type.lower() in ["integer", "bigint", "smallint", "numeric", "decimal", "float", "double precision"]

    if is_numeric:
        stats_sql = f"""
        SELECT MIN("{col_name}"::numeric), MAX("{col_name}"::numeric), AVG("{col_name}"::numeric)
        FROM "{table_name}"
        WHERE "{col_name}" IS NOT NULL
        LIMIT {PROFILING_DISTINCT_LIMIT}
        """
        try:
            cursor.execute(stats_sql)
            row = cursor.fetchone()
            if row:
                min_val, max_val, avg_val = row
        except Exception as e:
            logger.warning(f"{table_name}.{col_name}: could not compute min/max/avg: {e}")

    # Top N values
    top_values = []
    try:
        top_values_sql = f"""
        SELECT "{col_name}"::text, COUNT(*)
        FROM "{table_name}"
        WHERE "{col_name}" IS NOT NULL
        GROUP BY "{col_name}"
        ORDER BY COUNT(*) DESC
        LIMIT {PROFILING_TOP_VALUES_LIMIT}
        """
        cursor.execute(top_values_sql)
        top_values = [str(row[0]) for row in cursor.fetchall()]
    except Exception as e:
        logger.warning(f"{table_name}.{col_name}: could not retrieve top values: {e}")

    cursor.close()

    return ColumnProfile(
        table_name=table_name,
        col_name=col_name,
        col_type=col_type,
        null_percentage=null_percentage,
        distinct_count=distinct_count,
        min_value=min_val,
        max_value=max_val,
        avg_value=avg_val,
        top_values=top_values,
    )


def run_profiling(conn, schema_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Run profiling on all columns in schema.

    Args:
        conn: psycopg2 connection
        schema_dict: {table_name: {columns: [...]}}

    Returns:
        {table.column: profile_dict}
    """
    profiling = {}
    total_columns = sum(len(t.get("columns", [])) for t in schema_dict.values())

    logger.info(f"Starting data profiling on {total_columns} columns...")
    start_time = time.time()

    processed = 0
    for table_name, table_info in schema_dict.items():
        for col_info in table_info.get("columns", []):
            col_name = col_info.get("col_name") or col_info.get("name")
            col_type = col_info.get("data_type") or col_info.get("type")

            try:
                profile = profile_column(conn, table_name, col_name, col_type)
                key = f"{table_name}.{col_name}"
                profiling[key] = profile.to_dict()
                processed += 1

                if processed % 25 == 0:
                    logger.info(f"  Profiled {processed}/{total_columns} columns...")

            except Exception as e:
                logger.error(f"Failed to profile {table_name}.{col_name}: {e}")
                continue

    elapsed = time.time() - start_time
    logger.info(f"Profiling complete: {processed} columns in {elapsed:.1f}s")

    return profiling


def save_profiling(profiling: Dict[str, Dict[str, Any]], output_file: str = None):
    """Save profiling to JSON file."""
    if output_file is None:
        output_file = PROFILING_FILE

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(profiling, f, indent=2)

    logger.info(f"Profiling saved to {output_file}")


def load_profiling(input_file: str = None) -> Dict[str, Dict[str, Any]]:
    """Load profiling from JSON file."""
    if input_file is None:
        input_file = PROFILING_FILE

    if not os.path.exists(input_file):
        logger.warning(f"Profiling file not found: {input_file}")
        return {}

    with open(input_file, "r") as f:
        profiling = json.load(f)

    logger.info(f"Loaded profiling from {input_file}")
    return profiling
