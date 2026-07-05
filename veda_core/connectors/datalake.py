# =============================================================================
# connectors/datalake.py
# VEDA — Data Lake Connector (Phase 3)
#
# Implements BaseConnector for file-based data sources.
# Reads schema metadata from Delta Lake, Parquet, and CSV sources
# and returns RawSchema in the same format as relational connectors —
# the downstream pipeline (scanner → inference → encoder) is unchanged.
#
# FK edges: always empty — data graph discovers implicit joins via value overlap.
#
# All three engines are read via DuckDB in-process (no cluster needed).
# Optional dependency: duckdb — pip install duckdb
# Parquet/Delta/CSV are all handled natively by DuckDB with no extra libraries.
#
# Source path conventions:
#   delta   — path IS the Delta table root directory (one table per source)
#   parquet — path is a directory; each *.parquet file = one table
#   csv     — path is a directory; each *.csv file = one table
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from connectors.base import (
    BaseConnector,
    ColumnRole,
    ConnectorState,
    ConnectorStatus,
    QueryResult,
    RawColumn,
    RawSchema,
    RawTable,
    normalise_data_type,
    register_connector,
)
from config import (
    VEDA_INTERNAL_TABLES,
    DATALAKE_QUERY_ENGINE,
    DATALAKE_DUCKDB_MEMORY_LIMIT,
)


# =============================================================================
# DuckDB — optional dep, graceful fallback
# =============================================================================

try:
    import duckdb as _duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False


def _new_duckdb_conn():
    """Returns a fresh in-process DuckDB connection with memory limit set."""
    conn = _duckdb.connect()
    conn.execute(f"SET memory_limit='{DATALAKE_DUCKDB_MEMORY_LIMIT}';")
    return conn


# =============================================================================
# DuckDB type → VEDA normalised type
# =============================================================================

def _normalise_duckdb_type(duck_type: str) -> str:
    """Maps DuckDB type strings to VEDA's normalised vocabulary."""
    t = duck_type.upper().split("(")[0].strip()
    _DUCK_MAP = {
        "INTEGER":   "integer",  "INT":       "integer",  "INT4":     "integer",
        "BIGINT":    "bigint",   "INT8":      "bigint",   "HUGEINT":  "bigint",
        "SMALLINT":  "smallint", "INT2":      "smallint", "TINYINT":  "smallint",
        "FLOAT":     "numeric",  "FLOAT4":    "numeric",  "REAL":     "numeric",
        "DOUBLE":    "double",   "FLOAT8":    "double",
        "DECIMAL":   "numeric",  "NUMERIC":   "numeric",
        "VARCHAR":   "varchar",  "TEXT":      "text",     "STRING":   "varchar",
        "BLOB":      "bytea",    "BYTEA":     "bytea",
        "BOOLEAN":   "boolean",  "BOOL":      "boolean",
        "DATE":      "date",
        "TIME":      "time",
        "TIMESTAMP": "timestamp", "TIMESTAMPTZ": "timestamptz",
        "UUID":      "uuid",
        "JSON":      "json",     "MAP":       "json",     "STRUCT":   "json",
        "LIST":      "text",
    }
    return _DUCK_MAP.get(t, normalise_data_type(duck_type))


# =============================================================================
# Base DataLakeConnector
# =============================================================================

class DataLakeConnector(BaseConnector):
    """
    Base class for all data lake connectors.
    Subclasses override _read_schema() and get_table_scans().
    """

    def __init__(self, source_config: dict) -> None:
        super().__init__(source_config)
        self._path = Path(source_config.get("path", "."))

    @property
    def supports_schema(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return _DUCKDB_AVAILABLE

    def connect(self) -> ConnectorStatus:
        t0 = time.time()
        if not _DUCKDB_AVAILABLE:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="datalake",
                engine=self._engine,
                message="duckdb not installed — pip install duckdb",
                latency_ms=0.0,
            )
        try:
            if not self._path.exists():
                raise FileNotFoundError(f"Path does not exist: {self._path}")
            conn = _new_duckdb_conn()
            conn.execute("SELECT 1").fetchone()
            conn.close()
            self._state = ConnectorState.CONNECTED
            return ConnectorStatus(
                ok=True, source_id=self._source_id, source_type="datalake",
                engine=self._engine,
                message=f"Connected to {self._path}",
                latency_ms=round((time.time() - t0) * 1000, 2),
                metadata={"path": str(self._path)},
            )
        except Exception as e:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="datalake",
                engine=self._engine, message=str(e),
                latency_ms=round((time.time() - t0) * 1000, 2),
            )

    def disconnect(self) -> None:
        self._state = ConnectorState.DISCONNECTED

    def get_schema(self) -> RawSchema:
        """
        Reads schema metadata via DuckDB and returns RawSchema.
        The format is identical to relational connectors — downstream pipeline
        (schema_scanner → semantic_type_inference → encoder) runs unchanged.
        """
        tables = self._read_tables()
        total_cols = sum(len(t.columns) for t in tables)
        return RawSchema(
            source_id   = self._source_id,
            source_type = "datalake",
            engine      = self._engine,
            tables      = tables,
            fk_edges    = [],
            stats       = {
                "total_tables":   len(tables),
                "total_columns":  total_cols,
                "total_fk_edges": 0,
            },
        )

    def get_table_scans(self) -> Dict[str, str]:
        """
        Returns {table_name → DuckDB FROM expression} for the execution engine.
        The execution engine creates a view per table from these expressions.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_tables(self) -> List[RawTable]:
        raise NotImplementedError

    def _describe_scan(self, scan_expr: str, conn) -> List[tuple]:
        """
        Runs DuckDB DESCRIBE on a scan expression.
        Returns [(col_name, col_type), ...].
        Falls back to empty list on error.
        """
        try:
            rows = conn.execute(f"DESCRIBE SELECT * FROM {scan_expr} LIMIT 0").fetchall()
            return [(r[0], r[1]) for r in rows]
        except Exception:
            return []

    def _count_rows(self, scan_expr: str, conn) -> int:
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {scan_expr}").fetchone()[0]
        except Exception:
            return 0

    def _build_table(
        self,
        table_name: str,
        columns_raw: List[tuple],
        row_count:   int,
    ) -> RawTable:
        """Builds a RawTable from DuckDB DESCRIBE output."""
        table_id = str(uuid.uuid4())
        columns  = []
        for col_name, col_type in columns_raw:
            if col_name.lower() in VEDA_INTERNAL_TABLES:
                continue
            col_id   = str(uuid.uuid4())
            data_type = _normalise_duckdb_type(col_type)
            # Heuristic: columns ending in _id are potential keys
            name_lower = col_name.lower()
            is_pk = name_lower in ("id",) or name_lower == f"{table_name.lower()}_id"
            role  = ColumnRole.PRIMARY_KEY if is_pk else ColumnRole.REGULAR
            columns.append(RawColumn(
                col_id          = col_id,
                col_name        = col_name,
                table_id        = table_id,
                table_name      = table_name,
                data_type       = data_type,
                role            = role,
                is_pk           = is_pk,
                is_fk           = False,
                fk_ref_table    = None,
                fk_ref_col      = None,
                fk_ref_table_id = None,
                nullable        = True,
                cardinality     = None,
                source_id       = self._source_id,
            ))
        return RawTable(
            table_id   = table_id,
            table_name = table_name,
            row_count  = row_count,
            columns    = columns,
            source_id  = self._source_id,
        )

    def execute_query(
        self,
        query:       str,
        params:      Optional[list] = None,
        row_limit:   int = 1000,
        timeout_sec: int = 30,
    ) -> QueryResult:
        """Executes a SQL query via DuckDB against this datalake source."""
        if not _DUCKDB_AVAILABLE:
            return QueryResult(
                source_id=self._source_id, source_type="datalake",
                rows=[], row_count=0, columns=[],
                sql_or_query=query, duration_ms=0.0, truncated=False,
                error="duckdb not installed",
            )
        t0 = time.time()
        try:
            conn = _new_duckdb_conn()
            # Create views for all tables in this source
            for tname, scan_expr in self.get_table_scans().items():
                conn.execute(f'CREATE OR REPLACE VIEW "{tname}" AS SELECT * FROM {scan_expr};')
            # Convert %s placeholders to ? for DuckDB
            duckdb_sql = query.replace("%s", "?")
            relation   = conn.execute(duckdb_sql, params or [])
            cols       = [desc[0] for desc in relation.description]
            all_rows   = relation.fetchmany(row_limit + 1)
            truncated  = len(all_rows) > row_limit
            rows       = [dict(zip(cols, r)) for r in all_rows[:row_limit]]
            conn.close()
            return QueryResult(
                source_id    = self._source_id,
                source_type  = "datalake",
                rows         = rows,
                row_count    = len(rows),
                columns      = cols,
                sql_or_query = query,
                duration_ms  = round((time.time() - t0) * 1000, 2),
                truncated    = truncated,
                error        = None,
            )
        except Exception as e:
            return QueryResult(
                source_id=self._source_id, source_type="datalake",
                rows=[], row_count=0, columns=[],
                sql_or_query=query,
                duration_ms=round((time.time() - t0) * 1000, 2),
                truncated=False, error=str(e),
            )


# =============================================================================
# DeltaConnector — one Delta table per source
# =============================================================================

class DeltaConnector(DataLakeConnector):
    """
    Delta Lake connector. The source path IS the Delta table root.
    Table name defaults to the final path component.
    """

    def get_table_scans(self) -> Dict[str, str]:
        table_name = self._config.get("table_name") or self._path.name
        return {table_name: f"delta_scan('{self._path}')"}

    def _read_tables(self) -> List[RawTable]:
        if not _DUCKDB_AVAILABLE:
            return []
        table_name = self._config.get("table_name") or self._path.name
        conn       = _new_duckdb_conn()
        try:
            # Load delta extension
            try:
                conn.execute("INSTALL delta; LOAD delta;")
            except Exception:
                pass
            scan_expr  = f"delta_scan('{self._path}')"
            cols_raw   = self._describe_scan(scan_expr, conn)
            row_count  = self._count_rows(scan_expr, conn)
        finally:
            conn.close()
        if not cols_raw:
            return []
        return [self._build_table(table_name, cols_raw, row_count)]


# =============================================================================
# ParquetConnector — each *.parquet file = one table
# =============================================================================

class ParquetConnector(DataLakeConnector):
    """
    Parquet connector. Each .parquet file in the source path = one table.
    Table name = filename without extension.
    """

    def get_table_scans(self) -> Dict[str, str]:
        scans = {}
        for f in self._path.glob("*.parquet"):
            table_name = f.stem
            scans[table_name] = f"read_parquet('{f}')"
        return scans

    def _read_tables(self) -> List[RawTable]:
        if not _DUCKDB_AVAILABLE:
            return []
        tables = []
        conn   = _new_duckdb_conn()
        try:
            for f in self._path.glob("*.parquet"):
                table_name = f.stem
                scan_expr  = f"read_parquet('{f}')"
                cols_raw   = self._describe_scan(scan_expr, conn)
                row_count  = self._count_rows(scan_expr, conn)
                if cols_raw:
                    tables.append(self._build_table(table_name, cols_raw, row_count))
        finally:
            conn.close()
        return tables


# =============================================================================
# CSVConnector — each *.csv file = one table
# =============================================================================

class CSVConnector(DataLakeConnector):
    """
    CSV connector. Each .csv file in the source path = one table.
    Table name = filename without extension.
    Uses DuckDB auto_detect for schema inference.
    """

    def get_table_scans(self) -> Dict[str, str]:
        scans = {}
        for f in self._path.glob("*.csv"):
            table_name = f.stem
            scans[table_name] = f"read_csv('{f}', auto_detect=true)"
        return scans

    def _read_tables(self) -> List[RawTable]:
        if not _DUCKDB_AVAILABLE:
            return []
        tables = []
        conn   = _new_duckdb_conn()
        try:
            for f in self._path.glob("*.csv"):
                table_name = f.stem
                scan_expr  = f"read_csv('{f}', auto_detect=true)"
                cols_raw   = self._describe_scan(scan_expr, conn)
                row_count  = self._count_rows(scan_expr, conn)
                if cols_raw:
                    tables.append(self._build_table(table_name, cols_raw, row_count))
        finally:
            conn.close()
        return tables


# =============================================================================
# Connector registration
# =============================================================================

def _ensure_registered() -> None:
    pass   # registration happens at module level below


register_connector("datalake", "delta",   DeltaConnector)
register_connector("datalake", "parquet", ParquetConnector)
register_connector("datalake", "csv",     CSVConnector)
register_connector("datalake", "generic", ParquetConnector)   # fallback
