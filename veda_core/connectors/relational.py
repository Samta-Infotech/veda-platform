# =============================================================================
# connectors/relational.py
# VEDA — Relational Database Connector
#
# Implements BaseConnector for relational databases.
# Supports: PostgreSQL, MySQL, SQLite, Oracle, SQL Server (generic fallback)
#
# Design:
#   - One base class RelationalConnector handles the full lifecycle
#   - Engine-specific subclasses override only what differs:
#       _get_connection()       — returns a DB-API 2.0 connection
#       _get_tables_query()     — SQL to list tables
#       _get_columns_query()    — SQL to list columns for a table
#       _get_pk_query()         — SQL to list PKs
#       _get_fk_query()         — SQL to list FKs
#       _get_row_count_query()  — SQL to count rows
#   - All output is normalised to RawSchema using base.py's DATA_TYPE_MAP
#   - VEDA_INTERNAL_TABLES always excluded from scanning
#
# get_real_schema() compatibility shim:
#   Returns the legacy dict format for schema_scanner.py during
#   Phase 1 migration. Once schema_unifier.py is complete,
#   callers use get_schema() → RawSchema directly.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from connectors.base import (
    BaseConnector,
    ConnectorState,
    ConnectorStatus,
    ColumnRole,
    RawColumn,
    RawSchema,
    RawTable,
    QueryResult,
    normalise_data_type,
    register_connector,
)
from config import (
    SENSITIVE_PATTERNS,
    VEDA_INTERNAL_TABLES,
    get_primary_relational_source,
)


# =============================================================================
# RelationalConnector — base for all relational engines
# =============================================================================

class RelationalConnector(BaseConnector):
    """
    Base relational connector. Handles the full ingestion lifecycle.
    Subclasses override only the engine-specific query methods.
    """

    def __init__(self, source_config: dict) -> None:
        super().__init__(source_config)
        self._conn       = None
        self._schema     = source_config.get("schema", None)     # restrict to schema
        self._exclude    = set(source_config.get("exclude_tables", []))
        self._exclude   |= VEDA_INTERNAL_TABLES

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def supports_schema(self) -> bool:
        return True

    @property
    def supports_value_sampling(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return self._config.get("role") == "queryable"

    # ------------------------------------------------------------------
    # Engine-specific overrides — implement in subclasses
    # ------------------------------------------------------------------

    def _get_raw_connection(self):
        """Returns a DB-API 2.0 connection. Must be overridden per engine."""
        raise NotImplementedError(
            f"Engine '{self._engine}' must implement _get_raw_connection()"
        )

    def _q(self, name: str) -> str:
        """Double-quote identifier. Override for engines that use backticks."""
        return f'"{name.replace(chr(34), "")}"'

    def _get_tables_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        """
        Returns (sql, params) to list all user tables.
        Default: INFORMATION_SCHEMA — works on PostgreSQL, MySQL, SQL Server.
        """
        schema_filter = schema or "public"
        return (
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """,
            [schema_filter],
        )

    def _get_columns_sql(self, table_name: str, schema: Optional[str]) -> Tuple[str, list]:
        """Returns (sql, params) to list columns for a table."""
        schema_filter = schema or "public"
        return (
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
              AND table_schema = %s
            ORDER BY ordinal_position;
            """,
            [table_name, schema_filter],
        )

    def _get_pk_sql(self, table_name: str, schema: Optional[str]) -> Tuple[str, list]:
        """Returns (sql, params) to list PK column names for a table."""
        schema_filter = schema or "public"
        return (
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            WHERE tc.table_name   = %s
              AND tc.table_schema  = %s
              AND tc.constraint_type = 'PRIMARY KEY';
            """,
            [table_name, schema_filter],
        )

    def _get_fk_sql(self, table_name: str, schema: Optional[str]) -> Tuple[str, list]:
        """Returns (sql, params) to list FK relationships for a table.

        Uses REFERENTIAL_CONSTRAINTS + a second KEY_COLUMN_USAGE join to get
        the actual referenced column name.  constraint_column_usage alone is
        unreliable on PostgreSQL — it returns the constraining column name
        rather than the referenced column name for FK constraints.
        """
        schema_filter = schema or "public"
        return (
            """
            SELECT kcu.column_name                AS fk_col,
                   kcu2.table_name                AS ref_table,
                   kcu2.column_name               AS ref_col
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name  = kcu.constraint_name
             AND tc.table_schema     = kcu.table_schema
            JOIN information_schema.referential_constraints rc
              ON tc.constraint_name  = rc.constraint_name
             AND tc.table_schema     = rc.constraint_schema
            JOIN information_schema.key_column_usage kcu2
              ON rc.unique_constraint_name   = kcu2.constraint_name
             AND rc.unique_constraint_schema = kcu2.table_schema
             AND kcu.ordinal_position        = kcu2.ordinal_position
            WHERE tc.table_name    = %s
              AND tc.table_schema  = %s
              AND tc.constraint_type = 'FOREIGN KEY';
            """,
            [table_name, schema_filter],
        )

    # ------------------------------------------------------------------
    # Batched (schema-wide) introspection — one query per metadata type
    # instead of one-per-table. The per-table variants above are kept for
    # value sampling / single-table callers; get_schema() uses these.
    # ------------------------------------------------------------------

    def _get_all_columns_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        """(sql, params) → (table_name, column_name, data_type, is_nullable) for the whole schema."""
        schema_filter = schema or "public"
        return (
            """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position;
            """,
            [schema_filter],
        )

    def _get_all_pks_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        """(sql, params) → (table_name, column_name) for every PK column in the schema."""
        schema_filter = schema or "public"
        return (
            """
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            WHERE tc.table_schema = %s
              AND tc.constraint_type = 'PRIMARY KEY';
            """,
            [schema_filter],
        )

    def _get_all_fks_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        """(sql, params) → (src_table, fk_col, ref_table, ref_col) for every FK in the schema."""
        schema_filter = schema or "public"
        return (
            """
            SELECT tc.table_name  AS src_table,
                   kcu.column_name AS fk_col,
                   kcu2.table_name AS ref_table,
                   kcu2.column_name AS ref_col
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name  = kcu.constraint_name
             AND tc.table_schema     = kcu.table_schema
            JOIN information_schema.referential_constraints rc
              ON tc.constraint_name  = rc.constraint_name
             AND tc.table_schema     = rc.constraint_schema
            JOIN information_schema.key_column_usage kcu2
              ON rc.unique_constraint_name   = kcu2.constraint_name
             AND rc.unique_constraint_schema = kcu2.table_schema
             AND kcu.ordinal_position        = kcu2.ordinal_position
            WHERE tc.table_schema = %s
              AND tc.constraint_type = 'FOREIGN KEY';
            """,
            [schema_filter],
        )

    def _get_all_row_counts_sql(self, schema: Optional[str]) -> Tuple[Optional[str], list]:
        """(sql, params) → (table_name, row_count) for the whole schema, or (None, [])
        when the engine has no cheap batch path (caller falls back to per-table COUNT(*))."""
        return (None, [])

    def _get_row_count_sql(self, table_name: str) -> Tuple[str, list]:
        """Returns (sql, params) to count rows in a table."""
        return f'SELECT COUNT(*) FROM {self._q(table_name)};', []

    def _get_sample_values_sql(
        self, table_name: str, col_name: str, n: int
    ) -> Tuple[str, list]:
        """Returns (sql, params) to sample distinct non-null values."""
        return (
            f"SELECT DISTINCT {self._q(col_name)} "
            f"FROM {self._q(table_name)} "
            f"WHERE {self._q(col_name)} IS NOT NULL "
            f"LIMIT %s;",
            [n],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> ConnectorStatus:
        t0 = time.time()
        try:
            self._conn = self._get_raw_connection()
            # Quick ping — use direct execute for SQLite compatibility
            cur = self._conn.cursor()
            cur.execute("SELECT 1;")
            try:
                cur.close()
            except Exception:
                pass
            self._state = ConnectorState.CONNECTED
            return ConnectorStatus(
                ok          = True,
                source_id   = self._source_id,
                source_type = "relational",
                engine      = self._engine,
                message     = "Connection successful",
                latency_ms  = round((time.time() - t0) * 1000, 2),
            )
        except Exception as e:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok          = False,
                source_id   = self._source_id,
                source_type = "relational",
                engine      = self._engine,
                message     = f"Connection failed: {e}",
                latency_ms  = round((time.time() - t0) * 1000, 2),
            )

    def disconnect(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._state = ConnectorState.DISCONNECTED

    def _ensure_connected(self) -> None:
        """Reconnects if connection was dropped."""
        if self._state != ConnectorState.CONNECTED or self._conn is None:
            status = self.connect()
            if not status.ok:
                raise RuntimeError(
                    f"Cannot connect to source '{self._source_id}': {status.message}"
                )

    # ------------------------------------------------------------------
    # Schema extraction
    # ------------------------------------------------------------------

    def get_schema(self) -> RawSchema:
        """
        Extracts the full schema from the relational database.
        Returns RawSchema with RawTable and RawColumn objects.
        VEDA internal tables and sensitive columns are excluded.
        """
        self._ensure_connected()
        schema = self._schema

        name_to_id: Dict[str, str] = {}
        tables:     List[RawTable] = []
        fk_edges:   List[dict]     = []

        with self._conn.cursor() as cur:
            # ── 1. List tables ────────────────────────────────────────
            sql, params = self._get_tables_sql(schema)
            cur.execute(sql, params)
            table_names = [
                r[0] for r in cur.fetchall()
                if r[0] not in self._exclude
            ]

            # ── 2. Batched schema-wide introspection ──────────────────
            # One query per metadata type (columns / PKs / FKs / row-counts)
            # for the WHOLE schema, grouped in Python — instead of 3-4 queries
            # per table. On a large catalog the per-table information_schema
            # constraint joins cost seconds *each* (they re-scan every schema's
            # constraints before filtering), so N tables × that = ~an hour.
            # Batched (and pg_catalog on Postgres) takes the scan to sub-second.
            tset = set(table_names)

            cols_by_table: Dict[str, list] = defaultdict(list)
            sql, params = self._get_all_columns_sql(schema)
            cur.execute(sql, params)
            for tname, col_name, raw_type, nullable in cur.fetchall():
                if tname in tset:
                    cols_by_table[tname].append((col_name, raw_type, nullable))

            pks_by_table: Dict[str, set] = defaultdict(set)
            sql, params = self._get_all_pks_sql(schema)
            cur.execute(sql, params)
            for tname, col_name in cur.fetchall():
                if tname in tset:
                    pks_by_table[tname].add(col_name)

            fks_by_table: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
            sql, params = self._get_all_fks_sql(schema)
            cur.execute(sql, params)
            for tname, fk_col, ref_table, ref_col in cur.fetchall():
                if tname in tset:
                    fks_by_table[tname][fk_col] = (ref_table, ref_col)

            # Row counts: cheap batch path if the engine offers one (Postgres →
            # pg_class estimate), else per-table COUNT(*) below.
            row_counts: Dict[str, int] = {}
            rc_sql, rc_params = self._get_all_row_counts_sql(schema)
            if rc_sql:
                cur.execute(rc_sql, rc_params)
                for tname, rc in cur.fetchall():
                    if tname in tset:
                        row_counts[tname] = max(int(rc or 0), 0)

            # ── 3. Assemble tables from the grouped metadata ──────────
            # Deterministic UUID — same table always gets the same UUID so
            # pgvector UPSERTs overwrite instead of accumulating stale rows.
            for tname in table_names:
                tid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{schema}.{tname}"))
                name_to_id[tname] = tid

                pk_cols = pks_by_table.get(tname, set())
                fk_map: Dict[str, Tuple[str, str]] = fks_by_table.get(tname, {})

                # Row count: batched value when available, else per-table COUNT(*).
                if tname in row_counts:
                    row_count = row_counts[tname]
                else:
                    try:
                        sql, params = self._get_row_count_sql(tname)
                        cur.execute(sql, params)
                        row_count = cur.fetchone()[0]
                    except Exception:
                        row_count = 0

                columns: List[RawColumn] = []
                for col_name, raw_type, nullable in cols_by_table.get(tname, []):
                    # Sensitive column exclusion
                    if any(p in col_name.lower() for p in SENSITIVE_PATTERNS):
                        continue

                    fk_info   = fk_map.get(col_name)
                    is_pk     = col_name in pk_cols
                    is_fk     = col_name in fk_map
                    role      = (ColumnRole.PRIMARY_KEY if is_pk
                                 else ColumnRole.FOREIGN_KEY if is_fk
                                 else ColumnRole.REGULAR)

                    columns.append(RawColumn(
                        col_id          = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{schema}.{tname}.{col_name}")),
                        col_name        = col_name,
                        table_id        = tid,
                        table_name      = tname,
                        data_type       = normalise_data_type(raw_type),
                        role            = role,
                        is_pk           = is_pk,
                        is_fk           = is_fk,
                        fk_ref_table    = fk_info[0] if fk_info else None,
                        fk_ref_col      = fk_info[1] if fk_info else None,
                        fk_ref_table_id = None,   # resolved in post-processing
                        nullable        = (nullable == "YES"),
                        cardinality     = None,
                        source_id       = self._source_id,
                    ))

                tables.append(RawTable(
                    table_id   = tid,
                    table_name = tname,
                    row_count  = row_count,
                    columns    = columns,
                    source_id  = self._source_id,
                ))

        # ── 3. Resolve FK table IDs ──────────────────────────────────
        col_id_map: Dict[str, str] = {}  # (table_name, col_name) → col_id
        for table in tables:
            for col in table.columns:
                col_id_map[(table.table_name, col.col_name)] = col.col_id

        for table in tables:
            for col in table.columns:
                if col.is_fk and col.fk_ref_table:
                    ref_tid = name_to_id.get(col.fk_ref_table)
                    col.fk_ref_table_id = ref_tid

                    # Build fk_edges list in the format store_fk_adjacency expects
                    ref_col_id = col_id_map.get(
                        (col.fk_ref_table, col.fk_ref_col or ""), ""
                    )
                    fk_edges.append({
                        "from_col_id":   col.col_id,
                        "from_col_name": col.col_name,
                        "from_table_id": col.table_id,
                        "from_table":    col.table_name,
                        "to_col_id":     ref_col_id,
                        "to_col_name":   col.fk_ref_col or "",
                        "to_table_id":   ref_tid or "",
                        "to_table":      col.fk_ref_table,
                    })

        total_cols = sum(len(t.columns) for t in tables)
        total_fks  = len(fk_edges)

        return RawSchema(
            source_id   = self._source_id,
            source_type = "relational",
            engine      = self._engine,
            tables      = tables,
            fk_edges    = fk_edges,
            stats       = {
                "total_tables":   len(tables),
                "total_columns":  total_cols,
                "total_fk_edges": total_fks,
                "excluded_count": 0,
            },
        )

    # ------------------------------------------------------------------
    # Value sampling
    # ------------------------------------------------------------------

    def sample_column_values(
        self,
        table_name: str,
        col_name:   str,
        n:          int = 100,
    ) -> List[str]:
        """Samples up to n distinct non-null values from table.col."""
        self._ensure_connected()
        try:
            sql, params = self._get_sample_values_sql(table_name, col_name, n)
            cur = self._conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            try: cur.close()
            except Exception: pass
            return [str(r[0]) for r in rows if r[0] is not None]
        except Exception:
            return []

    def get_row_count(self, table_name: str) -> int:
        """Returns row count for a table."""
        self._ensure_connected()
        try:
            sql, params = self._get_row_count_sql(table_name)
            cur = self._conn.cursor()
            cur.execute(sql, params)
            result = cur.fetchone()[0]
            try: cur.close()
            except Exception: pass
            return result
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute_query(
        self,
        query:       str,
        params:      Optional[list] = None,
        row_limit:   int = 1000,
        timeout_sec: int = 30,
    ) -> QueryResult:
        """Executes a parameterised SQL query. Read-only enforcement via cursor."""
        self._ensure_connected()
        t0 = time.time()
        try:
            cur = self._conn.cursor()
            cur.execute(query, params or [])
            rows_raw  = cur.fetchmany(row_limit + 1)
            truncated = len(rows_raw) > row_limit
            rows_raw  = rows_raw[:row_limit]
            col_names = [desc[0] for desc in cur.description] if cur.description else []
            rows      = [dict(zip(col_names, row)) for row in rows_raw]
            try: cur.close()
            except Exception: pass
            return QueryResult(
                source_id    = self._source_id,
                source_type  = "relational",
                rows         = rows,
                row_count    = len(rows),
                columns      = col_names,
                sql_or_query = query,
                duration_ms  = round((time.time() - t0) * 1000, 2),
                truncated    = truncated,
                error        = None,
            )
        except Exception as e:
            return QueryResult(
                source_id    = self._source_id,
                source_type  = "relational",
                rows         = [],
                row_count    = 0,
                columns      = [],
                sql_or_query = query,
                duration_ms  = round((time.time() - t0) * 1000, 2),
                truncated    = False,
                error        = str(e),
            )

    # ------------------------------------------------------------------
    # Legacy shim — returns dict format for schema_scanner.py
    # Used during Phase 1 migration. Replaced by get_schema() + schema_unifier.
    # ------------------------------------------------------------------

    def get_raw_schema_dict(self) -> dict:
        """
        Returns the legacy dict format consumed by schema_scanner.py.
        Wraps get_schema() → RawSchema → legacy dict.
        Called by schema/real_schema.py get_real_schema() shim.
        """
        raw = self.get_schema()

        tables    = []
        name_to_id = {}

        for rt in raw.tables:
            name_to_id[rt.table_name] = rt.table_id
            cols = []
            for rc in rt.columns:
                cols.append({
                    "col_id":        rc.col_id,
                    "col_name":      rc.col_name,
                    "data_type":     rc.data_type,
                    "is_pk":         rc.is_pk,
                    "is_fk":         rc.is_fk,
                    "fk_ref_table":  rc.fk_ref_table,
                    "fk_ref_col":    rc.fk_ref_col,
                    "fk_ref_table_id": rc.fk_ref_table_id,
                    "nullable":      rc.nullable,
                    "cardinality":   rc.cardinality,
                })
            tables.append({
                "table_id":   rt.table_id,
                "table_name": rt.table_name,
                "row_count":  rt.row_count,
                "columns":    cols,
            })

        return {
            "tables":           tables,
            "name_to_id":       name_to_id,
            "excluded_columns": [],
            "stats":            raw.stats,
        }


# =============================================================================
# PostgreSQL connector
# =============================================================================

class PostgreSQLConnector(RelationalConnector):
    """
    PostgreSQL connector using psycopg2.
    Supports all INFORMATION_SCHEMA queries — uses the base implementation.
    """

    def _get_raw_connection(self):
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL: pip install psycopg2-binary"
            )
        cfg = self._config
        return psycopg2.connect(
            host     = cfg.get("host", "localhost"),
            port     = cfg.get("port", 5432),
            dbname   = cfg.get("dbname"),
            user     = cfg.get("user"),
            password = cfg.get("password"),
        )

    # -- Fast batched introspection via pg_catalog --------------------------
    # information_schema's constraint views (referential_constraints +
    # key_column_usage) are pathologically slow on large catalogs — a
    # schema-wide FK query can take *minutes*. pg_catalog is indexed and
    # returns the same data in milliseconds. Columns stay on information_schema
    # (fast, and preserves the exact data_type vocabulary, e.g. ARRAY→text).

    def _get_all_pks_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        return (
            """
            SELECT src.relname, att.attname
            FROM pg_constraint con
            JOIN pg_class src     ON src.oid = con.conrelid
            JOIN pg_namespace n   ON n.oid = con.connamespace
            JOIN unnest(con.conkey) AS ck(attnum) ON true
            JOIN pg_attribute att  ON att.attrelid = con.conrelid AND att.attnum = ck.attnum
            WHERE con.contype = 'p' AND n.nspname = %s;
            """,
            [schema or "public"],
        )

    def _get_all_fks_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        # unnest conkey/confkey WITH ORDINALITY and join on matching position so
        # composite FKs map each local column to its referenced column correctly.
        return (
            """
            SELECT src.relname, att.attname, tgt.relname, att2.attname
            FROM pg_constraint con
            JOIN pg_class src      ON src.oid = con.conrelid
            JOIN pg_class tgt      ON tgt.oid = con.confrelid
            JOIN pg_namespace n    ON n.oid = con.connamespace
            JOIN unnest(con.conkey)  WITH ORDINALITY AS ck(attnum, ord)  ON true
            JOIN unnest(con.confkey) WITH ORDINALITY AS cfk(attnum, ord) ON ck.ord = cfk.ord
            JOIN pg_attribute att   ON att.attrelid  = con.conrelid  AND att.attnum  = ck.attnum
            JOIN pg_attribute att2  ON att2.attrelid = con.confrelid AND att2.attnum = cfk.attnum
            WHERE con.contype = 'f' AND n.nspname = %s;
            """,
            [schema or "public"],
        )

    def _get_all_row_counts_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        # reltuples is the planner's estimate (accurate after ANALYZE). row_count
        # feeds only cardinality normalisation / log-scaled features downstream,
        # so an estimate is fine — and avoids 316 COUNT(*) full scans.
        return (
            """
            SELECT c.relname, c.reltuples::bigint
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relkind = 'r';
            """,
            [schema or "public"],
        )


# =============================================================================
# MySQL connector
# =============================================================================

class MySQLConnector(RelationalConnector):
    """
    MySQL connector using mysql-connector-python.
    Overrides schema queries — MySQL uses 'public' → actual dbname,
    and uses %s placeholders differently.
    """

    def _get_raw_connection(self):
        try:
            import mysql.connector
        except ImportError:
            raise ImportError(
                "mysql-connector-python is required for MySQL: "
                "pip install mysql-connector-python"
            )
        cfg = self._config
        return mysql.connector.connect(
            host     = cfg.get("host", "localhost"),
            port     = cfg.get("port", 3306),
            database = cfg.get("dbname"),
            user     = cfg.get("user"),
            password = cfg.get("password"),
        )

    def _q(self, name: str) -> str:
        """MySQL uses backtick quoting."""
        return f"`{name.replace('`', '')}`"

    def _get_tables_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        db = schema or self._config.get("dbname", "")
        return (
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """,
            [db],
        )

    def _get_columns_sql(self, table_name: str, schema: Optional[str]) -> Tuple[str, list]:
        db = schema or self._config.get("dbname", "")
        return (
            """
            SELECT column_name, column_type, is_nullable
            FROM information_schema.columns
            WHERE table_name   = %s
              AND table_schema = %s
            ORDER BY ordinal_position;
            """,
            [table_name, db],
        )

    def _get_pk_sql(self, table_name: str, schema: Optional[str]) -> Tuple[str, list]:
        db = schema or self._config.get("dbname", "")
        return (
            """
            SELECT column_name
            FROM information_schema.key_column_usage
            WHERE table_name        = %s
              AND constraint_schema = %s
              AND constraint_name   = 'PRIMARY';
            """,
            [table_name, db],
        )

    def _get_fk_sql(self, table_name: str, schema: Optional[str]) -> Tuple[str, list]:
        db = schema or self._config.get("dbname", "")
        return (
            """
            SELECT column_name,
                   referenced_table_name,
                   referenced_column_name
            FROM information_schema.key_column_usage
            WHERE table_name        = %s
              AND constraint_schema = %s
              AND referenced_table_name IS NOT NULL;
            """,
            [table_name, db],
        )

    # -- Batched (schema-wide) variants used by get_schema() ---------------
    def _get_all_columns_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        db = schema or self._config.get("dbname", "")
        return (
            """
            SELECT table_name, column_name, column_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position;
            """,
            [db],
        )

    def _get_all_pks_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        db = schema or self._config.get("dbname", "")
        return (
            """
            SELECT table_name, column_name
            FROM information_schema.key_column_usage
            WHERE constraint_schema = %s
              AND constraint_name   = 'PRIMARY';
            """,
            [db],
        )

    def _get_all_fks_sql(self, schema: Optional[str]) -> Tuple[str, list]:
        db = schema or self._config.get("dbname", "")
        return (
            """
            SELECT table_name, column_name,
                   referenced_table_name, referenced_column_name
            FROM information_schema.key_column_usage
            WHERE constraint_schema = %s
              AND referenced_table_name IS NOT NULL;
            """,
            [db],
        )


# =============================================================================
# SQLite connector
# =============================================================================

class SQLiteConnector(RelationalConnector):
    """
    SQLite connector using the stdlib sqlite3 module — no extra install needed.
    Uses PRAGMA statements instead of INFORMATION_SCHEMA.
    """

    def _get_raw_connection(self):
        import sqlite3
        path = self._config.get("path") or self._config.get("dbname", ":memory:")
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    def _q(self, name: str) -> str:
        return f'"{name.replace(chr(34), "")}"'

    def get_schema(self) -> RawSchema:
        """
        SQLite uses PRAGMA — override the full get_schema() method.
        PRAGMA table_info and PRAGMA foreign_key_list replace INFORMATION_SCHEMA.
        """
        self._ensure_connected()
        import sqlite3

        name_to_id: Dict[str, str] = {}
        tables:     List[RawTable] = []
        fk_edges:   List[dict]     = []

        cur = self._conn.cursor()

        # List tables
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name;"
        )
        table_names = [
            r[0] for r in cur.fetchall()
            if r[0] not in self._exclude
        ]

        for tname in table_names:
            tid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{self._source_id}.{tname}"))
            name_to_id[tname] = tid

            # Row count
            try:
                cur.execute(f'SELECT COUNT(*) FROM {self._q(tname)};')
                row_count = cur.fetchone()[0]
            except Exception:
                row_count = 0

            # Columns via PRAGMA table_info
            cur.execute(f"PRAGMA table_info({self._q(tname)});")
            col_rows = cur.fetchall()
            # Fields: cid, name, type, notnull, dflt_value, pk
            pk_cols = {r[1] for r in col_rows if r[5] > 0}

            # FKs via PRAGMA foreign_key_list
            cur.execute(f"PRAGMA foreign_key_list({self._q(tname)});")
            fk_rows = cur.fetchall()
            # Fields: id, seq, table, from, to, on_update, on_delete, match
            fk_map = {r[3]: (r[2], r[4]) for r in fk_rows}

            columns: List[RawColumn] = []
            for crow in col_rows:
                col_name = crow[1]
                raw_type = crow[2] or "text"

                if any(p in col_name.lower() for p in SENSITIVE_PATTERNS):
                    continue

                fk_info = fk_map.get(col_name)
                is_pk   = col_name in pk_cols
                is_fk   = col_name in fk_map
                role    = (ColumnRole.PRIMARY_KEY if is_pk
                           else ColumnRole.FOREIGN_KEY if is_fk
                           else ColumnRole.REGULAR)

                columns.append(RawColumn(
                    col_id          = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{self._source_id}.{tname}.{col_name}")),
                    col_name        = col_name,
                    table_id        = tid,
                    table_name      = tname,
                    data_type       = normalise_data_type(raw_type),
                    role            = role,
                    is_pk           = is_pk,
                    is_fk           = is_fk,
                    fk_ref_table    = fk_info[0] if fk_info else None,
                    fk_ref_col      = fk_info[1] if fk_info else None,
                    fk_ref_table_id = None,
                    nullable        = (crow[3] == 0),   # notnull=0 means nullable
                    cardinality     = None,
                    source_id       = self._source_id,
                ))

            tables.append(RawTable(
                table_id   = tid,
                table_name = tname,
                row_count  = row_count,
                columns    = columns,
                source_id  = self._source_id,
            ))

        cur.close()

        # Resolve FK table IDs
        col_id_map: Dict[str, str] = {}
        for table in tables:
            for col in table.columns:
                col_id_map[(table.table_name, col.col_name)] = col.col_id

        for table in tables:
            for col in table.columns:
                if col.is_fk and col.fk_ref_table:
                    ref_tid = name_to_id.get(col.fk_ref_table)
                    col.fk_ref_table_id = ref_tid
                    fk_edges.append({
                        "from_col_id":   col.col_id,
                        "from_col_name": col.col_name,
                        "from_table_id": col.table_id,
                        "from_table":    col.table_name,
                        "to_col_id":     col_id_map.get((col.fk_ref_table, col.fk_ref_col or ""), ""),
                        "to_col_name":   col.fk_ref_col or "",
                        "to_table_id":   ref_tid or "",
                        "to_table":      col.fk_ref_table,
                    })

        total_cols = sum(len(t.columns) for t in tables)

        return RawSchema(
            source_id   = self._source_id,
            source_type = "relational",
            engine      = "sqlite",
            tables      = tables,
            fk_edges    = fk_edges,
            stats       = {
                "total_tables":   len(tables),
                "total_columns":  total_cols,
                "total_fk_edges": len(fk_edges),
                "excluded_count": 0,
            },
        )


# =============================================================================
# Connector registration
# =============================================================================

def _ensure_registered() -> None:
    """Called by build_connector() to register all relational engine classes."""
    register_connector("relational", "postgresql", PostgreSQLConnector)
    register_connector("relational", "postgres",   PostgreSQLConnector)   # alias
    register_connector("relational", "mysql",      MySQLConnector)
    register_connector("relational", "mariadb",    MySQLConnector)        # alias
    register_connector("relational", "sqlite",     SQLiteConnector)
    register_connector("relational", "generic",    RelationalConnector)   # fallback


# Register immediately when module is imported
_ensure_registered()


# =============================================================================
# Legacy shim — get_real_schema()
#
# Maintains backward compatibility with schema_scanner.py and main.py.
# During Phase 1 migration, existing code continues calling get_real_schema().
# Once schema_unifier.py is complete, this shim is removed.
# =============================================================================

def get_real_schema() -> dict:
    """
    Legacy entry point used by schema_scanner.py and main.py.
    Instantiates a connector for the primary relational source from config
    and returns the legacy dict format.
    """
    src_cfg   = get_primary_relational_source()
    connector = _build_relational_connector(src_cfg)
    status    = connector.connect()
    if not status.ok:
        raise RuntimeError(
            f"Cannot connect to primary relational source "
            f"'{src_cfg['id']}': {status.message}"
        )
    try:
        return connector.get_raw_schema_dict()
    finally:
        connector.disconnect()


def _build_relational_connector(src_cfg: dict) -> RelationalConnector:
    """Instantiates the right connector class for the given source config."""
    engine = src_cfg.get("engine", "postgresql").lower()
    if engine in ("postgresql", "postgres"):
        return PostgreSQLConnector(src_cfg)
    if engine in ("mysql", "mariadb"):
        return MySQLConnector(src_cfg)
    if engine == "sqlite":
        return SQLiteConnector(src_cfg)
    # Generic fallback — uses INFORMATION_SCHEMA
    return RelationalConnector(src_cfg)


# =============================================================================
# Smoke test — python connectors/relational.py
# =============================================================================

if __name__ == "__main__":
    import sys
    from connectors.base import _CONNECTOR_REGISTRY

    print("Connector registry after import:")
    for key, cls in sorted(_CONNECTOR_REGISTRY.items()):
        print(f"  {key:<30} → {cls.__name__}")
    print()

    # SQLite smoke test — no external deps
    import tempfile, os

    db_path = os.path.join(tempfile.gettempdir(), "veda_test.db")
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, email TEXT);")
    conn.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), status TEXT, amount REAL);")
    conn.execute("INSERT OR IGNORE INTO users VALUES (1, 'alice', 'alice@example.com');")
    conn.execute("INSERT OR IGNORE INTO orders VALUES (1, 1, 'open', 99.99);")
    conn.commit()
    conn.close()

    src_cfg = {
        "id": "test_sqlite", "type": "relational",
        "engine": "sqlite", "path": db_path,
        "role": "queryable", "exclude_tables": [],
    }

    connector = SQLiteConnector(src_cfg)
    status    = connector.connect()
    print(f"Connect: ok={status.ok}  latency={status.latency_ms}ms  message={status.message}")

    if status.ok:
        schema = connector.get_schema()
        print(f"\nSchema: {schema.stats}")
        for tbl in schema.tables:
            print(f"\n  [{tbl.table_name}]  rows={tbl.row_count}")
            for col in tbl.columns:
                print(f"    {col.col_name:<20} {col.data_type:<12} pk={col.is_pk} fk={col.is_fk}")
        print(f"\nFK edges: {schema.fk_edges}")

        vals = connector.sample_column_values("orders", "status", 10)
        print(f"\nSampled orders.status: {vals}")

        result = connector.execute_query("SELECT * FROM users;")
        print(f"\nQuery result: {result.rows}  error={result.error}")

    connector.disconnect()
    os.unlink(db_path)
    print("\nSmoke test passed ✓")