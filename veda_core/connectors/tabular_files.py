# =============================================================================
# connectors/tabular_files.py
# VEDA — Tabular-file connector (Cross-source plan, Phase 2)
#
# A CSV/Excel/Parquet file is a TABLE, not a bag of chunks. This connector
# presents file-backed tabular data through the *relational* connector
# interface (get_schema → RawSchema/RawTable/RawColumn) so the standard
# L1–L5 pipeline (semantic types → value sampling → M3 embeddings → sparse
# index → graph persist) runs over it UNCHANGED. Downstream, a CSV column is
# indistinguishable from a Postgres column.
#
# Contrast with connectors/datalake.py (which also reads files via DuckDB):
# that connector feeds the lighter *schema* pipeline and mints random UUIDs.
# This one is built for the FULL relational pipeline and mints DETERMINISTIC
# UUIDv5 ids from (source_id + file/sheet + column) — identical id-stability
# guarantee to the relational connector, so re-ingesting the same file yields
# the same node ids (idempotent graph writes).
#
# Execution surface (Phase 5): materialize_parquet() writes each table to
# canonical typed Parquet under ARTIFACT_ROOT/<scope>/tables/<table>.parquet —
# the original CSV is never re-parsed at query time.
#
# Optional dependency: duckdb. Absent → connect() fails cleanly (never raises).
# xlsx additionally needs DuckDB's spatial extension (st_read) or a pandas/
# openpyxl fallback; both are attempted, in that order.
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
    register_connector,
)
from connectors.datalake import _normalise_duckdb_type
from config import VEDA_INTERNAL_TABLES, DATALAKE_DUCKDB_MEMORY_LIMIT


# Fixed namespace so UUIDv5 ids are stable across processes/re-ingests. Never
# change this value — it would re-key every file-backed table's graph nodes.
_TABULAR_NS = uuid.UUID("6f1e7c2a-1b3d-5e4f-8a90-abcdef012345")


try:
    import duckdb as _duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False


def _new_duckdb_conn():
    conn = _duckdb.connect()
    conn.execute(f"SET memory_limit='{DATALAKE_DUCKDB_MEMORY_LIMIT}';")
    return conn


def table_uuid(source_id: str, table_name: str) -> str:
    """Deterministic table id: uuid5(ns, '<source_id>:<table_name>')."""
    return str(uuid.uuid5(_TABULAR_NS, f"{source_id}:{table_name}"))


def column_uuid(table_id: str, col_name: str) -> str:
    """Deterministic column id: uuid5(ns, '<table_id>:<col_name>')."""
    return str(uuid.uuid5(_TABULAR_NS, f"{table_id}:{col_name}"))


class TabularFileConnector(BaseConnector):
    """Relational-interface connector over file-backed tables (CSV/Parquet/Excel).

    ``dialect`` selects the reader:
        csv_lake / csv  → read_csv_auto (one table per *.csv)
        parquet         → read_parquet  (one table per *.parquet)
        xlsx / excel    → one table per sheet in each workbook

    The source path may be a directory (each matching file = one table) or a
    single file. Table names are the file stem (plus ``__<sheet>`` for Excel).
    """

    _GLOBS = {
        "csv": ("*.csv",), "csv_lake": ("*.csv",),
        "parquet": ("*.parquet",),
        "xlsx": ("*.xlsx", "*.xls"), "excel": ("*.xlsx", "*.xls"),
    }

    def __init__(self, source_config: dict) -> None:
        super().__init__(source_config)
        self._path = Path(source_config.get("path") or source_config.get("source_path") or ".")
        # normalise the dialect/engine name to a reader family
        eng = (source_config.get("engine") or source_config.get("dialect") or "csv").lower()
        self._kind = "xlsx" if eng in ("xlsx", "excel") else \
                     "parquet" if eng == "parquet" else "csv"

    # ------------------------------------------------------------------ caps
    @property
    def supports_schema(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return _DUCKDB_AVAILABLE

    @property
    def supports_value_sampling(self) -> bool:
        return _DUCKDB_AVAILABLE

    # --------------------------------------------------------------- lifecycle
    def connect(self) -> ConnectorStatus:
        t0 = time.time()
        if not _DUCKDB_AVAILABLE:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="relational",
                engine=self._kind, message="duckdb not installed — pip install duckdb",
                latency_ms=0.0)
        try:
            if not self._path.exists():
                raise FileNotFoundError(f"Path does not exist: {self._path}")
            conn = _new_duckdb_conn()
            conn.execute("SELECT 1").fetchone()
            conn.close()
            self._state = ConnectorState.CONNECTED
            return ConnectorStatus(
                ok=True, source_id=self._source_id, source_type="relational",
                engine=self._kind, message=f"Connected to {self._path}",
                latency_ms=round((time.time() - t0) * 1000, 2),
                metadata={"path": str(self._path), "kind": self._kind})
        except Exception as e:
            self._state = ConnectorState.ERROR
            return ConnectorStatus(
                ok=False, source_id=self._source_id, source_type="relational",
                engine=self._kind, message=str(e),
                latency_ms=round((time.time() - t0) * 1000, 2))

    def disconnect(self) -> None:
        self._state = ConnectorState.DISCONNECTED

    # ------------------------------------------------------------------ files
    def _files(self) -> List[Path]:
        if self._path.is_file():
            return [self._path]
        out: List[Path] = []
        for pat in self._GLOBS.get(self._kind, ("*.csv",)):
            out.extend(sorted(self._path.glob(pat)))
        return out

    def get_table_scans(self) -> Dict[str, str]:
        """{table_name → DuckDB FROM expression}. For Excel, one entry per sheet."""
        scans: Dict[str, str] = {}
        for f in self._files():
            if self._kind == "parquet":
                scans[f.stem] = f"read_parquet('{f}')"
            elif self._kind == "xlsx":
                for sheet in self._excel_sheets(f):
                    scans[f"{f.stem}__{_slug(sheet)}"] = self._excel_scan(f, sheet)
            else:
                scans[f.stem] = f"read_csv_auto('{f}', SAMPLE_SIZE=-1)"
        return scans

    def _excel_sheets(self, f: Path) -> List[str]:
        """Sheet names in a workbook. Prefers openpyxl; empty list if unreadable."""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(f, read_only=True)
            names = list(wb.sheetnames)
            wb.close()
            return names
        except Exception:
            return []

    def _excel_scan(self, f: Path, sheet: str) -> str:
        """DuckDB FROM expression for one Excel sheet via the spatial st_read reader."""
        return f"st_read('{f}', layer='{sheet}')"

    # ----------------------------------------------------------------- schema
    def get_schema(self) -> RawSchema:
        tables = self._read_tables()
        total_cols = sum(len(t.columns) for t in tables)
        return RawSchema(
            source_id=self._source_id, source_type="relational", engine=self._kind,
            tables=tables, fk_edges=[],   # intra-file FKs discovered by value overlap
            stats={"total_tables": len(tables), "total_columns": total_cols,
                   "total_fk_edges": 0})

    def get_raw_schema_dict(self) -> dict:
        """Legacy dict shape consumed by ingestion.schema_scanner.run_schema_scanner —
        identical to RelationalConnector.get_raw_schema_dict, so the standard L1–L5
        pipeline treats a file table exactly like a relational table (Phase 2.1)."""
        raw = self.get_schema()
        tables, name_to_id = [], {}
        for rt in raw.tables:
            name_to_id[rt.table_name] = rt.table_id
            cols = [{
                "col_id": rc.col_id, "col_name": rc.col_name, "data_type": rc.data_type,
                "is_pk": rc.is_pk, "is_fk": rc.is_fk, "fk_ref_table": rc.fk_ref_table,
                "fk_ref_col": rc.fk_ref_col, "fk_ref_table_id": rc.fk_ref_table_id,
                "nullable": rc.nullable, "cardinality": rc.cardinality,
            } for rc in rt.columns]
            tables.append({"table_id": rt.table_id, "table_name": rt.table_name,
                           "row_count": rt.row_count, "columns": cols})
        return {"tables": tables, "name_to_id": name_to_id,
                "excluded_columns": [], "stats": raw.stats}

    def _read_tables(self) -> List[RawTable]:
        if not _DUCKDB_AVAILABLE:
            return []
        tables: List[RawTable] = []
        conn = _new_duckdb_conn()
        if self._kind == "xlsx":
            try:
                conn.execute("INSTALL spatial; LOAD spatial;")
            except Exception:
                pass
        try:
            for tname, scan in self.get_table_scans().items():
                cols_raw = self._describe(scan, conn)
                if not cols_raw:
                    continue
                rows = self._count(scan, conn)
                tables.append(self._build_table(tname, cols_raw, rows))
        finally:
            conn.close()
        return tables

    def _describe(self, scan: str, conn) -> List[tuple]:
        try:
            rows = conn.execute(f"DESCRIBE SELECT * FROM {scan} LIMIT 0").fetchall()
            return [(r[0], r[1]) for r in rows]
        except Exception:
            return []

    def _count(self, scan: str, conn) -> int:
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {scan}").fetchone()[0]
        except Exception:
            return 0

    def _build_table(self, table_name: str, columns_raw: List[tuple], row_count: int) -> RawTable:
        table_id = table_uuid(self._source_id, table_name)
        columns: List[RawColumn] = []
        for col_name, col_type in columns_raw:
            if col_name.lower() in VEDA_INTERNAL_TABLES:
                continue
            name_lower = col_name.lower()
            is_pk = name_lower in ("id",) or name_lower == f"{table_name.lower()}_id"
            columns.append(RawColumn(
                col_id=column_uuid(table_id, col_name), col_name=col_name,
                table_id=table_id, table_name=table_name,
                data_type=_normalise_duckdb_type(col_type),
                role=ColumnRole.PRIMARY_KEY if is_pk else ColumnRole.REGULAR,
                is_pk=is_pk, is_fk=False, fk_ref_table=None, fk_ref_col=None,
                fk_ref_table_id=None, nullable=True, cardinality=None,
                source_id=self._source_id))
        return RawTable(table_id=table_id, table_name=table_name, row_count=row_count,
                        columns=columns, source_id=self._source_id,
                        metadata={"kind": self._kind})

    # --------------------------------------------------------- value sampling
    def sample_column_values(self, table_name: str, col_name: str, n: int = 100) -> List[str]:
        if not _DUCKDB_AVAILABLE:
            return []
        scan = self.get_table_scans().get(table_name)
        if not scan:
            return []
        conn = _new_duckdb_conn()
        try:
            rows = conn.execute(
                f'SELECT DISTINCT "{col_name}" FROM {scan} '
                f'WHERE "{col_name}" IS NOT NULL LIMIT ?', [n]).fetchall()
            return [str(r[0]) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def get_row_count(self, table_name: str) -> int:
        scan = self.get_table_scans().get(table_name)
        if not (_DUCKDB_AVAILABLE and scan):
            return 0
        conn = _new_duckdb_conn()
        try:
            return self._count(scan, conn)
        finally:
            conn.close()

    # -------------------------------------------------- Parquet materialization
    def materialize_parquet(self, out_dir: str) -> Dict[str, str]:
        """Write each table to canonical typed Parquet (snappy) under ``out_dir``.

        Returns {table_name → parquet_path}. This is the Phase-5 execution surface:
        the source file is parsed once here, never at query time. Idempotent —
        COPY overwrites the destination.
        """
        if not _DUCKDB_AVAILABLE:
            return {}
        os.makedirs(out_dir, exist_ok=True)
        written: Dict[str, str] = {}
        conn = _new_duckdb_conn()
        if self._kind == "xlsx":
            try:
                conn.execute("INSTALL spatial; LOAD spatial;")
            except Exception:
                pass
        try:
            for tname, scan in self.get_table_scans().items():
                dest = os.path.join(out_dir, f"{tname}.parquet")
                try:
                    conn.execute(
                        f"COPY (SELECT * FROM {scan}) TO '{dest}' "
                        f"(FORMAT PARQUET, COMPRESSION SNAPPY);")
                    written[tname] = dest
                except Exception:
                    continue
        finally:
            conn.close()
        return written

    # ----------------------------------------------------------- query (tests)
    def execute_query(self, query: str, params: Optional[list] = None,
                      row_limit: int = 1000, timeout_sec: int = 30) -> QueryResult:
        if not _DUCKDB_AVAILABLE:
            return QueryResult(source_id=self._source_id, source_type="relational",
                               rows=[], row_count=0, columns=[], sql_or_query=query,
                               duration_ms=0.0, truncated=False,
                               error="duckdb not installed")
        t0 = time.time()
        try:
            conn = _new_duckdb_conn()
            for tname, scan in self.get_table_scans().items():
                conn.execute(f'CREATE OR REPLACE VIEW "{tname}" AS SELECT * FROM {scan};')
            rel = conn.execute(query.replace("%s", "?"), params or [])
            cols = [d[0] for d in rel.description]
            allr = rel.fetchmany(row_limit + 1)
            truncated = len(allr) > row_limit
            rows = [dict(zip(cols, r)) for r in allr[:row_limit]]
            conn.close()
            return QueryResult(source_id=self._source_id, source_type="relational",
                               rows=rows, row_count=len(rows), columns=cols,
                               sql_or_query=query,
                               duration_ms=round((time.time() - t0) * 1000, 2),
                               truncated=truncated, error=None)
        except Exception as e:
            return QueryResult(source_id=self._source_id, source_type="relational",
                               rows=[], row_count=0, columns=[], sql_or_query=query,
                               duration_ms=round((time.time() - t0) * 1000, 2),
                               truncated=False, error=str(e))


def _slug(name: str) -> str:
    """Filesystem/identifier-safe sheet slug."""
    import re
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower() or "sheet"


def _ensure_registered() -> None:
    pass  # registration happens at module import (below)


# Registered under the RELATIONAL source-type so build_connector routes file
# tables through the full L1–L5 pipeline (Phase 2.1), keyed by reader family.
register_connector("relational", "csv",      TabularFileConnector)
register_connector("relational", "csv_lake", TabularFileConnector)
register_connector("relational", "parquet",  TabularFileConnector)
register_connector("relational", "xlsx",     TabularFileConnector)
register_connector("relational", "excel",    TabularFileConnector)
