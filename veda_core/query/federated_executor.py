# =============================================================================
# query/federated_executor.py
# VEDA — Federated cross-source SQL execution (Cross-source plan, Phase 5.1/5.3)
#
# Executes ONE validated SQL statement that joins across sources. A DuckDB
# in-memory connection attaches every tabular surface in the request scope under
# a per-source catalog:
#
#   relational sources  → ATTACH 'postgres:…' AS src_<id> (READ_ONLY) via the
#                         postgres_scanner extension. Pushdown means Postgres still
#                         does the heavy filtering. Credentials are resolved
#                         server-side and NEVER appear in the generated SQL.
#   tabular/derived     → CREATE VIEW src_<id>.<table> AS
#                         SELECT * FROM read_parquet('<artifact path>').
#
# Firewall (mandatory, same gates as the single-source path, wider scope):
#   - SELECT-only, single statement (reject any DML/DDL/COPY/ATTACH at the AST)
#   - every referenced catalog must be in the request scope (whitelist)
#   - per-query source-count cap (FED_MAX_SOURCES) + row limit + statement timeout
#
# Single-source plans keep the existing direct path — this module is only used
# when a plan touches > 1 source (a routing decision, not a feature flag).
#
# Optional dependency: duckdb (+ its postgres extension for relational attaches).
# =============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import DATALAKE_DUCKDB_MEMORY_LIMIT

try:
    import duckdb as _duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False


FED_MAX_SOURCES = int(os.environ.get("FED_MAX_SOURCES", "4"))
FED_STATEMENT_TIMEOUT_MS = int(os.environ.get("FED_STATEMENT_TIMEOUT_MS", "15000"))
FED_ROW_LIMIT = int(os.environ.get("FED_ROW_LIMIT", "10000"))


import re as _re
import sqlglot
from sqlglot import exp
_IDENT_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")   # safe SQL identifier (group_by key)


def catalog_name(source_id) -> str:
    """The DuckDB catalog/schema a source is attached under. Underscore form so it
    is a legal single identifier in generated SQL (`src_<id>`)."""
    return f"src_{source_id}"


@dataclass
class SourceSurface:
    """One source's execution surface in a federated query."""
    source_id: str
    kind: str                                   # "parquet" | "postgres"
    tables: Dict[str, str] = field(default_factory=dict)   # table_name -> parquet path
    pg_dsn: Optional[str] = None                # postgres attach DSN (kind == postgres)


class FederatedError(RuntimeError):
    pass


# --------------------------------------------------------------------------- AST
def validate_federated_sql(sql: str, allowed_catalogs: set) -> dict:
    """Firewall: SELECT-only single statement, and every referenced catalog is in
    scope. Returns {catalogs: set, tables: set}. Raises FederatedError on violation.
    Mirrors the single-source AST gate (veda.validation) with catalog awareness."""
    try:
        trees = sqlglot.parse(sql, read="duckdb")
    except Exception as e:
        raise FederatedError(f"unparseable SQL: {e}")
    trees = [t for t in trees if t is not None]
    if len(trees) != 1:
        raise FederatedError("exactly one statement is allowed")
    tree = trees[0]
    if not isinstance(tree, (exp.Select, exp.Subquery, exp.With)):
        raise FederatedError(f"only SELECT is allowed, got {type(tree).__name__}")
    # reject any write/DDL/side-effecting node anywhere in the tree
    _FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
                  exp.Alter, exp.Command, exp.Copy)
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN):
            raise FederatedError(f"forbidden statement node: {type(node).__name__}")
    # collect referenced catalogs (the `src_<id>` qualifier on each table)
    catalogs, tables = set(), set()
    for tbl in tree.find_all(exp.Table):
        tables.add(tbl.name)
        cat = tbl.catalog or tbl.db          # duckdb: schema qualifier lands in .db
        if cat:
            catalogs.add(cat)
    stray = {c for c in catalogs if c.startswith("src_")} - allowed_catalogs
    if stray:
        raise FederatedError(f"out-of-scope catalog(s): {sorted(stray)}")
    return {"catalogs": catalogs, "tables": tables}


# --------------------------------------------------------------------- executor
def _join_stats(conn, table_names: List[str], quoted_group_key: str) -> Optional[dict]:
    """Cross-source match counts for the aggregate-pushdown join (traceability Part 10).

    CHEAP BY CONSTRUCTION: the per-source aggregates are ALREADY materialized as
    DuckDB temp tables by execute_plan, so this is two local counts over local
    data — it touches no source and issues no federated query.

    ``matched``   = group keys present in EVERY source (the INNER JOIN cardinality)
    ``unmatched`` = keys present in at least one source but not all (union - matched)

    Deliberately NOT offered for the plain ``execute()`` path: there the SQL is an
    arbitrary caller-supplied join, so the same numbers would need one or two EXTRA
    federated round trips against the sources — real added latency on a path whose
    measured median is already tens of seconds. A missing join block is honest;
    a guessed one is not.

    Flag-gated (FEDERATED_JOIN_STATS_ENABLED, default OFF) and best-effort: any
    failure returns None so the query result is completely unaffected.
    """
    try:
        import config as _cfg
        if not bool(getattr(_cfg, "FEDERATED_JOIN_STATS_ENABLED", False)):
            return None
    except Exception:
        return None
    if len(table_names) < 2:
        return None
    try:
        inner = table_names[0]
        for tn in table_names[1:]:
            inner += f" INNER JOIN {tn} USING ({quoted_group_key})"
        matched = conn.execute(f"SELECT COUNT(*) FROM {inner}").fetchone()[0]

        full = table_names[0]
        for tn in table_names[1:]:
            full += f" FULL JOIN {tn} USING ({quoted_group_key})"
        union = conn.execute(f"SELECT COUNT(*) FROM {full}").fetchone()[0]

        matched, union = int(matched), int(union)
        return {"join_used": True, "matched_count": matched,
                "unmatched_count": max(0, union - matched),
                "source_count": len(table_names)}
    except Exception:
        return None


class FederatedExecutor:
    def __init__(self, surfaces: List[SourceSurface]):
        if not _DUCKDB_AVAILABLE:
            raise FederatedError("duckdb not installed")
        if len(surfaces) > FED_MAX_SOURCES:
            raise FederatedError(
                f"federated query spans {len(surfaces)} sources > FED_MAX_SOURCES={FED_MAX_SOURCES}")
        self.surfaces = surfaces
        self.allowed_catalogs = {catalog_name(s.source_id) for s in surfaces}
        self._conn = None

    def _connect(self):
        conn = _duckdb.connect()
        conn.execute(f"SET memory_limit='{DATALAKE_DUCKDB_MEMORY_LIMIT}';")
        try:
            conn.execute("SET enable_external_access=true;")
        except Exception:
            pass
        for s in self.surfaces:
            cat = catalog_name(s.source_id)
            if s.kind == "postgres":
                # server-side DSN — never surfaced in generated SQL
                conn.execute("INSTALL postgres; LOAD postgres;")
                conn.execute(f"ATTACH '{s.pg_dsn}' AS {cat} (TYPE POSTGRES, READ_ONLY);")
            else:
                conn.execute(f"CREATE SCHEMA IF NOT EXISTS {cat};")
                for tname, path in s.tables.items():
                    conn.execute(
                        f'CREATE OR REPLACE VIEW {cat}."{tname}" AS '
                        f"SELECT * FROM read_parquet('{path}');")
        return conn

    def execute(self, sql: str, params: Optional[list] = None,
                row_limit: int = FED_ROW_LIMIT) -> dict:
        """Validate then execute a federated SELECT. Returns
        {columns, rows, row_count, truncated}. Raises FederatedError on firewall
        violations."""
        info = validate_federated_sql(sql, self.allowed_catalogs)
        conn = self._connect()
        try:
            try:
                conn.execute(f"SET statement_timeout='{FED_STATEMENT_TIMEOUT_MS}ms';")
            except Exception:
                pass  # older duckdb: timeout GUC may differ; memory_limit still applies
            rel = conn.execute(sql.replace("%s", "?"), params or [])
            cols = [d[0] for d in rel.description]
            allr = rel.fetchmany(row_limit + 1)
            truncated = len(allr) > row_limit
            rows = [dict(zip(cols, r)) for r in allr[:row_limit]]
            return {"columns": cols, "rows": rows, "row_count": len(rows),
                    "truncated": truncated, "catalogs": sorted(info["catalogs"])}
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def execute_plan(self, plan: dict, row_limit: int = FED_ROW_LIMIT) -> dict:
        """Aggregate-pushdown execution (correctness for multi-metric cross-source queries).

        ``plan`` = {group_by: <col-alias or None>, metrics: [{alias, sql}, ...]}. Each
        metric.sql is an INDEPENDENT single-aggregate SELECT (its own join, GROUP BY the
        group key AS <group_by>) — so no fan-out double-counting. We materialize each into a
        DuckDB temp table then FULL JOIN them on the group key. Joining plain temp tables (not
        postgres_scanner subqueries) sidesteps DuckDB's "non-inner join on subquery" limit AND
        is semantically correct (each aggregate computed once, over its own rows).

        A single-metric / non-grouped plan degrades to a plain execute()."""
        metrics = plan.get("metrics") or []
        group_by = plan.get("group_by")
        if not metrics:
            raise FederatedError("empty federated plan")
        if not group_by or len(metrics) == 1:
            return self.execute(metrics[0]["sql"], row_limit=row_limit)
        if not _IDENT_RE.match(str(group_by)):
            raise FederatedError(f"unsafe group_by identifier: {group_by!r}")
        # Firewall each metric's source SELECT (SELECT-only, catalogs in scope).
        cats: set = set()
        for m in metrics:
            info = validate_federated_sql(m["sql"], self.allowed_catalogs)
            cats |= set(info["catalogs"])
        conn = self._connect()
        try:
            try:
                conn.execute(f"SET statement_timeout='{FED_STATEMENT_TIMEOUT_MS}ms';")
            except Exception:
                pass
            names = []
            for i, m in enumerate(metrics):
                tn = f"agg_{i}"
                # CREATE TEMP TABLE is a LOCAL duckdb object; the source access is the
                # firewalled SELECT inside it (no source mutation).
                conn.execute(f'CREATE TEMP TABLE {tn} AS {m["sql"]}')
                names.append(tn)
            gk = f'"{group_by}"'
            join_sql = names[0]
            for tn in names[1:]:
                join_sql += f" FULL JOIN {tn} USING ({gk})"
            final = f"SELECT * FROM {join_sql} ORDER BY 1 LIMIT {int(row_limit)}"
            rel = conn.execute(final)
            cols = [d[0] for d in rel.description]
            allr = rel.fetchmany(row_limit + 1)
            rows = [dict(zip(cols, r)) for r in allr[:row_limit]]
            out = {"columns": cols, "rows": rows, "row_count": len(rows),
                   "truncated": len(allr) > row_limit, "catalogs": sorted(cats),
                   "pushdown": True}
            stats = _join_stats(conn, names, gk)
            if stats:
                out["join"] = stats
            return out
        finally:
            try:
                conn.close()
            except Exception:
                pass


def build_surfaces_from_scope(scope) -> List[SourceSurface]:
    """Build execution surfaces for a query scope. ``scope`` is a list of dicts:
    {source_id, kind, tables?, pg_dsn?}. Kept dependency-light so the caller
    (the query pipeline) resolves parquet paths / server-side DSNs and this module
    stays purely about attaching + validating + executing."""
    return [SourceSurface(source_id=str(s["source_id"]), kind=s.get("kind", "parquet"),
                          tables=s.get("tables", {}), pg_dsn=s.get("pg_dsn"))
            for s in scope]
