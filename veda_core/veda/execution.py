"""VEDA · L7 — read-only execution."""
import os, re, sys, time, json, logging, threading
from veda.runtime import get_db_config


def _scope_source_ids():
    """In-scope source ids from the ambient request context (both context module names —
    see veda_hybrid._current_ctx for why)."""
    import importlib
    for modname in ("veda_core.context", "context"):
        try:
            ctx = importlib.import_module(modname).try_current()
            if ctx is not None:
                return [str(s) for s in (ctx.source_ids or ())] or [str(ctx.source_id)], \
                       str(ctx.tenant)
        except Exception:
            continue
    return [], "default"


def _tabular_sources(source_ids, tenant):
    """{source_id: SourceSurface} for the tabular (parquet) sources in scope, else {}.
    A source resolving to kind='parquet' has no relational DB — its SQL must run on DuckDB."""
    out = {}
    try:
        from query.cross_source_composer import resolve_surface
    except Exception:
        return out
    for sid in source_ids:
        try:
            surf = resolve_surface(str(sid), tenant)
        except Exception:
            surf = None
        if surf is not None and getattr(surf, "kind", "") == "parquet":
            out[str(sid)] = surf
    return out


def _execute_duckdb(sql, params, surfaces):
    """Run a read-only SELECT against DuckDB with each tabular source's parquet tables
    registered under their BARE names (the single-source semantic model emits bare
    `FROM <table>`). Returns (cols, rows, err)."""
    try:
        import duckdb
    except Exception as e:
        return None, None, f"duckdb unavailable: {e}"
    conn = duckdb.connect()
    try:
        try:
            conn.execute("SET enable_external_access=true;")
        except Exception:
            pass
        for surf in surfaces.values():
            for tname, path in (surf.tables or {}).items():
                conn.execute(f'CREATE OR REPLACE VIEW "{tname}" AS '
                             f"SELECT * FROM read_parquet('{path}');")
        rel = conn.execute(sql.replace("%s", "?"), params or [])
        cols = [d[0] for d in rel.description]
        rows = rel.fetchmany(20)
        return cols, rows, None
    except Exception as e:
        return None, None, str(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def execute_sql(sql, params=None):
    # Tabular sources (CSV/parquet) have no relational DB — route their SQL to DuckDB
    # over the materialized parquet. Purely relational scopes keep the psycopg2 fast path.
    source_ids, tenant = _scope_source_ids()
    if source_ids:
        tabular = _tabular_sources(source_ids, tenant)
        if tabular and len(tabular) == len(source_ids):
            return _execute_duckdb(sql, params, tabular)
        # (mixed relational+tabular = federated execution — handled by the composer path.)

    import psycopg2
    cfg = get_db_config()
    conn = psycopg2.connect(
        host=cfg["host"], port=cfg["port"], dbname=cfg["database"],
        user=cfg["user"], password=cfg["password"])
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 30000")
            cur.execute(sql, params or [])   # parameterized — no value interpolation
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(20)
        return cols, rows, None
    except Exception as e:
        return None, None, str(e)
    finally:
        conn.close()
