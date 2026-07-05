# =============================================================================
# ingestion/db_abstraction.py
# VEDA — Database Abstraction Layer (DAL)
#
# Single point of truth for all database connections in the ingestion pipeline.
# Replaces direct psycopg2.connect(**DB_CONFIG) calls across all files.
#
# Two distinct connection types — NEVER confused:
#
#   Internal connections  → VEDA's own pgvector index (VEDA_INTERNAL_DB)
#     Used by: vector_store.py, data_graph.py, value_sampler.py
#     Always PostgreSQL + pgvector
#     get_internal_connection() / execute_internal() / internal_cursor()
#
#   Client connections    → Client's data sources (VEDA_SOURCES[n])
#     Used by: connectors/relational.py, data_graph.py (for value sampling)
#     Can be PostgreSQL, MySQL, SQLite, Delta, MongoDB, etc.
#     get_client_connection(source_id)
#     — NOTE: prefer using connectors/relational.py directly for client DBs
#
# Migration guide for existing files:
#   BEFORE:  import psycopg2; conn = psycopg2.connect(**DB_CONFIG)
#   AFTER:   from ingestion.db_abstraction import get_internal_connection
#            conn = get_internal_connection()
#
#   BEFORE:  psycopg2.extras.DictCursor
#   AFTER:   dal.DICT_CURSOR   (same class, zero change in call sites)
#
#   BEFORE:  PSYCOPG2_AVAILABLE
#   AFTER:   dal.INTERNAL_DB_AVAILABLE   (same semantics)
#
# All existing files keep their in-memory fallback logic unchanged.
# The DAL simply provides a cleaner, engine-aware connection factory.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import contextlib
from typing import Any, Dict, Generator, List, Optional

from config import VEDA_INTERNAL_DB, get_source, get_enabled_sources


# =============================================================================
# psycopg2 import — graceful fallback (same pattern as all existing files)
# =============================================================================

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

# Public availability flag — replaces PSYCOPG2_AVAILABLE in all files
INTERNAL_DB_AVAILABLE: bool = _PSYCOPG2_AVAILABLE

# Public DictCursor alias — replaces psycopg2.extras.DictCursor in all files
# None when psycopg2 not installed — callers guard with INTERNAL_DB_AVAILABLE
DICT_CURSOR = psycopg2.extras.DictCursor if _PSYCOPG2_AVAILABLE else None


# =============================================================================
# Internal DB connection pool
#
# A simple connection pool for VEDA_INTERNAL_DB.
# Avoids creating a new TCP connection on every vector store operation.
# Pool size = 1 for the POC (single-threaded ingestion).
# Production would use ThreadedConnectionPool with size = num_workers.
# =============================================================================

_INTERNAL_POOL = None


def _get_internal_pool():
    """Returns (creating if needed) the connection pool for VEDA_INTERNAL_DB."""
    global _INTERNAL_POOL
    if _INTERNAL_POOL is None and _PSYCOPG2_AVAILABLE:
        try:
            # minconn=0: don't eagerly open a connection at pool init.
            # Connections are created on first getconn() call, avoiding
            # "recovery mode" errors if postgres is mid-startup when the
            # module is first imported.
            _INTERNAL_POOL = psycopg2.pool.SimpleConnectionPool(
                minconn = 0,
                maxconn = 5,
                **VEDA_INTERNAL_DB,
            )
        except Exception:
            pass
    return _INTERNAL_POOL


def _reset_internal_pool() -> None:
    """Closes and resets the pool. Called on connection errors or test teardown."""
    global _INTERNAL_POOL
    if _INTERNAL_POOL is not None:
        try:
            _INTERNAL_POOL.closeall()
        except Exception:
            pass
        _INTERNAL_POOL = None


# =============================================================================
# Internal DB — public API
# =============================================================================

def get_internal_connection():
    """
    Returns a psycopg2 connection to VEDA_INTERNAL_DB.

    Drop-in replacement for:
        psycopg2.connect(**DB_CONFIG)

    Usage:
        conn = get_internal_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(...)
        finally:
            release_internal_connection(conn)

    For simpler one-shot usage, prefer the context manager:
        with internal_connection() as conn:
            ...
    """
    if not _PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for VEDA internal DB operations. "
            "Install with: pip install psycopg2-binary"
        )
    try:
        pool = _get_internal_pool()
        if pool:
            return pool.getconn()
        # Pool creation failed — fall back to direct connection
        return psycopg2.connect(**VEDA_INTERNAL_DB)
    except Exception:
        # Pool exhausted or broken — direct connect fallback
        _reset_internal_pool()
        return psycopg2.connect(**VEDA_INTERNAL_DB)


def release_internal_connection(conn) -> None:
    """
    Returns a connection to the pool (or closes it if pool unavailable).
    Always call this in a finally block after get_internal_connection().
    """
    if conn is None:
        return
    pool = _get_internal_pool()
    if pool:
        try:
            pool.putconn(conn)
            return
        except Exception:
            pass
    # Pool unavailable — close directly
    try:
        conn.close()
    except Exception:
        pass


@contextlib.contextmanager
def internal_connection() -> Generator:
    """
    Context manager for VEDA internal DB connections.
    Handles acquire + release automatically.

    Usage:
        with internal_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    conn = get_internal_connection()
    try:
        yield conn
    finally:
        release_internal_connection(conn)


def execute_internal(
    sql:    str,
    params: Optional[list] = None,
    fetch:  str = "none",    # "none" | "one" | "all"
    dict_cursor: bool = False,
) -> Any:
    """
    Executes a single SQL statement against VEDA_INTERNAL_DB.
    Convenience wrapper for simple one-shot operations.

    Parameters
    ----------
    sql    : SQL string with %s placeholders
    params : parameter list (or None)
    fetch  : "none" | "one" | "all"
    dict_cursor : if True, returns rows as dicts (like DictCursor)

    Returns
    -------
    None | dict | list[dict] — depending on fetch parameter
    """
    conn = get_internal_connection()
    try:
        cursor_factory = psycopg2.extras.DictCursor if dict_cursor else None
        with conn:
            cur = conn.cursor(cursor_factory=cursor_factory) if cursor_factory \
                  else conn.cursor()
            cur.execute(sql, params or [])
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return None
    finally:
        release_internal_connection(conn)


def ensure_pgvector_extension() -> None:
    """
    Ensures the pgvector extension is installed in VEDA_INTERNAL_DB.
    Safe to call multiple times — CREATE EXTENSION IF NOT EXISTS is idempotent.
    """
    if not _PSYCOPG2_AVAILABLE:
        return
    conn = get_internal_connection()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    finally:
        release_internal_connection(conn)


# =============================================================================
# Client DB connections
#
# For client data sources in VEDA_SOURCES.
# The preferred approach is to use connectors/relational.py directly.
# This function is provided as a lower-level escape hatch for data_graph.py
# which needs raw DB access for value sampling without going through connectors.
# =============================================================================

def get_client_connection(source_id: str):
    """
    Returns a DB-API 2.0 connection to the specified client source.

    Reads connection details from VEDA_SOURCES in config.py.
    Engine-specific — uses the right driver per engine.

    Parameters
    ----------
    source_id : str
        The 'id' field from a VEDA_SOURCES entry.

    Returns
    -------
    DB-API 2.0 connection object (engine-specific).

    Raises
    ------
    KeyError    : source_id not found in VEDA_SOURCES
    ImportError : required driver not installed
    RuntimeError: connection failed
    """
    src = get_source(source_id)
    engine = src.get("engine", "postgresql").lower()

    if engine in ("postgresql", "postgres"):
        return _connect_postgresql(src)
    if engine in ("mysql", "mariadb"):
        return _connect_mysql(src)
    if engine == "sqlite":
        return _connect_sqlite(src)

    # Generic fallback — try psycopg2 (PostgreSQL-compatible)
    return _connect_postgresql(src)


def _connect_postgresql(src: dict):
    if not _PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 is required for PostgreSQL: pip install psycopg2-binary"
        )
    return psycopg2.connect(
        host     = src.get("host", "localhost"),
        port     = src.get("port", 5432),
        dbname   = src.get("dbname"),
        user     = src.get("user"),
        password = src.get("password"),
    )


def _connect_mysql(src: dict):
    try:
        import mysql.connector
    except ImportError:
        raise ImportError(
            "mysql-connector-python is required for MySQL: "
            "pip install mysql-connector-python"
        )
    return mysql.connector.connect(
        host     = src.get("host", "localhost"),
        port     = src.get("port", 3306),
        database = src.get("dbname"),
        user     = src.get("user"),
        password = src.get("password"),
    )


def _connect_sqlite(src: dict):
    import sqlite3
    path = src.get("path") or src.get("dbname", ":memory:")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# Source-aware cursor factory
#
# Returns the right "dict cursor" class for each engine.
# Callers use this instead of hardcoding psycopg2.extras.DictCursor.
# =============================================================================

def get_dict_cursor_factory(source_id: str = None):
    """
    Returns the dict cursor factory for the given source.
    If source_id is None, returns the factory for VEDA_INTERNAL_DB (psycopg2).

    Usage:
        factory = get_dict_cursor_factory()
        cur = conn.cursor(cursor_factory=factory)
        row = cur.fetchone()
        value = row["col_name"]   # works for psycopg2 DictCursor

    Returns None for engines that use sqlite3.Row (already dict-like).
    """
    if source_id is None:
        # Internal DB is always PostgreSQL
        return psycopg2.extras.DictCursor if _PSYCOPG2_AVAILABLE else None

    src = get_source(source_id)
    engine = src.get("engine", "postgresql").lower()

    if engine in ("postgresql", "postgres"):
        return psycopg2.extras.DictCursor if _PSYCOPG2_AVAILABLE else None
    # sqlite3.Row is already subscriptable — no factory needed
    return None


# =============================================================================
# Health check
# =============================================================================

def check_internal_db_health() -> Dict[str, Any]:
    """
    Verifies connectivity to VEDA_INTERNAL_DB.
    Returns a health dict suitable for logging and monitoring.
    """
    if not _PSYCOPG2_AVAILABLE:
        return {
            "ok": False,
            "message": "psycopg2 not installed",
            "latency_ms": 0,
        }
    t0 = time.time()
    try:
        conn = get_internal_connection()
        cur  = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        try: cur.close()
        except Exception: pass
        release_internal_connection(conn)
        return {
            "ok":         True,
            "message":    "Connected",
            "version":    version,
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "host":       VEDA_INTERNAL_DB.get("host"),
            "dbname":     VEDA_INTERNAL_DB.get("dbname"),
        }
    except Exception as e:
        return {
            "ok":         False,
            "message":    str(e),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }


def check_client_source_health(source_id: str) -> Dict[str, Any]:
    """
    Verifies connectivity to a client data source.
    Wraps the connector's connect() method for consistency.
    """
    try:
        from connectors.base import build_connector
        connector = build_connector(get_source(source_id))
        status    = connector.connect()
        connector.disconnect()
        return {
            "ok":         status.ok,
            "source_id":  source_id,
            "engine":     status.engine,
            "message":    status.message,
            "latency_ms": status.latency_ms,
        }
    except Exception as e:
        return {
            "ok":        False,
            "source_id": source_id,
            "message":   str(e),
        }


# =============================================================================
# Compatibility shims
#
# These make it trivially easy to migrate existing files.
# Each file needs only to change its import, not its call sites.
#
# BEFORE (in vector_store.py):
#   import psycopg2
#   PSYCOPG2_AVAILABLE = True/False
#   def _get_connection(): return psycopg2.connect(**DB_CONFIG)
#
# AFTER:
#   from ingestion.db_abstraction import (
#       INTERNAL_DB_AVAILABLE as PSYCOPG2_AVAILABLE,
#       get_internal_connection as _get_connection,
#       DICT_CURSOR,
#   )
# =============================================================================

# Direct drop-in aliases
PSYCOPG2_AVAILABLE = INTERNAL_DB_AVAILABLE   # backward compat name


def _get_connection():
    """
    Drop-in replacement for the _get_connection() function defined locally
    in vector_store.py, data_graph.py, and value_sampler.py.
    Returns a connection to VEDA_INTERNAL_DB.
    """
    return get_internal_connection()


# =============================================================================
# Smoke test — python ingestion/db_abstraction.py
# =============================================================================

if __name__ == "__main__":
    print("=== VEDA Database Abstraction Layer ===")
    print(f"  psycopg2 available      : {_PSYCOPG2_AVAILABLE}")
    print(f"  INTERNAL_DB_AVAILABLE   : {INTERNAL_DB_AVAILABLE}")
    print(f"  DICT_CURSOR             : {DICT_CURSOR}")
    print()

    # Internal DB health
    health = check_internal_db_health()
    print("Internal DB health:")
    for k, v in health.items():
        print(f"  {k:<20} : {v}")
    print()

    # Client source health
    from config import get_enabled_sources
    for src in get_enabled_sources():
        h = check_client_source_health(src["id"])
        print(f"Source '{src['id']}' health:")
        for k, v in h.items():
            print(f"  {k:<20} : {v}")
    print()

    # Test context manager
    if INTERNAL_DB_AVAILABLE:
        try:
            with internal_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1 AS ping;")
                row = cur.fetchone()
                print(f"Context manager ping: {row[0]}")
                try: cur.close()
                except Exception: pass
        except Exception as e:
            print(f"Context manager test failed (expected if no local DB): {e}")
    else:
        print("psycopg2 not available — skipping connection test")

    print()
    print("DAL smoke test complete ✓")