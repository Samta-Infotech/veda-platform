"""VEDA · L7 — read-only execution."""
import os, re, sys, time, json, logging, threading
from veda.runtime import get_db_config


def execute_sql(sql, params=None):
    import psycopg2
    from psycopg2 import sql as _sql
    from config import EXECUTION_RESULT_LIMIT
    cfg = get_db_config()
    kw = {"host": cfg["host"], "port": cfg["port"], "dbname": cfg["database"],
          "user": cfg["user"], "password": cfg["password"]}
    if cfg.get("sslmode"):
        kw["sslmode"] = cfg["sslmode"]
    conn = psycopg2.connect(**kw)
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 30000")
            # Make the configured schema authoritative for this connection instead of
            # depending on the DB role's server-side search_path — sql_builder emits
            # unqualified table names, so whichever schema resolves first is where the
            # query actually runs. No-op when the source has no non-default schema
            # (per-request sources from storage_adapters.reader don't carry one).
            schema = cfg.get("schema")
            if schema:
                cur.execute(_sql.SQL("SET search_path TO {}, public").format(
                    _sql.Identifier(schema)))
            cur.execute(sql, params or [])   # parameterized — no value interpolation
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(EXECUTION_RESULT_LIMIT)
        return cols, rows, None
    except Exception as e:
        return None, None, str(e)
    finally:
        conn.close()
