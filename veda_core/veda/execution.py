"""VEDA · L7 — read-only execution."""
import os, re, sys, time, json, logging, threading
from veda.runtime import DB_CONFIG


def execute_sql(sql, params=None):
    import psycopg2
    conn = psycopg2.connect(
        host=DB_CONFIG["host"], port=DB_CONFIG["port"], dbname=DB_CONFIG["database"],
        user=DB_CONFIG["user"], password=DB_CONFIG["password"])
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
