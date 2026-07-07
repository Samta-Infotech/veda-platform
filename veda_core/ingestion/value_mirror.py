"""L5 PUBLISH · Redis value mirror (Q-5).

Mirrors the hot ``column_values`` rows into Redis hashes at ingestion so the
query-time value resolver/arbiter do a sub-ms Redis lookup instead of a Postgres
round trip on nearly every non-fast-path query (Postgres stays the fallback).

Key: ``value:{tenant}:{source}:{value_norm}`` → JSON list of {table, col, raw}.
Additive + non-fatal: if Redis or column_values is unavailable, ingestion continues
and the query tier simply keeps using Postgres.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


def _redis_client():
    import redis
    url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
    return redis.Redis.from_url(url)


def _key(tenant: str, source_id: str, value_norm: str) -> str:
    return f"value:{tenant}:{source_id}:{value_norm}"


def mirror_values_to_redis(source_id: str = "", tenant: str = "default",
                           verbose: bool = False) -> Dict:
    """Read column_values for this source and mirror value_norm → [(table,col,raw)]."""
    from ingestion.db_abstraction import get_internal_connection, release_internal_connection

    conn = get_internal_connection()
    grouped: Dict[str, List[dict]] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value_norm, table_name, column_name, value_raw "
                "FROM column_values WHERE source_id = %s",
                [source_id],
            )
            for value_norm, table_name, column_name, value_raw in cur.fetchall():
                grouped.setdefault(value_norm, []).append(
                    {"table": table_name, "col": column_name, "raw": value_raw})
    finally:
        release_internal_connection(conn)

    if not grouped:
        return {"values": 0, "mirrored": False}

    client = _redis_client()
    pipe = client.pipeline()
    for value_norm, entries in grouped.items():
        pipe.set(_key(tenant, source_id, value_norm), json.dumps(entries))
    pipe.execute()
    if verbose:
        print(f"  [value_mirror] {len(grouped)} value_norm keys → redis")
    return {"values": len(grouped), "mirrored": True}


def lookup_value(value_norm: str, source_id: str = "", tenant: str = "default") -> Optional[List[dict]]:
    """Query-tier fast lookup: Redis-first value resolution, None → Postgres fallback."""
    try:
        raw = _redis_client().get(_key(tenant, source_id, value_norm))
        return json.loads(raw) if raw else None
    except Exception:
        return None
