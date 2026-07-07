"""storage_adapters/reader.py — query-time read contracts (migration plan §3.3, §8.3).

The ENGINE calls these from the inference tier, which is Django-free — so reads use
raw psycopg2 against the Postgres Django manages (through PgBouncer) + Redis, never
the ORM (§8.3: FK map from Redis/pgvector, value samples from Redis SET, ANN from
raw pgvector). Signatures are FROZEN and read the ambient `(source, tenant)` from
`veda_core.context.current()` (§4.1, fail-closed) — no tenant parameter. Return
shapes match today's `vector_store.py` so the engine's callers are unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from veda_core import context


@dataclass
class FKEdge:
    """Structurally identical to `ingestion.vector_store.FKEdge` (same 8 fields),
    but defined here so the reader stays import-light (no numpy) and runs in the
    Django-free inference tier. Callers attribute-access these fields."""
    from_col_id: str
    from_col_name: str
    from_table_id: str
    from_table_name: str
    to_col_id: str
    to_col_name: str
    to_table_id: str
    to_table_name: str


_CONN = None


def _connection():
    """psycopg2 connection to the Django-managed Postgres (the `veda` DB, through
    PgBouncer). Reads the same POSTGRES_*/PGBOUNCER_* env the Django settings use."""
    global _CONN
    if _CONN is not None and _CONN.closed == 0:
        return _CONN
    import psycopg2
    _CONN = psycopg2.connect(
        host=os.environ.get("PGBOUNCER_HOST", "pgbouncer"),
        port=int(os.environ.get("PGBOUNCER_PORT", "6432")),
        dbname=os.environ.get("POSTGRES_DB", "veda"),
        user=os.environ.get("POSTGRES_USER", "veda"),
        password=os.environ.get("POSTGRES_PASSWORD", "change-me"),
    )
    # autocommit only — do NOT set a session-level READ ONLY default: under
    # PgBouncer transaction pooling that persists on the shared server connection
    # and poisons the next client (a Django write would fail read-only). The reader
    # is read-only by construction (SELECT only).
    _CONN.autocommit = True
    return _CONN


def _scope():
    ctx = context.current()  # fail-closed if unset (§4.1)
    return ctx.source_id, ctx.tenant


def source_connection() -> dict:
    """The client source DB connection for the current request's source_id, read
    from the `Source` registry row (§3.1, §5) — the single source of truth. This is
    the query-tier counterpart to ``apps.sources.models.Source.as_engine_env`` (the
    ingestion tier): both resolve the SAME row, so one warm engine serves N sources
    by connecting to whichever source the ambient context selects.

    Django-free — raw psycopg2 over the same Postgres the ORM manages (through
    PgBouncer), so the inference tier resolves the per-request source WITHOUT
    importing Django. Password resolves env-ref first (prod: Docker secret named by
    ``password_env``), then the dev inline value — mirrors ``Source.resolve_password``.
    Fail-closed: a missing/unset context raises; an unknown source_id raises — never
    a silent localhost/default-credential fallback (§3.1)."""
    source_id, _tenant = _scope()  # fail-closed if context unset (§4.1)
    with _connection().cursor() as cur:
        cur.execute(
            "SELECT host, port, dbname, db_user, password_env, password_inline "
            "FROM sources_source WHERE id = %s",
            [source_id],
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(
            f"Source id={source_id} not found in the sources registry — cannot resolve "
            "its connection (§3.1: the Source row is the single source of truth)."
        )
    host, port, dbname, db_user, password_env, password_inline = row
    password = os.environ.get(password_env, "") if password_env else (password_inline or "")
    return {"host": host, "port": port or 5432, "database": dbname,
            "user": db_user, "password": password}


def get_fk_adjacency(table_ids: Optional[List[str]] = None) -> List[FKEdge]:
    """FK edges (legacy FKEdge shape) for the current (source, tenant), optionally
    restricted to edges touching `table_ids`. Raw SQL join over the Django-owned
    substrate tables — same return shape as `vector_store.get_fk_adjacency`."""
    source_id, tenant = _scope()
    sql = """
        SELECT fc.id, fc.name, ft.id, ft.name, tc.id, tc.name, tt.id, tt.name
        FROM substrate_fkedge e
        JOIN substrate_schemacolumn fc ON fc.id = e.from_col_id
        JOIN substrate_schematable  ft ON ft.id = e.from_table_id
        JOIN substrate_schemacolumn tc ON tc.id = e.to_col_id
        JOIN substrate_schematable  tt ON tt.id = e.to_table_id
        WHERE e.source_id = %s AND e.tenant = %s
    """
    params: list = [source_id, tenant]
    if table_ids:
        ids = tuple(str(t) for t in table_ids)
        sql += " AND (e.from_table_id IN %s OR e.to_table_id IN %s)"
        params += [ids, ids]
    with _connection().cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [FKEdge(str(r[0]), r[1], str(r[2]), r[3], str(r[4]), r[5], str(r[6]), r[7]) for r in rows]


def glossary() -> dict:
    """term -> {canonical, definition} for the current (source, tenant)."""
    source_id, tenant = _scope()
    with _connection().cursor() as cur:
        cur.execute(
            "SELECT term, canonical, definition FROM substrate_glossaryentry "
            "WHERE source_id = %s AND tenant = %s", [source_id, tenant],
        )
        return {r[0]: {"canonical": r[1], "definition": r[2]} for r in cur.fetchall()}


def synonyms() -> dict:
    """term -> [synonyms] for the current (source, tenant)."""
    source_id, tenant = _scope()
    out: dict = {}
    with _connection().cursor() as cur:
        cur.execute(
            "SELECT term, synonym FROM substrate_synonym WHERE source_id = %s AND tenant = %s",
            [source_id, tenant],
        )
        for term, syn in cur.fetchall():
            out.setdefault(term, []).append(syn)
    return out


def value_samples(column_uuid: str) -> List[str]:
    """Sampled values for `column_uuid` (value grounding, §6.3)."""
    source_id, tenant = _scope()
    with _connection().cursor() as cur:
        cur.execute(
            "SELECT value FROM substrate_columnvaluesample "
            "WHERE column_id = %s AND source_id = %s AND tenant = %s",
            [column_uuid, source_id, tenant],
        )
        return [r[0] for r in cur.fetchall()]


_EF_SEARCH_CACHE: dict = {}   # (source_id, tenant) → int; cleared on rehydrate


def clear_ef_search_cache() -> None:
    """Called by the inference rehydrate subscriber so a re-ingest's newly tuned
    ef_search is picked up without a process restart."""
    _EF_SEARCH_CACHE.clear()


def _resolve_ef_search(source_id) -> int:
    """Per-source hnsw.ef_search (P7/Q-10 knob, review Finding 4 closed):
    VEDA_HNSW_EF_SEARCH_<source_id> env → SubstrateVersion.hnsw_ef_search (tuned at
    L5, persisted by writer.warm; cached per process so it is NOT a per-query DB
    hit) → VEDA_HNSW_EF_SEARCH → 40."""
    per_source = os.environ.get(f"VEDA_HNSW_EF_SEARCH_{source_id}")
    if per_source:
        try:
            return int(per_source)
        except ValueError:
            pass
    # Substrate lookup (Django-free — raw SQL, one query per (scope, process)).
    try:
        _, tenant = _scope()
        key = (str(source_id), str(tenant))
        if key not in _EF_SEARCH_CACHE:
            with _connection().cursor() as cur:
                cur.execute(
                    "SELECT hnsw_ef_search FROM substrate_substrateversion "
                    "WHERE source_id = %s AND tenant = %s "
                    "ORDER BY id DESC LIMIT 1",
                    [source_id, tenant],
                )
                row = cur.fetchone()
            _EF_SEARCH_CACHE[key] = int(row[0]) if row and row[0] else 0
        if _EF_SEARCH_CACHE[key] > 0:
            return _EF_SEARCH_CACHE[key]
    except Exception:
        pass  # table absent / no context — fall through to the global default
    return int(os.environ.get("VEDA_HNSW_EF_SEARCH", "40"))


def ann_search(mode: str, qvec: List[float], top_k: int) -> List[Any]:
    """Raw pgvector cosine ANN over the HNSW index for `mode`'s table (§6.4),
    scoped to the current (source, tenant)."""
    source_id, tenant = _scope()
    # WP3: one embedding space (BGE-M3) → one store. The legacy relgt/light_text/hybrid
    # modes and their column_embeddings/_lt/_hybrid tables were removed.
    table = "column_embeddings_bge"
    vec = "[" + ",".join(str(float(x)) for x in qvec) + "]"
    # Pin hnsw.ef_search to the §7.1a-tuned value (recall@k=1.0 on the home-schema
    # fixtures) so the served ANN ordering matches exact cosine — the shipped index IS
    # the gated index. Per-source override (tuned at L5 by source size, stored on
    # SubstrateVersion.hnsw_ef_search) via VEDA_HNSW_EF_SEARCH_<source_id>, then the
    # global VEDA_HNSW_EF_SEARCH, then 40 — so a large source can widen search without
    # changing the global (re-run scripts/hnsw_parity_sweep.py per source).
    ef_search = _resolve_ef_search(source_id)
    sql = (
        f'SELECT column_uuid, 1 - (embedding <=> %s::vector) AS score FROM "{table}" '
        f'WHERE source_id = %s AND tenant = %s ORDER BY embedding <=> %s::vector LIMIT %s'
    )
    # Explicit transaction so SET LOCAL is scoped to it and released at COMMIT — this is
    # PgBouncer-transaction-pool-safe (never leaks the GUC to the next pooled client),
    # unlike a session-level SET (see the readonly-poisoning fix in _connection).
    with _connection().cursor() as cur:
        cur.execute("BEGIN")
        cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
        cur.execute(sql, [vec, source_id, tenant, vec, top_k])
        rows = cur.fetchall()
        cur.execute("COMMIT")
        return rows


def verified_cache_lookup(qvec: List[float], threshold: float = 0.85) -> Optional[dict]:
    """Verified-query cache lookup (cosine >= threshold), §6.6."""
    source_id, tenant = _scope()
    vec = "[" + ",".join(str(float(x)) for x in qvec) + "]"
    with _connection().cursor() as cur:
        cur.execute(
            "SELECT verified_sql, columns_json, 1 - (query_embedding <=> %s::vector) "
            "FROM substrate_verifiedquerycache "
            "WHERE source_id = %s AND tenant = %s AND query_embedding IS NOT NULL "
            "ORDER BY query_embedding <=> %s::vector LIMIT 1",
            [vec, source_id, tenant, vec],
        )
        row = cur.fetchone()
    if row and row[2] is not None and row[2] >= threshold:
        return {"sql": row[0], "columns": row[1], "similarity": float(row[2])}
    return None


def _query_hash(query: str) -> str:
    import hashlib
    import re
    norm = re.sub(r"\s+", " ", (query or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:64]


def verified_cache_exact(query: str) -> Optional[dict]:
    """Q-8: exact-hash short-circuit — one indexed PK lookup on (source, tenant,
    query_hash) BEFORE any BGE encode + cosine ANN. Returns similarity 1.0 on hit.
    Repeat/identical queries skip the embed entirely."""
    source_id, tenant = _scope()
    qhash = _query_hash(query)
    with _connection().cursor() as cur:
        cur.execute(
            "SELECT verified_sql, columns_json FROM substrate_verifiedquerycache "
            "WHERE source_id = %s AND tenant = %s AND query_hash = %s LIMIT 1",
            [source_id, tenant, qhash],
        )
        row = cur.fetchone()
    if row:
        return {"sql": row[0], "columns": row[1], "similarity": 1.0}
    return None


def save_verified_query(query: str, qvec: List[float], sql: str, columns=None) -> bool:
    """The ONE documented inference-tier WRITE (§6.6): record a verified query on the hot
    path. Django-free (raw psycopg2, own writable connection). Idempotent under N replicas
    via INSERT … ON CONFLICT (source, tenant, query_hash) DO NOTHING — never
    read-modify-write. Publishes a cache-entry fan-out so peers warm it (§8.4). Skip rules
    (existence/fast-path/temporal never cached) stay in the caller (pipeline.py)."""
    import json as _json
    import os

    import psycopg2

    source_id, tenant = _scope()
    vec = "[" + ",".join(str(float(x)) for x in qvec) + "]"
    qhash = _query_hash(query)
    conn = psycopg2.connect(
        host=os.environ.get("PGBOUNCER_HOST", "pgbouncer"),
        port=int(os.environ.get("PGBOUNCER_PORT", "6432")),
        dbname=os.environ.get("POSTGRES_DB", "veda"),
        user=os.environ.get("POSTGRES_USER", "veda"),
        password=os.environ.get("POSTGRES_PASSWORD", "change-me"),
    )
    conn.autocommit = True
    inserted = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO substrate_verifiedquerycache "
                "(id, source_id, tenant, query_hash, query_text, verified_sql, columns_json, "
                " query_embedding, created_at, updated_at) "
                "VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s::jsonb, %s::vector, now(), now()) "
                "ON CONFLICT (source_id, tenant, query_hash) DO NOTHING",
                [source_id, tenant, qhash, query, sql, _json.dumps(columns or []), vec],
            )
            inserted = cur.rowcount > 0
    finally:
        conn.close()
    if inserted:
        try:
            import redis as _redis
            url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
            ch = f"veda:rehydrate:{source_id}:{tenant}:verified_cache"
            _redis.Redis.from_url(url).publish(
                ch, _json.dumps({"source_id": source_id, "tenant": tenant, "scope": "verified_cache"}))
        except Exception:
            pass
    return inserted
