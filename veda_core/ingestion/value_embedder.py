# =============================================================================
# ingestion/value_embedder.py
# VEDA — Semantic bridge Tier B: per-value embedding index (structured side)
# (docs/SEMANTIC_ENTITY_BRIDGE.md §3.1 Tier B)
#
# Tier A (semantic_linker.py) bridges a chunk to a COLUMN topically. Tier B bridges
# a chunk SPAN to the actual DB VALUE it paraphrases — the true "ACME Corporation"
# (doc) ↔ "ACME-CORP" (column value) link that exact matching misses.
#
# This module owns the STRUCTURED side: at structured ingest it embeds each eligible
# sampled DISPLAY value into `entity_value_embeddings` (a lazily-created table in the
# INTERNAL store, HNSW cosine index — same pattern as graph_node_embeddings, so NO
# migration). The DOC side (span extraction + ANN match) lives in semantic_linker.py
# and reads this index via `load_value_embeddings`.
#
# Eligibility: CATEGORY / FREE_TEXT columns only (SEMANTIC_VALUE_ELIGIBLE_TYPES) —
# identifiers/codes have no semantic synonyms and are already owned by the exact/
# pattern detectors. Sensitive + enum/metadata columns are excluded (same guards as
# the entity linker). Per-column cap bounds high-cardinality FREE_TEXT cost.
#
# Idempotent per source (scoped delete before re-insert). Graceful no-op when
# disabled, when M3 is unavailable, or when there are no eligible values.
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    SEMANTIC_VALUE_BRIDGE_ENABLED,
    SEMANTIC_VALUE_EMB_TABLE,
    SEMANTIC_VALUE_ELIGIBLE_TYPES,
    SEMANTIC_VALUE_MAX_PER_COL,
    SEMANTIC_VALUE_MAX_VECTORS,
    GRAPH_NODE_EMB_DIM,
    SENSITIVE_PATTERNS,
)
from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
    DICT_CURSOR,
)
from ingestion.column_sketches import normalize_value, value_class as _value_class
from ingestion.graph_persist import col_node_id
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_VALUE_LEN = 4


@dataclass
class ValueEmbedResult:
    values_embedded: int
    columns:         int
    source_id:       str
    backend:         str
    duration_sec:    float
    stats:           dict = field(default_factory=dict)


# --------------------------------------------------------------------------- vecs
def _vec_str(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in np.asarray(vec, dtype=np.float32).tolist()) + "]"


def _parse_vec(raw) -> Optional[np.ndarray]:
    if raw is None:
        return None
    if hasattr(raw, "tolist") or isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.float32)
    try:
        return np.asarray([float(x) for x in str(raw).strip("[]").split(",") if x.strip()],
                          dtype=np.float32)
    except Exception:
        return None


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32)


# ------------------------------------------------------------------- eligibility
# Reuse the entity-linker guards so the value index and the exact bridge exclude the
# same enum/metadata/sensitive columns (imported lazily to avoid an import cycle).
def _col_excluded(col_name: str) -> bool:
    n = (col_name or "").lower()
    if any(p in n for p in SENSITIVE_PATTERNS):
        return True
    try:
        from ingestion.entity_linker import _is_metadata_col
        return _is_metadata_col(n)
    except Exception:
        return False


def _eligible_values(sc) -> List[Tuple[str, str]]:
    """[(value_norm, display)] for one SampledColumn — eligible type, non-sensitive
    column, capped, deduped by normalized value, length-filtered. Display = original
    casing (raw_values) so M3 sees natural text; falls back to normalized."""
    st = (getattr(sc, "semantic_type", "") or "").upper()
    if st not in SEMANTIC_VALUE_ELIGIBLE_TYPES:
        return []
    if _col_excluded(getattr(sc, "col_name", "")):
        return []
    raws = list(getattr(sc, "raw_values", None) or getattr(sc, "values", None) or [])
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for raw in raws:
        display = str(raw).strip()
        vn = normalize_value(display)
        if len(vn) < _MIN_VALUE_LEN or vn in seen:
            continue
        seen.add(vn)
        out.append((vn, display))
        if len(out) >= SEMANTIC_VALUE_MAX_PER_COL:
            break
    return out


# --------------------------------------------------------------------------- ddl
def _create_value_emb_table(cursor) -> None:
    # Drop + recreate if the embedding dimension ever changes (mirrors graph_embedder).
    try:
        cursor.execute(f"""
            SELECT atttypmod - 4 AS dim FROM pg_attribute
            JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
            WHERE pg_class.relname = '{SEMANTIC_VALUE_EMB_TABLE}'
              AND pg_attribute.attname = 'embedding'
        """)
        row = cursor.fetchone()
        if row and row[0] != GRAPH_NODE_EMB_DIM:
            cursor.execute(f"DROP TABLE IF EXISTS {SEMANTIC_VALUE_EMB_TABLE};")
    except Exception:
        pass
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {SEMANTIC_VALUE_EMB_TABLE} (
            col_id      TEXT NOT NULL,
            source_id   TEXT NOT NULL,
            value_norm  TEXT NOT NULL,
            display     TEXT NOT NULL,
            value_class TEXT NOT NULL,
            embedding   vector({GRAPH_NODE_EMB_DIM}),
            PRIMARY KEY (col_id, value_norm)
        );
    """)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{SEMANTIC_VALUE_EMB_TABLE}_src "
                   f"ON {SEMANTIC_VALUE_EMB_TABLE} (source_id);")
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{SEMANTIC_VALUE_EMB_TABLE}_emb
        ON {SEMANTIC_VALUE_EMB_TABLE} USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200);
    """)


# ------------------------------------------------------------------- build (write)
def embed_source_values(source_id: str, sampled_columns, tenant: str = "default",
                        verbose: bool = False) -> ValueEmbedResult:
    """Embed a structured source's eligible sampled values into entity_value_embeddings.
    Idempotent per source. Best-effort: any failure returns a zero result rather than
    breaking the ingest. ``sampled_columns`` are value_sampler.SampledColumn records."""
    t0 = time.time()
    backend = "postgres" if INTERNAL_DB_AVAILABLE else "in_memory"
    if not SEMANTIC_VALUE_BRIDGE_ENABLED or not sampled_columns:
        return ValueEmbedResult(0, 0, str(source_id), "disabled", round(time.time() - t0, 3))
    if not INTERNAL_DB_AVAILABLE:
        # No structured store → nothing to persist against (in-memory fallback unused here).
        return ValueEmbedResult(0, 0, str(source_id), "in_memory", round(time.time() - t0, 3))

    # Collect (col_id, value_norm, display, value_class) across eligible columns.
    rows: List[Tuple[str, str, str, str]] = []
    n_cols = 0
    for sc in sampled_columns:
        vals = _eligible_values(sc)
        if not vals:
            continue
        n_cols += 1
        vclass = _value_class(getattr(sc, "semantic_type", ""), getattr(sc, "data_type", ""))
        cid = getattr(sc, "col_id", "")
        for vn, disp in vals:
            rows.append((cid, vn, disp, vclass))
            if len(rows) >= SEMANTIC_VALUE_MAX_VECTORS:
                break
        if len(rows) >= SEMANTIC_VALUE_MAX_VECTORS:
            break
    if not rows:
        return ValueEmbedResult(0, 0, str(source_id), backend, round(time.time() - t0, 3))

    # Batch-encode display forms (natural text) via the shared M3 singleton.
    try:
        from ingestion import m3_encoder
        embs = m3_encoder.encode_dense([disp for (_c, _v, disp, _cl) in rows])
    except Exception as e:
        logger.warning("value_embedder: M3 encode failed (%s) — skipped", e)
        return ValueEmbedResult(0, n_cols, str(source_id), "no_model", round(time.time() - t0, 3))
    embs = _l2_normalize(np.asarray(embs, dtype=np.float32))

    from psycopg2.extras import execute_values
    conn = get_internal_connection()
    written = 0
    try:
        with conn:
            with conn.cursor() as cur:
                _create_value_emb_table(cur)
                cur.execute(f"DELETE FROM {SEMANTIC_VALUE_EMB_TABLE} WHERE source_id = %s;",
                            (str(source_id),))
                payload = [
                    (cid, str(source_id), vn, disp, vclass, _vec_str(embs[i]))
                    for i, (cid, vn, disp, vclass) in enumerate(rows)
                ]
                execute_values(
                    cur,
                    f"""INSERT INTO {SEMANTIC_VALUE_EMB_TABLE}
                        (col_id, source_id, value_norm, display, value_class, embedding)
                        VALUES %s
                        ON CONFLICT (col_id, value_norm) DO UPDATE SET
                            source_id = EXCLUDED.source_id,
                            display = EXCLUDED.display,
                            value_class = EXCLUDED.value_class,
                            embedding = EXCLUDED.embedding""",
                    payload, template="(%s,%s,%s,%s,%s,%s::vector)",
                )
                written = len(payload)
    except Exception as e:
        logger.warning("value_embedder: persist failed (%s)", e)
        return ValueEmbedResult(0, n_cols, str(source_id), backend, round(time.time() - t0, 3))
    finally:
        release_internal_connection(conn)

    dur = round(time.time() - t0, 3)
    if verbose:
        logger.info("value_embedder: %d value vectors over %d cols, source=%s (%.2fs)",
                    written, n_cols, source_id, dur)
    return ValueEmbedResult(written, n_cols, str(source_id), backend, dur)


# ------------------------------------------------------------------- load (read)
def load_value_embeddings(exclude_source_id: Optional[str] = None
                          ) -> Tuple[List[str], List[str], List[str], List[str], np.ndarray]:
    """Value vectors for the doc-side matcher. Returns
    (col_node_ids, value_norms, displays, value_classes, matrix[n, dim] L2-normalized),
    tenant-wide, excluding the doc's own source. Capped at SEMANTIC_VALUE_MAX_VECTORS."""
    empty = ([], [], [], [], np.zeros((0, 0), dtype=np.float32))
    if not INTERNAL_DB_AVAILABLE:
        return empty
    try:
        conn = get_internal_connection()
    except Exception:
        return empty
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        params: list = []
        where = "1=1"
        if exclude_source_id is not None:
            where = "source_id <> %s"
            params.append(str(exclude_source_id))
        params.append(int(SEMANTIC_VALUE_MAX_VECTORS))
        cur.execute(
            f"SELECT col_id, value_norm, display, value_class, embedding "
            f"FROM {SEMANTIC_VALUE_EMB_TABLE} WHERE {where} LIMIT %s", params)
        rows = cur.fetchall()
        try: cur.close()
        except Exception: pass
    except Exception as e:
        # Table may not exist yet (no structured source embedded) — graceful empty.
        logger.debug("value_embedder: load skipped (%s)", e)
        return empty
    finally:
        release_internal_connection(conn)

    col_nodes: List[str] = []
    vnorms: List[str] = []
    displays: List[str] = []
    vclasses: List[str] = []
    vecs: List[np.ndarray] = []
    for r in rows:
        v = _parse_vec(r["embedding"])
        if v is None or v.size == 0:
            continue
        col_nodes.append(col_node_id(r["col_id"]))
        vnorms.append(r["value_norm"])
        displays.append(r["display"])
        vclasses.append(r["value_class"])
        vecs.append(v)
    if not vecs:
        return empty
    return col_nodes, vnorms, displays, vclasses, _l2_normalize(np.vstack(vecs))
