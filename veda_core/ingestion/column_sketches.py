# =============================================================================
# ingestion/column_sketches.py
# VEDA — Per-column MinHash sketches (Cross-source plan, Phase 2.4)
#
# For every IDENTIFIER / CATEGORY / text column (ANY source kind) we compute a
# 128-permutation MinHash sketch over its normalized distinct values and persist
# it to `column_sketches`. Phase 4's cross-source join discovery
# (ingestion/cross_source_graph.py) reads these sketches to estimate Jaccard /
# containment between columns of DIFFERENT sources — the cheap, tenant-wide
# value-overlap signal that produces `cross_source_fk` edges.
#
# Cost is one pass over values already being sampled by the value sampler, so
# this rides alongside L1 value sampling rather than re-scanning the source.
#
# Storage mirrors ingestion/value_sampler.py's column_values table: a lazily
# CREATE-TABLE-IF-NOT-EXISTS table in the INTERNAL store (not a Django-managed
# table), so there is no migration to apply and the writer is self-bootstrapping.
#
# Optional dependency: datasketch. Absent → every entry point is a graceful
# no-op (returns None / 0), so importing this module and running ingestion never
# breaks before the dependency lands.
# =============================================================================

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, List, Optional, Tuple

from config import (
    COLUMN_SKETCHES_TABLE_NAME, MINHASH_NUM_PERM, SENSITIVE_PATTERNS,
    CROSS_SOURCE_EXACT_CONTAINMENT_CAP,
)
from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE as PSYCOPG2_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
)
from utils.logger import get_logger

logger = get_logger(__name__)


try:
    from datasketch import MinHash as _MinHash
    _DATASKETCH_AVAILABLE = True
except ImportError:
    _DATASKETCH_AVAILABLE = False


def _new_minhash(num_perm: int, hashvalues=None):
    """Construct a MinHash pinned to the 'legacy' hashing scheme so sketches are
    comparable regardless of the installed datasketch version. datasketch 2.0.0
    made `scheme` mandatory when rehydrating from stored hashvalues and changed the
    default; 'legacy' is the classic pre-2.0 behavior and is stable across 2.x.
    Falls back to a scheme-less constructor on older datasketch (<2.0)."""
    kw = {"num_perm": num_perm}
    if hashvalues is not None:
        kw["hashvalues"] = hashvalues
    try:
        return _MinHash(scheme="legacy", **kw)
    except TypeError:
        return _MinHash(**kw)


# Semantic types that carry join-key-shaped values worth sketching. Kept broad on
# purpose: identifiers are the primary join keys, categories/text widen recall for
# shared vocabularies across sources.
SKETCHABLE_TYPES = ("IDENTIFIER", "CATEGORY", "FREE_TEXT")

_NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
_ISO_DATE_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def sketches_available() -> bool:
    """True when MinHash sketching can actually run (datasketch present)."""
    return _DATASKETCH_AVAILABLE


def normalize_value(val) -> str:
    """Canonical form shared by the sketch pipeline and Phase-4 entity linking:
    NFC unicode, casefold, whitespace-collapsed; numbers and ISO dates are
    canonicalized so `01` / `1` and `2024-1-3` / `2024-01-03` sketch as equal."""
    if val is None:
        return ""
    s = unicodedata.normalize("NFC", str(val)).strip()
    s = re.sub(r"\s+", " ", s).casefold()
    if _NUMERIC_RE.match(s):
        try:
            f = float(s)
            return str(int(f)) if f.is_integer() else repr(f)
        except ValueError:
            pass
    m = _ISO_DATE_RE.match(s)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return s


def value_class(semantic_type: str, data_type: str = "") -> str:
    """Coarse compatibility class used to gate cross-source candidate pairs
    (Phase 4.2 compares only same-class columns). Distinct from the fine-grained
    semantic type so an IDENTIFIER in a CSV and an IDENTIFIER in Postgres match."""
    st = (semantic_type or "").upper()
    if st == "IDENTIFIER":
        return "id"
    if st == "CATEGORY":
        return "category"
    if (data_type or "").lower() in ("integer", "bigint", "smallint", "numeric", "double"):
        return "numeric"
    return "text"


def _value_hash(value_norm: str) -> int:
    """Stable 64-bit hash of a normalized value (process/version independent — unlike
    Python's salted hash()). Used to build the compact exact value set for containment."""
    return int.from_bytes(
        hashlib.blake2b(value_norm.encode("utf-8"), digest_size=8).digest(), "little")


def pack_value_hashes(value_norms: Iterable[str],
                      cap: int = CROSS_SOURCE_EXACT_CONTAINMENT_CAP) -> Optional[bytes]:
    """Sorted uint64 blob of the DISTINCT value hashes, or None when the set exceeds
    ``cap`` (fall back to the MinHash estimate above the cap)."""
    hs = {_value_hash(v) for v in value_norms if v}
    if not hs or len(hs) > cap:
        return None
    import numpy as np
    return np.array(sorted(hs), dtype=np.uint64).tobytes()


def unpack_value_hashes(blob: bytes):
    """Rehydrate a packed value-hash blob to a numpy uint64 array (None if empty)."""
    if not blob:
        return None
    import numpy as np
    return np.frombuffer(blob, dtype=np.uint64)


def compute_sketch(values: Iterable[str], num_perm: int = MINHASH_NUM_PERM
                   ) -> Tuple[Optional[bytes], int, Optional[bytes]]:
    """Compute a MinHash sketch AND (when small enough) an exact value-hash set over
    normalized distinct ``values``.

    Returns (sketch_bytes, n_distinct, value_hashes_bytes). sketch_bytes is None when
    datasketch is unavailable or there is nothing to sketch; value_hashes_bytes is None
    when the distinct set exceeds CROSS_SOURCE_EXACT_CONTAINMENT_CAP.
    """
    if not _DATASKETCH_AVAILABLE:
        return None, 0, None
    seen = set()
    mh = _new_minhash(num_perm)
    for v in values:
        nv = normalize_value(v)
        if nv and nv not in seen:
            seen.add(nv)
            mh.update(nv.encode("utf-8"))
    if not seen:
        return None, 0, None
    import numpy as np
    return (mh.hashvalues.astype(np.uint64).tobytes(), len(seen),
            pack_value_hashes(seen))


def sketch_from_bytes(blob: bytes, num_perm: int = MINHASH_NUM_PERM):
    """Rehydrate a datasketch MinHash from stored hashvalue bytes (Phase 4 read)."""
    if not _DATASKETCH_AVAILABLE or not blob:
        return None
    import numpy as np
    hv = np.frombuffer(blob, dtype=np.uint64).copy()  # datasketch mutates; need writable
    return _new_minhash(num_perm, hashvalues=hv)


def _create_table(cur) -> None:
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {COLUMN_SKETCHES_TABLE_NAME} (
            col_id       TEXT NOT NULL,
            source_id    TEXT NOT NULL,
            tenant       TEXT NOT NULL,
            table_name   TEXT NOT NULL,
            col_name     TEXT NOT NULL,
            n_distinct   INTEGER NOT NULL,
            value_class  TEXT NOT NULL,
            num_perm     INTEGER NOT NULL,
            sketch       BYTEA NOT NULL,
            value_hashes BYTEA,
            PRIMARY KEY (col_id, source_id, tenant)
        );
    """)
    # Lazy migration for stores created before the exact-containment column landed.
    cur.execute(f"ALTER TABLE {COLUMN_SKETCHES_TABLE_NAME} "
                f"ADD COLUMN IF NOT EXISTS value_hashes BYTEA;")
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{COLUMN_SKETCHES_TABLE_NAME}_tenant_class
        ON {COLUMN_SKETCHES_TABLE_NAME} (tenant, value_class);
    """)


def persist_sketches(rows: List[dict], source_id: str, tenant: str) -> int:
    """Upsert sketch rows for ``(source_id, tenant)``. Each row:
    {col_id, table_name, col_name, n_distinct, value_class, sketch(bytes)}.
    Idempotent per column via ON CONFLICT. Returns rows written. No-op (0) when
    datasketch/psycopg2 is unavailable so ingestion never fails on this stage."""
    if not (_DATASKETCH_AVAILABLE and PSYCOPG2_AVAILABLE) or not rows:
        return 0
    import psycopg2
    conn = get_internal_connection()
    written = 0
    try:
        with conn:
            with conn.cursor() as cur:
                _create_table(cur)
                # Fresh emit for this source: drop its prior sketches, then insert.
                cur.execute(
                    f"DELETE FROM {COLUMN_SKETCHES_TABLE_NAME} "
                    f"WHERE source_id = %s AND tenant = %s", (str(source_id), str(tenant)))
                for r in rows:
                    if not r.get("sketch"):
                        continue
                    vh = r.get("value_hashes")
                    cur.execute(f"""
                        INSERT INTO {COLUMN_SKETCHES_TABLE_NAME}
                            (col_id, source_id, tenant, table_name, col_name,
                             n_distinct, value_class, num_perm, sketch, value_hashes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (col_id, source_id, tenant) DO UPDATE SET
                            n_distinct = EXCLUDED.n_distinct,
                            value_class = EXCLUDED.value_class,
                            num_perm = EXCLUDED.num_perm,
                            sketch = EXCLUDED.sketch,
                            value_hashes = EXCLUDED.value_hashes;
                    """, (
                        str(r["col_id"]), str(source_id), str(tenant),
                        r.get("table_name", ""), r.get("col_name", ""),
                        int(r.get("n_distinct", 0)), r.get("value_class", "text"),
                        MINHASH_NUM_PERM, psycopg2.Binary(r["sketch"]),
                        psycopg2.Binary(vh) if vh else None,
                    ))
                    written += 1
    finally:
        release_internal_connection(conn)
    logger.info("column_sketches: wrote %d sketches for source=%s tenant=%s",
                written, source_id, tenant)
    return written


def read_columns_from_graph(source_id) -> List[dict]:
    """A source's column nodes from graph_nodes:
    [{col_id, table_name, col_name, semantic_type, data_type, is_pk, is_fk}].
    Used by the file-ingest sketch pass + the backfill to sketch a source without a
    live relational connection."""
    if not PSYCOPG2_AVAILABLE:
        return []
    from config import GRAPH_NODES_TABLE
    conn = get_internal_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT ref_id, table_name, name, semantic_type, data_type, is_pk, is_fk "
                f"FROM {GRAPH_NODES_TABLE} WHERE node_type='column' AND source_id=%s",
                [str(source_id)])
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("read_columns_from_graph failed (%s)", e)
        return []
    finally:
        release_internal_connection(conn)
    return [{"col_id": r[0], "table_name": r[1], "col_name": r[2],
             "semantic_type": r[3] or "", "data_type": r[4] or "",
             "is_pk": bool(r[5]), "is_fk": bool(r[6])} for r in rows]


def _column_value_class(col: dict) -> str:
    return "id" if (col["is_pk"] or col["is_fk"]) else value_class(col["semantic_type"], col["data_type"])


def _is_join_key_shaped(col: dict) -> bool:
    name = (col["col_name"] or "").lower()
    if any(p in name for p in SENSITIVE_PATTERNS):
        return False
    return bool(col["is_pk"] or col["is_fk"] or (col["semantic_type"] or "").upper() in SKETCHABLE_TYPES)


def sketch_columns_via_sampler(source_id, tenant, sampler, sample_size=None) -> int:
    """Sketch a source's join-key-shaped columns by reading them from the graph and
    sampling DISTINCT values through ``sampler(table, col, n) -> [values]``. Persists
    to column_sketches. Reused by the file-ingest path (sampler = tabular connector)
    and the backfill (sampler = relational SELECT DISTINCT). No-op without datasketch."""
    from config import CROSS_SOURCE_SKETCH_SAMPLE_SIZE
    if not sketches_available():
        return 0
    n = sample_size or CROSS_SOURCE_SKETCH_SAMPLE_SIZE
    rows: List[dict] = []
    for c in read_columns_from_graph(source_id):
        if not _is_join_key_shaped(c):
            continue
        vals = sampler(c["table_name"], c["col_name"], n)
        sketch, nd, vhashes = compute_sketch(vals)
        if sketch is None:
            continue
        rows.append({"col_id": c["col_id"], "table_name": c["table_name"],
                     "col_name": c["col_name"], "n_distinct": nd,
                     "value_class": _column_value_class(c), "sketch": sketch,
                     "value_hashes": vhashes})
    return persist_sketches(rows, source_id=source_id, tenant=tenant)


def build_sketch_rows(sampled_columns, is_sensitive=None) -> List[dict]:
    """Turn value_sampler SampledColumn records into sketch rows. A column is
    skipped when its name looks sensitive (PII never gets a value-overlap sketch,
    mirroring the entity-layer PII guard). ``sampled_columns`` items must expose
    col_id, col_name, table_name, semantic_type, and values (normalized list)."""
    rows: List[dict] = []
    for sc in sampled_columns:
        st = getattr(sc, "semantic_type", "")
        if st not in SKETCHABLE_TYPES:
            continue
        name = (getattr(sc, "col_name", "") or "").lower()
        sensitive = is_sensitive(name) if is_sensitive else \
            any(p in name for p in SENSITIVE_PATTERNS)
        if sensitive:
            continue
        sketch, n, vhashes = compute_sketch(getattr(sc, "values", []) or [])
        if sketch is None:
            continue
        rows.append({
            "col_id": getattr(sc, "col_id", ""),
            "table_name": getattr(sc, "table_name", ""),
            "col_name": getattr(sc, "col_name", ""),
            "n_distinct": n,
            "value_class": value_class(st, getattr(sc, "data_type", "")),
            "sketch": sketch,
            "value_hashes": vhashes,
        })
    return rows
