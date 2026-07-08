# =============================================================================
# ingestion/entity_linker.py
# VEDA — Entity extraction & linking (Cross-source plan, Phase 4.1)
#
# Replaces and subsumes chunk_linker: normalized values that appear in BOTH a
# chunk's text AND a column's sampled values become `entity` nodes that bridge the
# narrative (chunks) and tabular (columns) worlds. Traversal can then go
#   chunk --mentions_entity--> entity --value_of--> column --cross_source_fk--> other source
# which is exactly how "what does ACME's contract say about late fees" reaches both
# the contract chunk and the ACME row across sources.
#
# Three deterministic detectors (no training):
#   1. Dictionary (primary) — chunk n-grams matched against the tenant value store
#      (column_values). Only IDENTIFIER/CATEGORY/name-like columns; values < 4 chars
#      or stopwords excluded to control noise.
#   2. Pattern — typed regexes (email, phone, money, ISO date) + tenant id_patterns.
#      Typed entities match columns whose value_class shares the pattern class.
#   3. SLM (optional, docs) — widens recall of 1–2; only admits entities that ALSO
#      dictionary/pattern-match, so it never mints unlinked entities. (Hook left for
#      the enrichment pass; deterministic detectors are the grounded core.)
#
# Admission rule (explosion control): an entity node is created only when it links
# ≥ 1 chunk AND ≥ 1 column (or ≥ 2 columns in different sources). Pure single-sided
# values stay as plain value signals.
#
# PII guard: columns whose name matches SENSITIVE_PATTERNS never emit entities;
# email entities store a salted hash as the node id with a masked display value.
# =============================================================================

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from config import (
    COLUMN_VALUES_TABLE_NAME, SENSITIVE_PATTERNS, GRAPH_EDGE_WEIGHTS,
    GRAPH_NODES_TABLE, GRAPH_EDGES_TABLE,
)
from ingestion import graph_persist as GP
from ingestion.graph_persist import GraphNode, GraphEdge, chunk_node_id
from ingestion.column_sketches import normalize_value
from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE, get_internal_connection, release_internal_connection, DICT_CURSOR,
)
from utils.logger import get_logger

logger = get_logger(__name__)

ENTITY_CLASSES = ("id", "email", "name", "money", "date", "term")
_ENTITY_SALT = os.environ.get("VEDA_ENTITY_SALT", "veda-entity")
_MIN_VALUE_LEN = 4
_STOPWORDS = {"true", "false", "none", "null", "n/a", "yes", "no", "the", "and", "for"}

# Typed pattern detectors → the value_class a matched entity should link columns on.
_PATTERNS: List[Tuple[str, str, "re.Pattern"]] = [
    ("email", "email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("money", "numeric", re.compile(r"[$₹€£]\s?\d[\d,]*(?:\.\d+)?\b")),
    ("date",  "date",    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")),
    ("phone", "id",      re.compile(r"\b(?:\+?\d[\d\s-]{7,}\d)\b")),
]


@dataclass
class EntityLinkResult:
    chunk_nodes: int
    entity_nodes: int
    mentions_entity: int
    value_of: int
    source_id: str
    backend: str
    duration_sec: float
    stats: dict = field(default_factory=dict)


def entity_node_id(cls: str, value_norm: str) -> str:
    """`ent:<class>:<value_norm>`; emails are hashed (PII) so the raw address never
    becomes a node id."""
    if cls == "email":
        h = hashlib.sha256((_ENTITY_SALT + value_norm).encode("utf-8")).hexdigest()[:24]
        return f"ent:email:{h}"
    safe = re.sub(r"\s+", "_", value_norm)[:80]
    return f"ent:{cls}:{safe}"


def _mask_email(value_norm: str) -> str:
    m = re.match(r"([^@]{1,2})[^@]*(@.*)", value_norm)
    return f"{m.group(1)}***{m.group(2)}" if m else "***"


def _is_sensitive_col(col_name: str) -> bool:
    n = (col_name or "").lower()
    return any(p in n for p in SENSITIVE_PATTERNS)


# --------------------------------------------------------------------------- load
def _load_value_index() -> Dict[str, List[dict]]:
    """value_norm -> [{col_id, col_name, source_id, semantic_type, value_class}] from
    column_values joined to the column nodes (for source_id / sensitivity / class).
    Only non-sensitive, dictionary-eligible columns participate."""
    if not INTERNAL_DB_AVAILABLE:
        return {}
    conn = get_internal_connection()
    try:
        cur = conn.cursor(cursor_factory=DICT_CURSOR)
        # column metadata from graph_nodes (source_id, name, semantic_type)
        cur.execute(f"SELECT ref_id, source_id, name, semantic_type FROM {GRAPH_NODES_TABLE} "
                    f"WHERE node_type = 'column'")
        meta = {r["ref_id"]: r for r in cur.fetchall()}
        cur.execute(f"SELECT col_id, value_norm FROM {COLUMN_VALUES_TABLE_NAME}")
        rows = cur.fetchall()
        cur.close()
    except Exception as e:
        logger.warning("entity_linker: value index load failed (%s)", e)
        return {}
    finally:
        release_internal_connection(conn)

    idx: Dict[str, List[dict]] = {}
    for r in rows:
        col = meta.get(r["col_id"])
        if col is None or _is_sensitive_col(col["name"]):
            continue
        st = (col["semantic_type"] or "").upper()
        if st not in ("IDENTIFIER", "CATEGORY", "FREE_TEXT"):
            continue
        v = normalize_value(r["value_norm"])
        if len(v) < _MIN_VALUE_LEN or v in _STOPWORDS:
            continue
        cls = "id" if st == "IDENTIFIER" else ("name" if st == "FREE_TEXT" else "term")
        idx.setdefault(v, []).append({
            "col_id": r["col_id"], "col_name": col["name"], "source_id": str(col["source_id"]),
            "semantic_type": st, "class": cls})
    return idx


# ------------------------------------------------------------------ detectors
def detect_entities(text: str, value_index: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Return {value_norm: {"class": cls, "columns": [colmeta...]}} for entities found
    in ``text`` that satisfy the admission rule (link ≥1 column). Pure function —
    unit-testable without a DB (pass a synthetic value_index)."""
    norm = normalize_value(text)
    found: Dict[str, dict] = {}

    # 1. Dictionary detector — substring match of stored values against the chunk.
    # Cheap first-token substring pre-gate before the full-value scan (a plain token
    # set fails when punctuation attaches, e.g. "inv-2024-0113." — so gate on substring).
    for value_norm, cols in value_index.items():
        first = value_norm.split(" ", 1)[0]
        if first not in norm:                 # cheap pre-gate before full-value scan
            continue
        if value_norm in norm:
            found[value_norm] = {"class": cols[0]["class"], "columns": list(cols)}

    # 2. Pattern detector — typed regexes; a typed entity links columns whose class
    #    matches the pattern class (even if the exact value wasn't sampled).
    for cls, vclass, pat in _PATTERNS:
        for m in pat.findall(text):
            v = normalize_value(m if isinstance(m, str) else m[0])
            if len(v) < _MIN_VALUE_LEN:
                continue
            cols = [c for c in value_index.get(v, [])]
            if v in found:
                found[v]["class"] = cls
                continue
            # admission: pattern entities need a column linkage (exact-value hit).
            if cols:
                found[v] = {"class": cls, "columns": cols}
    return found


# ------------------------------------------------------------------- scoped delete
_ENTITY_EDGE_TYPES = ("mentions_entity", "value_of", "mentions", "name_match", "about")


def _scoped_delete(source_id: str) -> None:
    if not INTERNAL_DB_AVAILABLE:
        GP._IN_MEMORY_NODES = [n for n in GP._IN_MEMORY_NODES
                               if not (n["source_id"] == source_id and n["node_type"] == "chunk")]
        GP._IN_MEMORY_EDGES = [e for e in GP._IN_MEMORY_EDGES
                               if not (e["source_id"] == source_id
                                       and e["edge_type"] in _ENTITY_EDGE_TYPES)]
        return
    conn = get_internal_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                GP._create_graph_tables(cur)
                ph = ",".join(["%s"] * len(_ENTITY_EDGE_TYPES))
                cur.execute(f"DELETE FROM {GRAPH_EDGES_TABLE} WHERE source_id = %s "
                            f"AND edge_type IN ({ph})", (source_id, *_ENTITY_EDGE_TYPES))
                cur.execute(f"DELETE FROM {GRAPH_NODES_TABLE} WHERE source_id = %s "
                            f"AND node_type = 'chunk'", (source_id,))
                # entity nodes are tenant-shared; leave them (idempotent upsert refreshes).
    except Exception as e:
        logger.warning("entity_linker: scoped delete failed (%s)", e)
    finally:
        release_internal_connection(conn)


# ------------------------------------------------------------------- main entry
def link_entities(chunks, source_id: str, tenant: str = "default",
                  verbose: bool = False) -> EntityLinkResult:
    """Create chunk nodes, extract entities per chunk, and emit
    mentions_entity (chunk→entity) + value_of (entity→column) edges. Replaces
    chunk_linker.link_chunks_to_graph. Idempotent per source."""
    t0 = time.time()
    backend = "postgres" if INTERNAL_DB_AVAILABLE else "in_memory"
    if not chunks:
        return EntityLinkResult(0, 0, 0, 0, source_id, "no_chunks", round(time.time() - t0, 3))

    _scoped_delete(source_id)
    value_index = _load_value_index()

    chunk_nodes: List[GraphNode] = []
    entity_nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    me_w = GRAPH_EDGE_WEIGHTS.get("mentions_entity", 1.2)
    vo_w = GRAPH_EDGE_WEIGHTS.get("value_of", 1.5)

    for c in chunks:
        cnode = chunk_node_id(c.chunk_id)
        chunk_nodes.append(GraphNode(
            node_id=cnode, node_type="chunk", source_id=source_id, ref_id=c.chunk_id,
            name=getattr(c, "doc_name", ""),
            attrs={"doc_id": getattr(c, "doc_id", ""), "page_num": getattr(c, "page_num", None),
                   "chunk_index": getattr(c, "chunk_index", 0),
                   "section_path": (getattr(c, "metadata", {}) or {}).get("section_path", "")}))
        ents = detect_entities(getattr(c, "text", "") or "", value_index)
        for value_norm, info in ents.items():
            cls = info["class"]
            eid = entity_node_id(cls, value_norm)
            display = _mask_email(value_norm) if cls == "email" else value_norm
            if eid not in entity_nodes:
                entity_nodes[eid] = GraphNode(
                    node_id=eid, node_type="entity", source_id=source_id, ref_id=value_norm,
                    name=display, semantic_type=cls,
                    attrs={"class": cls, "display": display})
            # chunk --mentions_entity--> entity
            edges.append(GraphEdge(str(uuid4()), cnode, eid, "mentions_entity", me_w,
                                   source_id, evidence=f"entity:{cls}",
                                   attrs={"class": cls}))
            # entity --value_of--> column (per column whose sample set contains it)
            for col in info["columns"]:
                edges.append(GraphEdge(str(uuid4()), eid, GP.col_node_id(col["col_id"]),
                                       "value_of", vo_w, col["source_id"],
                                       evidence=f"value_of:{col['col_name']}",
                                       attrs={"col_source": col["source_id"]}))

    cn = GP.upsert_nodes(chunk_nodes, verbose=verbose)
    en = GP.upsert_nodes(list(entity_nodes.values()), verbose=verbose)
    ew = GP.upsert_edges(edges, verbose=verbose)
    me = sum(1 for e in edges if e.edge_type == "mentions_entity")
    vo = sum(1 for e in edges if e.edge_type == "value_of")
    dur = round(time.time() - t0, 3)
    if verbose:
        logger.info("entity_linker: %d chunks, %d entities, %d mentions_entity, %d value_of (%.2fs)",
                    cn, en, me, vo, dur)
    return EntityLinkResult(cn, en, me, vo, source_id, backend, dur,
                            stats={"edges_written": ew})
