# =============================================================================
# ingestion/cross_source_graph.py
# VEDA — Cross-source join discovery (Cross-source plan, Phase 4.2/4.3)
#
# The tenant-wide stage that makes the graph "know how to connect sources". Runs
# at the END of every ingestion (cheap — sketch comparisons only) over ALL ready
# sources of the tenant:
#
#   1. Load all column_sketches for the tenant.
#   2. Candidate pairs = columns from DIFFERENT sources with compatible value_class
#      and comparable cardinality (ratio within CROSS_SOURCE_CARDINALITY_RATIO).
#   3. MinHash Jaccard estimate per pair; containment estimate for asymmetric
#      FK-like relations (small set ⊂ large set).
#   4. Emit `cross_source_fk` edges (col → col) with HIGH/MEDIUM tiers mirroring
#      discovered_fk. Evidence attrs (jaccard, containment, cardinalities) so a
#      cross-source join is always explainable (the join planner + answer composer
#      surface them).
#   5. Idempotent: scoped delete of the tenant's cross_source_fk edges before
#      re-emit, so re-ingesting any one source re-runs the whole pass.
#
# The discovery core (`discover_edges`) is a pure function over sketch rows — no
# DB, no datasketch import at call sites — so it is unit-testable in isolation.
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (
    COLUMN_SKETCHES_TABLE_NAME,
    CROSS_SOURCE_CARDINALITY_RATIO,
    CROSS_SOURCE_FK_HIGH_CONTAINMENT,
    CROSS_SOURCE_FK_HIGH_MIN_DISTINCT,
    CROSS_SOURCE_FK_MEDIUM_CONTAINMENT,
    CROSS_SOURCE_FK_TIER_WEIGHT,
    GRAPH_EDGES_TABLE,
)
from ingestion.column_sketches import sketch_from_bytes, sketches_available
from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
)
from ingestion import graph_persist as GP
from utils.logger import get_logger

logger = get_logger(__name__)

CROSS_SOURCE_FK_EDGE = "cross_source_fk"


@dataclass
class SketchRow:
    col_id: str
    source_id: str
    tenant: str
    table_name: str
    col_name: str
    n_distinct: int
    value_class: str
    num_perm: int
    sketch: bytes
    _mh: object = field(default=None, repr=False)

    def minhash(self):
        if self._mh is None:
            self._mh = sketch_from_bytes(self.sketch, self.num_perm)
        return self._mh


def estimate_containment(jaccard: float, n_a: int, n_b: int) -> float:
    """Estimate containment of the SMALLER column into the larger from a MinHash
    Jaccard estimate + the distinct counts.

    MinHash gives J = |A∩B| / |A∪B|. With |A∪B| = n_a + n_b − |A∩B|, solving for
    the intersection: |A∩B| ≈ J·(n_a + n_b) / (1 + J). Containment is that
    intersection over the smaller cardinality — the FK-like signal (child ⊂ parent).
    """
    if jaccard <= 0 or n_a <= 0 or n_b <= 0:
        return 0.0
    inter = jaccard * (n_a + n_b) / (1.0 + jaccard)
    return min(1.0, inter / min(n_a, n_b))


def _tier(containment: float, n_small: int) -> Optional[str]:
    if containment >= CROSS_SOURCE_FK_HIGH_CONTAINMENT and n_small >= CROSS_SOURCE_FK_HIGH_MIN_DISTINCT:
        return "HIGH"
    if containment >= CROSS_SOURCE_FK_MEDIUM_CONTAINMENT:
        return "MEDIUM"
    return None


def _cardinality_compatible(n_a: int, n_b: int) -> bool:
    if n_a <= 0 or n_b <= 0:
        return False
    lo, hi = CROSS_SOURCE_CARDINALITY_RATIO
    ratio = n_a / n_b
    return lo <= ratio <= hi


def discover_edges(rows: List[SketchRow], verbose: bool = False) -> List[dict]:
    """Pure discovery core. Returns a list of edge dicts:
    {from_col_id, to_col_id, from_source, to_source, tier, weight, jaccard,
     containment, n_from, n_to, value_class}. The edge points child→parent
    (smaller cardinality → larger), mirroring fk_to directionality. Each unordered
    pair is emitted at most once."""
    edges: List[dict] = []
    n = len(rows)
    for i in range(n):
        a = rows[i]
        for j in range(i + 1, n):
            b = rows[j]
            if str(a.source_id) == str(b.source_id):
                continue                          # cross-source only
            if a.value_class != b.value_class:
                continue                          # compatible class only
            if not _cardinality_compatible(a.n_distinct, b.n_distinct):
                continue
            mha, mhb = a.minhash(), b.minhash()
            if mha is None or mhb is None:
                continue
            jac = float(mha.jaccard(mhb))
            if jac <= 0:
                continue
            containment = estimate_containment(jac, a.n_distinct, b.n_distinct)
            # child = smaller cardinality (the set more likely contained in the other)
            child, parent = (a, b) if a.n_distinct <= b.n_distinct else (b, a)
            tier = _tier(containment, child.n_distinct)
            if tier is None:
                continue
            edges.append({
                "from_col_id": child.col_id, "to_col_id": parent.col_id,
                "from_source": str(child.source_id), "to_source": str(parent.source_id),
                "tier": tier, "weight": CROSS_SOURCE_FK_TIER_WEIGHT[tier],
                "jaccard": round(jac, 4), "containment": round(containment, 4),
                "n_from": child.n_distinct, "n_to": parent.n_distinct,
                "value_class": child.value_class,
            })
            if verbose:
                logger.info("cross_source_fk %s.%s → %s.%s  [%s] jac=%.3f cont=%.3f",
                            child.source_id, child.col_name, parent.source_id,
                            parent.col_name, tier, jac, containment)
    return edges


def _load_sketches(tenant: str, source_ids: Optional[List[str]] = None) -> List[SketchRow]:
    """Read every column_sketch for the tenant (optionally restricted to a source
    set) from the internal store. Empty when the table doesn't exist yet."""
    if not INTERNAL_DB_AVAILABLE:
        return []
    conn = get_internal_connection()
    try:
        with conn.cursor() as cur:
            sql = (f"SELECT col_id, source_id, tenant, table_name, col_name, "
                   f"n_distinct, value_class, num_perm, sketch "
                   f"FROM {COLUMN_SKETCHES_TABLE_NAME} WHERE tenant = %s")
            params: list = [str(tenant)]
            if source_ids:
                sql += " AND source_id = ANY(%s)"
                params.append([str(s) for s in source_ids])
            cur.execute(sql, params)
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("cross_source_graph: could not load sketches (%s)", e)
        return []
    finally:
        release_internal_connection(conn)
    return [SketchRow(col_id=r[0], source_id=str(r[1]), tenant=r[2], table_name=r[3],
                      col_name=r[4], n_distinct=int(r[5]), value_class=r[6],
                      num_perm=int(r[7]), sketch=bytes(r[8])) for r in rows]


def _delete_tenant_edges(tenant_source_ids: List[str]) -> int:
    """Idempotency: drop the tenant's existing cross_source_fk edges before re-emit.
    graph_edges has no tenant column, so scope by the tenant's source id set (every
    cross_source_fk edge stores its child column's source_id)."""
    if not (INTERNAL_DB_AVAILABLE and tenant_source_ids):
        return 0
    conn = get_internal_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {GRAPH_EDGES_TABLE} WHERE edge_type = %s "
                    f"AND source_id = ANY(%s)",
                    [CROSS_SOURCE_FK_EDGE, [str(s) for s in tenant_source_ids]])
                return cur.rowcount or 0
    except Exception as e:
        logger.warning("cross_source_graph: edge cleanup skipped (%s)", e)
        return 0
    finally:
        release_internal_connection(conn)


def _to_graph_edges(edge_dicts: List[dict]) -> List[GP.GraphEdge]:
    import uuid
    out: List[GP.GraphEdge] = []
    for e in edge_dicts:
        out.append(GP.GraphEdge(
            edge_id=str(uuid.uuid4()),
            src_node_id=GP.col_node_id(e["from_col_id"]),
            dst_node_id=GP.col_node_id(e["to_col_id"]),
            edge_type=CROSS_SOURCE_FK_EDGE,
            weight=float(e["weight"]),
            source_id=e["from_source"],
            evidence=f"{e['tier']} jac={e['jaccard']} cont={e['containment']}",
            attrs={k: e[k] for k in ("tier", "jaccard", "containment", "from_source",
                                     "to_source", "n_from", "n_to", "value_class")},
        ))
    return out


def discover_and_persist(tenant: str, source_ids: Optional[List[str]] = None,
                         verbose: bool = False) -> Dict:
    """Tenant-wide L5+ stage: load sketches → discover cross-source joins → emit
    idempotent cross_source_fk edges. Returns a stats dict. Safe no-op when
    datasketch is unavailable or no sketches exist."""
    t0 = time.time()
    if not sketches_available():
        return {"ok": False, "reason": "datasketch unavailable", "edges": 0}
    rows = _load_sketches(tenant, source_ids)
    tenant_source_ids = sorted({r.source_id for r in rows})
    if len({r.source_id for r in rows}) < 2:
        # Nothing to link across yet (single source). Still clear stale edges so a
        # source removal doesn't leave dangling cross links.
        deleted = _delete_tenant_edges(tenant_source_ids)
        return {"ok": True, "edges": 0, "sources": len(tenant_source_ids),
                "deleted": deleted, "duration_sec": round(time.time() - t0, 3)}
    edge_dicts = discover_edges(rows, verbose=verbose)
    deleted = _delete_tenant_edges(tenant_source_ids)
    written = GP.upsert_edges(_to_graph_edges(edge_dicts), verbose=verbose)
    tiers = {"HIGH": 0, "MEDIUM": 0}
    for e in edge_dicts:
        tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
    stats = {"ok": True, "edges": written, "deleted": deleted,
             "sources": len(tenant_source_ids), "tiers": tiers,
             "duration_sec": round(time.time() - t0, 3)}
    logger.info("cross_source_graph: %s", stats)
    return stats
