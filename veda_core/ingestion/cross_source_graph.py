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

import re

from config import (
    COLUMN_SKETCHES_TABLE_NAME,
    CROSS_SOURCE_AFFINITY_REQUIRED_CLASSES,
    CROSS_SOURCE_CARDINALITY_RATIO,
    CROSS_SOURCE_FK_HIGH_CONTAINMENT,
    CROSS_SOURCE_FK_HIGH_MIN_DISTINCT,
    CROSS_SOURCE_FK_MEDIUM_CONTAINMENT,
    CROSS_SOURCE_FK_TIER_WEIGHT,
    CROSS_SOURCE_GENERIC_NAME_TOKENS,
    GRAPH_EDGES_TABLE,
)
from ingestion.column_sketches import (
    sketch_from_bytes, sketches_available, unpack_value_hashes)
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
    value_hashes: bytes = None            # packed sorted uint64, when n_distinct <= cap
    _mh: object = field(default=None, repr=False)
    _vh: object = field(default=False, repr=False)   # False = not yet unpacked

    def minhash(self):
        if self._mh is None:
            self._mh = sketch_from_bytes(self.sketch, self.num_perm)
        return self._mh

    def vhashes(self):
        """numpy uint64 array of exact value hashes, or None when not stored."""
        if self._vh is False:
            self._vh = unpack_value_hashes(self.value_hashes) if self.value_hashes else None
        return self._vh


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


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _meaningful_tokens(*names: str) -> set:
    """Identity-bearing tokens of one or more names (snake/camel split, generic
    join-key tokens like id/name/code/type stripped)."""
    toks: set = set()
    for name in names:
        if not name:
            continue
        camel = _CAMEL_RE.sub(" ", str(name))
        for t in _TOKEN_SPLIT_RE.split(camel.lower()):
            if t and len(t) >= 2 and t not in CROSS_SOURCE_GENERIC_NAME_TOKENS:
                toks.add(t)
    return toks


_DESCRIPTOR_TOKENS = frozenset({"name", "label", "title", "description", "desc"})


def _is_descriptor_col(name: str) -> bool:
    toks = set(_TOKEN_SPLIT_RE.split((name or "").lower()))
    return bool(toks & _DESCRIPTOR_TOKENS)


def _affine_target_ok(target_col: str, other_col: str) -> bool:
    """For the FK/entity → table path, the target column must be the table's PRIMARY key
    `id`, an identically-named FK (same key on both sides), or a descriptor (name/label).
    This stops an `<e>_id` FK matching an unrelated `<other>_id` of the same table
    (`asset_id` → `assets_asset.project_id`) merely because the table carries the entity
    token — only the actual join target (the PK, the twin FK, or the entity name) counts."""
    t = (target_col or "").strip().lower()
    return t == "id" or t == (other_col or "").strip().lower() or _is_descriptor_col(t)


def _name_affinity(a: "SketchRow", b: "SketchRow") -> bool:
    """True when the two columns share join identity. Two ways:
      1. Direct column-token overlap — `city` ↔ `city_name`, `asset_id` ↔ `asset_id`.
      2. FK/entity-to-table — one side's column entity token names the OTHER side's
         table AND that side's column is the join target (PK `id`, the twin FK, or a
         descriptor):  `asset_id` → `assets_asset.id`,  `amenity_name` → `assets_amenity.name`.
    Exact tokens keep the app-prefix apart (`asset` ≠ `assets`), and _affine_target_ok
    keeps it off arbitrary same-table attributes (`assets_asset.project_id`)."""
    a_col = _meaningful_tokens(a.col_name)
    b_col = _meaningful_tokens(b.col_name)
    if a_col & b_col:
        return True
    if (a_col & _meaningful_tokens(b.table_name)) and _affine_target_ok(b.col_name, a.col_name):
        return True
    if (b_col & _meaningful_tokens(a.table_name)) and _affine_target_ok(a.col_name, b.col_name):
        return True
    return False


def _exact_containment(child: "SketchRow", parent: "SketchRow"):
    """Exact (jaccard, containment) from the stored value-hash sets when BOTH sides
    have them, else None. containment = |child ∩ parent| / |child| (child = smaller)."""
    cv, pv = child.vhashes(), parent.vhashes()
    if cv is None or pv is None:
        return None
    import numpy as np
    inter = int(np.intersect1d(cv, pv, assume_unique=True).size)
    nc, npar = int(cv.size), int(pv.size)
    union = nc + npar - inter
    jac = (inter / union) if union else 0.0
    cont = (inter / nc) if nc else 0.0
    return jac, cont


def _tier(containment: float, n_small: int, affine: bool) -> Optional[str]:
    if containment < CROSS_SOURCE_FK_MEDIUM_CONTAINMENT:
        return None
    # A genuine FK is often a small child ⊂ large parent, so the distinct-count floor
    # would keep it out of HIGH forever; a strong name affinity is the corroborating
    # signal that promotes it to the executable tier without that floor.
    if containment >= CROSS_SOURCE_FK_HIGH_CONTAINMENT and (
            n_small >= CROSS_SOURCE_FK_HIGH_MIN_DISTINCT or affine):
        return "HIGH"
    return "MEDIUM"


def _cardinality_compatible(n_a: int, n_b: int, exact: bool) -> bool:
    """Loose sanity ratio for the MinHash path (its containment estimate degrades at
    extreme asymmetry). Bypassed entirely when exact value-hash containment is
    available, since exact set math is correct at any asymmetry — which is precisely
    the small-child ⊂ large-parent FK the MinHash estimate cannot see."""
    if n_a <= 0 or n_b <= 0:
        return False
    if exact:
        return True
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
            # child = smaller cardinality (the set more likely contained in the other)
            child, parent = (a, b) if a.n_distinct <= b.n_distinct else (b, a)
            # Exact containment from stored value-hash sets when available (correct at
            # any asymmetry); else the MinHash Jaccard estimate (resolution-limited).
            exact = _exact_containment(child, parent)
            if exact is not None:
                jac, containment = exact
            else:
                if not _cardinality_compatible(a.n_distinct, b.n_distinct, exact=False):
                    continue
                mha, mhb = a.minhash(), b.minhash()
                if mha is None or mhb is None:
                    continue
                jac = float(mha.jaccard(mhb))
                if jac <= 0:
                    continue
                containment = estimate_containment(jac, a.n_distinct, b.n_distinct)
            if containment <= 0:
                continue
            affine = _name_affinity(child, parent)
            # id/numeric key domains overlap by coincidence; require a name relation so a
            # value match alone (created_by_id vs asset_id) never mints an edge.
            if child.value_class in CROSS_SOURCE_AFFINITY_REQUIRED_CLASSES and not affine:
                continue
            tier = _tier(containment, child.n_distinct, affine)
            if tier is None:
                continue
            edges.append({
                "from_col_id": child.col_id, "to_col_id": parent.col_id,
                "from_source": str(child.source_id), "to_source": str(parent.source_id),
                "tier": tier, "weight": CROSS_SOURCE_FK_TIER_WEIGHT[tier],
                "jaccard": round(jac, 4), "containment": round(containment, 4),
                "n_from": child.n_distinct, "n_to": parent.n_distinct,
                "value_class": child.value_class, "name_affinity": affine,
                "exact": exact is not None,
            })
            if verbose:
                logger.info("cross_source_fk %s.%s → %s.%s  [%s] jac=%.3f cont=%.3f "
                            "affine=%s exact=%s", child.source_id, child.col_name,
                            parent.source_id, parent.col_name, tier, jac, containment,
                            affine, exact is not None)
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
                   f"n_distinct, value_class, num_perm, sketch, value_hashes "
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
                      num_perm=int(r[7]), sketch=bytes(r[8]),
                      value_hashes=bytes(r[9]) if r[9] is not None else None)
            for r in rows]


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
            evidence=f"{e['tier']} jac={e['jaccard']} cont={e['containment']}"
                     f" affine={e.get('name_affinity')} exact={e.get('exact')}",
            attrs={k: e[k] for k in ("tier", "jaccard", "containment", "from_source",
                                     "to_source", "n_from", "n_to", "value_class",
                                     "name_affinity", "exact")},
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
