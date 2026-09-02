"""query/source_evidence.py — the SourceEvidence contract + group-by-source (multi-source routing, Phase 2.3).

The routing coordinator (Phase 3) reasons over EVIDENCE, not opaque scores. This module turns the
raw retrieval output — tabular columns (``ingestion.vector_store.RetrievalResult`` /
``retrieval_engine_phase3.RetrievalResult``, both now source_id-tagged) and document chunks
(``ingestion.chunk_embedder.ChunkRetrievalResult``) — into one ``SourceEvidence`` per source_id.

Design (see docs/multisource_routing/MEMORY.md):
- **One logical interface, group by source_id.** Both columns and chunks already carry source_id;
  we bucket them per source rather than running N independent per-source retrievals.
- **Never compare raw cross-type scores.** Column cosine and chunk cosine live in different
  distributions. Each item keeps its own ``score`` for WITHIN-kind ordering only; the routing
  decision uses ``presence_tier`` (STRONG / WEAK / NONE), computed per evidence KIND against its
  own config-driven floor (``ROUTING_TIER_*`` in config.py — provisional, tune in Phase 6) with
  the source taking the best of its per-kind tiers.
- Pure / duck-typed (getattr on objects OR dict keys), so it is unit-testable with no DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Routing-level presence tiers. Comparison is only ever tier-vs-tier or within one kind — never a
# raw float compared across source types.
TIER_STRONG = "STRONG"
TIER_WEAK = "WEAK"
TIER_NONE = "NONE"


@dataclass
class EvidenceItem:
    kind: str                 # "column" | "chunk"
    ref: str                  # col_id (columns) or chunk_id (chunks)
    name: str                 # col_name (columns) or doc_name (chunks)
    table_name: str = ""      # columns only
    score: float = 0.0        # WITHIN-kind score (cosine sim); never cross-kind comparable
    retrieval_method: str = ""  # "bi-encoder" | "graph" | "chunk" | "inject" (best-effort)


@dataclass
class SourceEvidence:
    source_id: str
    items: List[EvidenceItem] = field(default_factory=list)
    column_count: int = 0
    chunk_count: int = 0
    top_column_score: float = 0.0
    top_chunk_score: float = 0.0
    # Source-level routing PRIOR: top cosine of query ↔ this source's ITEM descriptions
    # (source_item_embeddings). Robust to a big DB's spurious column match — a source whose table/
    # document is semantically ABOUT the query scores here even when its raw columns don't.
    top_item_score: float = 0.0
    # Source-DESCRIPTION prior (SOURCE_DESC_PRIOR_ENABLED, default OFF): top cosine of query ↔ this
    # source's own `sources_source.description`. A single focused source-level summary, immune to the
    # MAX-over-N population bias of top_item_score. Recorded for traceability; when the flag is on the
    # coordinator blends it into top_item_score (max) BEFORE tiering. Default 0.0 → no effect when off.
    top_desc_score: float = 0.0
    # Computed per evidence kind against config-driven floors (Phase 2.4); source takes the best.
    presence_tier: str = TIER_NONE

    def summary(self) -> dict:
        """Compact, JSON-friendly view for the SLM/trace (names only, no embeddings/scores dump)."""
        return {
            "source_id": self.source_id,
            "presence_tier": self.presence_tier,
            "columns": [i.name for i in self.items if i.kind == "column"][:12],
            "tables": sorted({i.table_name for i in self.items
                              if i.kind == "column" and i.table_name})[:12],
            "documents": sorted({i.name for i in self.items if i.kind == "chunk"})[:8],
            "column_count": self.column_count,
            "chunk_count": self.chunk_count,
        }


def _get(obj, key, default=""):
    """Read ``key`` from an object attribute or a dict — the retrieval result shapes vary."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _col_id(c) -> str:
    return str(_get(c, "col_id", "") or "")


def _source_of(obj) -> str:
    return str(_get(obj, "source_id", "") or "")


# ── Presence-tier calibration (Phase 2.4) ─────────────────────────────────────────────────────
# Column (bi-encoder) and chunk (RAG) cosine scores live in different distributions, so each kind
# is tiered against its OWN floor (config, env-overridable, PROVISIONAL — tune in Phase 6). The
# source's tier is the best of its per-kind tiers. Floors are read lazily so config env overrides
# and test monkeypatching both take effect.
_TIER_RANK = {TIER_NONE: 0, TIER_WEAK: 1, TIER_STRONG: 2}


def _floors() -> dict:
    """Per-kind (strong, weak) similarity floors from config, with hard-coded fallbacks so this
    module stays importable even if config is unavailable."""
    try:
        import config as _cfg
        return {
            "column": (float(_cfg.ROUTING_TIER_TABULAR_STRONG), float(_cfg.ROUTING_TIER_TABULAR_WEAK)),
            "chunk": (float(_cfg.ROUTING_TIER_DOCUMENT_STRONG), float(_cfg.ROUTING_TIER_DOCUMENT_WEAK)),
        }
    except Exception:
        return {"column": (0.55, 0.30), "chunk": (0.50, 0.30)}


def _tier_for_kind(top_score: float, count: int, kind: str, floors: dict) -> str:
    """Tier one evidence kind by its own floor. Any present evidence is at least WEAK (a real hit
    below the weak floor still means the source has *something* on-topic); a high hit is STRONG."""
    if not count:
        return TIER_NONE
    strong, _weak = floors[kind]
    return TIER_STRONG if top_score >= strong else TIER_WEAK


def _source_tier(top_column_score: float, top_chunk_score: float,
                 column_count: int, chunk_count: int) -> str:
    floors = _floors()
    col_tier = _tier_for_kind(top_column_score, column_count, "column", floors)
    chunk_tier = _tier_for_kind(top_chunk_score, chunk_count, "chunk", floors)
    return col_tier if _TIER_RANK[col_tier] >= _TIER_RANK[chunk_tier] else chunk_tier


def group_evidence_by_source(columns, chunks) -> Dict[str, SourceEvidence]:
    """Bucket source_id-tagged columns + chunks into one ``SourceEvidence`` per source.

    Columns/chunks with a blank source_id are skipped (they cannot be attributed — the Phase 2.2
    fixes ensure real candidates carry one). Returns ``{source_id: SourceEvidence}``.
    """
    buckets: Dict[str, SourceEvidence] = {}

    for c in columns or []:
        sid = _source_of(c)
        if not sid:
            continue
        ev = buckets.setdefault(sid, SourceEvidence(source_id=sid))
        sim = float(_get(c, "similarity", 0.0) or 0.0)
        ev.items.append(EvidenceItem(
            kind="column", ref=_col_id(c), name=str(_get(c, "col_name", "") or ""),
            table_name=str(_get(c, "table_name", "") or ""), score=sim,
            retrieval_method=str(_get(c, "retrieval_method", "") or ""),
        ))
        ev.column_count += 1
        ev.top_column_score = max(ev.top_column_score, sim)

    for ch in chunks or []:
        sid = _source_of(ch)
        if not sid:
            continue
        ev = buckets.setdefault(sid, SourceEvidence(source_id=sid))
        sim = float(_get(ch, "similarity", 0.0) or 0.0)
        ev.items.append(EvidenceItem(
            kind="chunk", ref=str(_get(ch, "chunk_id", "") or ""),
            name=str(_get(ch, "doc_name", "") or ""), score=sim, retrieval_method="chunk",
        ))
        ev.chunk_count += 1
        ev.top_chunk_score = max(ev.top_chunk_score, sim)

    for ev in buckets.values():
        ev.presence_tier = _source_tier(
            ev.top_column_score, ev.top_chunk_score, ev.column_count, ev.chunk_count)
    return buckets
