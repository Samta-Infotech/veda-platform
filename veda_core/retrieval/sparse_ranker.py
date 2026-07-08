"""retrieval/sparse_ranker.py — learned-sparse Signal 2 (WP3, replaces bm25_ranker).

BGE-M3's learned-sparse (lexical) weights replace BM25. On engine warm we load the
source's persisted sparse rows (``column_sparse_v1`` / ``table_sparse_v1``, written at L4
by ingestion/sparse_index.py) and build an in-memory inverted index
``{token: [(col_key, weight)]}`` — a few thousand columns fit trivially in memory. Per
query we encode the raw query's sparse weights once (reusing the same M3 forward pass as
the dense query encode via ``encode_query``) and score columns by sparse dot product.

Enriched expansion phrases are ALSO scored — each unique phrase is sparse-encoded and its
per-column contribution is MAX-pooled, then added to the raw-query score. This is where
query enrichment now lives (the dense Signal 1 encodes the raw query only, WP1).

Keys are "table.col" — the identical fusion key space BM25 used via retrieval_documents —
so the RRF merger fuses sparse with the subgraph/fk/value signals unchanged.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

from config import COLUMN_SPARSE_TABLE, TABLE_SPARSE_TABLE

logger = logging.getLogger(__name__)


def _as_weight_dict(raw) -> Dict[str, float]:
    """JSONB comes back as a dict from psycopg2; tolerate a JSON string too."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    return {str(k): float(v) for k, v in (raw or {}).items()}


def _build_inverted(weights: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
    inv: Dict[str, List[Tuple[str, float]]] = {}
    for key, w in weights.items():
        for tok, wt in w.items():
            inv.setdefault(tok, []).append((key, wt))
    return inv


class SparseRanker:
    """Learned-sparse column/table ranker (Signal 2)."""

    def __init__(self):
        self.col_weights: Dict[str, Dict[str, float]] = {}     # "table.col" -> {token: w}
        self.table_weights: Dict[str, Dict[str, float]] = {}   # table_id    -> {token: w}
        self._col_inv: Dict[str, List[Tuple[str, float]]] = {}
        self._table_inv: Dict[str, List[Tuple[str, float]]] = {}

    # ── warm-load from the persisted store (primary path) ────────────────────
    def warm_from_store(self, source_ids: List[str]) -> int:
        """Load column_sparse_v1 / table_sparse_v1 for the scope; build inverted indexes.
        Returns the number of columns loaded (0 → caller should fall back to fit())."""
        try:
            from ingestion.db_abstraction import (
                get_internal_connection, release_internal_connection,
            )
        except Exception:
            return 0
        try:
            conn = get_internal_connection()
        except Exception:
            return 0
        try:
            cur = conn.cursor()
            if source_ids:
                ph = ",".join(["%s"] * len(source_ids))
                cur.execute(f"SELECT col_id, weights FROM {COLUMN_SPARSE_TABLE} "
                            f"WHERE source_id IN ({ph})", list(source_ids))
            else:
                cur.execute(f"SELECT col_id, weights FROM {COLUMN_SPARSE_TABLE}")
            self.col_weights = {r[0]: _as_weight_dict(r[1]) for r in cur.fetchall()}
            # Key table sparse weights by table_name so they combine with the dense
            # table prior (which keys by table_name) in the WP4 table_sim map.
            if source_ids:
                ph = ",".join(["%s"] * len(source_ids))
                cur.execute(f"SELECT table_name, weights FROM {TABLE_SPARSE_TABLE} "
                            f"WHERE source_id IN ({ph})", list(source_ids))
            else:
                cur.execute(f"SELECT table_name, weights FROM {TABLE_SPARSE_TABLE}")
            self.table_weights = {r[0]: _as_weight_dict(r[1]) for r in cur.fetchall()}
            try: cur.close()
            except Exception: pass
        except Exception as e:
            logger.warning(f"sparse warm-load failed ({e}); caller falls back to fit()")
            try: conn.rollback()
            except Exception: pass
            return 0
        finally:
            release_internal_connection(conn)

        self._col_inv = _build_inverted(self.col_weights)
        self._table_inv = _build_inverted(self.table_weights)
        logger.info(f"✓ SparseRanker warm-loaded {len(self.col_weights)} cols, "
                    f"{len(self.table_weights)} tables")
        return len(self.col_weights)

    # ── fallback: encode retrieval_documents in-memory (dev / no persisted rows) ─
    def fit(self, semantic_model: Dict) -> "SparseRanker":
        from ingestion import m3_encoder
        rdocs = semantic_model.get("retrieval_documents", {}) or {}
        keys = list(rdocs.keys())
        texts = [rdocs[k] for k in keys]
        weights = m3_encoder.encode_sparse(texts) if texts else []
        self.col_weights = {k: w for k, w in zip(keys, weights)}
        self._col_inv = _build_inverted(self.col_weights)
        # No table passages in the semantic model → table sparse degrades to empty
        # (the WP4 table prior still has the dense half).
        self.table_weights = {}
        self._table_inv = {}
        logger.info(f"✓ SparseRanker fit on {len(self.col_weights)} retrieval documents")
        return self

    # ── scoring ──────────────────────────────────────────────────────────────
    @staticmethod
    def _dot(sparse: Dict[str, float], inv: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for tok, qw in sparse.items():
            for key, cw in inv.get(tok, ()):  # noqa: B905
                scores[key] = scores.get(key, 0.0) + qw * cw
        return scores

    def _combined(self, q_sparse, phrase_sparses, inv) -> Dict[str, float]:
        scores = self._dot(q_sparse, inv)
        if phrase_sparses:
            pooled: Dict[str, float] = {}
            for ps in phrase_sparses:
                for key, v in self._dot(ps, inv).items():
                    if v > pooled.get(key, 0.0):
                        pooled[key] = v
            for key, v in pooled.items():
                scores[key] = scores.get(key, 0.0) + v
        return scores

    def _encode(self, query: str, enriched_tokens):
        from ingestion import m3_encoder
        _, q_sparse = m3_encoder.encode_query(query)
        phrase_sparses = []
        if enriched_tokens:
            ql = query.lower()
            uniq = [t for t in dict.fromkeys(enriched_tokens) if t and t.lower() not in ql]
            if uniq:
                phrase_sparses = m3_encoder.encode_sparse(uniq)
        return q_sparse, phrase_sparses

    def rank(self, query: str, enriched_tokens: List[str] = None,
             top_k: int = 50) -> List[Tuple[str, float]]:
        """Return top_k [(table.col, sparse_score)] — same interface BM25Ranker.rank had."""
        q_sparse, phrase_sparses = self._encode(query, enriched_tokens)
        scores = self._combined(q_sparse, phrase_sparses, self._col_inv)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def table_scores(self, query: str, enriched_tokens: List[str] = None) -> Dict[str, float]:
        """Sparse scores per table_name (for the WP4 table prior). Empty when tables
        weren't warm-loaded (fit() fallback)."""
        if not self._table_inv:
            return {}
        q_sparse, phrase_sparses = self._encode(query, enriched_tokens)
        return self._combined(q_sparse, phrase_sparses, self._table_inv)
