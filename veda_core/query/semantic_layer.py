# =============================================================================
# query/semantic_layer.py
# VEDA POC — Query Pipeline Layer 2: Semantic Layer
#
# Responsibility:
#   - Accepts a raw user query string
#   - Encodes it into a query vector using the same strategy as ingestion
#   - Performs cosine similarity search against stored embeddings
#   - Resolves JOIN paths from FK relationships between top-K columns
#   - Returns SemanticLayerResult consumed by L3 (SLM)
#
# Encoding strategy — selected by config.ENCODER_MODE:
#
#   "relgt_only"  — structural keyword projection → 256-dim (POC Run 1)
#   "light_text"  — TF-IDF + SVD using fitted ingestion models → 256-dim (POC Run 2)
#   "hybrid"      — MiniLM(384) + RELGT(256) concat → 640-dim (POC Run 3)
#   "ensemble"    — both light_text AND hybrid encoded independently,
#                   top-K fetched from both stores, merged via RRF (POC Run 4)
#
# FK Bridge Injection (all modes):
#   After cosine search, _inject_bridge_columns() queries the FK adjacency
#   store to find bridge PKs needed to JOIN the retrieved tables together.
#   These are injected into top_k_results before JOIN resolution.
#   Controlled by FK_BRIDGE_INJECTION_ENABLED in config.py.
#
# ENCODER_MODE in config.py is the single switch that keeps ingestion and
# query encoding in sync.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ingestion.value_sampler import expand_query_tokens
from ingestion.vector_store import (
    retrieve_top_k,
    retrieve_top_k_lt,
    retrieve_top_k_hybrid,
    retrieve_temporal_cols_for_tables,
    retrieve_cols_by_name_keywords,
    RetrievalResult,
    get_fk_adjacency,
    get_display_columns,
    FKEdge,
)
from config import (
    ENCODER_MODE,
    TOP_K,
    VECTOR_DIM,
    SEMANTIC_TYPES,
    MONETARY_KEYWORDS,
    METRIC_KEYWORDS,
    IDENTIFIER_SUFFIXES,
    RELGT_HIDDEN_DIM,
    RELGT_OUTPUT_DIM,
    RELGT_NUM_LAYERS,
    RELGT_EMBEDDING_DIM,
    LIGHT_TEXT_SENTENCE_TEMPLATE,
    LIGHT_TEXT_CHAR_SPLIT,
    LIGHT_TEXT_EMBEDDING_DIM,
    MINILM_SENTENCE_TEMPLATE,
    MINILM_DEVICE,
    MINILM_EMBEDDING_DIM,
    HYBRID_EMBEDDING_DIM,
    ENSEMBLE_RRF_K,
    ENSEMBLE_LIGHT_TEXT_WEIGHT,
    FK_BRIDGE_INJECTION_ENABLED,
    FK_MAX_HOP_DEPTH,
    FK_MAX_INJECTED_COLS,
    ENSEMBLE_HYBRID_WEIGHT,
    ENSEMBLE_CANDIDATES_PER_STORE,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Re-entry guard: select_retrieval calls run_semantic_layer for its Step 4
# semantic layer result. Without this flag that creates infinite recursion.
_IN_V2_DISPATCH: bool = False

# Module-level glossary singleton — loaded once per process
_DOMAIN_GLOSSARY: dict = None


def _get_domain_glossary() -> dict:
    global _DOMAIN_GLOSSARY
    if _DOMAIN_GLOSSARY is None:
        try:
            from ingestion.domain_glossary import load_glossary
            _DOMAIN_GLOSSARY = load_glossary()
        except Exception:
            _DOMAIN_GLOSSARY = {}
    return _DOMAIN_GLOSSARY


# =============================================================================
# Output data structures — unchanged across all encoder modes
# =============================================================================

@dataclass
class JoinEdge:
    """A single JOIN hop between two tables."""
    from_table_id:   str
    from_table_name: str
    from_col_id:     str
    from_col_name:   str
    to_table_id:     str
    to_table_name:   str
    to_col_id:       str
    to_col_name:     str
    join_type:       str    # "INNER" | "LEFT"


@dataclass
class SemanticLayerResult:
    """Output of L2 — passed directly to L3 (SLM)."""
    query:              str
    query_vector:       np.ndarray      # primary vector (LT for ensemble)
    top_k_columns:      List[RetrievalResult]
    join_path:          List[JoinEdge]
    tables_involved:    List[str]
    encoding_strategy:  str
    duration_ms:        float
    stats:              dict = field(default_factory=dict)


# =============================================================================
# Shared tokeniser — used by all encoder paths
# Synonym resolution is handled by MiniLM semantic embeddings, not a static map.
# =============================================================================

_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "by",
    "is", "are", "was", "were", "be", "been", "being",
    "show", "get", "find", "list", "give", "tell", "me", "us",
    "all", "any", "how", "many", "much", "what", "which", "where",
    "and", "or", "not", "with", "from", "into", "have", "has",
    "that", "this", "those", "these", "i", "we", "my", "our",
    # Aggregation intent words — these describe what to DO with results,
    # not which columns to retrieve. Stripping them prevents false matches
    # to columns named total_*, count_*, sum_* etc.
    "total", "count", "sum", "average", "avg",
}


def _inject_bridge_columns(
    top_k_results: List[RetrievalResult],
    verbose:       bool = False,
) -> List[RetrievalResult]:
    """
    Injects missing bridge PK columns into top_k_results via FK traversal.

    Returns the enriched result list. Original results are always preserved
    at the front — injected columns are appended at the end.
    """
    if not FK_BRIDGE_INJECTION_ENABLED:
        return top_k_results

    if not top_k_results:
        return top_k_results

    # --- Step 1: collect tables and col_ids already in results ---
    retrieved_table_ids   = list(dict.fromkeys(r.table_id   for r in top_k_results))
    retrieved_table_names = {r.table_id: r.table_name for r in top_k_results}
    retrieved_col_ids     = {r.col_id for r in top_k_results}
    retrieved_id_set      = set(retrieved_table_ids)   # defined here for both branches

    # --- Step 2: query FK adjacency for all edges touching retrieved tables ---
    # Done even for single-table results — needed to inject missing PKs (Step 5)
    fk_edges = get_fk_adjacency(retrieved_table_ids, verbose=verbose)

    if not fk_edges:
        return top_k_results

    if len(retrieved_table_ids) < 2:
        # Single table: skip bridge finding, but still inject missing PKs below
        injections: List[RetrievalResult] = []
        injected_count = 0
        for edge in fk_edges:
            if injected_count >= FK_MAX_INJECTED_COLS:
                break
            if (edge.to_table_id in retrieved_id_set
                    and edge.to_col_id
                    and edge.to_col_id not in retrieved_col_ids):
                injections.append(RetrievalResult(
                    col_id        = edge.to_col_id,
                    col_name      = edge.to_col_name,
                    table_id      = edge.to_table_id,
                    table_name    = edge.to_table_name,
                    semantic_type = "IDENTIFIER",
                    similarity    = 0.0,
                    embedding     = None,
                ))
                retrieved_col_ids.add(edge.to_col_id)
                injected_count += 1
        if verbose and injections:
            print(f"  [FK Bridge] Single-table PK injection: {len(injections)} cols")
            for inj in injections:
                print(f"    + {inj.table_name}.{inj.col_name}")
        return top_k_results + injections

    # --- Step 3: find bridge tables ---
    # A bridge table is one that has FK edges to TWO OR MORE of the retrieved tables
    # but is NOT itself in the retrieved table set.
    # Build: table_id → set of retrieved tables it connects to
    bridge_candidates: Dict[str, dict] = {}  # table_id → {table_name, connected_to: set}

    retrieved_id_set = set(retrieved_table_ids)

    for edge in fk_edges:
        # Check from_table → to_table direction
        from_id   = edge.from_table_id
        from_name = edge.from_table_name
        to_id     = edge.to_table_id
        to_name   = edge.to_table_name

        # If from_table is NOT retrieved but to_table IS retrieved
        # → from_table is a potential bridge
        if from_id not in retrieved_id_set and to_id in retrieved_id_set:
            if from_id not in bridge_candidates:
                bridge_candidates[from_id] = {
                    "table_name":   from_name,
                    "connected_to": set(),
                    "pk_col_id":    "",
                    "pk_col_name":  "",
                }
            bridge_candidates[from_id]["connected_to"].add(to_id)

        # If to_table is NOT retrieved but from_table IS retrieved
        # → to_table is a potential bridge
        if to_id not in retrieved_id_set and from_id in retrieved_id_set:
            if to_id not in bridge_candidates:
                bridge_candidates[to_id] = {
                    "table_name":   to_name,
                    "connected_to": set(),
                    "pk_col_id":    "",
                    "pk_col_name":  "",
                }
            bridge_candidates[to_id]["connected_to"].add(from_id)

    # A bridge table must connect to at least 2 retrieved tables
    bridges = {
        tid: info
        for tid, info in bridge_candidates.items()
        if len(info["connected_to"]) >= 2
    }

    if not bridges:
        # No bridge tables found — but still inject direct FK PKs
        # for tables that ARE in results but whose PKs are missing
        pass

    # --- Step 4: find PK cols for bridge tables from FK edges ---
    # Also collect PKs of retrieved tables that are missing from results
    # (e.g. payment_id when payments table is in results but only paid_amount was retrieved)
    injections: List[RetrievalResult] = []
    injected_count = 0

    # Build: table_id → list of FK edges where it's the "to" (PK) side
    pk_edges_by_table: Dict[str, List[FKEdge]] = {}
    for edge in fk_edges:
        pk_edges_by_table.setdefault(edge.to_table_id, []).append(edge)

    # Inject bridge table PKs
    for bridge_id, bridge_info in bridges.items():
        if injected_count >= FK_MAX_INJECTED_COLS:
            break

        # Find the PK col_id for this bridge table from fk_edges
        pk_col_id   = ""
        pk_col_name = ""
        for edge in fk_edges:
            # The "to" side of a fk_to edge is the PK column
            if edge.to_table_id == bridge_id and edge.to_col_id:
                pk_col_id   = edge.to_col_id
                pk_col_name = edge.to_col_name
                break
            # Also check from side — some schemas have the bridge as FK source
            if edge.from_table_id == bridge_id and edge.from_col_id:
                # Use the from_col if it looks like a PK (_id suffix)
                if edge.from_col_name.endswith("_id"):
                    pk_col_id   = edge.from_col_id
                    pk_col_name = edge.from_col_name
                    break

        if not pk_col_id or pk_col_id in retrieved_col_ids:
            continue

        injections.append(RetrievalResult(
            col_id        = pk_col_id,
            col_name      = pk_col_name,
            table_id      = bridge_id,
            table_name    = bridge_info["table_name"],
            semantic_type = "IDENTIFIER",
            similarity    = 0.0,       # injected — not retrieved by search
            embedding     = None,
        ))
        retrieved_col_ids.add(pk_col_id)
        injected_count += 1

    # --- Step 5: also inject missing PKs for tables already in results ---
    # Uses its own counter — bridge injection (Step 4) must not exhaust the budget
    # before PKs of the primary retrieved tables are injected.
    pk_injected_count = 0
    for edge in fk_edges:
        if pk_injected_count >= FK_MAX_INJECTED_COLS:
            break
        # The PK side (to_col) of edges pointing INTO retrieved tables
        if (edge.to_table_id in retrieved_id_set
                and edge.to_col_id
                and edge.to_col_id not in retrieved_col_ids):
            injections.append(RetrievalResult(
                col_id        = edge.to_col_id,
                col_name      = edge.to_col_name,
                table_id      = edge.to_table_id,
                table_name    = edge.to_table_name,
                semantic_type = "IDENTIFIER",
                similarity    = 0.0,
                embedding     = None,
            ))
            retrieved_col_ids.add(edge.to_col_id)
            pk_injected_count += 1

    if verbose and injections:
        print(f"  [FK Bridge] Injected {len(injections)} bridge columns:")
        for inj in injections:
            print(f"    + {inj.table_name}.{inj.col_name}")

    return top_k_results + injections


# =============================================================================
# JOIN path resolver — shared across all encoder modes, unchanged
# =============================================================================

def _resolve_join_path(
    top_k_results: List[RetrievalResult],
    fk_edges: List = None,
) -> List[JoinEdge]:
    if not top_k_results:
        return []

    table_map: Dict[str, str] = {r.table_name: r.table_id for r in top_k_results}
    tables_in_results = set(table_map.keys())
    join_edges: List[JoinEdge] = []
    seen_pairs: set = set()

    # Prefer real FK adjacency data when available — gives accurate to_col_name.
    if fk_edges:
        for edge in fk_edges:
            if edge.from_table_name in tables_in_results and edge.to_table_name in tables_in_results:
                pair = (table_map.get(edge.from_table_name, ""), table_map.get(edge.to_table_name, ""))
                if pair and pair not in seen_pairs:
                    seen_pairs.add(pair)
                    join_edges.append(JoinEdge(
                        from_table_id   = table_map.get(edge.from_table_name, ""),
                        from_table_name = edge.from_table_name,
                        from_col_id     = edge.from_col_id,
                        from_col_name   = edge.from_col_name,
                        to_table_id     = table_map.get(edge.to_table_name, ""),
                        to_table_name   = edge.to_table_name,
                        to_col_id       = edge.to_col_id or "",
                        to_col_name     = edge.to_col_name,
                        join_type       = "INNER",
                    ))
        return join_edges

    # Fallback heuristic: infer join from _id column name prefix matching.
    for r in top_k_results:
        col_lower = r.col_name.lower()
        if col_lower.endswith("_id") and r.semantic_type == "IDENTIFIER":
            prefix = col_lower.replace("_id", "")
            referenced_table = None
            for tname in tables_in_results:
                if tname != r.table_name:
                    if prefix in tname or tname.startswith(prefix):
                        referenced_table = tname
                        break
            if referenced_table:
                pair = (r.table_id, table_map[referenced_table])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    # Look for the actual PK column of the referenced table in top_k
                    pk_col = next(
                        (rr for rr in top_k_results
                         if rr.table_name == referenced_table
                         and rr.semantic_type == "IDENTIFIER"
                         and (rr.col_name == "id" or rr.col_name.endswith("_id"))),
                        None,
                    )
                    join_edges.append(JoinEdge(
                        from_table_id   = r.table_id,
                        from_table_name = r.table_name,
                        from_col_id     = r.col_id,
                        from_col_name   = r.col_name,
                        to_table_id     = table_map[referenced_table],
                        to_table_name   = referenced_table,
                        to_col_id       = pk_col.col_id if pk_col else "",
                        to_col_name     = pk_col.col_name if pk_col else r.col_name,
                        join_type       = "INNER",
                    ))
    return join_edges

