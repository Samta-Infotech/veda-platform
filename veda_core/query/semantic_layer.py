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


def _tokenise(query: str) -> List[str]:
    text   = query.lower()
    text   = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# =============================================================================
# ENCODER PATH A — RELGT structural projection
# Used when ENCODER_MODE = "relgt_only"
# Seeds MUST match relgt_encoder.py _xavier_init calls exactly.
# =============================================================================

_SEMANTIC_TYPES_LIST = SEMANTIC_TYPES
_DATA_TYPES_LIST = [
    "integer", "varchar", "numeric", "timestamp",
    "boolean", "timestamptz", "date"
]

_MONETARY_SIGNAL_TOKENS = set(MONETARY_KEYWORDS + [
    "earnings", "income", "charges", "fees", "payment", "payments",
    "money", "cost", "costs", "price", "spend", "spending",
])
_TEMPORAL_SIGNAL_TOKENS = {
    "date", "time", "when", "period", "start", "end", "due",
    "created", "updated", "timestamp", "month", "year", "quarter",
    "day", "last", "this", "recent", "latest", "since", "before", "after",
    "past", "week", "ago", "yesterday", "today", "tomorrow",
    "hours", "hour", "days", "weeks", "months", "years",
}
_METRIC_SIGNAL_TOKENS = set(METRIC_KEYWORDS + [
    "size", "area", "floor", "duration", "count", "total", "average",
    "sum", "max", "min", "number", "quantity",
])
_CATEGORY_SIGNAL_TOKENS = {
    "type", "kind", "category", "state", "condition", "status",
    "active", "inactive", "pending", "approved", "rejected",
}
_IDENTIFIER_SIGNAL_TOKENS = {
    "id", "uuid", "key", "tenant", "owner", "unit", "lease",
    "agent", "project", "invoice", "payment", "request",
}


def _build_semantic_type_scores(tokens: List[str]) -> np.ndarray:
    scores  = np.zeros(len(_SEMANTIC_TYPES_LIST), dtype=np.float32)
    sig_map = {
        "MONETARY":   _MONETARY_SIGNAL_TOKENS,
        "TEMPORAL":   _TEMPORAL_SIGNAL_TOKENS,
        "CATEGORY":   _CATEGORY_SIGNAL_TOKENS,
        "IDENTIFIER": _IDENTIFIER_SIGNAL_TOKENS,
        "METRIC":     _METRIC_SIGNAL_TOKENS,
        "FREE_TEXT":  set(),
    }
    for token in tokens:
        for i, stype in enumerate(_SEMANTIC_TYPES_LIST):
            if token in sig_map[stype]:
                scores[i] += 1.0
    exp_s = np.exp(scores - scores.max())
    total = exp_s.sum()
    return exp_s / total if total > 0 else scores


def _build_data_type_scores(tokens: List[str]) -> np.ndarray:
    scores = np.zeros(len(_DATA_TYPES_LIST), dtype=np.float32)
    dt_signals = {
        "integer":   {"id", "count", "number", "floor", "quantity"},
        "varchar":   {"name", "type", "status", "category", "city", "address"},
        "numeric":   _MONETARY_SIGNAL_TOKENS | _METRIC_SIGNAL_TOKENS,
        "timestamp": _TEMPORAL_SIGNAL_TOKENS,
        "boolean":   {"active", "inactive", "is", "has", "enabled", "disabled"},
    }
    for token in tokens:
        for i, dtype in enumerate(_DATA_TYPES_LIST):
            if token in dt_signals.get(dtype, set()):
                scores[i] += 1.0
    total = scores.sum()
    return scores / total if total > 0 else scores


def _build_query_feature_vector(tokens: List[str]) -> np.ndarray:
    return np.concatenate([
        _build_semantic_type_scores(tokens),
        _build_data_type_scores(tokens),
        np.array([0.0],  dtype=np.float32),
        np.array([0.5],  dtype=np.float32),
        np.array([0.5],  dtype=np.float32),
        np.array([0.65], dtype=np.float32),
    ])


def _xavier_init(fan_in: int, fan_out: int, seed: int) -> np.ndarray:
    """Must match relgt_encoder._xavier_init exactly — same seeds."""
    rng   = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def _get_col_feat_dim() -> int:
    """Returns the column feature dim from the saved graph (49 with name hash, 17 without)."""
    try:
        from ingestion.graph_store import graph_available, load_graph
        if graph_available():
            _graph, _ = load_graph()
            if _graph is not None:
                return _graph.column_feature_matrix.shape[1]
    except Exception:
        pass
    return 17


def _encode_relgt_only(tokens: List[str]) -> np.ndarray:
    base_vec     = _build_query_feature_vector(tokens)   # always 17-dim
    col_feat_dim = _get_col_feat_dim()
    if col_feat_dim > base_vec.shape[0]:
        pad      = np.zeros(col_feat_dim - base_vec.shape[0], dtype=np.float32)
        feat_vec = np.concatenate([base_vec, pad])
    else:
        feat_vec = base_vec
    W_in          = _xavier_init(col_feat_dim,     RELGT_HIDDEN_DIM, seed=42)
    W_out         = _xavier_init(RELGT_HIDDEN_DIM, RELGT_OUTPUT_DIM, seed=99)
    layer_weights = [_xavier_init(RELGT_HIDDEN_DIM, RELGT_HIDDEN_DIM, seed=100 + i)
                     for i in range(RELGT_NUM_LAYERS)]
    h = _relu(feat_vec @ W_in)
    for i in range(RELGT_NUM_LAYERS):
        h_new = _relu(h @ layer_weights[i]) + h
        mean  = h_new.mean()
        std   = h_new.std() + 1e-6
        h     = (h_new - mean) / std
    return _l2_normalise(h @ W_out).astype(np.float32)


# =============================================================================
# ENCODER PATH B — Light Text TF-IDF + SVD
# Used when ENCODER_MODE = "light_text" or as sub-encoder in "ensemble"
# =============================================================================

def _split_identifier(name: str) -> str:
    tokens = name.replace("_", " ")
    tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", tokens)
    return tokens.lower().strip()


def _build_query_sentence(query_tokens: List[str]) -> str:
    pseudo_col = " ".join(query_tokens)
    sem_scores = _build_semantic_type_scores(query_tokens)
    top_sem    = _SEMANTIC_TYPES_LIST[int(np.argmax(sem_scores))].lower()
    return LIGHT_TEXT_SENTENCE_TEMPLATE.format(
        col_name      = pseudo_col,
        table_name    = "schema",
        semantic_type = top_sem,
    )


def _encode_light_text(tokens: List[str]) -> np.ndarray:
    """Encodes tokens using the fitted TF-IDF + SVD from ingestion."""
    from ingestion.relgt_encoder import get_light_text_models
    try:
        from sklearn.preprocessing import normalize as sklearn_normalize
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False

    vectorizer, svd = get_light_text_models()
    if vectorizer is None or svd is None:
        raise RuntimeError(
            "Light text models not fitted. Run ingestion before querying."
        )

    sentence  = _build_query_sentence(tokens)
    tfidf_vec = vectorizer.transform([sentence])
    reduced   = svd.transform(tfidf_vec)

    if reduced.shape[1] < LIGHT_TEXT_EMBEDDING_DIM:
        pad     = np.zeros((1, LIGHT_TEXT_EMBEDDING_DIM - reduced.shape[1]), dtype=np.float32)
        reduced = np.hstack([reduced, pad])

    if sklearn_ok:
        from sklearn.preprocessing import normalize as sklearn_normalize
        reduced = sklearn_normalize(reduced, norm="l2")

    vec  = reduced[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# =============================================================================
# ENCODER PATH C — Hybrid: MiniLM(384) + RELGT(256) → 640
# Used when ENCODER_MODE = "hybrid" or as sub-encoder in "ensemble"
# =============================================================================

def _encode_hybrid(
    raw_query:      str,
    mapped_tokens:  List[str],
    top_k_col_ids:  Optional[List[str]] = None,
) -> np.ndarray:
    """
    MiniLM encodes the raw query → 384-dim.
    Structural component (256-dim):
      - If top_k_col_ids provided and graph available: subgraph RELGT on those
        columns, mean-pooled to a single vector (contextual structural signal).
      - Otherwise: token projection fallback (same Xavier-init GNN, no graph).
    Concat + L2 normalise → 640-dim.
    """
    from ingestion.relgt_encoder import get_minilm_model
    model = get_minilm_model()
    if model is None:
        raise RuntimeError(
            "MiniLM model not loaded. Run ingestion before querying."
        )
    minilm_vec = model.encode(
        [raw_query],
        normalize_embeddings = True,
        device               = MINILM_DEVICE,
        show_progress_bar    = False,
    )[0].astype(np.float32)

    relgt_vec = None
    if top_k_col_ids:
        try:
            from ingestion.graph_store import graph_available, load_graph
            from ingestion.kuzu_store import get_subgraph_for_cols
            from ingestion.relgt_encoder import _run_numpy_encoder
            if graph_available():
                _graph, _col_idx = load_graph()
                if _graph is not None and _col_idx is not None:
                    _sub = get_subgraph_for_cols(top_k_col_ids, _graph, _col_idx)
                    if _sub is not None and len(_sub.column_nodes) > 0:
                        _out      = _run_numpy_encoder(_sub)
                        relgt_vec = _out.mean(axis=0).astype(np.float32)
                        _norm     = np.linalg.norm(relgt_vec)
                        if _norm > 0:
                            relgt_vec = relgt_vec / _norm
        except Exception as _e:
            logger.debug("Subgraph encoding failed (%s) — fallback", _e)

    if relgt_vec is None:
        relgt_vec = _encode_relgt_only(mapped_tokens)

    combined = np.concatenate([minilm_vec, relgt_vec])
    return _l2_normalise(combined).astype(np.float32)


def _encode_hybrid_bge(
    raw_query:     str,
    mapped_tokens: List[str],
    top_k_col_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """BGE(1024) + RELGT(256) → 1280-dim. Used in RETRIEVAL_V2 path."""
    from config import RELGT_IN_HYBRID

    bge_vec = None
    try:
        from ingestion.biencoder import _get_biencoder
        from config import BIENCODER_QUERY_PREFIX
        model = _get_biencoder()
        if model is not None:
            bge_vec = model.encode(
                [BIENCODER_QUERY_PREFIX + raw_query],
                normalize_embeddings=True,
            )[0].astype(np.float32)
    except Exception as e:
        logger.debug("BGE encode failed (%s) — fallback to MiniLM hybrid", e)

    if bge_vec is None:
        return _encode_hybrid(raw_query, mapped_tokens, top_k_col_ids)

    # RELGT structural vector
    if RELGT_IN_HYBRID:
        relgt_vec = _get_relgt_structural_vec(mapped_tokens, top_k_col_ids)
    else:
        relgt_vec = np.zeros(RELGT_OUTPUT_DIM, dtype=np.float32)

    return _l2_normalise(np.concatenate([bge_vec, relgt_vec]))


def _get_relgt_structural_vec(
    mapped_tokens:  List[str],
    top_k_col_ids:  Optional[List[str]] = None,
) -> np.ndarray:
    """Subgraph RELGT if available, else token projection fallback."""
    if top_k_col_ids:
        try:
            from ingestion.graph_store import load_graph, graph_available
            from ingestion.kuzu_store import get_subgraph_for_cols
            if graph_available():
                graph, col_id_to_idx = load_graph()
                if graph is not None:
                    subgraph = get_subgraph_for_cols(
                        top_k_col_ids, graph, col_id_to_idx)
                    if subgraph is not None and len(subgraph.column_nodes) > 0:
                        from ingestion.relgt_encoder import _run_numpy_encoder
                        out = _run_numpy_encoder(subgraph)
                        return out.mean(axis=0).astype(np.float32)
        except Exception as e:
            logger.debug("Subgraph RELGT failed (%s) — token projection", e)
    return _encode_relgt_only(mapped_tokens)


# =============================================================================
# ENCODER PATH E — Ensemble: dual encoding + RRF fusion
#
# At query time:
#   1. Encode query with light_text → 256-dim vector (lt_vec)
#   2. Encode query with hybrid → 640-dim vector (h_vec)
#   3. Retrieve top-ENSEMBLE_CANDIDATES_PER_STORE from light_text store using lt_vec
#   4. Retrieve top-ENSEMBLE_CANDIDATES_PER_STORE from hybrid store using h_vec
#   5. Apply Reciprocal Rank Fusion (RRF) to merge both ranked lists
#   6. Return the top-K columns from the fused ranked list
#
# RRF formula:
#   score(col) = w_lt / (rank_lt + K) + w_hybrid / (rank_hybrid + K)
#   where rank is 1-based (∞ = ENSEMBLE_CANDIDATES_PER_STORE + 1 if absent)
#   K = ENSEMBLE_RRF_K (default 60)
#   w_lt, w_hybrid = configurable weights (default 1.0 each)
#
# Why RRF over score normalisation:
#   - Cosine scores from 256-dim and 640-dim spaces are not comparable
#   - RRF uses only rank positions — rank 1 in either list is equally valuable
#   - Robust to one encoder dominating by raw score magnitude
#   - Empirically outperforms score fusion in multi-source IR (Cormack 2009)
# =============================================================================

def _rrf_merge(
    lt_results:     List[RetrievalResult],
    hybrid_results: List[RetrievalResult],
    top_k:          int,
    rrf_k:          int   = ENSEMBLE_RRF_K,
    w_lt:           float = ENSEMBLE_LIGHT_TEXT_WEIGHT,
    w_hybrid:       float = ENSEMBLE_HYBRID_WEIGHT,
    n_candidates:   int   = ENSEMBLE_CANDIDATES_PER_STORE,
) -> List[RetrievalResult]:
    """
    Merges two ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    lt_results     : ranked list from light_text store
    hybrid_results : ranked list from hybrid store
    top_k          : number of results to return after fusion
    rrf_k          : smoothing constant (default 60)
    w_lt           : weight for light_text ranks
    w_hybrid       : weight for hybrid ranks
    n_candidates   : total candidates per store (used as ∞ rank for absent cols)

    Returns
    -------
    List[RetrievalResult]
        Fused list ordered by RRF score descending, length = top_k.
        similarity field is replaced with the RRF score for transparency.
    """
    # Absent rank — columns not in a list get this penalty rank
    absent_rank = n_candidates + 1

    # Build col_id → rank mapping for each list (1-based)
    lt_rank     = {r.col_id: i + 1 for i, r in enumerate(lt_results)}
    hybrid_rank = {r.col_id: i + 1 for i, r in enumerate(hybrid_results)}

    # Collect all unique col_ids across both lists
    all_col_ids = list(dict.fromkeys(
        [r.col_id for r in lt_results] +
        [r.col_id for r in hybrid_results]
    ))

    # Build lookup: col_id → RetrievalResult (prefer lt for metadata)
    col_metadata: Dict[str, RetrievalResult] = {}
    for r in hybrid_results:
        col_metadata[r.col_id] = r
    for r in lt_results:                     # lt overwrites — preferred source
        col_metadata[r.col_id] = r

    # Compute RRF score per column
    scored = []
    for col_id in all_col_ids:
        rank_lt  = lt_rank.get(col_id,     absent_rank)
        rank_h   = hybrid_rank.get(col_id, absent_rank)
        rrf_score = (w_lt / (rank_lt + rrf_k)) + (w_hybrid / (rank_h + rrf_k))
        scored.append((rrf_score, col_id))

    # Sort descending by RRF score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build final result list — replace similarity with RRF score
    results = []
    for rrf_score, col_id in scored[:top_k]:
        if col_id not in col_metadata:
            continue
        r = col_metadata[col_id]
        results.append(RetrievalResult(
            col_id        = r.col_id,
            col_name      = r.col_name,
            table_id      = r.table_id,
            table_name    = r.table_name,
            semantic_type = r.semantic_type,
            similarity    = round(rrf_score, 6),   # RRF score in similarity field
            source_id     = r.source_id,
            embedding     = r.embedding,
        ))

    return results


def _encode_ensemble(
    raw_query:        str,
    mapped_tokens:    List[str],
    top_k:            int,
    verbose:          bool = False,
) -> Tuple[List[RetrievalResult], np.ndarray, dict]:
    """
    Dual encoding + dual retrieval + RRF merge.

    Returns (fused_top_k_results, lt_query_vector, ensemble_stats).
    lt_query_vector is stored in SemanticLayerResult.query_vector for
    consistency — it's the primary (cheaper) encoding.
    """
    # --- Step 1: encode with light_text → 256-dim ---
    lt_vec = _encode_light_text(mapped_tokens)         # (256,)

    # --- Step 2: Pass-1 retrieval — fast TF-IDF results feed the subgraph ---
    # These results are also reused as lt_results in the final RRF merge so
    # there is no extra DB round-trip.
    lt_results    = retrieve_top_k_lt(lt_vec, top_k=ENSEMBLE_CANDIDATES_PER_STORE)
    pass1_col_ids = [r.col_id for r in lt_results]

    # --- Step 3: encode with hybrid → 640-dim ---
    # pass1_col_ids ground the subgraph in actual retrieval results: the GNN
    # runs on the neighbourhood of retrieved columns rather than a fixed token
    # projection, giving a more contextual structural signal.
    # Pass cleaned tokens (stopwords + aggregation words stripped) as the
    # MiniLM sentence so that words like "total" don't pull similarity toward
    # columns named total_*. MiniLM still gets a meaningful entity string.
    search_query   = " ".join(mapped_tokens) if mapped_tokens else raw_query
    h_vec          = _encode_hybrid(search_query, mapped_tokens,
                                    top_k_col_ids=pass1_col_ids)   # (640,)

    # --- Step 4: retrieve from hybrid store ---
    hybrid_results = retrieve_top_k_hybrid(h_vec, top_k=ENSEMBLE_CANDIDATES_PER_STORE)

    if verbose:
        print(f"  LT candidates    : {len(lt_results)}")
        print(f"  Hybrid candidates: {len(hybrid_results)}")

    # --- Step 5: RRF merge → top_k ---
    fused = _rrf_merge(
        lt_results     = lt_results,
        hybrid_results = hybrid_results,
        top_k          = top_k,
    )

    # --- Stats for evaluation ---
    stats = {
        "lt_candidates":     len(lt_results),
        "hybrid_candidates": len(hybrid_results),
        "fused_count":       len(fused),
        "rrf_k":             ENSEMBLE_RRF_K,
        "w_lt":              ENSEMBLE_LIGHT_TEXT_WEIGHT,
        "w_hybrid":          ENSEMBLE_HYBRID_WEIGHT,
        # Track which encoder contributed each top result
        "lt_col_ids":     {r.col_id for r in lt_results[:top_k]},
        "hybrid_col_ids": {r.col_id for r in hybrid_results[:top_k]},
        "both_col_ids":   {r.col_id for r in lt_results[:top_k]}
                         & {r.col_id for r in hybrid_results[:top_k]},
    }

    return fused, lt_vec, stats



# =============================================================================
# FK Bridge Column Injector
#
# After cosine search returns top-K semantic columns, this step asks:
# "Given the tables we found, what PKs and bridge tables are needed
#  to actually JOIN them together?"
#
# Example:
#   Query: "total rent collected per project"
#   Cosine search finds: lease_transactions.rent, projects.project_id
#   Missing bridge:      units.unit_id  (needed: lease_transactions → units → projects)
#   Injected:            units.unit_id added to top_k_results
#
# Algorithm:
#   1. Collect unique table_ids from top_k_results
#   2. Query fk_adjacency for all FK edges touching those tables
#   3. Find tables that bridge two or more retrieved tables (hop depth ≤ FK_MAX_HOP_DEPTH)
#   4. For each bridge table not yet in results, find its PK col in the embedding store
#   5. Inject up to FK_MAX_INJECTED_COLS bridge columns into the result list
#      with similarity = 0.0 (injected, not retrieved by search)
#
# Controlled by FK_BRIDGE_INJECTION_ENABLED in config.py.
# Works identically for all four ENCODER_MODEs.
# =============================================================================

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


# =============================================================================
# Public entry point
# =============================================================================

# =============================================================================
# Domain synonym expansion map
# Maps query terms that differ structurally from schema column names.
# Defined at module level so it's built once, not on every query call.
# Extend this for your specific domain — MiniLM handles most semantic
# synonyms, these cover structural mismatches only.
# =============================================================================
_DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    # ── Generic structural synonyms ───────────────────────────────────────
    # These are schema-naming conventions, not domain-specific.
    # They work across any tenant schema.
    "number":         ["no"],
    "names":          ["name"],
    "modules":        ["module"],
    "entries":        ["id", "log"],
    "records":        ["id"],
    "items":          ["id"],

    # ── AML / Compliance domain synonyms ─────────────────────────────────
    # Maps analyst vocabulary to schema table/column concepts.
    # Extend this section for new tenant domains.
    "case":           ["incident"],
    "cases":          ["incident"],
    "alert":          ["incident"],
    "alerts":         ["incident"],
    "investigation":  ["incident", "investigation"],
    "investigations": ["incident", "investigation"],
    "analyst":        ["assigned", "user"],
    "escalated":      ["incident", "status"],
    "flagged":        ["incident", "transaction"],
    "trail":          ["log", "audit"],
    "incidents":      ["incident"],

    # ── RFI (Request for Information) synonyms ────────────────────────────
    # 'requests for information', 'RFI', 'rfi' all map to rfi_objects.
    # These are multi-word phrases — the tokeniser splits them, so we
    # cover each token individually.
    "rfi":            ["rfi_objects", "rfi"],
    "request":        ["rfi_objects", "rfi"],
    "requests":       ["rfi_objects", "rfi"],
    "information":    ["rfi_objects", "rfi"],
    "review":         ["workflow_state", "status"],
}


def run_semantic_layer(
    query:      str,
    top_k:      int            = TOP_K,
    verbose:    bool           = False,
    source_ids: Optional[List[str]] = None,
) -> SemanticLayerResult:
    """
    Main entry point for Query Pipeline Layer 2.

    Routes to the correct encoding + retrieval path based on ENCODER_MODE.
    Output contract (SemanticLayerResult) is identical for all modes.

    Parameters
    ----------
    query         : str  — Raw user query.
    top_k         : int  — Number of top columns to return.
    verbose       : bool — Print per-step progress.
    """
    # RETRIEVAL_V2 path — delegates to schema_linker + bi-encoder + reranker
    # pipeline via retrieval_select.py.
    # _IN_V2_DISPATCH guard prevents infinite recursion: select_retrieval calls
    # run_semantic_layer for its own Step 4, which must use the legacy path.
    global _IN_V2_DISPATCH
    try:
        from config import RETRIEVAL_V2_ENABLED
    except ImportError:
        RETRIEVAL_V2_ENABLED = False

    if RETRIEVAL_V2_ENABLED and not _IN_V2_DISPATCH:
        _IN_V2_DISPATCH = True
        try:
            from query.retrieval_select import select_retrieval
            _sel = select_retrieval(
                query      = query,
                source_ids = source_ids,
                intent     = "sql",
                verbose    = verbose,
            )
            result = _sel.semantic_layer_result
            if _sel.columns:
                result.top_k_columns   = list(_sel.columns)
                result.tables_involved = list(dict.fromkeys(r.table_name for r in _sel.columns))
            return result
        except Exception as _e:
            logger.warning("RETRIEVAL_V2 failed (%s) — falling back to legacy retrieval", _e)
        finally:
            _IN_V2_DISPATCH = False

    t0 = time.time()

    logger.debug("L2 semantic: query=%r, top_k=%d, mode=%s", query[:120], top_k, ENCODER_MODE)

    if verbose:
        print(f"[SemanticLayer] Query       : '{query}'")
        print(f"  Encoder mode             : {ENCODER_MODE}")

    # ------------------------------------------------------------------
    # Step 1 — Tokenise
    # ------------------------------------------------------------------
    tokens = _tokenise(query)
    if verbose:
        print(f"  Raw tokens               : {tokens}")

    # ------------------------------------------------------------------
    # Step 1b — Value-based token expansion
    # Checks query tokens against the value sampler index.
    # 'escalated' → injects 'workflow_state' token before encoding.
    # No-op if value_sampler store is empty.
    # ------------------------------------------------------------------
    tokens, value_expansion_map = expand_query_tokens(
        tokens, verbose=verbose, full_query=query
    )

    # ------------------------------------------------------------------
    # Step 1c — Domain glossary expansion
    # ------------------------------------------------------------------
    try:
        from ingestion.domain_glossary import expand_query_with_glossary
        _glossary = _get_domain_glossary()
        if _glossary:
            _glossary_tokens = expand_query_with_glossary(tokens, _glossary)
            if _glossary_tokens:
                tokens = list(dict.fromkeys(tokens + _glossary_tokens))
                logger.debug("Glossary expansion: +%d tokens", len(_glossary_tokens))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 2 — Lightweight domain synonym expansion
    # ------------------------------------------------------------------
    expanded: List[str] = []
    synonym_hits: List[str] = []
    for tok in tokens:
        expanded.append(tok)
        extra = _DOMAIN_SYNONYMS.get(tok)
        if extra:
            for e in extra:
                if e not in expanded:
                    expanded.append(e)
            synonym_hits.append(tok)
    mapped_tokens = expanded

    # ------------------------------------------------------------------
    # Step 3 — Encode + Retrieve
    # ------------------------------------------------------------------
    ensemble_stats = {}

    # Backstop: the hybrid/ensemble encoders need MiniLM. When BGE (RETRIEVAL_V2) is the
    # primary path, MiniLM is intentionally not loaded — so degrade to an empty result
    # instead of raising deep in _encode_hybrid. Any caller (not just retrieval_select)
    # then proceeds without this legacy signal rather than crashing.
    if ENCODER_MODE in ("hybrid", "ensemble"):
        try:
            from ingestion.relgt_encoder import get_minilm_model
            _minilm_ok = get_minilm_model() is not None
        except Exception:
            _minilm_ok = False
        if not _minilm_ok:
            logger.warning("MiniLM unavailable under ENCODER_MODE=%s — returning empty "
                           "semantic-layer result (BGE is primary)", ENCODER_MODE)
            return SemanticLayerResult(
                query=query, query_vector=np.zeros(0, dtype=np.float32),
                top_k_columns=[], join_path=[], tables_involved=[],
                encoding_strategy="unavailable_minilm", duration_ms=0.0, stats={})

    if ENCODER_MODE == "relgt_only":
        query_vector      = _encode_relgt_only(mapped_tokens)
        top_k_results     = retrieve_top_k(query_vector, top_k=top_k)
        encoding_strategy = "relgt_structural_keyword_projection_256dim"

    elif ENCODER_MODE == "light_text":
        query_vector      = _encode_light_text(mapped_tokens)
        top_k_results     = retrieve_top_k(query_vector, top_k=top_k)
        encoding_strategy = "light_text_tfidf_svd_256dim"

    elif ENCODER_MODE == "hybrid":
        # Pass original query to MiniLM — it needs full natural language intent.
        # mapped_tokens used only for the RELGT structural projection component.
        query_vector      = _encode_hybrid(query, mapped_tokens)
        top_k_results     = retrieve_top_k(query_vector, top_k=top_k)
        encoding_strategy = "hybrid_minilm384_relgt256_concat640dim"

    elif ENCODER_MODE == "ensemble":
        top_k_results, query_vector, ensemble_stats = _encode_ensemble(
            raw_query        = query,
            mapped_tokens    = mapped_tokens,
            top_k            = top_k,
            verbose          = verbose,
        )
        encoding_strategy = "ensemble_light_text_plus_hybrid_rrf"

    else:
        raise ValueError(f"Unknown ENCODER_MODE: {ENCODER_MODE}")

    # Shape check for single-encoder modes only
    if ENCODER_MODE != "ensemble":
        assert query_vector.shape == (VECTOR_DIM,), (
            f"Query vector shape mismatch: expected ({VECTOR_DIM},), "
            f"got {query_vector.shape}"
        )

    if verbose:
        print(f"  Query vector dim         : {query_vector.shape[0]}")
        print(f"  Encoding strategy        : {encoding_strategy}")

    # ------------------------------------------------------------------
    # Step 3b — RRF score normalisation (ensemble mode only)
    #
    # RRF scores live in [~0.016, ~0.033] — a narrow range that looks
    # like near-zero confidence to L3. The SLM was trained on cosine
    # similarity scores in [0.0, 1.0]. Passing raw RRF scores corrupts
    # the model's internal confidence calibration, causing flat 0.5
    # confidence on all queries regardless of actual column relevance.
    #
    # Fix: min-max rescale to [0.1, 1.0] so the model sees meaningful
    # gradients. Only applied to non-injected columns (similarity > 0.0).
    # Applies to ensemble mode; cosine similarities are already in [0,1].
    # ------------------------------------------------------------------
    if ENCODER_MODE == "ensemble" and top_k_results:
        _scored = [r for r in top_k_results if r.similarity > 0.0]
        if _scored:
            _min_s = min(r.similarity for r in _scored)
            _max_s = max(r.similarity for r in _scored)
            _range = _max_s - _min_s
            for r in top_k_results:
                if r.similarity > 0.0:
                    # Rescale to [0.1, 1.0] preserving relative ordering
                    r.similarity = round(
                        0.1 + 0.9 * (r.similarity - _min_s) / (_range + 1e-9),
                        6,
                    )

    # ------------------------------------------------------------------
    # Step 3c — Source-ID filter
    #
    # The vector stores contain embeddings from ALL ingested sources mixed
    # together. The router has already decided which sources are relevant
    # for this query (e.g. ['primary_db'] for SQL, or ['primary_db','dmt']
    # for hybrid). Filter here so downstream steps never see columns from
    # sources that weren't selected — this stops analytics_lake CSV columns
    # (e.g. permissions_list_export) from polluting primary_db SQL queries.
    # ------------------------------------------------------------------
    if source_ids:
        top_k_results = [
            r for r in top_k_results
            if not r.source_id or r.source_id in source_ids
        ]

    # ------------------------------------------------------------------
    # Step 4 — FK bridge injection
    # Injects missing bridge PKs before JOIN resolution
    # ------------------------------------------------------------------
    top_k_results = _inject_bridge_columns(top_k_results, verbose=verbose)

    # ------------------------------------------------------------------
    # Step 4a — Keyword name-match injection
    #
    # Vector similarity can fail to retrieve columns whose names are exact
    # keyword matches in the query (e.g. 'workflow state' → workflow_state).
    # This step guarantees any column whose every name-part appears as a
    # query keyword is always present in top_k_results, regardless of rank.
    # ------------------------------------------------------------------
    _kw_stopwords = frozenset({
        "show", "list", "get", "find", "give", "me", "all", "the", "a", "an",
        "and", "or", "for", "of", "in", "on", "at", "to", "with", "by",
        "from", "that", "their", "each", "per", "where", "which", "is", "are",
        "was", "were", "has", "have", "its", "user", "users",
    })
    _kw_tokens = [
        w for w in re.sub(r"[^\w]", " ", query.lower()).split()
        if w not in _kw_stopwords and len(w) > 2
    ]
    # Value-vs-Column arbitration (retrieval side, OFF by default): drop tokens grounded
    # as categorical VALUES from the keyword name-match set, so a value word cannot inject
    # a same-named column ("open" → no open_date; "active" → no is_active). Data-driven
    # (column_values EXACT match). Any failure → no exclusion (current behaviour preserved).
    try:
        from config import VALUE_ARBITER_RETRIEVAL_FILTER as _VARF
    except Exception:
        _VARF = False
    if _VARF and _kw_tokens:
        try:
            from query.value_arbiter import arbitrate, column_values_typed_lookup
            from config import DB_CONFIG as _DBC

            def _arb_conn():
                import psycopg2
                return psycopg2.connect(
                    host=_DBC["host"], port=_DBC["port"], dbname=_DBC["database"],
                    user=_DBC["user"], password=_DBC["password"])

            _arb_res = arbitrate(query, column_values_typed_lookup(_arb_conn))
            _value_toks = _arb_res.value_tokens
            if _value_toks:
                _dropped = [w for w in _kw_tokens if w in _value_toks]
                _kw_tokens = [w for w in _kw_tokens if w not in _value_toks]
                if verbose and _dropped:
                    print(f"  [Arbiter] excluded value tokens from KW-inject: {_dropped}")
        except Exception as _e:
            logger.debug("value-arbiter retrieval filter skipped (%s)", _e)
    if _kw_tokens:
        _kw_injected = retrieve_cols_by_name_keywords(_kw_tokens)
        _existing_kw_ids = {r.col_id for r in top_k_results}
        _new_kw = [
            r for r in _kw_injected
            if r.col_id not in _existing_kw_ids
            and (not source_ids or not r.source_id or r.source_id in source_ids)
        ]
        if _new_kw:
            top_k_results = top_k_results + _new_kw
            if verbose:
                print(f"  [KW-inject] +{len(_new_kw)} columns via name-keyword match:")
                for inj in _new_kw:
                    print(f"    + {inj.table_name}.{inj.col_name}")

    # ------------------------------------------------------------------
    # Step 4b — Standalone PK injection
    #
    # The FK bridge injector handles bridge tables and FK-connected PKs.
    # But some tables are retrieved semantically (via their data columns)
    # yet their own PK has no incoming FK edge in the result set.
    # e.g. payments table found via paid_amount, but payment_id missing
    # because no other retrieved table has an FK pointing to payments.pk.
    #
    # Fix: for every table in top_k_results, ensure its PK column appears.
    # PK is identified by is_pk=True which is carried in SemanticType=IDENTIFIER
    # + col_name ending in _id/_uuid/_key (heuristic — no schema lookup needed).
    # Works on any relational schema generically.
    # ------------------------------------------------------------------
    _standalone_injected = 0
    _retrieved_col_ids_now = {r.col_id for r in top_k_results}
    _retrieved_table_ids   = list(dict.fromkeys(r.table_id for r in top_k_results))
    _table_has_pk: Dict[str, bool] = {}  # table_id → True if PK already present

    # First pass: mark which tables already have a PK-like column
    for r in top_k_results:
        if r.semantic_type == "IDENTIFIER" and r.similarity > 0.0:
            _table_has_pk[r.table_id] = True

    # Second pass: for each table without a PK, scan fk_adjacency for it
    # If no FK adjacency available, use col_name heuristic from existing results
    _fk_edges_for_pk = get_fk_adjacency(_retrieved_table_ids) if _retrieved_table_ids else []
    _pk_by_table: Dict[str, tuple] = {}  # table_id → (col_id, col_name)
    for edge in _fk_edges_for_pk:
        # The 'to' side of any FK edge is a PK column
        if edge.to_table_id and edge.to_col_id and edge.to_table_id not in _pk_by_table:
            _pk_by_table[edge.to_table_id] = (edge.to_col_id, edge.to_col_name,
                                              edge.to_table_name)

    _standalone_injections: List[RetrievalResult] = []
    for tid in _retrieved_table_ids:
        if _table_has_pk.get(tid):
            continue   # already has a PK-like column — skip
        if _standalone_injected >= FK_MAX_INJECTED_COLS:
            break
        pk_info = _pk_by_table.get(tid)
        if not pk_info:
            continue
        pk_col_id, pk_col_name, pk_table_name = pk_info
        if pk_col_id in _retrieved_col_ids_now:
            continue
        _standalone_injections.append(RetrievalResult(
            col_id        = pk_col_id,
            col_name      = pk_col_name,
            table_id      = tid,
            table_name    = pk_table_name,
            semantic_type = "IDENTIFIER",
            similarity    = 0.0,
            embedding     = None,
        ))
        _retrieved_col_ids_now.add(pk_col_id)
        _standalone_injected += 1

    if _standalone_injections:
        top_k_results = top_k_results + _standalone_injections
        if verbose:
            print(f"  [Standalone PK] Injected {len(_standalone_injections)} orphaned PKs:")
            for inj in _standalone_injections:
                print(f"    + {inj.table_name}.{inj.col_name}")

    # ------------------------------------------------------------------
    # Step 4c — Display column injection
    #
    # For every table in top_k_results, inject its primary display column
    # if not already present. Display columns are the human-readable
    # identifiers users expect in results (incident_no, order_number, etc.)
    # They have no FK signal so FK bridge injection misses them.
    # Identified at ingestion time and stored in table_metadata.
    # Generic — works on any schema without domain knowledge.
    # ------------------------------------------------------------------
    _current_col_ids  = {r.col_id for r in top_k_results}
    _current_tbl_ids  = list(dict.fromkeys(r.table_id for r in top_k_results))
    _display_info     = get_display_columns(_current_tbl_ids)
    _display_injected: List[RetrievalResult] = []

    for tid, info in _display_info.items():
        dcol_id = info.get("col_id")
        if not dcol_id or dcol_id in _current_col_ids:
            continue   # already in results
        _display_injected.append(RetrievalResult(
            col_id        = dcol_id,
            col_name      = info["col_name"],
            table_id      = tid,
            table_name    = info["table_name"],
            semantic_type = "IDENTIFIER",
            similarity    = 0.0,   # injected — not retrieved by search
            embedding     = None,
        ))
        _current_col_ids.add(dcol_id)

    if _display_injected:
        top_k_results = top_k_results + _display_injected
        if verbose:
            print(f"  [Display Col] Injected {len(_display_injected)} display columns:")
            for inj in _display_injected:
                print(f"    ★  {inj.table_name}.{inj.col_name}")

    # ------------------------------------------------------------------
    # Step 4d — Temporal column injection
    #
    # When the query carries temporal language (last week, past month,
    # ago, yesterday …) L3 needs at least one TEMPORAL-typed column per
    # primary table so it can emit a valid date-range filter_tree.
    # Normal cosine retrieval often misses these (low TF-IDF overlap, dim
    # mismatch) so we inject them directly from the semantic_type index.
    # Only tables whose table_id appears in the scored (non-injected)
    # portion of top_k_results are considered — we don't expand the table
    # set, just fill in the missing TEMPORAL columns for tables already
    # deemed relevant.
    # ------------------------------------------------------------------
    _has_temporal_query = any(t in _TEMPORAL_SIGNAL_TOKENS for t in tokens)
    if _has_temporal_query:
        _scored_tids     = list(dict.fromkeys(
            r.table_id for r in top_k_results if r.similarity > 0.0
        ))
        _existing_col_ids = {r.col_id for r in top_k_results}
        _temporal_hits   = retrieve_temporal_cols_for_tables(_scored_tids)
        _temporal_injected: List[RetrievalResult] = []
        for tr in _temporal_hits:
            if tr.col_id not in _existing_col_ids:
                _temporal_injected.append(tr)
                _existing_col_ids.add(tr.col_id)
        if _temporal_injected:
            top_k_results = top_k_results + _temporal_injected
            if verbose:
                print(f"  [Temporal] Injected {len(_temporal_injected)} TEMPORAL columns:")
                for inj in _temporal_injected:
                    print(f"    ⏱  {inj.table_name}.{inj.col_name}")

    # ------------------------------------------------------------------
    # Step 5 — Resolve JOIN path using real FK adjacency data
    # ------------------------------------------------------------------
    _join_path_tids    = list(dict.fromkeys(r.table_id for r in top_k_results))
    _fk_edges_for_join = get_fk_adjacency(_join_path_tids)
    join_path = _resolve_join_path(top_k_results, fk_edges=_fk_edges_for_join)

    # ------------------------------------------------------------------
    # Step 6 — Unique tables
    # ------------------------------------------------------------------
    tables_involved = list(dict.fromkeys(r.table_name for r in top_k_results))

    duration_ms = round((time.time() - t0) * 1000, 2)

    stats = {
        "encoder_mode":      ENCODER_MODE,
        "encoding_strategy": encoding_strategy,
        "tokens_raw":        tokens,
        "tokens_mapped":     mapped_tokens,
        "synonym_hits":          synonym_hits,
        "top_k_returned":        len(top_k_results),
        "join_edges_found":  len(join_path),
        "tables_involved":   tables_involved,
        "duration_ms":       duration_ms,
        **ensemble_stats,    # empty dict for non-ensemble modes
    }

    if verbose:
        print(f"  Top-K returned           : {len(top_k_results)}")
        print(f"  JOIN edges found         : {len(join_path)}")
        print(f"  Tables involved          : {tables_involved}")
        if ensemble_stats:
            both = len(ensemble_stats.get("both_col_ids", set()))
            print(f"  RRF — cols in both lists : {both}")
        print(f"  Duration                 : {duration_ms}ms")
        print(f"[SemanticLayer] Done.\n")

    top_cols_preview = [(r.table_name, r.col_name, round(r.similarity, 4))
                        for r in top_k_results[:5]]
    logger.info(
        "L2 semantic: returned %d columns, tables=%s, JOIN edges=%d, %dms | top5=%s",
        len(top_k_results), tables_involved, len(join_path), duration_ms, top_cols_preview,
    )

    return SemanticLayerResult(
        query             = query,
        query_vector      = query_vector,
        top_k_columns     = top_k_results,
        join_path         = join_path,
        tables_involved   = tables_involved,
        encoding_strategy = encoding_strategy,
        duration_ms       = duration_ms,
        stats             = stats,
    )


# =============================================================================
# Smoke test — python query/semantic_layer.py
# =============================================================================

if __name__ == "__main__":
    from ingestion.vector_store import run_vector_store

    print(f"Running ingestion pipeline  [ENCODER_MODE = {ENCODER_MODE}]...\n")
    run_vector_store(verbose=True)

    test_queries = [
        "show incident status and workflow state",
        "list all alerts and their current queue status",
        "find transactions with high risk score",
        "which counterparties have open incidents",
        "show signal rules and their triggered count",
        "audit log entries by action type",
        "sla breach incidents pending review",
    ]

    print("=" * 70)
    print(f"VEDA POC — Semantic Layer (L2)  [{ENCODER_MODE}]")
    print("=" * 70)

    for query in test_queries:
        result = run_semantic_layer(query, verbose=False)
        print(f"\nQuery : '{query}'")
        print(f"  Strategy  : {result.encoding_strategy}")
        print(f"  Tables    : {result.tables_involved[:4]}")
        if ENCODER_MODE == "ensemble":
            both = len(result.stats.get("both_col_ids", set()))
            print(f"  RRF cols in both: {both}")
        print(f"  Top-3 cols:")
        for r in result.top_k_columns[:3]:
            print(
                f"    {r.table_name}.{r.col_name:<28} "
                f"rrf/sim={r.similarity:.4f}  {r.semantic_type}"
            )
        if result.join_path:
            print(f"  JOINs:")
            for edge in result.join_path[:2]:
                print(f"    {edge.from_table_name}.{edge.from_col_name}"
                      f" → {edge.to_table_name}.{edge.to_col_name}")