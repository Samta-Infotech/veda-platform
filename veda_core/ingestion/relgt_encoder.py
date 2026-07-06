# =============================================================================
# ingestion/relgt_encoder.py
# VEDA POC — Step 4: Encoder
#
# Responsibility:
#   - Accepts REGGraph from reg_builder.py
#   - Reads ENCODER_MODE from config.py and routes to the correct encoder
#   - Produces one fixed-dim embedding vector per column
#   - Passes EncoderResult to vector_store.py
#
# Encoder paths — selected by config.ENCODER_MODE:
#
#   "relgt_only"  — RELGT structural GNN (PyTorch HGT or numpy fallback)
#                   256-dim, no text signal
#
#   "light_text"  — TF-IDF + TruncatedSVD over column sentences
#                   256-dim, pure text signal, no graph
#                   sklearn only — no GPU needed
#
#   "hybrid"      — MiniLM 384-dim + RELGT 256-dim concatenated = 640-dim
#                   Matches v1.0 architecture exactly
#                   Requires: pip install sentence-transformers
#
#   "ensemble"    — Runs BOTH light_text AND hybrid encoders
#                   Produces two separate embedding sets (256-dim + 640-dim)
#                   vector_store.py writes to two separate pgvector tables
#                   semantic_layer.py queries both and merges via RRF
#
# All single-encoder paths produce EncoderResult with List[ColumnEmbedding].
# Ensemble path produces EnsembleEncoderResult with two separate lists.
# vector_store.py handles both types.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ingestion.reg_builder import REGGraph, ColumnNode, run_reg_builder
from ingestion.column_text import build_enriched_column_text
from config import (
    ENCODER_MODE,
    VECTOR_DIM,
    # RELGT params
    RELGT_HIDDEN_DIM,
    RELGT_NUM_LAYERS,
    RELGT_OUTPUT_DIM,
    RELGT_EMBEDDING_DIM,
    RELGT_FK_EDGE_WEIGHT,
    # MiniLM params
    MINILM_MODEL_NAME,
    MINILM_SENTENCE_TEMPLATE,
    MINILM_BATCH_SIZE,
    MINILM_DEVICE,
    MINILM_EMBEDDING_DIM,
    # Light text params
    LIGHT_TEXT_SENTENCE_TEMPLATE,
    LIGHT_TEXT_TFIDF_MAX_FEATURES,
    LIGHT_TEXT_TFIDF_NGRAM_RANGE,
    LIGHT_TEXT_SVD_COMPONENTS,
    LIGHT_TEXT_CHAR_SPLIT,
    LIGHT_TEXT_EMBEDDING_DIM,
    # Ensemble params
    HYBRID_EMBEDDING_DIM,
    SYNTHETIC_PAIRS_PATH,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Optional imports — graceful fallbacks
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import HGTConv, Linear as PyGLinear
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize as sklearn_normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class ColumnEmbedding:
    """
    Final output unit — one per column.
    Consumed directly by vector_store.py for persistence.
    embedding shape = (VECTOR_DIM,) — exact dim depends on ENCODER_MODE.
    """
    col_id:        str
    col_name:      str
    table_id:      str
    table_name:    str
    semantic_type: str
    embedding:     np.ndarray


@dataclass
class EncoderResult:
    """
    Top-level output for single-encoder modes (relgt_only, light_text, hybrid).
    """
    embeddings:    List[ColumnEmbedding]
    embedding_dim: int
    encoder_type:  str
    stats:         dict = field(default_factory=dict)


@dataclass
class EnsembleEncoderResult:
    """
    Top-level output for ensemble mode.
    Carries two independent embedding sets — one per sub-encoder.
    vector_store.py writes each to its own pgvector table.
    semantic_layer.py queries both tables at query time.
    """
    # Light text embeddings — 256-dim, stored in column_embeddings_lt
    lt_embeddings:    List[ColumnEmbedding]
    lt_embedding_dim: int

    # Hybrid embeddings — 640-dim, stored in column_embeddings_hybrid
    hybrid_embeddings:    List[ColumnEmbedding]
    hybrid_embedding_dim: int

    encoder_type: str   # always "ensemble_light_text_plus_hybrid"
    stats:        dict = field(default_factory=dict)



# =============================================================================
# Shared numpy helpers (used by RELGT paths and hybrid)
# =============================================================================

def _xavier_init(fan_in: int, fan_out: int, seed: int) -> np.ndarray:
    """Deterministic Xavier uniform — seed must match semantic_layer.py."""
    rng   = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _l2_normalise_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


# =============================================================================
# PATH A — RELGT PyTorch HGT
# =============================================================================

def _build_hgt_model(table_feat_dim, column_feat_dim, hidden_dim, output_dim, num_layers):
    class RELGTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.table_proj = PyGLinear(table_feat_dim,  hidden_dim)
            self.col_proj   = PyGLinear(column_feat_dim, hidden_dim)
            metadata = (
                ["table", "column"],
                [("table", "has_column", "column"), ("column", "fk_to", "column")],
            )
            self.convs = nn.ModuleList([
                HGTConv(in_channels=hidden_dim, out_channels=hidden_dim,
                        metadata=metadata, heads=4)
                for _ in range(num_layers)
            ])
            self.out_proj    = PyGLinear(hidden_dim, output_dim)
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                "table":  F.relu(self.table_proj(x_dict["table"])),
                "column": F.relu(self.col_proj(x_dict["column"])),
            }
            for i, conv in enumerate(self.convs):
                residual   = x_dict["column"]
                x_dict_new = conv(x_dict, edge_index_dict)
                col_out    = self.layer_norms[i](x_dict_new.get("column", x_dict["column"]) + residual)
                x_dict     = {"table": x_dict_new.get("table", x_dict["table"]), "column": col_out}
            return F.normalize(self.out_proj(x_dict["column"]), p=2, dim=-1)

    model = RELGTModel()

    # Load trained weights if available — produced by training/train_relgt.py
    weights_path = os.path.join(
        os.path.dirname(__file__),
        '../training/relgt_trained.pt'
    )
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"[RELGT] Pretrained weights loaded from {weights_path} ✅")
            logger.info("RELGT pretrained weights loaded from %s", weights_path)
        except Exception as e:
            print(f"[RELGT] ⚠️  Weight load failed ({e}) — using random init")
            logger.warning("RELGT weight load failed (%s) — using random init", e)
    else:
        print(f"[RELGT] ⚠️  No trained weights found at {weights_path} — using random init")
        print(f"[RELGT]     Run: python training/train_relgt.py")
        logger.warning("RELGT no trained weights at %s — using random Xavier init", weights_path)

    model.eval()
    return model


def _run_torch_encoder(graph: REGGraph) -> np.ndarray:
    hetero = graph.hetero_data
    if hetero is None:
        raise RuntimeError("hetero_data is None")
    model = _build_hgt_model(
        graph.table_feature_matrix.shape[1],
        graph.column_feature_matrix.shape[1],
        RELGT_HIDDEN_DIM, RELGT_OUTPUT_DIM, RELGT_NUM_LAYERS,
    )
    with torch.no_grad():
        x_dict = {"table": hetero["table"].x, "column": hetero["column"].x}
        edge_index_dict = {}
        for key in [("table", "has_column", "column"), ("column", "fk_to", "column")]:
            try:
                edge_index_dict[key] = hetero[key].edge_index
            except (KeyError, AttributeError):
                pass
        return model(x_dict, edge_index_dict).numpy()


# =============================================================================
# PATH B — RELGT NumPy fallback GNN
# =============================================================================

def _build_adjacency(n_cols, has_column_edges, fk_to_edges, n_tables):
    adj = np.zeros((n_cols, n_cols), dtype=np.float32)
    table_to_cols: Dict[int, List[int]] = {}
    for (t_idx, c_idx) in has_column_edges:
        table_to_cols.setdefault(t_idx, []).append(c_idx)
    for col_indices in table_to_cols.values():
        for i in col_indices:
            for j in col_indices:
                if i != j:
                    adj[i, j] = 1.0
    for (from_idx, to_idx) in fk_to_edges:
        adj[from_idx, to_idx] = RELGT_FK_EDGE_WEIGHT
        adj[to_idx, from_idx] = RELGT_FK_EDGE_WEIGHT
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return adj / row_sums


def _run_numpy_encoder(graph: REGGraph) -> np.ndarray:
    col_feats    = graph.column_feature_matrix
    n_cols       = col_feats.shape[0]
    col_feat_dim = col_feats.shape[1]
    adj = _build_adjacency(n_cols, graph.has_column_edges, graph.fk_to_edges,
                           graph.table_feature_matrix.shape[0])
    W_in          = _xavier_init(col_feat_dim,     RELGT_HIDDEN_DIM, seed=42)
    W_out         = _xavier_init(RELGT_HIDDEN_DIM, RELGT_OUTPUT_DIM, seed=99)
    layer_weights = [_xavier_init(RELGT_HIDDEN_DIM, RELGT_HIDDEN_DIM, seed=100 + i)
                     for i in range(RELGT_NUM_LAYERS)]
    H = _relu(col_feats @ W_in)
    for i in range(RELGT_NUM_LAYERS):
        H_new = _relu((adj @ H) @ layer_weights[i]) + H
        mean  = H_new.mean(axis=0, keepdims=True)
        std   = H_new.std(axis=0,  keepdims=True) + 1e-6
        H     = (H_new - mean) / std
    return _l2_normalise_rows(H @ W_out).astype(np.float32)


# =============================================================================
# PATH C — Light Text Encoder (TF-IDF + TruncatedSVD)
# =============================================================================

_TFIDF_VECTORIZER = None
_SVD_MODEL        = None


def _split_identifier(name: str) -> str:
    tokens = name.replace("_", " ")
    tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", tokens)
    return tokens.lower().strip()


def _build_column_sentence(col_node: ColumnNode, sibling_names: list = None, sampled=None) -> str:
    name_str = _split_identifier(col_node.col_name) if LIGHT_TEXT_CHAR_SPLIT else col_node.col_name
    return build_enriched_column_text(
        col_name      = name_str,
        table_name    = col_node.table_name,
        semantic_type = col_node.semantic_type,
        is_pk         = col_node.is_pk,
        is_fk         = col_node.is_fk,
        fk_ref_table  = col_node.fk_ref_table,
        fk_ref_col    = col_node.fk_ref_col,
        sampled       = sampled,
        sibling_names = sibling_names,
        style         = "light_text",
    )


_SIBLING_TYPES = {"CATEGORY", "TEMPORAL", "METRIC"}


def _run_light_text_encoder(graph: REGGraph) -> np.ndarray:
    global _TFIDF_VECTORIZER, _SVD_MODEL

    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    sibling_map: Dict[str, List[str]] = {}
    for cn in graph.column_nodes:
        if cn.semantic_type in _SIBLING_TYPES:
            sibling_map.setdefault(cn.table_id, []).append(cn.col_name)

    sentences = [""] * len(graph.column_nodes)
    try:
        from ingestion.value_sampler import _VALUE_STORE as _vs
    except Exception:
        _vs = {}
    for col_node in graph.column_nodes:
        siblings = [s for s in sibling_map.get(col_node.table_id, []) if s != col_node.col_name][:4]
        sentences[col_node.node_index] = _build_column_sentence(
            col_node, siblings, sampled=_vs.get(col_node.col_id)
        )

    vectorizer   = TfidfVectorizer(
        max_features  = LIGHT_TEXT_TFIDF_MAX_FEATURES,
        ngram_range   = LIGHT_TEXT_TFIDF_NGRAM_RANGE,
        sublinear_tf  = True,
        analyzer      = "word",
        strip_accents = "unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)

    n_components = min(LIGHT_TEXT_SVD_COMPONENTS, tfidf_matrix.shape[1] - 1)
    svd          = TruncatedSVD(n_components=n_components, algorithm="randomized",
                                n_iter=10, random_state=42)
    reduced      = svd.fit_transform(tfidf_matrix)

    if reduced.shape[1] < LIGHT_TEXT_EMBEDDING_DIM:
        pad     = np.zeros((reduced.shape[0], LIGHT_TEXT_EMBEDDING_DIM - reduced.shape[1]),
                           dtype=np.float32)
        reduced = np.hstack([reduced, pad])

    reduced = sklearn_normalize(reduced, norm="l2").astype(np.float32)

    _TFIDF_VECTORIZER = vectorizer
    _SVD_MODEL        = svd

    # Persist for query-time reuse without re-ingestion
    import pickle as _pkl, os as _os
    _schema_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'schema')
    _os.makedirs(_schema_dir, exist_ok=True)
    with open(_os.path.join(_schema_dir, 'tfidf_vectorizer.pkl'), 'wb') as _f:
        _pkl.dump(_TFIDF_VECTORIZER, _f, protocol=_pkl.HIGHEST_PROTOCOL)
    with open(_os.path.join(_schema_dir, 'svd_transformer.pkl'), 'wb') as _f:
        _pkl.dump(_SVD_MODEL, _f, protocol=_pkl.HIGHEST_PROTOCOL)
    print("[TF-IDF] Vectorizer + SVD saved to schema/")

    return reduced


def get_light_text_models():
    """Returns (vectorizer, svd) fitted during ingestion; loads from disk if not in memory."""
    global _TFIDF_VECTORIZER, _SVD_MODEL
    if _TFIDF_VECTORIZER is not None and _SVD_MODEL is not None:
        return _TFIDF_VECTORIZER, _SVD_MODEL
    import pickle, os
    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'schema')
    tp = os.path.join(schema_dir, 'tfidf_vectorizer.pkl')
    sp = os.path.join(schema_dir, 'svd_transformer.pkl')
    if os.path.exists(tp) and os.path.exists(sp):
        with open(tp, 'rb') as f: _TFIDF_VECTORIZER = pickle.load(f)
        with open(sp, 'rb') as f: _SVD_MODEL = pickle.load(f)
        print("[TF-IDF] Loaded from schema/")
    return _TFIDF_VECTORIZER, _SVD_MODEL


# =============================================================================
# PATH D — Hybrid: MiniLM (384-dim) + RELGT (256-dim) = 640-dim
# =============================================================================

_MINILM_MODEL = None


def _get_minilm_model() -> "SentenceTransformer":
    """
    Returns the MiniLM model singleton (base weights).
    The fine-tune chain was removed — both query tiers use base MiniLM.
    """
    global _MINILM_MODEL
    if _MINILM_MODEL is None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for hybrid/ensemble encoder. "
                "Install with: pip install sentence-transformers"
            )
        # Load base MiniLM (fine-tune chain removed — both tiers use base weights).
        _MINILM_MODEL = SentenceTransformer(MINILM_MODEL_NAME, device=MINILM_DEVICE)
    return _MINILM_MODEL


def _build_minilm_sentence(col_node: ColumnNode, sibling_names: list = None, sampled=None) -> str:
    return build_enriched_column_text(
        col_name      = col_node.col_name,
        table_name    = col_node.table_name,
        semantic_type = col_node.semantic_type,
        is_pk         = col_node.is_pk,
        is_fk         = col_node.is_fk,
        fk_ref_table  = col_node.fk_ref_table,
        fk_ref_col    = col_node.fk_ref_col,
        sampled       = sampled,
        sibling_names = sibling_names,
        style         = "minilm",
    )


def _run_minilm_encoder(graph: REGGraph) -> np.ndarray:
    sibling_map: Dict[str, List[str]] = {}
    for cn in graph.column_nodes:
        if cn.semantic_type in _SIBLING_TYPES:
            sibling_map.setdefault(cn.table_id, []).append(cn.col_name)

    model     = _get_minilm_model()
    sentences = [""] * len(graph.column_nodes)
    try:
        from ingestion.value_sampler import _VALUE_STORE as _vs
    except Exception:
        _vs = {}
    for col_node in graph.column_nodes:
        siblings = [s for s in sibling_map.get(col_node.table_id, []) if s != col_node.col_name][:4]
        sentences[col_node.node_index] = _build_minilm_sentence(
            col_node, siblings, sampled=_vs.get(col_node.col_id)
        )
    embeddings = model.encode(
        sentences,
        batch_size           = MINILM_BATCH_SIZE,
        show_progress_bar    = False,
        normalize_embeddings = True,
        device               = MINILM_DEVICE,
    )
    return embeddings.astype(np.float32)


def _run_hybrid_encoder(graph: REGGraph) -> np.ndarray:
    minilm_embeddings = _run_minilm_encoder(graph)
    if TORCH_GEOMETRIC_AVAILABLE and graph.hetero_data is not None:
        relgt_embeddings = _run_torch_encoder(graph)
    else:
        relgt_embeddings = _run_numpy_encoder(graph)
    combined = np.concatenate([minilm_embeddings, relgt_embeddings], axis=1)
    return _l2_normalise_rows(combined).astype(np.float32)


def get_minilm_model():
    """Returns the loaded MiniLM model singleton for query-time reuse."""
    return _MINILM_MODEL


# =============================================================================
# PATH E — Ensemble: runs BOTH light_text AND hybrid
#
# Both encoders run on the same graph in sequence.
# MiniLM is loaded once and reused for the hybrid component — no double load.
# Returns EnsembleEncoderResult with two separate ColumnEmbedding lists.
#
# Why run both instead of just using hybrid:
#   - Light text has better synonym precision (proved in Run 2 vs Run 3)
#   - Hybrid has better multi-table / structural recall (proved in Run 3)
#   - RRF fusion in semantic_layer.py combines both strengths
#   - Running both at ingestion is a one-time cost
# =============================================================================

def _wrap_embeddings(
    raw: np.ndarray,
    graph: REGGraph,
) -> List[ColumnEmbedding]:
    """Wraps a raw embedding matrix into a list of ColumnEmbedding objects."""
    result = []
    for col_node in graph.column_nodes:
        result.append(ColumnEmbedding(
            col_id        = col_node.col_id,
            col_name      = col_node.col_name,
            table_id      = col_node.table_id,
            table_name    = col_node.table_name,
            semantic_type = col_node.semantic_type,
            embedding     = raw[col_node.node_index],
        ))
    return result


def _run_ensemble_encoder(
    graph:   REGGraph,
    verbose: bool = False,
) -> "EnsembleEncoderResult":
    """
    Runs light_text and hybrid encoders on the same graph.
    Both models are loaded/fitted once and reused.

    Returns EnsembleEncoderResult with two ColumnEmbedding lists.
    """
    if verbose:
        print(f"  [Ensemble] Step 1/2 — Light Text encoder (TF-IDF + SVD → 256-dim)")

    # --- Light text: 256-dim ---
    lt_raw = _run_light_text_encoder(graph)     # also fits + saves TF-IDF/SVD singletons

    if verbose:
        print(f"  [Ensemble] Step 2/2 — Hybrid encoder (MiniLM + RELGT → 640-dim)")
        print(f"             MiniLM model: {MINILM_MODEL_NAME}")

    # --- Hybrid: 640-dim ---
    # MiniLM is loaded once here — get_minilm_model() reuses the singleton at query time
    hybrid_raw = _run_hybrid_encoder(graph)

    # --- Wrap into ColumnEmbedding lists ---
    lt_embeddings     = _wrap_embeddings(lt_raw,     graph)
    hybrid_embeddings = _wrap_embeddings(hybrid_raw, graph)

    # --- Norms for stats ---
    lt_norms     = np.linalg.norm(lt_raw,     axis=1)
    hybrid_norms = np.linalg.norm(hybrid_raw, axis=1)

    stats = {
        "total_embeddings":      len(lt_embeddings),
        "encoder_type":          "ensemble_light_text_plus_hybrid",
        "encoder_mode":          "ensemble",
        "lt_embedding_dim":      LIGHT_TEXT_EMBEDDING_DIM,
        "hybrid_embedding_dim":  HYBRID_EMBEDDING_DIM,
        "lt_norm_mean":          round(float(lt_norms.mean()),     4),
        "lt_norm_std":           round(float(lt_norms.std()),      4),
        "hybrid_norm_mean":      round(float(hybrid_norms.mean()), 4),
        "hybrid_norm_std":       round(float(hybrid_norms.std()),  4),
        "minilm_used":           True,
        "relgt_used":            True,
        "light_text_used":       True,
    }

    return EnsembleEncoderResult(
        lt_embeddings        = lt_embeddings,
        lt_embedding_dim     = LIGHT_TEXT_EMBEDDING_DIM,
        hybrid_embeddings    = hybrid_embeddings,
        hybrid_embedding_dim = HYBRID_EMBEDDING_DIM,
        encoder_type         = "ensemble_light_text_plus_hybrid",
        stats                = stats,
    )


# =============================================================================
# Query-time subgraph encoder (used by semantic_layer.py via graph_store.py)
# =============================================================================

def run_relgt_on_graph(graph) -> np.ndarray:
    """
    Runs the numpy GNN on any graph object that exposes the same fields as
    REGGraph (column_feature_matrix, has_column_edges, fk_to_edges,
    table_feature_matrix). Accepts both REGGraph and SubGraph.

    Returns L2-normalised embeddings of shape (n_cols, RELGT_OUTPUT_DIM).
    Seeds match _run_numpy_encoder exactly — same Xavier weights as ingestion.
    """
    return _run_numpy_encoder(graph)


# =============================================================================
# Public entry point
# =============================================================================

def run_relgt_encoder(
    graph:   REGGraph = None,
    verbose: bool     = False,
):
    """
    Main entry point for Step 4.

    Routes to the correct encoder based on config.ENCODER_MODE.

    Returns
    -------
    EncoderResult        — for relgt_only, light_text, hybrid
    EnsembleEncoderResult — for ensemble
    """
    if graph is None:
        graph = run_reg_builder(verbose=verbose)

    logger.debug("Starting encoder: mode=%s, columns=%d", ENCODER_MODE, len(graph.column_nodes))

    if verbose:
        print(f"[Encoder] Mode             : {ENCODER_MODE}")
        print(f"  Column nodes             : {len(graph.column_nodes)}")

    # ------------------------------------------------------------------
    # Route to correct encoder
    # ------------------------------------------------------------------
    if ENCODER_MODE == "relgt_only":
        if verbose:
            print(f"  Backend                  : {'PyTorch HGT' if TORCH_GEOMETRIC_AVAILABLE else 'NumPy GNN fallback'}")
        if TORCH_GEOMETRIC_AVAILABLE and graph.hetero_data is not None:
            raw_embeddings = _run_torch_encoder(graph)
            encoder_type   = "relgt_torch"
        else:
            raw_embeddings = _run_numpy_encoder(graph)
            encoder_type   = "relgt_numpy_fallback"

    elif ENCODER_MODE == "light_text":
        if verbose:
            print(f"  Backend                  : TF-IDF + TruncatedSVD (sklearn)")
        raw_embeddings = _run_light_text_encoder(graph)
        encoder_type   = "light_text_tfidf_svd"

    elif ENCODER_MODE == "hybrid":
        if verbose:
            print(f"  Backend                  : MiniLM (384) + RELGT (256) → 640")
            print(f"  sentence-transformers    : {'available' if SENTENCE_TRANSFORMERS_AVAILABLE else 'NOT installed'}")
        raw_embeddings = _run_hybrid_encoder(graph)
        encoder_type   = "hybrid_minilm_relgt"

    elif ENCODER_MODE == "ensemble":
        if verbose:
            print(f"  Backend                  : Light Text (256) + Hybrid (640) dual store")
            print(f"  sentence-transformers    : {'available' if SENTENCE_TRANSFORMERS_AVAILABLE else 'NOT installed'}")
        # Ensemble returns its own result type — early return
        return _run_ensemble_encoder(graph, verbose=verbose)

    else:
        raise ValueError(f"Unknown ENCODER_MODE: {ENCODER_MODE}")

    # ------------------------------------------------------------------
    # Single-encoder: validate shape, wrap, return EncoderResult
    # ------------------------------------------------------------------
    assert raw_embeddings.shape == (len(graph.column_nodes), VECTOR_DIM), (
        f"Encoder output shape mismatch: got {raw_embeddings.shape}, "
        f"expected ({len(graph.column_nodes)}, {VECTOR_DIM})"
    )

    embeddings = []
    for col_node in graph.column_nodes:
        embeddings.append(ColumnEmbedding(
            col_id        = col_node.col_id,
            col_name      = col_node.col_name,
            table_id      = col_node.table_id,
            table_name    = col_node.table_name,
            semantic_type = col_node.semantic_type,
            embedding     = raw_embeddings[col_node.node_index],
        ))

    norms = np.linalg.norm(raw_embeddings, axis=1)
    stats = {
        "total_embeddings": len(embeddings),
        "embedding_dim":    VECTOR_DIM,
        "encoder_type":     encoder_type,
        "encoder_mode":     ENCODER_MODE,
        "minilm_used":      ENCODER_MODE == "hybrid",
        "relgt_used":       ENCODER_MODE in ("relgt_only", "hybrid"),
        "light_text_used":  ENCODER_MODE == "light_text",
        "norm_mean":        round(float(norms.mean()), 4),
        "norm_std":         round(float(norms.std()),  4),
        "norm_min":         round(float(norms.min()),  4),
        "norm_max":         round(float(norms.max()),  4),
    }

    if verbose:
        print(f"  Embeddings built         : {stats['total_embeddings']}")
        print(f"  Encoder type             : {encoder_type}")
        print(f"  Norm mean / std          : {stats['norm_mean']} / {stats['norm_std']}")
        print(f"[Encoder] Done.\n")

    logger.info(
        "Encoder complete: mode=%s, type=%s, embeddings=%d, dim=%d, norm_mean=%.4f",
        ENCODER_MODE, encoder_type, stats["total_embeddings"],
        stats["embedding_dim"], stats["norm_mean"],
    )

    return EncoderResult(
        embeddings    = embeddings,
        embedding_dim = VECTOR_DIM,
        encoder_type  = encoder_type,
        stats         = stats,
    )


# =============================================================================
# Smoke test — python ingestion/relgt_encoder.py
# =============================================================================

if __name__ == "__main__":
    result = run_relgt_encoder(verbose=True)

    print("=" * 60)
    print(f"VEDA POC — Encoder Output  [{ENCODER_MODE}]")
    print("=" * 60)

    if ENCODER_MODE == "ensemble":
        print(f"  Total columns    : {result.stats['total_embeddings']}")
        print(f"  LT dim           : {result.lt_embedding_dim}  norm_mean={result.stats['lt_norm_mean']}")
        print(f"  Hybrid dim       : {result.hybrid_embedding_dim}  norm_mean={result.stats['hybrid_norm_mean']}")
        print()
        print("  LT sample (first 3):")
        for emb in result.lt_embeddings[:3]:
            print(f"    {emb.table_name}.{emb.col_name:<26} vec[:3]={np.round(emb.embedding[:3], 3)}")
        print("  Hybrid sample (first 3):")
        for emb in result.hybrid_embeddings[:3]:
            print(f"    {emb.table_name}.{emb.col_name:<26} vec[:3]={np.round(emb.embedding[:3], 3)}")
    else:
        print(f"  Total embeddings : {result.stats['total_embeddings']}")
        print(f"  Embedding dim    : {result.stats['embedding_dim']}")
        print(f"  Encoder type     : {result.stats['encoder_type']}")
        print(f"  Norm mean / std  : {result.stats['norm_mean']} / {result.stats['norm_std']}")
        print()
        print("Sample embeddings (first 5):")
        for emb in result.embeddings[:5]:
            print(
                f"  {emb.table_name}.{emb.col_name:<28} "
                f"{emb.semantic_type:<12} "
                f"vec[:4]={np.round(emb.embedding[:4], 3)}"
            )

    if ENCODER_MODE in ("light_text", "ensemble"):
        vectorizer, svd = get_light_text_models()
        if vectorizer:
            print(f"\n  TF-IDF vocab size            : {len(vectorizer.vocabulary_)}")
            if svd:
                print(f"  SVD explained variance sum   : {svd.explained_variance_ratio_.sum():.4f}")

    if ENCODER_MODE in ("hybrid", "ensemble"):
        model = get_minilm_model()
        if model:
            print(f"\n  MiniLM loaded : {MINILM_MODEL_NAME}  max_seq={model.max_seq_length}")