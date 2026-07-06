# =============================================================================
# ingestion/reg_builder.py
# VEDA POC — Step 3: REG Builder (Relational Entity Graph)
#
# Responsibility:
#   - Accepts InferenceResult from semantic_type_inference.py
#   - Constructs a PyTorch Geometric HeteroData graph in memory
#   - Table nodes  : one per table  — features: name embedding, row_count, col_count
#   - Column nodes : one per column — features: semantic_type, data_type, is_pk,
#                                               is_fk, cardinality (normalised)
#   - has_column edges : table  → column
#   - fk_to edges      : fk_col → referenced pk_col
#   - Graph is passed directly to relgt_encoder.py
#   - Graph is NEVER persisted — discarded after encoding (architecture constraint)
#
# NOTE: PyTorch Geometric is a heavy dependency. For the POC we implement a
# lightweight fallback (pure numpy adjacency representation) that activates
# automatically if torch_geometric is not installed. The RELGT encoder uses
# whichever representation is available.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ingestion.semantic_type_inference import (
    InferenceResult,
    TypedColumn,
    run_semantic_type_inference,
)
from config import (
    SEMANTIC_TYPES,
    TABLE_NAME_EMBED_DIM,
    RELGT_EMBEDDING_DIM,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# PyTorch Geometric import — graceful fallback
# =============================================================================

try:
    import torch
    from torch_geometric.data import HeteroData
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


# =============================================================================
# Feature encoding helpers
# =============================================================================

# One-hot index for semantic types — must match SEMANTIC_TYPES order in config
SEMANTIC_TYPE_INDEX = {st: i for i, st in enumerate(SEMANTIC_TYPES)}

# One-hot index for data types
DATA_TYPES = ["integer", "varchar", "numeric", "timestamp", "boolean",
              "timestamptz", "date"]
DATA_TYPE_INDEX = {dt: i for i, dt in enumerate(DATA_TYPES)}


def _encode_name_to_vector(name: str, dim: int = TABLE_NAME_EMBED_DIM) -> np.ndarray:
    """
    Deterministic name → fixed-dim vector using character-level hashing.
    No model required. Produces a stable vector for the same name every run.

    This is intentionally simple for the POC. In production this would be
    replaced by a lightweight sentence encoder or cached lookup.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for i, char in enumerate(name):
        # spread character values across the vector deterministically
        idx = (ord(char) * 31 + i * 17) % dim
        vec[idx] += 1.0

    # L2 normalise so magnitude doesn't vary with name length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _encode_semantic_type(semantic_type: str) -> np.ndarray:
    """One-hot vector of length len(SEMANTIC_TYPES)."""
    vec = np.zeros(len(SEMANTIC_TYPES), dtype=np.float32)
    idx = SEMANTIC_TYPE_INDEX.get(semantic_type, 0)
    vec[idx] = 1.0
    return vec


def _encode_data_type(data_type: str) -> np.ndarray:
    """One-hot vector of length len(DATA_TYPES)."""
    vec = np.zeros(len(DATA_TYPES), dtype=np.float32)
    idx = DATA_TYPE_INDEX.get(data_type.lower(), 0)
    vec[idx] = 1.0
    return vec


def _normalise_cardinality(cardinality: Optional[int], row_count: int) -> float:
    """
    Returns cardinality as a fraction of row_count (0.0 – 1.0).
    None → 0.5 (unknown, mid-range assumption).
    """
    if cardinality is None:
        return 0.5
    if row_count <= 0:
        return 0.0
    return min(float(cardinality) / float(row_count), 1.0)


def _normalise_row_count(row_count: int, max_rows: int = 250_000) -> float:
    """Log-normalised row count → 0.0 – 1.0."""
    if row_count <= 0:
        return 0.0
    import math
    return min(math.log1p(row_count) / math.log1p(max_rows), 1.0)


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class TableNode:
    """Feature vector + metadata for a single table node."""
    table_id:    str
    table_name:  str
    node_index:  int                  # position in the table node matrix
    row_count:   int
    col_count:   int
    feature_vec: np.ndarray           # shape: (TABLE_NAME_EMBED_DIM + 2,)


@dataclass
class ColumnNode:
    """Feature vector + metadata for a single column node."""
    col_id:        str
    col_name:      str
    table_id:      str
    table_name:    str
    node_index:    int                # position in the column node matrix
    semantic_type: str
    data_type:     str
    is_pk:         bool
    is_fk:         bool
    confidence:    float
    feature_vec:   np.ndarray         # shape: (len(SEMANTIC_TYPES) + len(DATA_TYPES) + 4,)
    fk_ref_table:  Optional[str] = None   # table this FK column references
    fk_ref_col:    Optional[str] = None   # column this FK column references


@dataclass
class REGGraph:
    """
    In-memory Relational Entity Graph.

    Contains:
      - table_nodes / column_nodes : node metadata + feature vectors
      - has_column_edges           : (table_node_index, column_node_index) pairs
      - fk_to_edges                : (from_col_node_index, to_col_node_index) pairs
      - table_feature_matrix       : np.ndarray shape (n_tables, table_feat_dim)
      - column_feature_matrix      : np.ndarray shape (n_cols, col_feat_dim)
      - hetero_data                : torch_geometric.HeteroData or None
      - col_id_to_node_index       : dict for fast lookup in encoder
    """
    table_nodes:          List[TableNode]
    column_nodes:         List[ColumnNode]
    has_column_edges:     List[Tuple[int, int]]   # (table_idx, col_idx)
    fk_to_edges:          List[Tuple[int, int]]   # (from_col_idx, to_col_idx)
    table_feature_matrix: np.ndarray
    column_feature_matrix: np.ndarray
    hetero_data:          object                  # HeteroData or None
    col_id_to_node_index: Dict[str, int]
    table_id_to_node_index: Dict[str, int]
    stats: dict = field(default_factory=dict)


# =============================================================================
# Node feature construction
# =============================================================================

def _build_table_nodes(
    inference_result: InferenceResult,
) -> Tuple[List[TableNode], Dict[str, int]]:
    """
    Builds one TableNode per unique table.
    Returns (list of TableNode, table_id → node_index mapping).
    """
    # Collect unique tables preserving order
    seen = {}
    ordered_tables = []
    for tc in inference_result.typed_columns:
        if tc.table_id not in seen:
            seen[tc.table_id] = True
            ordered_tables.append((tc.table_id, tc.table_name))

    # Count columns and rows per table
    col_counts  = {}
    row_counts  = {}
    for tc in inference_result.typed_columns:
        col_counts[tc.table_id]  = col_counts.get(tc.table_id, 0) + 1

    # row_count is not on TypedColumn — rebuild from scanner via table_name
    # We use a fixed lookup from simulate_schema row_counts embedded in col metadata
    # For POC: approximate from cardinality of PK column
    pk_cardinality = {}
    for tc in inference_result.typed_columns:
        if tc.is_pk and tc.cardinality:
            pk_cardinality[tc.table_id] = tc.cardinality

    table_nodes   = []
    table_id_map  = {}

    for idx, (table_id, table_name) in enumerate(ordered_tables):
        col_count = col_counts.get(table_id, 1)
        row_count = pk_cardinality.get(table_id, 1000)   # fallback estimate

        # Feature vector: name_hash_embed | norm_row_count | norm_col_count
        name_vec     = _encode_name_to_vector(table_name, TABLE_NAME_EMBED_DIM)
        row_count_f  = np.array([_normalise_row_count(row_count)],  dtype=np.float32)
        col_count_f  = np.array([float(col_count) / 30.0],          dtype=np.float32)  # /30 normalise
        feature_vec  = np.concatenate([name_vec, row_count_f, col_count_f])

        node = TableNode(
            table_id   = table_id,
            table_name = table_name,
            node_index = idx,
            row_count  = row_count,
            col_count  = col_count,
            feature_vec = feature_vec,
        )
        table_nodes.append(node)
        table_id_map[table_id] = idx

    return table_nodes, table_id_map


def _build_column_nodes(
    inference_result: InferenceResult,
    table_id_to_row_count: Dict[str, int],
) -> Tuple[List[ColumnNode], Dict[str, int]]:
    """
    Builds one ColumnNode per typed column.
    Returns (list of ColumnNode, col_id → node_index mapping).
    """
    column_nodes = []
    col_id_map   = {}

    for idx, tc in enumerate(inference_result.typed_columns):
        row_count   = table_id_to_row_count.get(tc.table_id, 1000)
        card_norm   = _normalise_cardinality(tc.cardinality, row_count)

        # Feature vector components:
        # [semantic_type onehot | data_type onehot | is_pk | is_fk | cardinality_norm | confidence]
        sem_vec    = _encode_semantic_type(tc.semantic_type)
        dtype_vec  = _encode_data_type(tc.data_type)
        flags_vec  = np.array([
            float(tc.is_pk),
            float(tc.is_fk),
            card_norm,
            tc.confidence,
        ], dtype=np.float32)

        col_name_hash = _encode_name_to_vector(tc.col_name, dim=32)
        feature_vec   = np.concatenate([sem_vec, dtype_vec, flags_vec, col_name_hash])

        node = ColumnNode(
            col_id        = tc.col_id,
            col_name      = tc.col_name,
            table_id      = tc.table_id,
            table_name    = tc.table_name,
            node_index    = idx,
            semantic_type = tc.semantic_type,
            data_type     = tc.data_type,
            is_pk         = tc.is_pk,
            is_fk         = tc.is_fk,
            confidence    = tc.confidence,
            feature_vec   = feature_vec,
            fk_ref_table  = tc.fk_ref_table if tc.is_fk else None,
            fk_ref_col    = tc.fk_ref_col   if tc.is_fk else None,
        )
        column_nodes.append(node)
        col_id_map[tc.col_id] = idx

    return column_nodes, col_id_map


# =============================================================================
# Edge construction
# =============================================================================

def _build_has_column_edges(
    table_nodes:    List[TableNode],
    column_nodes:   List[ColumnNode],
    table_id_map:   Dict[str, int],
) -> List[Tuple[int, int]]:
    """
    has_column edges: one edge per (table, column) pair.
    Direction: table_node → column_node.
    """
    edges = []
    for col_node in column_nodes:
        t_idx = table_id_map.get(col_node.table_id)
        if t_idx is not None:
            edges.append((t_idx, col_node.node_index))
    return edges


def _build_fk_to_edges(
    inference_result: InferenceResult,
    col_id_map:       Dict[str, int],
) -> List[Tuple[int, int]]:
    """
    fk_to edges: one edge per FK column → referenced PK column.
    Direction: fk_col_node → pk_col_node.
    Skips edges where either endpoint is not found in col_id_map.
    """
    edges     = []
    skipped   = 0

    # Build lookup: (table_name, col_name) → col_id
    name_to_col_id = {
        (tc.table_name, tc.col_name): tc.col_id
        for tc in inference_result.typed_columns
    }

    for tc in inference_result.typed_columns:
        if not tc.is_fk:
            continue
        if not tc.fk_ref_table or not tc.fk_ref_col:
            skipped += 1
            continue

        from_idx = col_id_map.get(tc.col_id)
        to_col_id = name_to_col_id.get((tc.fk_ref_table, tc.fk_ref_col))

        if to_col_id is None:
            skipped += 1
            continue

        to_idx = col_id_map.get(to_col_id)
        if from_idx is None or to_idx is None:
            skipped += 1
            continue

        edges.append((from_idx, to_idx))

    return edges


# =============================================================================
# PyTorch Geometric HeteroData construction (optional)
# =============================================================================

def _build_hetero_data(
    table_feature_matrix:  np.ndarray,
    column_feature_matrix: np.ndarray,
    has_column_edges:      List[Tuple[int, int]],
    fk_to_edges:           List[Tuple[int, int]],
) -> object:
    """
    Wraps numpy matrices into a torch_geometric HeteroData object.
    Returns None if torch_geometric is not available.
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None

    data = HeteroData()

    data["table"].x  = torch.tensor(table_feature_matrix,  dtype=torch.float)
    data["column"].x = torch.tensor(column_feature_matrix, dtype=torch.float)

    if has_column_edges:
        src = [e[0] for e in has_column_edges]
        dst = [e[1] for e in has_column_edges]
        data["table", "has_column", "column"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )

    if fk_to_edges:
        src = [e[0] for e in fk_to_edges]
        dst = [e[1] for e in fk_to_edges]
        data["column", "fk_to", "column"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )

    return data


# =============================================================================
# Public entry point
# =============================================================================

def run_reg_builder(
    inference_result: InferenceResult = None,
    verbose: bool = False,
) -> REGGraph:
    """
    Main entry point for Step 3.

    Parameters
    ----------
    inference_result : InferenceResult, optional
        Output of run_semantic_type_inference(). If None, runs inference internally.
    verbose : bool
        Print progress to stdout if True.

    Returns
    -------
    REGGraph
        Graph passed to relgt_encoder.py and persisted to schema/ for
        query-time subgraph RELGT encoding.
    """
    if inference_result is None:
        inference_result = run_semantic_type_inference(verbose=verbose)

    logger.debug("Building REG: %d typed columns, torch_geometric=%s",
                 len(inference_result.typed_columns), TORCH_GEOMETRIC_AVAILABLE)

    if verbose:
        print("[REGBuilder] Building Relational Entity Graph...")
        print(f"  Torch Geometric  : {'available' if TORCH_GEOMETRIC_AVAILABLE else 'NOT available — using numpy fallback'}")
        print(f"  Typed columns    : {len(inference_result.typed_columns)}")

    # ------------------------------------------------------------------
    # Build table nodes
    # ------------------------------------------------------------------
    table_nodes, table_id_map = _build_table_nodes(inference_result)

    # Row count lookup for column cardinality normalisation
    table_id_to_row_count = {n.table_id: n.row_count for n in table_nodes}

    # ------------------------------------------------------------------
    # Build column nodes
    # ------------------------------------------------------------------
    column_nodes, col_id_map = _build_column_nodes(
        inference_result, table_id_to_row_count
    )

    # ------------------------------------------------------------------
    # Build edges
    # ------------------------------------------------------------------
    has_column_edges = _build_has_column_edges(table_nodes, column_nodes, table_id_map)
    fk_to_edges      = _build_fk_to_edges(inference_result, col_id_map)

    # ------------------------------------------------------------------
    # Build feature matrices
    # ------------------------------------------------------------------
    table_feature_matrix  = np.stack(
        [n.feature_vec for n in table_nodes], axis=0
    ).astype(np.float32)

    column_feature_matrix = np.stack(
        [n.feature_vec for n in column_nodes], axis=0
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Optional: build HeteroData for PyTorch Geometric
    # ------------------------------------------------------------------
    hetero_data = _build_hetero_data(
        table_feature_matrix,
        column_feature_matrix,
        has_column_edges,
        fk_to_edges,
    )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    stats = {
        "num_table_nodes":      len(table_nodes),
        "num_column_nodes":     len(column_nodes),
        "num_has_column_edges": len(has_column_edges),
        "num_fk_to_edges":      len(fk_to_edges),
        "table_feat_dim":       table_feature_matrix.shape[1],
        "column_feat_dim":      column_feature_matrix.shape[1],
        "torch_geometric":      TORCH_GEOMETRIC_AVAILABLE,
    }

    if verbose:
        print(f"  Table nodes      : {stats['num_table_nodes']}  (feat_dim={stats['table_feat_dim']})")
        print(f"  Column nodes     : {stats['num_column_nodes']}  (feat_dim={stats['column_feat_dim']})")
        print(f"  has_column edges : {stats['num_has_column_edges']}")
        print(f"  fk_to edges      : {stats['num_fk_to_edges']}")
        print("[REGBuilder] Done.\n")

    logger.info(
        "REG built: %d table nodes, %d column nodes, %d has_column edges, %d FK edges",
        stats["num_table_nodes"], stats["num_column_nodes"],
        stats["num_has_column_edges"], stats["num_fk_to_edges"],
    )

    graph = REGGraph(
        table_nodes           = table_nodes,
        column_nodes          = column_nodes,
        has_column_edges      = has_column_edges,
        fk_to_edges           = fk_to_edges,
        table_feature_matrix  = table_feature_matrix,
        column_feature_matrix = column_feature_matrix,
        hetero_data           = hetero_data,
        col_id_to_node_index  = col_id_map,
        table_id_to_node_index = table_id_map,
        stats                 = stats,
    )


    return graph


# =============================================================================
# Smoke test — python ingestion/reg_builder.py
# =============================================================================

if __name__ == "__main__":
    graph = run_reg_builder(verbose=True)

    print("=" * 60)
    print("VEDA POC — REG Builder Output")
    print("=" * 60)
    print(f"  Table nodes      : {graph.stats['num_table_nodes']}")
    print(f"  Column nodes     : {graph.stats['num_column_nodes']}")
    print(f"  has_column edges : {graph.stats['num_has_column_edges']}")
    print(f"  fk_to edges      : {graph.stats['num_fk_to_edges']}")
    print(f"  Table feat dim   : {graph.stats['table_feat_dim']}")
    print(f"  Column feat dim  : {graph.stats['column_feat_dim']}")
    print(f"  Torch Geometric  : {graph.stats['torch_geometric']}")
    print()

    print("Table nodes:")
    for tn in graph.table_nodes:
        print(f"  [{tn.node_index}] {tn.table_name:<30} rows≈{tn.row_count:<8} cols={tn.col_count}")

    print("\nFK edges (column → column):")
    col_index = {n.node_index: n for n in graph.column_nodes}
    for (from_idx, to_idx) in graph.fk_to_edges:
        fc = col_index[from_idx]
        tc = col_index[to_idx]
        print(f"  {fc.table_name}.{fc.col_name}  →  {tc.table_name}.{tc.col_name}")

    print("\nFeature matrix shapes:")
    print(f"  table_feature_matrix  : {graph.table_feature_matrix.shape}")
    print(f"  column_feature_matrix : {graph.column_feature_matrix.shape}")