# =============================================================================
# ingestion/graph_store.py
# VEDA POC — REG Graph Persistence + Subgraph Extraction
#
# Responsibility:
#   - save_graph()               : pickle REGGraph + col_id_to_idx to schema/
#   - load_graph()               : cached load; returns (graph, col_id_to_idx)
#   - graph_available()          : fast existence check
#   - extract_subgraph()         : builds a SubGraph from a list of col_ids
#   - get_query_structural_vec() : mean-pools RELGT output to a single vector
#
# SubGraph is accepted by run_relgt_on_graph() and _run_numpy_encoder() via
# duck typing — it exposes the same fields the numpy GNN reads.
# =============================================================================

import os
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import REG_GRAPH_PATH, COL_ID_IDX_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


_ABS_GRAPH_PATH = _abs(REG_GRAPH_PATH)
_ABS_IDX_PATH   = _abs(COL_ID_IDX_PATH)

_cached_graph: Optional[object]         = None
_cached_col_idx: Optional[Dict[str, int]] = None


@dataclass
class SubGraph:
    """
    Minimal graph object accepted by _run_numpy_encoder() and run_relgt_on_graph().
    Exposes the same attribute names the numpy GNN reads from REGGraph.
    """
    column_feature_matrix: np.ndarray          # (n_sub_cols, col_feat_dim)
    has_column_edges:       List[Tuple[int, int]]  # (orig_table_idx, new_col_idx)
    fk_to_edges:            List[Tuple[int, int]]  # (new_from_col, new_to_col)
    table_feature_matrix:   np.ndarray          # placeholder — shape[0] used as n_tables arg
    col_id_to_node_index:   Dict[str, int]      # col_id → new index in this subgraph


# =============================================================================
# Persistence
# =============================================================================

def save_graph(graph: object, col_id_to_idx: Dict[str, int]) -> None:
    """Pickles the full REGGraph and col_id → index mapping to disk."""
    os.makedirs(os.path.dirname(_ABS_GRAPH_PATH), exist_ok=True)
    with open(_ABS_GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(_ABS_IDX_PATH, "wb") as f:
        pickle.dump(col_id_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Graph saved: %s (%d columns)", _ABS_GRAPH_PATH, len(col_id_to_idx))
    print(f"[REG] Graph saved to {_ABS_GRAPH_PATH}")


def graph_available() -> bool:
    return os.path.exists(_ABS_GRAPH_PATH)


def load_graph() -> Tuple[Optional[object], Optional[Dict[str, int]]]:
    """
    Returns (graph, col_id_to_idx) with in-process caching.
    Returns (None, None) if no graph has been saved or load fails.
    """
    global _cached_graph, _cached_col_idx
    if _cached_graph is not None:
        return _cached_graph, _cached_col_idx
    if not graph_available():
        return None, None
    try:
        with open(_ABS_GRAPH_PATH, "rb") as f:
            _cached_graph = pickle.load(f)
        with open(_ABS_IDX_PATH, "rb") as f:
            _cached_col_idx = pickle.load(f)
        logger.info("Graph loaded: %d columns", len(_cached_col_idx))
        return _cached_graph, _cached_col_idx
    except Exception as e:
        logger.warning("Graph load failed: %s", e)
        return None, None


# =============================================================================
# Subgraph extraction
# =============================================================================

def extract_subgraph(
    graph:          object,
    col_id_to_idx:  Dict[str, int],
    query_col_ids:  List[str],
) -> Optional[SubGraph]:
    """
    Extracts a subgraph from the full schema graph containing only the columns
    in query_col_ids (col_ids not found in the graph are silently skipped).

    Index remapping:
      - column indices are remapped to 0 … len(valid_col_ids)-1
      - table indices in has_column_edges are kept as original (used only as
        grouping keys in _build_adjacency — the actual value does not matter,
        only that co-table columns share the same key)
      - fk_to_edges are remapped; edges where either endpoint is outside
        query_col_ids are dropped

    Returns None if no valid col_ids were found.
    """
    valid_col_ids = [cid for cid in query_col_ids if cid in col_id_to_idx]
    if not valid_col_ids:
        return None

    old_indices = [col_id_to_idx[cid] for cid in valid_col_ids]
    old_to_new  = {old: new for new, old in enumerate(old_indices)}

    new_col_feat = graph.column_feature_matrix[old_indices]

    new_has_col = [
        (t_idx, old_to_new[c_idx])
        for (t_idx, c_idx) in graph.has_column_edges
        if c_idx in old_to_new
    ]

    new_fk = [
        (old_to_new[f], old_to_new[t])
        for (f, t) in graph.fk_to_edges
        if f in old_to_new and t in old_to_new
    ]

    # table_feature_matrix is only used as shape[0] (n_tables) in the GNN call;
    # passing a 1-row slice is safe and avoids index-out-of-bounds on subgraphs
    # that don't include all original tables.
    tbl_feat = graph.table_feature_matrix[:1]

    return SubGraph(
        column_feature_matrix = new_col_feat,
        has_column_edges      = new_has_col,
        fk_to_edges           = new_fk,
        table_feature_matrix  = tbl_feat,
        col_id_to_node_index  = {cid: new for new, cid in enumerate(valid_col_ids)},
    )


# =============================================================================
# Aggregation helper
# =============================================================================

def get_query_structural_vec(subgraph: SubGraph, relgt_output: np.ndarray) -> np.ndarray:
    """
    Mean-pools the (n_sub_cols, 256) RELGT output to a single (256,) vector,
    then L2-normalises. Used as the structural component of the hybrid query vector.
    """
    vec  = relgt_output.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
