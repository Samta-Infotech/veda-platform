"""
ingestion/kuzu_store.py
Stores and queries the REG graph using Kùzu embedded graph database.
Used at query time for subgraph extraction.
Falls back gracefully if kuzu not installed.
"""
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

KUZU_DB_PATH = "schema/kuzu_graph"

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

_db   = None
_conn = None


def kuzu_available() -> bool:
    return KUZU_AVAILABLE and os.path.exists(KUZU_DB_PATH)


def _get_conn():
    global _db, _conn
    if _conn is not None:
        return _conn
    if not KUZU_AVAILABLE:
        return None
    _db   = kuzu.Database(KUZU_DB_PATH)
    _conn = kuzu.Connection(_db)
    return _conn


def save_graph_to_kuzu(graph) -> bool:
    """
    Persist REGGraph to Kùzu.

    Returns True on success, False on failure.
    """
    if not KUZU_AVAILABLE:
        logger.info("kuzu not installed — skipping graph save")
        return False

    # Let Kùzu create its own directory — pre-creating an empty dir causes
    # "Database path cannot be a directory" on Kùzu 0.11+
    os.makedirs(os.path.dirname(KUZU_DB_PATH), exist_ok=True)
    try:
        db   = kuzu.Database(KUZU_DB_PATH)
        conn = kuzu.Connection(db)

        for stmt in [
            "DROP TABLE IF EXISTS FK_TO",
            "DROP TABLE IF EXISTS HAS_COLUMN",
            "DROP TABLE IF EXISTS ColNode",
            "DROP TABLE IF EXISTS TableNode",
        ]:
            try:
                conn.execute(stmt)
            except Exception:
                pass

        conn.execute("""
            CREATE NODE TABLE ColNode(
                col_id STRING,
                col_name STRING,
                table_id STRING,
                table_name STRING,
                semantic_type STRING,
                is_pk BOOLEAN,
                is_fk BOOLEAN,
                node_index INT64,
                PRIMARY KEY(col_id)
            )
        """)
        conn.execute("""
            CREATE NODE TABLE TableNode(
                table_id STRING,
                table_name STRING,
                node_index INT64,
                PRIMARY KEY(table_id)
            )
        """)
        conn.execute("CREATE REL TABLE HAS_COLUMN(FROM TableNode TO ColNode)")
        conn.execute("CREATE REL TABLE FK_TO(FROM ColNode TO ColNode)")

        for tn in graph.table_nodes:
            conn.execute(
                "CREATE (:TableNode {table_id: $tid, table_name: $tname, node_index: $idx})",
                {"tid": tn.table_id, "tname": tn.table_name, "idx": tn.node_index}
            )

        for cn in graph.column_nodes:
            conn.execute(
                """CREATE (:ColNode {col_id: $cid, col_name: $cname,
                    table_id: $tid, table_name: $tname,
                    semantic_type: $st, is_pk: $pk, is_fk: $fk,
                    node_index: $idx})""",
                {"cid": cn.col_id, "cname": cn.col_name,
                 "tid": cn.table_id, "tname": cn.table_name,
                 "st": cn.semantic_type, "pk": cn.is_pk,
                 "fk": cn.is_fk, "idx": cn.node_index}
            )

        table_idx_to_id = {tn.node_index: tn.table_id for tn in graph.table_nodes}
        col_idx_to_id   = {cn.node_index: cn.col_id   for cn in graph.column_nodes}

        for (t_idx, c_idx) in graph.has_column_edges:
            tid = table_idx_to_id.get(t_idx)
            cid = col_idx_to_id.get(c_idx)
            if tid and cid:
                conn.execute(
                    "MATCH (t:TableNode {table_id: $tid}), (c:ColNode {col_id: $cid}) "
                    "CREATE (t)-[:HAS_COLUMN]->(c)",
                    {"tid": tid, "cid": cid}
                )

        for (from_idx, to_idx) in graph.fk_to_edges:
            from_id = col_idx_to_id.get(from_idx)
            to_id   = col_idx_to_id.get(to_idx)
            if from_id and to_id:
                conn.execute(
                    "MATCH (a:ColNode {col_id: $from_id}), (b:ColNode {col_id: $to_id}) "
                    "CREATE (a)-[:FK_TO]->(b)",
                    {"from_id": from_id, "to_id": to_id}
                )

        n_cols   = len(graph.column_nodes)
        n_tables = len(graph.table_nodes)
        n_fk     = len(graph.fk_to_edges)
        logger.info("Saved: %d tables, %d columns, %d FK edges → %s", n_tables, n_cols, n_fk, KUZU_DB_PATH)
        return True

    except Exception as e:
        logger.error("Save failed: %s", e)
        return False


@dataclass
class _Subgraph:
    column_feature_matrix: np.ndarray
    table_feature_matrix:  np.ndarray
    has_column_edges:      list
    fk_to_edges:           list
    col_id_to_node_index:  dict
    column_nodes:          list
    table_nodes:           list


def get_subgraph_for_cols(
    query_col_ids: List[str],
    graph,
    col_id_to_idx: Dict[str, int],
) -> Optional[_Subgraph]:
    """
    Extract subgraph for given col_ids.

    Tries Kùzu first (Cypher for FK-connected neighbours).
    Falls back to in-memory graph filtering if Kùzu unavailable.
    """
    if KUZU_AVAILABLE and os.path.exists(KUZU_DB_PATH):
        return _subgraph_from_kuzu(query_col_ids, graph, col_id_to_idx)
    return _subgraph_from_memory(query_col_ids, graph, col_id_to_idx)


def _subgraph_from_kuzu(
    query_col_ids: List[str],
    graph,
    col_id_to_idx: Dict[str, int],
) -> Optional[_Subgraph]:
    """Use Kùzu Cypher to find FK-connected columns, then build subgraph."""
    try:
        conn = _get_conn()
        if conn is None:
            return _subgraph_from_memory(query_col_ids, graph, col_id_to_idx)

        col_id_list = ", ".join(f'"{cid}"' for cid in query_col_ids)
        result = conn.execute(f"""
            MATCH (c:ColNode)
            WHERE c.col_id IN [{col_id_list}]
            OPTIONAL MATCH (c)-[:FK_TO]->(fk:ColNode)
            OPTIONAL MATCH (c)<-[:FK_TO]-(rfk:ColNode)
            RETURN c.col_id AS col_id,
                   fk.col_id AS fk_col_id,
                   rfk.col_id AS rfk_col_id
        """)

        relevant_ids = set(query_col_ids)
        rows = result.get_as_df() if hasattr(result, 'get_as_df') else []
        if hasattr(rows, 'iterrows'):
            for _, row in rows.iterrows():
                if row.get('fk_col_id'):
                    relevant_ids.add(row['fk_col_id'])
                if row.get('rfk_col_id'):
                    relevant_ids.add(row['rfk_col_id'])

        return _subgraph_from_memory(list(relevant_ids), graph, col_id_to_idx)

    except Exception as e:
        logger.warning("Kuzu subgraph query failed (%s) — using memory fallback", e)
        return _subgraph_from_memory(query_col_ids, graph, col_id_to_idx)


def _subgraph_from_memory(
    query_col_ids: List[str],
    graph,
    col_id_to_idx: Dict[str, int],
) -> Optional[_Subgraph]:
    """Build subgraph from in-memory REGGraph with 1-hop FK expansion."""
    if not query_col_ids or not graph:
        return None

    global_col_indices = set()
    for cid in query_col_ids:
        if cid in col_id_to_idx:
            global_col_indices.add(col_id_to_idx[cid])

    if not global_col_indices:
        return None

    for (from_idx, to_idx) in graph.fk_to_edges:
        if from_idx in global_col_indices:
            global_col_indices.add(to_idx)
        if to_idx in global_col_indices:
            global_col_indices.add(from_idx)

    global_col_indices = sorted(global_col_indices)
    col_to_table = {}
    for (t_idx, c_idx) in graph.has_column_edges:
        col_to_table[c_idx] = t_idx
    global_table_indices = sorted(set(
        col_to_table[c] for c in global_col_indices if c in col_to_table
    ))

    col_global_to_local   = {g: l for l, g in enumerate(global_col_indices)}
    table_global_to_local = {g: l for l, g in enumerate(global_table_indices)}

    sub_col_feats = graph.column_feature_matrix[global_col_indices]
    sub_col_nodes = [graph.column_nodes[i] for i in global_col_indices]
    col_id_to_node_index = {
        cn.col_id: col_global_to_local[cn.node_index]
        for cn in sub_col_nodes
    }

    if global_table_indices:
        sub_table_feats = graph.table_feature_matrix[global_table_indices]
        sub_table_nodes = [graph.table_nodes[i] for i in global_table_indices]
    else:
        sub_table_feats = np.zeros((1, graph.table_feature_matrix.shape[1]), dtype=np.float32)
        sub_table_nodes = []

    sub_has_col = [
        (table_global_to_local[t_idx], col_global_to_local[c_idx])
        for (t_idx, c_idx) in graph.has_column_edges
        if c_idx in col_global_to_local and t_idx in table_global_to_local
    ]

    sub_fk = [
        (col_global_to_local[f], col_global_to_local[t])
        for (f, t) in graph.fk_to_edges
        if f in col_global_to_local and t in col_global_to_local
    ]

    return _Subgraph(
        column_feature_matrix = sub_col_feats,
        table_feature_matrix  = sub_table_feats,
        has_column_edges      = sub_has_col,
        fk_to_edges           = sub_fk,
        col_id_to_node_index  = col_id_to_node_index,
        column_nodes          = sub_col_nodes,
        table_nodes           = sub_table_nodes,
    )
