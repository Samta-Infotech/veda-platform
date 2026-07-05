# query/retrieval_v2.py
# VEDA — V2 retrieval: first-stage bi-encoder + cross-encoder rerank + bidirectional merge
# Gate: RETRIEVAL_V2_ENABLED and BIENCODER_ENABLED

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    BIENCODER_MODEL,
    BIENCODER_DIM,
    BIENCODER_DEVICE,
    BIENCODER_BATCH_SIZE,
    BIENCODER_QUERY_PREFIX,
    BIENCODER_CANDIDATE_COLS,
    BIENCODER_CANDIDATE_TABLES,
    BIENCODER_COL_TABLE,
    BIENCODER_TABLE_TABLE,
    RERANKER_ENABLED,
    RERANKER_TOP_COLS,
    RERANKER_TOP_TABLES,
    BIDIRECTIONAL_ENABLED,
    VEDA_INTERNAL_DB,
)
try:
    from config import GRAPH_EXPAND_ENABLED, GRAPH_EXPAND_MAX
except ImportError:                       # graph feature optional / older config
    GRAPH_EXPAND_ENABLED, GRAPH_EXPAND_MAX = False, 12
from ingestion.vector_store import RetrievalResult

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------
BIENCODER_QUERY_AVAILABLE = False
_QUERY_ENCODER = None

try:
    from sentence_transformers import SentenceTransformer
    BIENCODER_QUERY_AVAILABLE = True
except ImportError:
    pass


def _get_query_encoder():
    global _QUERY_ENCODER
    if _QUERY_ENCODER is not None:
        return _QUERY_ENCODER
    if not BIENCODER_QUERY_AVAILABLE:
        return None
    try:
        # Reuse the ONE shared BGE instance when it's the same model/device — avoid a
        # duplicate ~1.3GB load of bge-large (same model as veda.runtime._get_bge).
        try:
            from config import BGE_MODEL_NAME as _shared_name
            if BIENCODER_MODEL == _shared_name and str(BIENCODER_DEVICE) == "cpu":
                from veda.runtime import _get_bge
                _QUERY_ENCODER = _get_bge()
                return _QUERY_ENCODER
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        _QUERY_ENCODER = SentenceTransformer(BIENCODER_MODEL, device=BIENCODER_DEVICE,
                                             local_files_only=True)
        return _QUERY_ENCODER
    except Exception as e:
        warnings.warn(f"[RetrievalV2] Could not load bi-encoder '{BIENCODER_MODEL}': {e}")
        return None


def _get_pg_conn():
    import psycopg2
    cfg = VEDA_INTERNAL_DB
    return psycopg2.connect(
        host=cfg["host"], port=cfg["port"], dbname=cfg["dbname"],
        user=cfg["user"], password=cfg["password"],
    )


def _cosine_search_v2(
    conn,
    table: str,
    query_vec: list,
    top_k: int,
    source_ids: Optional[List[str]] = None,
) -> List[RetrievalResult]:
    """Cosine similarity search against a v2 pgvector table."""
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    if source_ids:
        placeholders = ",".join(["%s"] * len(source_ids))
        source_filter = f"WHERE source_id IN ({placeholders})"
        params = [vec_str] + list(source_ids) + [vec_str, top_k]
    else:
        source_filter = ""
        params = [vec_str, vec_str, top_k]

    sql = f"""
        SELECT col_id, col_name, table_id, table_name, source_id, semantic_type,
               1 - (embedding <=> %s::vector) AS similarity
        FROM {table}
        {source_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        RetrievalResult(
            col_id        = row[0],
            col_name      = row[1],
            table_id      = row[2],
            table_name    = row[3],
            source_id     = row[4],
            semantic_type = row[5] or "UNKNOWN",
            similarity    = round(float(row[6]), 6),
        )
        for row in rows
    ]


@dataclass
class FirstStageResult:
    candidate_columns: List[RetrievalResult] = field(default_factory=list)
    candidate_tables:  List[RetrievalResult] = field(default_factory=list)
    stats: dict                              = field(default_factory=dict)


def first_stage_retrieve(
    query:      str,
    source_ids: Optional[List[str]] = None,
    verbose:    bool = False,
) -> FirstStageResult:
    """
    First-stage bi-encoder retrieval: recall-oriented candidate set.
    Query and column/table both encoded by the same bi-encoder model.
    """
    encoder = _get_query_encoder()
    if encoder is None:
        return FirstStageResult(stats={"error": "encoder_unavailable"})

    try:
        conn = _get_pg_conn()
    except Exception as e:
        warnings.warn(f"[RetrievalV2] DB unavailable: {e}")
        return FirstStageResult(stats={"error": f"db: {e}"})

    try:
        query_text = BIENCODER_QUERY_PREFIX + query
        q_vec = encoder.encode(
            [query_text],
            normalize_embeddings=True,
            device=BIENCODER_DEVICE,
            show_progress_bar=False,
        )[0].tolist()

        col_candidates = _cosine_search_v2(
            conn, BIENCODER_COL_TABLE, q_vec, BIENCODER_CANDIDATE_COLS, source_ids
        )
        tbl_candidates = _cosine_search_v2(
            conn, BIENCODER_TABLE_TABLE, q_vec, BIENCODER_CANDIDATE_TABLES, source_ids
        )
        conn.close()

        if verbose:
            print(f"  [RetrievalV2] First-stage cols   : {len(col_candidates)}")
            print(f"  [RetrievalV2] First-stage tables : {len(tbl_candidates)}")

        return FirstStageResult(
            candidate_columns = col_candidates,
            candidate_tables  = tbl_candidates,
            stats={
                "col_candidates":   len(col_candidates),
                "table_candidates": len(tbl_candidates),
            },
        )
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        err_msg = str(e)
        if "does not exist" in err_msg:
            raise RuntimeError(
                f"v2 stores not populated — run ingestion first. Detail: {err_msg}"
            )
        warnings.warn(f"[RetrievalV2] First stage failed: {e}")
        return FirstStageResult(stats={"error": err_msg})


def graph_expand(
    query:             str,
    candidate_columns: List[RetrievalResult],
    candidate_tables:  List[RetrievalResult],
    source_ids:        Optional[List[str]] = None,
    trace=None,
    verbose:           bool = False,
) -> List[RetrievalResult]:
    """Unified-graph recall booster (Phase 4). PURELY ADDITIVE: returns the original
    candidate columns UNION graph-suggested columns the bi-encoder may have missed
    (synonym/alias resolution + FK-neighbour columns). The cross-encoder reranker still
    decides the final cut, so this can only raise recall — never drop a candidate.

    Flag-guarded (GRAPH_EXPAND_ENABLED) and fully try/except'd: on ANY failure it
    returns candidate_columns unchanged → zero regression risk.
    """
    if not GRAPH_EXPAND_ENABLED:
        return candidate_columns
    try:
        from graph.query_graph import get_graph
        g = get_graph()
        if g is None:
            return candidate_columns

        have = {(c.table_name, c.col_name) for c in candidate_columns}

        # 1) synonym/alias resolution of query tokens → column node ids
        import re as _re
        tokens = [t for t in _re.findall(r"[a-zA-Z_]+", query.lower()) if len(t) > 2]
        seeds, suggested = [], []   # suggested = [(table, col)]
        for tok in tokens:
            cids = g.resolve_term(tok)
            if cids:
                seeds.append(tok)
            for cid in cids:
                node = g.node(cid)
                if node and "." in node["name"]:
                    t, c = node["name"].split(".", 1)
                    if (t, c) not in have:
                        suggested.append((t, c))

        # 2) FK-neighbour columns of the already-retrieved tables (join reach)
        for t in {c.table_name for c in candidate_columns}:
            for rel_t in g.get_related_tables(t):
                for cid in g.get_related_columns(rel_t):
                    node = g.node(cid)
                    if node and "." in node["name"]:
                        rt, rc = node["name"].split(".", 1)
                        if (rt, rc) not in have:
                            suggested.append((rt, rc))

        # dedupe, cap (token/latency bound)
        seen, capped = set(), []
        for tc in suggested:
            if tc not in seen:
                seen.add(tc); capped.append(tc)
            if len(capped) >= GRAPH_EXPAND_MAX:
                break
        if not capped:
            if trace is not None:
                trace.set("graph_expansion", seeds=seeds, added=[], note="no new columns")
            return candidate_columns

        # 3) resolve real vector-store ids for the suggested columns (one bounded query)
        added = _fetch_columns_by_name(capped, source_ids)
        if not added:
            return candidate_columns
        # neutral similarity → the reranker re-scores them on real text
        for c in added:
            c.similarity = 0.0

        if verbose:
            print(f"  [GraphExpand] seeds={seeds} +{len(added)} cols "
                  f"({', '.join(f'{c.table_name}.{c.col_name}' for c in added[:6])})")
        if trace is not None:
            trace.set("graph_expansion",
                      seeds=seeds,
                      synonyms={t: g.get_synonyms(t)[:6] for t in seeds},
                      added=[f"{c.table_name}.{c.col_name}" for c in added])

        return candidate_columns + added
    except Exception as e:        # never break retrieval
        warnings.warn(f"[GraphExpand] skipped: {e}")
        return candidate_columns


def _fetch_columns_by_name(
    pairs:      List[tuple],
    source_ids: Optional[List[str]] = None,
) -> List[RetrievalResult]:
    """Look up (table_name, col_name) pairs in the bi-encoder column store, returning
    RetrievalResult rows. Bounded single query; [] on any failure."""
    if not pairs:
        return []
    try:
        conn = _get_pg_conn()
    except Exception:
        return []
    try:
        conds = " OR ".join(["(table_name = %s AND col_name = %s)"] * len(pairs))
        params: List = []
        for t, c in pairs:
            params += [t, c]
        src = ""
        if source_ids:
            src = " AND source_id IN (" + ",".join(["%s"] * len(source_ids)) + ")"
            params += list(source_ids)
        sql = (f"SELECT col_id, col_name, table_id, table_name, source_id, semantic_type "
               f"FROM {BIENCODER_COL_TABLE} WHERE ({conds}){src}")
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        conn.close()
        return [RetrievalResult(col_id=r[0], col_name=r[1], table_id=r[2],
                                table_name=r[3], source_id=r[4],
                                semantic_type=r[5] or "UNKNOWN", similarity=0.0)
                for r in rows]
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []


def retrieve_v2(
    query:      str,
    source_ids: Optional[List[str]] = None,
    verbose:    bool = False,
    trace=None,
):
    """
    Full two-stage retrieve: first-stage bi-encoder -> graph expand -> cross-encoder rerank.
    When BIDIRECTIONAL_ENABLED, also does table-first retrieval and merges.
    Returns (cols, tables) as List[RetrievalResult].
    """
    from query.reranker import rerank_columns, rerank_tables, RERANKER_AVAILABLE

    fs = first_stage_retrieve(query, source_ids=source_ids, verbose=verbose)

    # Phase 4 — unified-graph recall booster (additive; reranker still cuts).
    fs.candidate_columns = graph_expand(
        query, fs.candidate_columns, fs.candidate_tables,
        source_ids=source_ids, trace=trace, verbose=verbose,
    )

    if RERANKER_ENABLED and RERANKER_AVAILABLE:
        cols   = rerank_columns(query, fs.candidate_columns, RERANKER_TOP_COLS,   verbose=verbose)
        tables = rerank_tables( query, fs.candidate_tables,  RERANKER_TOP_TABLES, verbose=verbose)
    else:
        cols   = fs.candidate_columns[:RERANKER_TOP_COLS]
        tables = fs.candidate_tables[:RERANKER_TOP_TABLES]

    if BIDIRECTIONAL_ENABLED and tables:
        # Table-first: for each top table, pull its columns and rerank
        try:
            conn = _get_pg_conn()
            encoder = _get_query_encoder()
            if encoder is not None:
                query_text = BIENCODER_QUERY_PREFIX + query
                q_vec = encoder.encode(
                    [query_text], normalize_embeddings=True,
                    device=BIENCODER_DEVICE, show_progress_bar=False,
                )[0].tolist()

                table_first_cols = []
                for tbl in tables:
                    vec_str = "[" + ",".join(str(v) for v in q_vec) + "]"
                    with conn.cursor() as cur:
                        cur.execute(
                            f"SELECT col_id, col_name, table_id, table_name, source_id, semantic_type, "
                            f"1 - (embedding <=> %s::vector) AS similarity "
                            f"FROM {BIENCODER_COL_TABLE} WHERE table_id = %s "
                            f"ORDER BY embedding <=> %s::vector LIMIT %s",
                            (vec_str, tbl.table_id, vec_str, BIENCODER_CANDIDATE_COLS)
                        )
                        for row in cur.fetchall():
                            table_first_cols.append(RetrievalResult(
                                col_id=row[0], col_name=row[1], table_id=row[2],
                                table_name=row[3], source_id=row[4],
                                semantic_type=row[5] or "UNKNOWN",
                                similarity=round(float(row[6]), 6),
                            ))
                conn.close()

                if table_first_cols and (RERANKER_ENABLED and RERANKER_AVAILABLE):
                    from query.reranker import rerank_columns as _rc
                    table_first_reranked = _rc(query, table_first_cols, RERANKER_TOP_COLS, verbose=False)
                else:
                    table_first_reranked = table_first_cols[:RERANKER_TOP_COLS]

                # Merge: union, dedupe by col_id, keep higher score
                existing = {c.col_id: c for c in cols}
                for c in table_first_reranked:
                    if c.col_id not in existing or c.similarity > existing[c.col_id].similarity:
                        existing[c.col_id] = c
                cols = sorted(existing.values(), key=lambda x: x.similarity, reverse=True)[:RERANKER_TOP_COLS]
        except Exception as e:
            warnings.warn(f"[RetrievalV2] Bidirectional merge failed: {e}")

    # Value-matched column injection
    # Force-inject columns matched by value expansion
    # These bypass similarity threshold — exact value matches
    try:
        import re as _re
        import ingestion.value_sampler as _vs
        if not _vs._VALUE_STORE:
            _vs.rebuild_value_index_from_db()

        q_tokens = _re.findall(r'\w+', query.lower())
        phrases  = list(q_tokens)
        for i in range(len(q_tokens) - 1):
            phrases.append(q_tokens[i] + " " + q_tokens[i+1])

        existing_col_ids = {c.col_id for c in cols}
        to_inject = []

        # Also check substring matches in value index
        matched_col_ids = set()
        for phrase in phrases:
            if len(phrase) < 4:
                continue
            # Exact match
            if phrase in _vs._VALUE_INDEX:
                for col_id in _vs._VALUE_INDEX[phrase]:
                    matched_col_ids.add((col_id, phrase))
            # Substring match — phrase appears in stored value
            for val, col_ids in _vs._VALUE_INDEX.items():
                if phrase in val and len(phrase) >= 5:
                    for col_id in col_ids:
                        matched_col_ids.add((col_id, phrase))

        for col_id, phrase in matched_col_ids:
            if col_id in existing_col_ids:
                continue
            sc = _vs._VALUE_STORE.get(col_id)
            if not sc:
                continue
            if sc.semantic_type not in ("CATEGORY", "IDENTIFIER", "FREE_TEXT"):
                continue
            to_inject.append(RetrievalResult(
                col_id        = col_id,
                col_name      = sc.col_name,
                table_id      = sc.table_id,
                table_name    = sc.table_name,
                source_id     = source_ids[0] if source_ids else "primary_db",
                semantic_type = sc.semantic_type,
                similarity    = 0.85,
            ))
            existing_col_ids.add(col_id)

        if to_inject:
            cols = cols + to_inject
            cols = sorted(cols, key=lambda x: x.similarity, reverse=True)
            if verbose:
                print(f"  [RetrievalV2] Value-injected: {[c.col_name for c in to_inject]}")
    except Exception as _ve:
        logger.debug("Value inject failed: %s", _ve)

    return cols, tables
