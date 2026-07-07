"""L4 INDEX · learned-sparse index (WP3 — replaces the BM25 index stage).

Encodes every column/table passage with BGE-M3's learned-sparse (lexical) head and
persists the token→weight maps to the internal DB (``column_sparse_v1`` /
``table_sparse_v1``) with the usual scoped delete-then-insert. The query tier's
``retrieval/sparse_ranker.py`` loads these rows at warm and builds an in-memory inverted
index — the learned-sparse replacement for BM25 as Signal 2, and the sparse half of the
WP4 table prior.

Passage text is built by the SAME helpers the BGE dense biencoder uses (``_passage_text``
/ the table-text template), so the dense and sparse signals describe the identical text.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from config import (
    COLUMN_SPARSE_TABLE, TABLE_SPARSE_TABLE, BIENCODER_PASSAGE_PREFIX,
)


@dataclass
class SparseIndexResult:
    cols_indexed:   int = 0
    tables_indexed: int = 0
    duration_s:     float = 0.0
    error:          Optional[str] = None


def _ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {COLUMN_SPARSE_TABLE} (
                col_id    TEXT,
                source_id TEXT,
                table_id  TEXT,
                weights   JSONB,
                PRIMARY KEY (source_id, col_id)
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_SPARSE_TABLE} (
                table_id   TEXT,
                table_name TEXT,
                source_id  TEXT,
                weights    JSONB,
                PRIMARY KEY (source_id, table_id)
            )
        """)
    conn.commit()


def build_sparse_index(inference_result, source_id: str, verbose: bool = False) -> SparseIndexResult:
    """Encode column + table passages with M3 sparse and persist per (source, key).

    Idempotent: scoped DELETE by source_id before INSERT, mirroring the biencoder store.
    Non-fatal — returns an error-carrying result rather than raising, so a missing model
    or DB degrades the stage without failing the ingest.
    """
    t0 = time.time()
    # Reuse the biencoder's exact passage-text construction so dense + sparse agree.
    from ingestion.biencoder import _load_retrieval_docs, _passage_text, _get_pg_conn
    from ingestion import m3_encoder

    try:
        conn = _get_pg_conn()
    except Exception as e:
        return SparseIndexResult(error=f"db_unavailable: {e}")

    try:
        _ensure_tables(conn)

        rdocs = _load_retrieval_docs()
        col_ids, col_tids, col_texts = [], [], []
        table_map: dict = {}
        for col in inference_result.typed_columns:
            text = BIENCODER_PASSAGE_PREFIX + _passage_text(col, rdocs)
            # Key by the fusion identity "table.col" (same key space BM25 used via
            # retrieval_documents), NOT the UUID col.col_id — so sparse_ranker returns
            # keys the RRF merger fuses with the subgraph/fk/value signals. table_id
            # keeps the UUID for the WP4 table-prior join.
            col_ids.append(f"{col.table_name}.{col.col_name}")
            col_tids.append(col.table_id)
            col_texts.append(text)
            if col.table_id not in table_map:
                table_map[col.table_id] = {"table_name": col.table_name, "col_names": []}
            table_map[col.table_id]["col_names"].append(col.col_name)

        n_cols = 0
        if col_texts:
            col_weights = m3_encoder.encode_sparse(col_texts)
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {COLUMN_SPARSE_TABLE} WHERE source_id = %s", (source_id,))
                for cid, tid, w in zip(col_ids, col_tids, col_weights):
                    cur.execute(
                        f"INSERT INTO {COLUMN_SPARSE_TABLE} (col_id, source_id, table_id, weights) "
                        f"VALUES (%s, %s, %s, %s)",
                        (cid, source_id, tid, json.dumps(w)),
                    )
            conn.commit()
            n_cols = len(col_texts)

        # --- Table passages (same template the biencoder uses) ---
        tbl_ids, tbl_names, tbl_texts = [], [], []
        for tid, info in table_map.items():
            col_list = ", ".join(info["col_names"][:20])
            tbl_ids.append(tid)
            tbl_names.append(info["table_name"])
            tbl_texts.append(BIENCODER_PASSAGE_PREFIX + f"{info['table_name']}: columns {col_list}")

        n_tables = 0
        if tbl_texts:
            tbl_weights = m3_encoder.encode_sparse(tbl_texts)
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {TABLE_SPARSE_TABLE} WHERE source_id = %s", (source_id,))
                for tid, tname, w in zip(tbl_ids, tbl_names, tbl_weights):
                    cur.execute(
                        f"INSERT INTO {TABLE_SPARSE_TABLE} (table_id, table_name, source_id, weights) "
                        f"VALUES (%s, %s, %s, %s)",
                        (tid, tname, source_id, json.dumps(w)),
                    )
            conn.commit()
            n_tables = len(tbl_texts)

        conn.close()
        if verbose:
            print(f"  [sparse_index] {n_cols} cols, {n_tables} tables → "
                  f"{COLUMN_SPARSE_TABLE}/{TABLE_SPARSE_TABLE}")
        return SparseIndexResult(cols_indexed=n_cols, tables_indexed=n_tables,
                                 duration_s=round(time.time() - t0, 2))
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        return SparseIndexResult(error=str(e))
