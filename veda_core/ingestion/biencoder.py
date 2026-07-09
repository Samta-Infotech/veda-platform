# ingestion/biencoder.py
# VEDA — Bi-encoder ingestion: embeds columns + tables into v2 pgvector stores
# Gate: RETRIEVAL_V2_ENABLED and BIENCODER_ENABLED
# Uses BAAI/bge-m3 dense (WP3) via ingestion/m3_encoder.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import time
from dataclasses import dataclass
from typing import Optional

from config import (
    BIENCODER_MODEL,
    BIENCODER_DIM,
    BIENCODER_DEVICE,
    BIENCODER_BATCH_SIZE,
    BIENCODER_PASSAGE_PREFIX,
    BIENCODER_COL_TABLE,
    BIENCODER_TABLE_TABLE,
    VEDA_INTERNAL_DB,
)
from ingestion.column_text import build_enriched_column_text

# ---------------------------------------------------------------------------
# Dense encoding is done by the shared BGE-M3 singleton (ingestion/m3_encoder.py) —
# the SAME model that produces the learned-sparse weights (WP3). No separate
# SentenceTransformer copy of a bge-large model is loaded anymore.
# ---------------------------------------------------------------------------


def _get_pg_conn():
    import psycopg2
    cfg = VEDA_INTERNAL_DB
    return psycopg2.connect(
        host=cfg["host"], port=cfg["port"], dbname=cfg["dbname"],
        user=cfg["user"], password=cfg["password"],
    )


def _ensure_v2_table(conn, table_name: str, dim: int):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id            SERIAL PRIMARY KEY,
                col_id        TEXT,
                col_name      TEXT,
                table_id      TEXT,
                table_name    TEXT,
                source_id     TEXT,
                semantic_type TEXT DEFAULT 'UNKNOWN',
                text          TEXT,
                embedding     vector({dim})
            )
        """)
        # Migration: add semantic_type to tables created before this column existed.
        try:
            cur.execute(f"""
                ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS semantic_type TEXT DEFAULT 'UNKNOWN'
            """)
        except Exception:
            pass
        try:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_emb_idx
                ON {table_name} USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)
        except Exception:
            pass
    conn.commit()


@dataclass
class BiEncoderResult:
    cols_embedded:   int   = 0
    tables_embedded: int   = 0
    backend:         str   = "pgvector"
    duration_s:      float = 0.0
    error:           Optional[str] = None


def _load_retrieval_docs() -> dict:
    """Rich per-column semantic text from the semantic model (DEFINITION / TERMS /
    SEARCH questions / …), keyed 'table.col'. Built FOR BGE embedding. Empty dict
    when the strategy is structural or the model is absent → caller falls back."""
    try:
        from config import EMBED_TEXT_STRATEGY, SEMANTIC_MODEL_FILE
        if EMBED_TEXT_STRATEGY == "structural":
            return {}
        import os
        import json as _json
        if os.path.exists(SEMANTIC_MODEL_FILE):
            return _json.load(open(SEMANTIC_MODEL_FILE)).get("retrieval_documents", {}) or {}
    except Exception:
        pass
    return {}


def _load_table_purposes() -> dict:
    """table_name -> business_purpose from the semantic model (semantic_layer_v2's
    LLM-authored one-sentence table summary). Empty dict when the model is absent —
    caller falls back to the bare column-list passage, same as before this existed."""
    try:
        from config import SEMANTIC_MODEL_FILE
        import os
        import json as _json
        if os.path.exists(SEMANTIC_MODEL_FILE):
            tables = _json.load(open(SEMANTIC_MODEL_FILE)).get("tables", {}) or {}
            return {n: (m.get("business_purpose") or "") for n, m in tables.items()}
    except Exception:
        pass
    return {}


def _passage_text(col, rdocs: dict) -> str:
    """Build the BGE passage text per config.EMBED_TEXT_STRATEGY. Always falls back
    to the structural string when no retrieval_document exists for the column."""
    from config import EMBED_TEXT_STRATEGY
    structural = build_enriched_column_text(
        col_name=col.col_name, table_name=col.table_name,
        semantic_type=col.semantic_type, is_pk=col.is_pk, is_fk=col.is_fk,
        style="minilm",
    )
    if EMBED_TEXT_STRATEGY == "structural":
        return structural
    doc = rdocs.get(f"{col.table_name}.{col.col_name}")
    if not doc:
        return structural                      # no doc → grounded fallback
    if EMBED_TEXT_STRATEGY == "doc":
        return doc
    return doc + "\n" + structural             # "hybrid" — rich NL + grounding tokens


def run_biencoder_ingestion(
    inference_result,
    source_id: str,
    verbose: bool = False,
) -> BiEncoderResult:
    """
    Embeds all columns and tables from inference_result into the v2 pgvector stores.
    Scoped delete by source_id before re-inserting (idempotent).
    Falls back gracefully if model or DB unavailable.
    """
    t0 = time.time()
    from ingestion import m3_encoder

    try:
        conn = _get_pg_conn()
    except Exception as e:
        warnings.warn(f"[BiEncoder] DB unavailable: {e}")
        return BiEncoderResult(error=f"db_unavailable: {e}")

    try:
        _ensure_v2_table(conn, BIENCODER_COL_TABLE,   BIENCODER_DIM)
        _ensure_v2_table(conn, BIENCODER_TABLE_TABLE, BIENCODER_DIM)

        # --- Column embeddings ---
        col_texts  = []
        col_metas  = []
        table_map: dict = {}  # table_id → {table_name, col_names, source_id}

        _rdocs = _load_retrieval_docs()
        for col in inference_result.typed_columns:
            text = BIENCODER_PASSAGE_PREFIX + _passage_text(col, _rdocs)
            col_texts.append(text)
            col_metas.append({
                "col_id":        col.col_id,
                "col_name":      col.col_name,
                "table_id":      col.table_id,
                "table_name":    col.table_name,
                "source_id":     source_id,
                "semantic_type": col.semantic_type,
                "text":          text,
            })
            # Accumulate for table-level text
            if col.table_id not in table_map:
                table_map[col.table_id] = {
                    "table_name": col.table_name,
                    "col_names":  [],
                    "source_id":  source_id,
                }
            table_map[col.table_id]["col_names"].append(col.col_name)

        if col_texts:
            from psycopg2.extras import execute_values
            col_embeddings = m3_encoder.encode_dense(col_texts)
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {BIENCODER_COL_TABLE} WHERE source_id = %s", (source_id,))
                # F2: batched insert — was one execute() per column (thousands
                # of round trips for a wide schema). Same rows, one statement.
                col_rows = [
                    (meta["col_id"], meta["col_name"], meta["table_id"],
                     meta["table_name"], meta["source_id"], meta["semantic_type"],
                     meta["text"], emb.tolist())
                    for meta, emb in zip(col_metas, col_embeddings)
                ]
                execute_values(
                    cur,
                    f"INSERT INTO {BIENCODER_COL_TABLE} "
                    f"(col_id, col_name, table_id, table_name, source_id, semantic_type, text, embedding) "
                    f"VALUES %s",
                    col_rows,
                    page_size=500,   # embedding vectors are large — smaller pages than value inserts
                )
            conn.commit()

        # --- Table embeddings ---
        tbl_texts = []
        tbl_metas = []
        _tbl_purposes = _load_table_purposes()
        for tid, info in table_map.items():
            col_list = ", ".join(info["col_names"][:20])
            purpose = _tbl_purposes.get(info["table_name"])
            # Bare column names retrieve poorly for a semantic query ("who's overdue on
            # rent" never lexically matches `columns tenant_id, due_date, amount`) — the
            # one-sentence business purpose gives the table embedding an actual semantic
            # anchor. Falls back to the old bare passage when the model has none.
            text = BIENCODER_PASSAGE_PREFIX + (
                f"{info['table_name']}: {purpose}. columns {col_list}" if purpose
                else f"{info['table_name']}: columns {col_list}")
            tbl_texts.append(text)
            tbl_metas.append({
                "col_id":     tid,
                "col_name":   info["table_name"],
                "table_id":   tid,
                "table_name": info["table_name"],
                "source_id":  source_id,
                "text":       text,
            })

        if tbl_texts:
            from psycopg2.extras import execute_values
            tbl_embeddings = m3_encoder.encode_dense(tbl_texts)
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {BIENCODER_TABLE_TABLE} WHERE source_id = %s", (source_id,))
                # F3: batched insert — table count is small (≤ a few hundred),
                # but batching costs nothing and keeps the pattern consistent.
                tbl_rows = [
                    (meta["col_id"], meta["col_name"], meta["table_id"],
                     meta["table_name"], meta["source_id"], meta["text"], emb.tolist())
                    for meta, emb in zip(tbl_metas, tbl_embeddings)
                ]
                execute_values(
                    cur,
                    f"INSERT INTO {BIENCODER_TABLE_TABLE} "
                    f"(col_id, col_name, table_id, table_name, source_id, text, embedding) "
                    f"VALUES %s",
                    tbl_rows,
                    page_size=500,
                )
            conn.commit()

        conn.close()
        return BiEncoderResult(
            cols_embedded=len(col_texts),
            tables_embedded=len(tbl_texts),
            duration_s=round(time.time() - t0, 2),
        )

    except Exception as e:
        warnings.warn(f"[BiEncoder] Ingestion failed: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return BiEncoderResult(error=str(e))
