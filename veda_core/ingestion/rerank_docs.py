"""L4 INDEX · precomputed rerank documents (Q-4).

Materialises the exact cross-encoder document text per column/table candidate at
ingestion, so the reranker reads ready-made strings instead of re-stitching
`_col_text`/`_table_text` (gloss + type + sampled values, and a `SELECT name FROM
graph_nodes` per table) on every query. The cross-encoder scoring itself is
unchanged — only document assembly moves earlier.

Column text reuses the semantic model's ``retrieval_documents`` (the same enriched
vocabulary used at indexing time). Table text is "<name>: columns c1, c2, …".
Pure transform of the on-disk semantic model → no source-DB touch, non-fatal.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional


def _index_path() -> str:
    from config import artifact_path
    return artifact_path("veda_rerank_docs.json")


def build_rerank_docs(source_id: str = "", verbose: bool = False) -> Dict:
    from config import SEMANTIC_MODEL_FILE

    if not os.path.exists(SEMANTIC_MODEL_FILE):
        raise FileNotFoundError(f"semantic model not found: {SEMANTIC_MODEL_FILE}")
    with open(SEMANTIC_MODEL_FILE) as f:
        sm = json.load(f)

    col_docs = dict(sm.get("retrieval_documents", {}))   # col_id -> enriched text

    # table_id -> "name: columns a, b, c" (mirrors reranker._table_text)
    table_docs: Dict[str, str] = {}
    for tid, tinfo in (sm.get("tables", {}) or {}).items():
        name = tinfo.get("table_name") or tinfo.get("name") or tid
        cols = tinfo.get("columns", {})
        col_names = list(cols.keys()) if isinstance(cols, dict) else [
            c.get("col_name") or c.get("name") for c in cols]
        col_names = [c for c in col_names if c][:20]
        table_docs[tid] = f"{name}: columns {', '.join(col_names)}" if col_names else str(name)

    out = {"columns": col_docs, "tables": table_docs}
    path = _index_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    if verbose:
        print(f"  [rerank_docs] {len(col_docs)} cols, {len(table_docs)} tables → {path}")
    return {"cols": len(col_docs), "tables": len(table_docs), "path": path}


_RERANK_DOCS_CACHE: Optional[dict] = None


def load_rerank_docs() -> Optional[dict]:
    """Query-tier loader: {"columns": {col_id: text}, "tables": {table_id: text}} or None."""
    global _RERANK_DOCS_CACHE
    if _RERANK_DOCS_CACHE is not None:
        return _RERANK_DOCS_CACHE
    path = _index_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            _RERANK_DOCS_CACHE = json.load(f)
        return _RERANK_DOCS_CACHE
    except Exception:
        return None
