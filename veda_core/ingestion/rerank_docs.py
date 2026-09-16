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


def _index_path(source_id=None, tenant: str = "default") -> str:
    """Per-(tenant, source) when source_id is given (P0-5, 2026-09-10) — unconditionally,
    via config.source_artifact_path(), not gated behind the VEDA_ARTIFACT_SCOPE flag
    (P0-7). source_id-less callers (legacy/dev-CLI) keep the flat artifact_path()."""
    if source_id is not None:
        from config import source_artifact_path
        return source_artifact_path("veda_rerank_docs.json", source_id, tenant)
    from config import artifact_path
    return artifact_path("veda_rerank_docs.json")


def build_rerank_docs(source_id: str = "", tenant: str = "default", verbose: bool = False,
                      semantic_model: Optional[Dict] = None) -> Dict:
    """``semantic_model`` (P0-4-style fix, 2026-09-10): pass the CALLER's own in-memory
    semantic model (``state["semantic_model"]`` from ``layers/l4_index.py``) when
    available, rather than always re-reading the flat ``SEMANTIC_MODEL_FILE`` — the
    same file another source's ingest could have written last (the semantic model
    itself isn't per-source-scoped yet; see P0-5 in
    docs/backlog/query-engine-open-items.md). Falls back to the flat file only when
    no in-memory model is given (dev-CLI / legacy callers), unchanged from before."""
    if semantic_model is not None:
        sm = semantic_model
    else:
        # M1 close-out (2026-09-15): THIS source's scoped model, never the flat file.
        from config import resolve_source_artifact
        _p = resolve_source_artifact("veda_semantic_model.json", source_id or None, tenant)
        if not _p or not os.path.exists(_p):
            raise FileNotFoundError(
                f"semantic model not materialized for source {source_id!r}: {_p}")
        with open(_p) as f:
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
    path = _index_path(source_id or None, tenant)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    if verbose:
        print(f"  [rerank_docs] {len(col_docs)} cols, {len(table_docs)} tables → {path}")

    # P0-6: drop this source's cached docs so the next read picks up what was just
    # written instead of a stale in-process copy.
    try:
        invalidate_rerank_docs_cache(source_id or None)
    except Exception:
        pass

    return {"cols": len(col_docs), "tables": len(table_docs), "path": path}


# (tenant, str(source_id) | "") -> loaded dict (or None for "checked, missing").
# Per-source (P0-5, 2026-09-10) — a single global here meant source B's query-time
# reranker could silently read source A's precomputed rerank text (or vice versa,
# whichever ingested/queried first in this worker process).
_RERANK_DOCS_CACHE: dict = {}


def load_rerank_docs(source_id=None, tenant=None) -> Optional[dict]:
    """Query-tier loader: {"columns": {col_id: text}, "tables": {table_id: text}} or
    None. Resolves source_id/tenant from the ambient request context when not given
    (the normal query-time call shape); a source_id-less, context-less call (dev-CLI)
    falls back to the legacy flat path, matching pre-fix behaviour."""
    if source_id is None:
        from veda_core import context
        ctx = context.try_current()
        if ctx is not None:
            source_id = ctx.source_id
            tenant = ctx.tenant or tenant
    key = (tenant or "default", str(source_id) if source_id is not None else "")
    if key in _RERANK_DOCS_CACHE:
        return _RERANK_DOCS_CACHE[key]
    path = _index_path(source_id, tenant or "default")
    if not os.path.exists(path):
        _RERANK_DOCS_CACHE[key] = None
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        _RERANK_DOCS_CACHE[key] = data
        return data
    except Exception:
        return None


def invalidate_rerank_docs_cache(source_id=None):
    """Drop the cached rerank-docs artifact for one source, or every source when
    source_id is None. Call after build_rerank_docs() rewrites the artifact, and from
    both rehydrate paths (P0-6)."""
    if source_id is None:
        _RERANK_DOCS_CACHE.clear()
        return
    sid = str(source_id)
    for key in [k for k in _RERANK_DOCS_CACHE if k[1] == sid]:
        del _RERANK_DOCS_CACHE[key]
