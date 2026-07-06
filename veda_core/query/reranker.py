# query/reranker.py
# VEDA — Cross-encoder reranker (Step 2)
# Gate: RETRIEVAL_V2_ENABLED and RERANKER_ENABLED
# Rescores first-stage candidates using full query<->column attention.
#
# Gap 1: dynamic cutoff (score cliff, not fixed top-15)
# Gap 2: table reranker uses real table text, not doubled name
# Gap 3: column reranker uses enriched text (gloss + type + sample values)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from typing import List, Optional

from config import (
    RERANKER_MODEL,
    RERANKER_DEVICE,
    RERANKER_BATCH_SIZE,
    RERANKER_MAX_TEXT_LEN,
    RERANKER_USE_ENRICHED_TEXT,
    RERANKER_DYNAMIC_CUTOFF,
    RERANKER_RELATIVE_DROP,
    RERANKER_SCORE_MIN,
    RERANKER_MIN_COLS,
    RERANKER_MAX_COLS,
    RERANKER_CUTOFF_BY_INTENT,
    RERANKER_RELATIVE_DROP_DIRECT,
    RERANKER_RELATIVE_DROP_MULTI,
    RERANKER_MAX_COLS_MULTI,
)
from ingestion.vector_store import RetrievalResult

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------
RERANKER_AVAILABLE = False
_RERANKER_INSTANCE = None

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    pass


def _get_reranker():
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is not None:
        return _RERANKER_INSTANCE
    if not RERANKER_AVAILABLE:
        return None
    try:
        from sentence_transformers import CrossEncoder
        _RERANKER_INSTANCE = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)
        return _RERANKER_INSTANCE
    except Exception as e:
        warnings.warn(f"[Reranker] Could not load model '{RERANKER_MODEL}': {e}")
        return None


def _columns_of_table(table_id: str) -> List[str]:
    """Column names for a given table_id from the graph node store. Falls back to []."""
    if not table_id:
        return []
    try:
        from ingestion.db_abstraction import (
            INTERNAL_DB_AVAILABLE, get_internal_connection,
            release_internal_connection, DICT_CURSOR,
        )
        from ingestion.graph_persist import GRAPH_NODES_TABLE
        if not INTERNAL_DB_AVAILABLE:
            return []
        conn = get_internal_connection()
        try:
            cur = conn.cursor(cursor_factory=DICT_CURSOR)
            cur.execute(
                f"SELECT name FROM {GRAPH_NODES_TABLE} "
                f"WHERE node_type='column' AND table_id=%s ORDER BY name LIMIT 30",
                (table_id,),
            )
            rows = cur.fetchall()
            try:
                cur.close()
            except Exception:
                pass
            return [r["name"] for r in rows]
        finally:
            release_internal_connection(conn)
    except Exception:
        return []


def _precomputed_rerank_text(item_id, is_table: bool):
    """Q-4: return the precomputed rerank text for a column/table id, or None (flag off,
    artifact absent, or id not covered → caller assembles text live)."""
    if not item_id:
        return None
    try:
        from config import RERANK_DOCS_ENABLED
        if not RERANK_DOCS_ENABLED:
            return None
        from ingestion.rerank_docs import load_rerank_docs
        docs = load_rerank_docs()
        if not docs:
            return None
        bucket = docs.get("tables" if is_table else "columns", {})
        return bucket.get(item_id)
    except Exception:
        return None


def _table_text(c: RetrievalResult) -> str:
    """
    Describe a table candidate for the cross-encoder.
    Uses the table name + its column names so the reranker has real content.
    Without this, table candidates score ~0 from doubled-name text like 'incident incident'.
    """
    # Q-4: use the precomputed rerank text (built at ingestion) when available —
    # avoids the per-query column-name stitch + the SELECT name FROM graph_nodes.
    _pre = _precomputed_rerank_text(getattr(c, "table_id", None), is_table=True)
    if _pre is not None:
        return _pre
    col_names = _columns_of_table(getattr(c, "table_id", None) or "")
    if col_names:
        names_str = ", ".join(col_names[:20])
        return f"{c.table_name}: columns {names_str}"
    return c.table_name or c.col_name or ""


def _col_text(c: RetrievalResult, sampled: dict) -> str:
    """
    Enriched text for a column candidate: gloss + type + sample values.
    Uses build_enriched_column_text so the cross-encoder has the same
    vocabulary that was used at indexing time.
    """
    # Q-4: precomputed enriched column text (built at ingestion) when available.
    _pre = _precomputed_rerank_text(c.col_id, is_table=False)
    if _pre is not None:
        return _pre[:RERANKER_MAX_TEXT_LEN]
    if not RERANKER_USE_ENRICHED_TEXT:
        return (c.col_name + " " + (c.table_name or ""))[:RERANKER_MAX_TEXT_LEN]
    try:
        from ingestion.column_text import build_enriched_column_text
        s = sampled.get(c.col_id)
        return build_enriched_column_text(
            c.col_name,
            c.table_name or "",
            getattr(c, "semantic_type", None) or "",
            sampled=s,
            style="minilm",
        )[:RERANKER_MAX_TEXT_LEN]
    except Exception:
        return (c.col_name + " " + (c.table_name or ""))[:RERANKER_MAX_TEXT_LEN]


def _apply_cutoff(scored: list, top_n: int, n_candidate_tables: int = 1) -> list:
    """
    Dynamic cutoff: keep cols above the score cliff.
    When RERANKER_CUTOFF_BY_INTENT, uses a looser threshold for multi-table
    queries (≥2 candidate tables) so join/filter cols are not dropped.
    Falls back to fixed top_n when RERANKER_DYNAMIC_CUTOFF is False.
    """
    if not RERANKER_DYNAMIC_CUTOFF or not scored:
        return scored[:top_n]

    if RERANKER_CUTOFF_BY_INTENT and n_candidate_tables >= 2:
        rel_drop = RERANKER_RELATIVE_DROP_MULTI
        max_cols = RERANKER_MAX_COLS_MULTI
    else:
        rel_drop = RERANKER_RELATIVE_DROP_DIRECT
        max_cols = RERANKER_MAX_COLS

    top_score = float(scored[0][0])
    # Relative-drop threshold is only meaningful for positive scores.
    # Cross-encoder logits can be negative; multiplying by rel_drop then
    # produces a less-negative threshold that almost nothing fails, letting
    # everything through.  Fall back to the absolute floor instead (B2 fix).
    rel_threshold = top_score * rel_drop if top_score > 0 else RERANKER_SCORE_MIN
    kept = []
    for sc, cand in scored:
        s = float(sc)
        if len(kept) >= RERANKER_MIN_COLS and (
            s < RERANKER_SCORE_MIN or s < rel_threshold
        ):
            break
        kept.append((sc, cand))
        if len(kept) >= max_cols:
            break
    return kept


_DOMAIN_SYN_CACHE = {"v": None}


def _domain_synonyms() -> dict:
    """The GENERATED domain synonyms ({phrase: [table.column, ...]}) — the ONE synonym
    source for the whole system. Absolute path (CWD-independent). Used PHRASE-level below
    so a synonym re-adds only its mapped columns, never a loose token match."""
    if _DOMAIN_SYN_CACHE["v"] is None:
        try:
            import json as _json
            _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "veda_domain_synonyms.json")
            d = _json.load(open(_p))
            _DOMAIN_SYN_CACHE["v"] = {str(k).lower(): {str(c).lower() for c in (v or [])}
                                     for k, v in d.items()}
        except Exception:
            _DOMAIN_SYN_CACHE["v"] = {}
    return _DOMAIN_SYN_CACHE["v"]


def _query_named_columns(
    query:      str,
    candidates: List[RetrievalResult],
) -> List[RetrievalResult]:
    """
    Return candidates whose column name tokens overlap with query tokens (synonym-expanded).
    Used to re-add explicitly-requested columns that the score cutoff dropped.
    E.g. "incident number" → synonym "no" matches incident_no even if its score is low.
    """
    import re as _re
    # GENERIC morphological abbreviations only (language-level, DB-agnostic: number↔no).
    # NO business/domain synonyms hardcoded here anymore — those come from the GENERATED
    # domain_synonyms (the one synonym source), applied PHRASE-level below.
    _ABBREV = {
        "number": {"no", "num"}, "identifier": {"id"}, "amount": {"amt"},
        "quantity": {"qty"}, "description": {"desc"}, "category": {"cat", "type"},
        "email": {"mail"}, "address": {"addr"}, "status": {"state"},
        "timestamp": {"time", "date"}, "created": {"create"},
        "updated": {"update", "modified"}, "assigned": {"assignee"},
        "priority": {"pri"}, "reference": {"ref"}, "reason": {"cause", "note"},
        "names": {"name"}, "types": {"type"}, "dates": {"date", "datetime"},
    }
    ql = query.lower()
    q_tokens = set(_re.findall(r"\w+", ql))
    for t in list(q_tokens):
        q_tokens |= _ABBREV.get(t, set())

    # GENERATED domain synonyms, PHRASE-level → exact target columns. A synonym phrase
    # present in the query re-adds ONLY its mapped columns (e.g. "asset type" → object_type),
    # never every column merely containing a token like "type". No token explosion = no over-broad.
    targets = set()
    for phrase, cols in _domain_synonyms().items():
        if phrase in ql:
            targets |= cols                      # cols are "table.column" (lowercased)

    named = []
    for c in candidates:
        cid = f"{c.table_name}.{c.col_name}".lower()
        if cid in targets:                       # generated-synonym phrase → this exact column
            named.append(c)
            continue
        raw_parts = set((c.col_name or "").lower().split("_"))
        # Filter short structural prefixes (id, no, is_, …) so "assigned_to_id" matches on
        # "assigned"; require ALL meaningful col-name parts covered (incident_no, not every incident.*).
        col_tokens = {t for t in raw_parts if len(t) > 2}
        if col_tokens and col_tokens <= q_tokens:
            named.append(c)
    return named


def _rerank(
    query:      str,
    candidates: List[RetrievalResult],
    top_n:      int,
    is_table:   bool = False,
    verbose:    bool = False,
) -> List[RetrievalResult]:
    """
    Internal reranker. Builds enriched pair text, scores with cross-encoder,
    applies dynamic cutoff, returns reranked RetrievalResult list.
    """
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return candidates[:top_n]

    # Load sampled values once (used for col enrichment — table path skips)
    sampled: dict = {}
    if RERANKER_USE_ENRICHED_TEXT and not is_table:
        try:
            from ingestion.value_sampler import get_sampled_columns
            sampled = get_sampled_columns()
        except Exception:
            pass

    pairs = [
        [query, (_table_text(c) if is_table else _col_text(c, sampled))[:RERANKER_MAX_TEXT_LEN]]
        for c in candidates
    ]

    try:
        scores = reranker.predict(pairs, batch_size=RERANKER_BATCH_SIZE)
    except Exception as e:
        warnings.warn(f"[Reranker] predict failed: {e} — falling back to first-stage order")
        return candidates[:top_n]

    scored = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)
    n_tables = len({c.table_id for c in candidates}) if not is_table else 1
    scored = _apply_cutoff(scored, top_n, n_candidate_tables=n_tables)

    # Re-add any candidate explicitly named in the query that the cutoff dropped.
    # The cutoff may drop "incident_no" when enriched-text flattened its score,
    # even though the query said "incident number" → must never drop those.
    if not is_table:
        named = _query_named_columns(query, candidates)
        kept_ids = {cand.col_id for _, cand in scored}
        floor_score = float(scored[-1][0]) if scored else 0.01
        for nc in named:
            if nc.col_id not in kept_ids and len(scored) < RERANKER_MAX_COLS:
                scored.append((floor_score, nc))
                kept_ids.add(nc.col_id)

    result = []
    for score, cand in scored:
        result.append(RetrievalResult(
            col_id        = cand.col_id,
            col_name      = cand.col_name,
            table_id      = cand.table_id,
            table_name    = cand.table_name,
            semantic_type = cand.semantic_type,
            similarity    = round(float(score), 6),
            source_id     = cand.source_id,
            embedding     = cand.embedding,
        ))

    if verbose:
        label = "tables" if is_table else "cols"
        print(f"  [Reranker] Reranked {len(candidates)} -> top {len(result)} {label}")
        for i, r in enumerate(result[:20]):
            print(f"    {i+1}. {r.table_name}.{r.col_name}  score={r.similarity:.4f}")

    return result


def rerank_columns(
    query:      str,
    candidates: List[RetrievalResult],
    top_n:      int,
    verbose:    bool = False,
) -> List[RetrievalResult]:
    """Re-scores column candidates with enriched text + dynamic cutoff."""
    return _rerank(query, candidates, top_n, is_table=False, verbose=verbose)


def rerank_tables(
    query:      str,
    candidates: List[RetrievalResult],
    top_n:      int,
    verbose:    bool = False,
) -> List[RetrievalResult]:
    """Re-scores table candidates using real table text (column names), not doubled name."""
    return _rerank(query, candidates, top_n, is_table=True, verbose=verbose)
