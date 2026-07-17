# =============================================================================
# query/rag_layer.py
# VEDA — RAG + Hybrid Query Layer
#
# Responsibility:
#   - Embeds the user query using BGE-M3 (same model as chunk_embedder, WP3)
#   - Improvement 1: Applies L1 temporal filter to restrict chunk retrieval
#     by document date — documents outside the time window are excluded
#   - Improvement 2: Applies value expansion (from value_sampler) before
#     embedding so DB column values in the query boost doc retrieval accuracy
#   - Retrieves top-K document chunks via cosine search on doc_chunks
#   - run_rag_layer(): pure document synthesis (RAG intent)
#   - run_hybrid_layer(): Improvement 3 — fuses SQL columns + doc chunks via
#     RRF, builds unified context, single SLM call produces combined answer
#
# Called by main.py query mode based on query_router intent classification.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ingestion.chunk_embedder import ChunkRetrievalResult, retrieve_top_k_chunks
from config import (
    SLM_MODEL_NAME,
    SLM_OLLAMA_BASE_URL,
    SLM_TEMPERATURE,
    SLM_MAX_TOKENS,
    SLM_TIMEOUT_SECS,
    RAG_TOP_K,
    HYBRID_RRF_K,
    HYBRID_SQL_WEIGHT,
    HYBRID_RAG_WEIGHT,
    HYBRID_MAX_RESULT_ROWS,
    TOP_K_TO_LLM,
)


def _emit(on_event, phase, message, **extra):
    """Fire the optional SSE progress callback — same contract as
    veda_hybrid.py's own _emit (self-contained copy, not a shared import:
    veda_hybrid.py imports FROM this module, so importing _emit back from
    there would be circular). A no-op when on_event is None, and it never
    raises into the pipeline — progress reporting must not be able to fail
    a query."""
    if on_event is None:
        return
    try:
        on_event(phase, message, extra)
    except Exception:
        pass


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class RAGResult:
    """Output of run_rag_layer() — pure document retrieval + synthesis."""
    answer:       str
    chunks:       List[ChunkRetrievalResult]
    citations:    List[str]           # "doc_name (p.N)" strings for UI
    confidence:   float               # top chunk similarity score
    duration_ms:  float
    error:        Optional[str] = None
    stats:        dict = field(default_factory=dict)


@dataclass
class HybridItem:
    """
    A single item in the unified RRF-fused result list for hybrid queries.
    Can be either a SQL column or a document chunk.
    """
    item_type:   str     # "sql_column" | "doc_chunk"
    rrf_score:   float
    # SQL column fields (set when item_type == "sql_column")
    col_id:      Optional[str]   = None
    col_name:    Optional[str]   = None
    table_name:  Optional[str]   = None
    semantic_type: Optional[str] = None
    col_similarity: Optional[float] = None
    # Doc chunk fields (set when item_type == "doc_chunk")
    chunk:       Optional[ChunkRetrievalResult] = None


@dataclass
class HybridResult:
    """Output of run_hybrid_layer() — unified SQL + document answer."""
    answer:          str
    sql_columns:     list                   # top SQL columns used in context
    doc_chunks:      List[ChunkRetrievalResult]  # top doc chunks used
    citations:       List[str]
    confidence:      float
    intent:          str = "hybrid"
    duration_ms:     float = 0.0
    error:           Optional[str] = None
    stats:           dict = field(default_factory=dict)
    # Populated by the caller (veda_hybrid.py) from the deterministic SQL head's
    # OWN executed rows/explain (never regenerated here) when the "hybrid" intent's
    # SQL sub-run succeeded — lets apps/chat/services.py chart/table/explain a
    # hybrid answer exactly like a plain SQL one, instead of it silently having
    # neither (the SQL rows existed, they just weren't attached to this object).
    cols:            list = field(default_factory=list)
    rows:            list = field(default_factory=list)
    explain:         Optional[dict] = None
    # Deterministic post-execution analysis (result_analyzer.analytics_summary),
    # populated by the caller from the SAME attached SQL-head (cols, rows) — parity
    # with Tier-1/Tier-2/federated so a hybrid answer gets the same charts + the
    # "Analysis:" summary fold-in. None when there were no tabular rows to analyze.
    analytics:       Optional[dict] = None


# =============================================================================
# Query encoding — reuses the shared BGE-M3 model (WP3)
# Improvement 2: value expansion applied before encoding
# =============================================================================

def _encode_rag_query(
    query:   str,
    verbose: bool = False,
) -> Optional[Any]:
    """
    Encodes the query using BGE-M3 (1024-dim, WP3) — the SAME model that embedded the
    document chunks, so query and chunk vectors share one space.

    Document search runs over free text, so the SQL value-expansion (which maps a query
    word to a DB COLUMN NAME, e.g. "document" → "workflow_state") injects an irrelevant
    snake_case token into the embedding query and hurts chunk relevance. Use the raw
    query for RAG; value-expansion stays in the SQL path.
    """
    try:
        from ingestion import m3_encoder
        return m3_encoder.encode_dense([query])[0]
    except Exception as e:
        if verbose:
            print(f"  [RAG] BGE-M3 encoding failed: {e}")
        return None


# =============================================================================
# Ollama synthesis call
# =============================================================================

_RAG_SYSTEM_PROMPT = (
    "You are a precise document Q&A assistant. "
    "Answer the user's question using ONLY the provided context passages. "
    "If the answer cannot be found in the context, say so explicitly. "
    "Cite the document name and page number when available. "
    "Be concise and factual."
)

_HYBRID_SYSTEM_PROMPT = (
    "You are a data and document analyst. "
    "Answer the user's question using ONLY the provided context below. "
    "STRICT RULES:\n"
    "1. You MUST prefix any database insights with '[DB]' and any document insights with '[DOC]'.\n"
    "2. SQL EXECUTION RESULTS are the absolute ground truth. If they are provided, assume they directly answer the user's query. You MUST state the exact data or count as your primary answer, even if the database column names (like 'count_result') differ from the query terms.\n"
    "3. Document chunks are supplementary. If they do not directly answer the user's question, IGNORE THEM completely. Do not summarize irrelevant documents.\n"
    "4. Only cite documents that appear in the context below — never invent citations.\n"
    "5. NEVER invent, fabricate, or assume any content not shown in context.\n"
    "6. Do not complain about missing information if the SQL EXECUTION RESULTS provide a valid data point or count."
)


def _call_ollama(system_prompt: str, user_message: str) -> str:
    """Single SLM call with configurable system prompt (§10 seam)."""
    from slm import call_slm
    return call_slm(
        user_message,
        system=system_prompt,
        purpose="rag_synthesis",
        temperature=SLM_TEMPERATURE,
        num_predict=SLM_MAX_TOKENS,
        timeout=SLM_TIMEOUT_SECS,
    ).strip()


# =============================================================================
# Context builders — format retrieved items for the SLM prompt
# =============================================================================

def _build_rag_user_message(
    query:  str,
    chunks: List[ChunkRetrievalResult],
) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        loc = chunk.doc_name
        if chunk.page_num:
            loc += f", page {chunk.page_num}"
        context_parts.append(f"[{i}] ({loc})\n{chunk.text}")
    return f"Context:\n\n{'chr(10)chr(10)'.join(context_parts)}\n\nQuestion: {query}"


def _build_hybrid_user_message(
    query:        str,
    sql_columns:  list,             # List[RetrievalResult] from semantic_layer
    doc_chunks:   List[ChunkRetrievalResult],
    sql_result:   object = None,    # ExecutionResult from L6 (Phase 1B)
) -> str:
    """
    Builds the unified context message for a hybrid query.
    To defeat LLM position bias and ensure ground truth isn't truncated by SLM context limits,
    SQL execution results (ground truth) are placed at the very top.
    Schema and Documents follow.
    """
    parts = []

    if sql_result is not None and not getattr(sql_result, "error", None):
        rows = getattr(sql_result, "rows", []) or []
        cols = getattr(sql_result, "columns", []) or []
        parts.append("=== EXECUTED DATABASE QUERY RESULTS (GROUND TRUTH) ===")
        parts.append("The database was queried to answer the user's question and returned this exact data:")
        if rows and cols:
            for row in rows[:HYBRID_MAX_RESULT_ROWS]:
                row_str = ", ".join(f"{c}: {row.get(c, '')}" for c in cols)
                parts.append(f"  - {row_str}")
            if len(rows) > HYBRID_MAX_RESULT_ROWS:
                parts.append(
                    f"  ... ({len(rows)} rows total, showing first {HYBRID_MAX_RESULT_ROWS})"
                )
            # Plain-language gloss for single-row aggregate results so the LLM
            # can connect a generic alias like "count_result" to the query intent.
            if len(rows) == 1:
                for col in cols:
                    val = rows[0].get(col)
                    if val is None:
                        continue
                    col_lc = col.lower()
                    if col_lc.startswith("count") or col_lc in ("n", "total", "num", "sum"):
                        parts.append(
                            f"*** PLAIN ANSWER: The database counted {val} records "
                            f"matching this query. '{col}' = {val} IS the answer. "
                            f"State {val} as your answer. ***"
                        )
                        break
        else:
            parts.append("  (Query executed successfully but returned 0 rows)")

    if sql_columns:
        parts.append("\n=== DATABASE SCHEMA CONTEXT ===")
        for col in sql_columns[:TOP_K_TO_LLM]:
            parts.append(
                f"Column: {col.table_name}.{col.col_name} "
                f"(type: {col.semantic_type})"
            )

    if doc_chunks:
        parts.append("\n=== DOCUMENT CONTEXT ===")
        for i, chunk in enumerate(doc_chunks[:RAG_TOP_K], 1):
            loc = chunk.doc_name
            if chunk.page_num:
                loc += f", page {chunk.page_num}"
            parts.append(f"[{i}] ({loc})\n{chunk.text}")

    parts.append(f"\nQuestion: {query}")
    parts.append("\nAnswer based ONLY on the context above (prioritize GROUND TRUTH if available):")
    return "\n".join(parts)


# =============================================================================
# Improvement 3 — RRF fusion for hybrid queries
#
# Merges ranked SQL columns and ranked doc chunks into one unified list.
# Uses the same RRF algorithm as ensemble retrieval in semantic_layer.py.
#
# score(item) = w_sql  / (rank_sql  + K)   (0 if item not in SQL results)
#             + w_rag  / (rank_rag  + K)   (0 if item not in RAG results)
#
# After fusion, the top-ranked items from BOTH sources form the unified
# context passed to the SLM. This ensures the SLM always sees the most
# relevant information regardless of which source it came from.
# =============================================================================

def _rrf_fuse_hybrid(
    sql_columns:  list,                       # List[RetrievalResult]
    doc_chunks:   List[ChunkRetrievalResult],
    rrf_k:        float = HYBRID_RRF_K,
    w_sql:        float = HYBRID_SQL_WEIGHT,
    w_rag:        float = HYBRID_RAG_WEIGHT,
) -> Tuple[list, List[ChunkRetrievalResult]]:
    """
    Applies RRF to merge SQL columns and doc chunks.
    Returns (top_sql_cols, top_doc_chunks) selected by RRF scores.
    Items are selected alternately from the RRF ranking to ensure both
    source types are represented in the final context.
    """
    scored: List[HybridItem] = []
    absent_rank = max(len(sql_columns), len(doc_chunks)) + 1

    # SQL columns — rank by their similarity score position
    sql_rank = {col.col_id: i + 1 for i, col in enumerate(sql_columns)}
    for col in sql_columns:
        rrf = w_sql / (sql_rank[col.col_id] + rrf_k)
        scored.append(HybridItem(
            item_type     = "sql_column",
            rrf_score     = rrf,
            col_id        = col.col_id,
            col_name      = col.col_name,
            table_name    = col.table_name,
            semantic_type = col.semantic_type,
            col_similarity = col.similarity,
        ))

    # Doc chunks — rank by similarity
    for i, chunk in enumerate(doc_chunks):
        # Check if any SQL column already exists from same concept (dedup)
        rag_rank = i + 1
        rrf = w_rag / (rag_rank + rrf_k)
        scored.append(HybridItem(
            item_type = "doc_chunk",
            rrf_score = rrf,
            chunk     = chunk,
        ))

    scored.sort(key=lambda x: x.rrf_score, reverse=True)

    # Separate back into typed lists, preserving RRF-driven ordering
    top_sql  = [s for s in scored if s.item_type == "sql_column"]
    top_docs = [s for s in scored if s.item_type == "doc_chunk"]

    # Return original objects (not HybridItem wrappers) for context builders
    top_sql_cols = [
        next(c for c in sql_columns if c.col_id == s.col_id)
        for s in top_sql[:TOP_K_TO_LLM]
    ]
    top_doc_chunks = [s.chunk for s in top_docs[:RAG_TOP_K]]

    return top_sql_cols, top_doc_chunks


# =============================================================================
# Public entry point 1 — Pure RAG
# =============================================================================

def run_rag_layer(
    query:           str,
    source_ids:      List[str] = None,
    top_k:           int = RAG_TOP_K,
    temporal_filter: object = None,    # Improvement 1: TemporalFilter from L1
    verbose:         bool = False,
    on_event:        Any = None,
) -> RAGResult:
    """
    Pure document retrieval + synthesis.
    Called when query_router classifies intent as 'rag'.

    Improvement 1: temporal_filter restricts chunk retrieval by doc_date.
    Improvement 2: value expansion applied inside _encode_rag_query().

    Parameters
    ----------
    query           : user's natural language question
    source_ids      : restrict to these document source IDs (None = all)
    top_k           : chunks to retrieve
    temporal_filter : TemporalFilter from L1 temporal_parser (or None)
    verbose         : print progress
    on_event        : optional progress callback (phase, message, extra) —
                       previously this whole function was a silent black box
                       between the caller's "Retrieving relevant documents..."
                       and "Synthesized answer..." ticks (veda_hybrid.py), with
                       no visibility into the actual retrieval or the SLM
                       synthesis call — often the slowest, most opaque step.
    """
    t0 = time.time()

    if verbose:
        print(f"[RAGLayer] Query: '{query}'")
        print(f"  source_ids       : {source_ids}")
        print(f"  top_k            : {top_k}")
        print(f"  temporal_filter  : {temporal_filter is not None}")

    # Encode query (with value expansion — Improvement 2)
    query_vec = _encode_rag_query(query, verbose=verbose)
    if query_vec is None:
        return RAGResult(
            answer="", chunks=[], citations=[], confidence=0.0,
            duration_ms=round((time.time() - t0) * 1000, 2),
            error="Query embedding failed — BGE-M3 not loaded",
        )

    # Retrieve chunks (with temporal filter — Improvement 1)
    chunks = retrieve_top_k_chunks(
        query_vector    = query_vec,
        source_ids      = source_ids,
        top_k           = top_k,
        temporal_filter = temporal_filter,
        verbose         = verbose,
    )

    if not chunks:
        msg = "No relevant document passages found"
        if temporal_filter is not None:
            msg += " within the specified time range"
        return RAGResult(
            answer=msg, chunks=[], citations=[], confidence=0.0,
            duration_ms=round((time.time() - t0) * 1000, 2),
            stats={"chunks_retrieved": 0},
        )

    confidence = chunks[0].similarity if chunks else 0.0
    citations  = list(dict.fromkeys(
        (c.doc_name + (f" (p.{c.page_num})" if c.page_num else ""))
        for c in chunks
    ))

    if verbose:
        print(f"  Chunks retrieved : {len(chunks)}")
        print(f"  Top similarity   : {confidence:.4f}")
    _emit(on_event, "rag_retrieve", f"Found {len(chunks)} relevant passage(s)", chunks=len(chunks))

    # Synthesise answer
    _emit(on_event, "rag_synthesize", "Reading through what was found")
    try:
        user_msg = _build_rag_user_message(query, chunks)
        answer   = _call_ollama(_RAG_SYSTEM_PROMPT, user_msg)
    except Exception as e:
        answer = "\n\n".join(f"[{c.doc_name}] {c.text[:300]}" for c in chunks)
        if verbose:
            print(f"  ⚠ SLM synthesis failed ({e}) — returning raw chunks")

    duration_ms = round((time.time() - t0) * 1000, 2)
    if verbose:
        print(f"  Duration         : {duration_ms}ms")
        print("[RAGLayer] Done.\n")

    return RAGResult(
        answer      = answer,
        chunks      = chunks,
        citations   = citations,
        confidence  = round(confidence, 4),
        duration_ms = duration_ms,
        stats       = {
            "chunks_retrieved":  len(chunks),
            "citations":         len(citations),
            "temporal_filtered": temporal_filter is not None,
            "value_expanded":    True,
        },
    )


# =============================================================================
# Public entry point 2 — Hybrid (Improvement 3)
# =============================================================================

def run_hybrid_layer(
    query:           str,
    sql_columns:     list,             # List[RetrievalResult] from semantic_layer
    source_ids:      List[str] = None,
    temporal_filter: object = None,
    sql_result:      object = None,    # Phase 1B: ExecutionResult from L6
    verbose:         bool = False,
    on_event:        Any = None,
) -> HybridResult:
    """
    Hybrid query: fuses SQL column context and document chunk context via RRF,
    then synthesises a single answer in one SLM call (Option A).

    Called when query_router classifies intent as 'hybrid'.

    Parameters
    ----------
    query           : user's natural language question
    sql_columns     : top-K RetrievalResult objects from run_semantic_layer()
    source_ids      : restrict doc retrieval to these source IDs
    temporal_filter : TemporalFilter from L1 (applied to both SQL and docs)
    sql_result      : ExecutionResult from L6 — when present, actual query rows
                      are included in the synthesis prompt for definitive answers
    verbose         : print progress
    on_event        : optional progress callback (phase, message, extra) —
                       same rationale as run_rag_layer's own on_event: the
                       RRF fusion + SLM synthesis steps were previously a
                       silent black box between the caller's outer ticks.
    """
    t0 = time.time()

    if verbose:
        print(f"[HybridLayer] Query: '{query}'")
        print(f"  SQL columns in   : {len(sql_columns)}")

    # Step 1 — Encode query (with value expansion)
    query_vec = _encode_rag_query(query, verbose=verbose)
    if query_vec is None:
        return HybridResult(
            answer="", sql_columns=[], doc_chunks=[], citations=[],
            confidence=0.0, duration_ms=round((time.time() - t0) * 1000, 2),
            error="Query embedding failed — BGE-M3 not loaded",
        )

    # Step 2 — Retrieve doc chunks (with temporal filter)
    doc_chunks = retrieve_top_k_chunks(
        query_vector    = query_vec,
        source_ids      = source_ids,
        top_k           = RAG_TOP_K,
        temporal_filter = temporal_filter,
        verbose         = verbose,
    )

    if verbose:
        print(f"  Doc chunks found : {len(doc_chunks)}")

    # Step 3 — RRF fusion of SQL columns + doc chunks
    top_sql_cols, top_doc_chunks = _rrf_fuse_hybrid(
        sql_columns = sql_columns,
        doc_chunks  = doc_chunks,
    )

    if verbose:
        print(f"  After RRF fusion : {len(top_sql_cols)} SQL cols + {len(top_doc_chunks)} doc chunks")
    _emit(on_event, "hybrid_retrieve",
          f"Combined {len(top_sql_cols)} data point(s) and {len(top_doc_chunks)} document passage(s)",
          sql_cols=len(top_sql_cols), doc_chunks=len(top_doc_chunks))

    # When SQL ground truth is available, drop doc chunks below this similarity
    # threshold — low-relevance documents add noise that causes the LLM to
    # override a definitive SQL answer with "no information found" statements.
    _MIN_DOC_SIM_WITH_SQL = 0.50
    if sql_result is not None and not getattr(sql_result, "error", None):
        filtered = [c for c in top_doc_chunks if c.similarity >= _MIN_DOC_SIM_WITH_SQL]
        if verbose and len(filtered) < len(top_doc_chunks):
            print(
                f"  Doc chunks after SQL-filter : {len(filtered)} "
                f"(dropped {len(top_doc_chunks) - len(filtered)} below sim={_MIN_DOC_SIM_WITH_SQL})"
            )
        top_doc_chunks = filtered

    # Step 4 — Single SLM call with unified context (Option A)
    citations = list(dict.fromkeys(
        (c.doc_name + (f" (p.{c.page_num})" if c.page_num else ""))
        for c in top_doc_chunks
    ))
    _emit(on_event, "hybrid_synthesize", "Piecing together an answer from everything found")

    # Hallucination guard (Fix 2):
    # When zero doc chunks were retrieved, the SLM has no document context.
    # Phase 1B: if SQL execution results are available, synthesise from them
    # instead of degrading; otherwise fall back to schema-only answer.
    if not top_doc_chunks:
        if sql_result is not None and not getattr(sql_result, "error", None):
            try:
                user_msg = _build_hybrid_user_message(
                    query, top_sql_cols, [], sql_result
                )
                answer = _call_ollama(_HYBRID_SYSTEM_PROMPT, user_msg)
            except Exception as e:
                rows = getattr(sql_result, "rows", [])
                answer = "[DB] SQL results: " + str(rows[:5])
                if verbose:
                    print(f"  ⚠ SLM synthesis failed ({e}) — returning raw SQL rows")
            return HybridResult(
                answer      = answer,
                sql_columns = top_sql_cols,
                doc_chunks  = [],
                citations   = [],
                confidence  = top_sql_cols[0].similarity if top_sql_cols else 0.0,
                intent      = "hybrid",
                duration_ms = round((time.time() - t0) * 1000, 2),
                stats       = {
                    "sql_cols_used":   len(top_sql_cols),
                    "doc_chunks_used": 0,
                    "sql_executed":    True,
                    "sql_rows":        getattr(sql_result, "row_count", 0),
                },
            )
        col_summary = ", ".join(
            f"{c.table_name}.{c.col_name}"
            for c in top_sql_cols[:5]
        )
        return HybridResult(
            answer      = (
                f"No relevant document content was found for this query.\n\n"
                f"[DB] The most relevant database columns are: {col_summary}.\n"
                f"Run a SQL query against these columns to get structured data."
            ),
            sql_columns = top_sql_cols,
            doc_chunks  = [],
            citations   = [],
            confidence  = 0.0,
            intent      = "sql",    # degrade to SQL intent — no docs found
            duration_ms = round((time.time() - t0) * 1000, 2),
            stats       = {
                "sql_cols_used":   len(top_sql_cols),
                "doc_chunks_used": 0,
                "degraded_to_sql": True,
                "reason":          "no doc chunks retrieved",
            },
        )

    try:
        user_msg = _build_hybrid_user_message(query, top_sql_cols, top_doc_chunks, sql_result)
        answer   = _call_ollama(_HYBRID_SYSTEM_PROMPT, user_msg)
    except Exception as e:
        # Graceful fallback — raw context
        answer = (
            "DATABASE: " + ", ".join(
                f"{c.table_name}.{c.col_name}" for c in top_sql_cols[:5]
            ) + "\n\n" +
            "DOCUMENTS: " + " | ".join(
                c.text[:200] for c in top_doc_chunks[:3]
            )
        )
        if verbose:
            print(f"  ⚠ SLM synthesis failed ({e}) — returning raw context")

    top_sim = top_doc_chunks[0].similarity if top_doc_chunks else (
        top_sql_cols[0].similarity if top_sql_cols else 0.0
    )

    duration_ms = round((time.time() - t0) * 1000, 2)
    if verbose:
        print(f"  Duration         : {duration_ms}ms")
        print("[HybridLayer] Done.\n")

    return HybridResult(
        answer       = answer,
        sql_columns  = top_sql_cols,
        doc_chunks   = top_doc_chunks,
        citations    = citations,
        confidence   = round(top_sim, 4),
        intent       = "hybrid",
        duration_ms  = duration_ms,
        stats        = {
            "sql_cols_used":     len(top_sql_cols),
            "doc_chunks_used":   len(top_doc_chunks),
            "citations":         len(citations),
            "temporal_filtered": temporal_filter is not None,
            "sql_executed":      sql_result is not None and not getattr(sql_result, "error", None),
            "sql_rows":          getattr(sql_result, "row_count", 0) if sql_result else 0,
        },
    )