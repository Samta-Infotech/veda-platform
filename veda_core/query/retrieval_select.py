"""Single source of truth for which columns/tables go to L3.
Both main.py (interactive) and evaluation/evaluator.py call this so the two
entry points can never diverge again."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass, field
from typing import List, Optional

from config import TOP_K


@dataclass
class SelectedRetrieval:
    columns:               list    # final cols for L3/scoring (List[RetrievalResult])
    tables:                list    # table names derived from final columns
    join_path:             list    # pruned join path from semantic layer
    short_circuit:         bool
    source:                str     # 'schema_link' | 'v2_rerank' | 'graph' | 'legacy'
    semantic_layer_result: object  # SemanticLayerResult (encoding_strategy, stats, etc.)
    graph_result:          object = None  # GraphRetrievalResult (chunks for hybrid fusion)
    stats:                 dict = field(default_factory=dict)


def select_retrieval(
    query:      str,
    source_ids: Optional[List[str]] = None,
    intent:     str = "sql",
    graph_result = None,    # pass pre-computed GraphRetrievalResult to avoid a double run
    verbose:    bool = False,
) -> SelectedRetrieval:
    """
    Unified column-selection logic. Mirrors main.py's retrieval block in one place.

    Flow:
      1. Schema linker  (RETRIEVAL_V2_ENABLED + SCHEMA_LINK_ENABLED)
         - high-confidence exact match → short-circuit, skip bi-encoder
      2. Bi-encoder + reranker  (RETRIEVAL_V2_ENABLED + BIENCODER_ENABLED)
         - skipped when schema-link short-circuited
      3. Graph retrieval  (UNIFIED_GRAPH_ENABLED + GRAPH_RETRIEVAL_ENABLED + GRAPH_EMBED_ENABLED)
         - skipped when graph_result already provided by caller
      4. Semantic layer  (always — needed for join_path, encoding_strategy, synonym stats)
      5. Override decision:
           V2 cols (schema-link or bi-encoder) → highest priority
           Graph   (hybrid intent or has chunks) → second
           Legacy  (sl.top_k_columns)           → fallback
      6. Join-path pruned to selected tables.
    """
    from config import (
        RETRIEVAL_V2_ENABLED, SCHEMA_LINK_ENABLED, BIENCODER_ENABLED,
        UNIFIED_GRAPH_ENABLED, GRAPH_RETRIEVAL_ENABLED, GRAPH_EMBED_ENABLED,
    )
    from query.semantic_layer import run_semantic_layer

    stats: dict = {}
    _v2_cols      = None
    _short_circuit = False
    _source        = "legacy"

    # ── Step 1: Schema linker ─────────────────────────────────────────────
    if RETRIEVAL_V2_ENABLED and SCHEMA_LINK_ENABLED:
        try:
            from query.schema_linker import run_schema_linker
            _sl = run_schema_linker(query=query, source_ids=source_ids, verbose=verbose)
            if _sl.short_circuit:
                _short_circuit = True
                _v2_cols = _sl.matched_columns
                _source  = "schema_link"
                stats["schema_link_short_circuit"] = True
                stats["schema_link_table_id"]      = _sl.primary_table_id
                stats["schema_link_n_cols"]        = len(_v2_cols)
        except Exception as e:
            stats["schema_link_failed"] = True
            stats["schema_link_error"]  = str(e)

    # ── Step 2: Bi-encoder + cross-encoder reranker ───────────────────────
    if RETRIEVAL_V2_ENABLED and BIENCODER_ENABLED and not _short_circuit:
        try:
            from query.retrieval_v2 import retrieve_v2
            _v2_cols, _v2_tables = retrieve_v2(
                query=query, source_ids=source_ids, verbose=verbose,
            )
            if _v2_cols:
                _source = "v2_rerank"
                stats["v2_rerank_ran"]    = True
                stats["v2_rerank_n_cols"] = len(_v2_cols)
            else:
                stats["v2_rerank_ran"] = True
                _v2_cols = None  # empty list → treat as no result, fall back to L2
        except Exception as e:
            stats["v2_rerank_failed"] = True
            stats["v2_error"]         = str(e)
            _v2_cols = None

    # ── Step 3: Graph retrieval ───────────────────────────────────────────
    _use_graph    = UNIFIED_GRAPH_ENABLED and GRAPH_RETRIEVAL_ENABLED and GRAPH_EMBED_ENABLED
    _graph_result = graph_result
    if _use_graph and _graph_result is None:
        try:
            from query.graph_retriever import run_graph_retrieval
            _graph_result = run_graph_retrieval(
                query=query, source_ids=source_ids, verbose=verbose,
            )
            stats["graph_ran"]    = True
            stats["graph_cols"]   = len(_graph_result.columns)
            stats["graph_chunks"] = len(_graph_result.chunks)
            stats["graph_edges"]  = len(_graph_result.edges_used)
        except Exception as e:
            stats["graph_failed"] = True
            stats["graph_error"]  = str(e)
            _graph_result = None

    # ── Step 4: Semantic layer (always) ───────────────────────────────────
    # Legacy ensemble signal. Under ENCODER_MODE=ensemble it needs MiniLM, which is NOT
    # loaded when BGE (RETRIEVAL_V2) is the primary path — so this one signal can fail
    # ("MiniLM model not loaded"). Degrade instead of crashing: skip it and let the BGE +
    # BM25 + FK + value signals carry retrieval (BGE already overrides sl.top_k_columns
    # below, and sl.join_path is recomputed for the V2 columns). Never fatal.
    # Pin the V2-dispatch guard BEFORE calling run_semantic_layer for Step 4. Without this,
    # run_semantic_layer sees the guard unset and RE-DISPATCHES back into select_retrieval —
    # running the entire BGE first-stage + cross-encoder rerank a SECOND time (the duplicate
    # "[Reranker] Reranked 80 -> top 20" seen in traces). Setting it here forces Step 4 down
    # the legacy path directly, so the expensive V2 retrieval runs exactly once.
    import query.semantic_layer as _sl_mod
    _prev_dispatch = _sl_mod._IN_V2_DISPATCH
    _sl_mod._IN_V2_DISPATCH = True
    try:
        sl = run_semantic_layer(
            query=query,
            top_k=TOP_K,
            verbose=verbose,
            source_ids=source_ids,
        )
    except Exception as _sle:
        import numpy as _np
        from query.semantic_layer import SemanticLayerResult
        stats["semantic_layer_skipped"] = True
        stats["semantic_layer_error"]   = str(_sle)
        if verbose:
            print(f"  [RetrievalV2] legacy semantic signal skipped "
                  f"({type(_sle).__name__}) — fusing BGE+BM25+FK+value")
        sl = SemanticLayerResult(
            query=query, query_vector=_np.zeros(0, dtype=_np.float32),
            top_k_columns=[], join_path=[], tables_involved=[],
            encoding_strategy="unavailable", duration_ms=0.0, stats={})
    finally:
        _sl_mod._IN_V2_DISPATCH = _prev_dispatch    # restore (re-entrancy safe)

    # ── Step 5: Graph-helps flag — mirrors main.py logic exactly ─────────
    _graph_helps = bool(_graph_result) and (
        intent == "hybrid"
        or len(getattr(_graph_result, "chunks", []) or []) > 0
    )

    # ── Step 6: Build the override (graph applied first, V2 takes final priority)
    if _graph_helps:
        from query.graph_retriever import _subgraph_to_retrieval_results
        _graph_cols = _subgraph_to_retrieval_results(_graph_result.columns)
        if getattr(_graph_result, "short_circuited", False) and _graph_cols:
            l2_top_k_override = _graph_cols[:TOP_K]
        elif _graph_cols:
            # Append-only: L2 keeps position 0, graph fills leftover slots.
            _l2_ids = {r.col_id for r in sl.top_k_columns}
            _extra  = [c for c in _graph_cols if c.col_id not in _l2_ids]
            l2_top_k_override = (sl.top_k_columns + _extra)[:TOP_K]
        else:
            l2_top_k_override = sl.top_k_columns
        # Prune join_path to tables that are still present after graph merge
        if sl.join_path and l2_top_k_override:
            _selected = {r.table_name for r in l2_top_k_override}
            sl.join_path = [
                e for e in sl.join_path
                if e.from_table_name in _selected and e.to_table_name in _selected
            ]
        _source = "graph"
    else:
        l2_top_k_override = sl.top_k_columns

    # V2 cols take final priority (schema-link or bi-encoder output).
    # Re-set source to reflect actual column origin after any graph merge.
    if _v2_cols:
        l2_top_k_override = _v2_cols
        _source = "schema_link" if _short_circuit else "v2_rerank"

        # ── V2 supplement: keyword / FK-PK / display inject ──────────────
        # LEGACY_RETRIEVAL_DISABLED=True disables Steps 4a/4b/4c in
        # semantic_layer.py, so sl.top_k_columns has no keyword-injected
        # (sim=0.0) cols.  The prior E4 carry-back iterated sl looking for
        # sim=0.0 cols that never exist — it was a no-op.  These three steps
        # replace it by running the injections directly here.
        import re as _re
        from ingestion.vector_store import (
            retrieve_cols_by_name_keywords as _rcbnk,
            get_display_columns            as _gdc,
            get_fk_adjacency               as _gfka,
            RetrievalResult                as _RR,
        )

        # A: keyword injection ─────────────────────────────────────────────
        # Token set = raw query words (+ singulars) + synonym-expansion parts.
        # "queue" → synonym target "workflow_state" → parts "workflow","state"
        # so incident.workflow_state is found even though neither word is in
        # the raw query.  Singular forms fix "types"→"type", "names"→"name".
        _inj_stop = frozenset({
            "show", "list", "get", "find", "give", "me", "all", "the", "a", "an",
            "and", "or", "for", "of", "in", "on", "at", "to", "with", "by",
            "from", "that", "their", "each", "per", "where", "which", "is", "are",
            "was", "were", "has", "have", "its", "user", "users",
        })
        _inj_raw = {w for w in _re.sub(r"[^\w]", " ", query.lower()).split()
                    if w not in _inj_stop and len(w) > 2}
        _inj_raw |= {t[:-1] for t in _inj_raw if t.endswith("s") and len(t) > 3}
        _inj_syn: set = set()
        for _mt in sl.stats.get("tokens_mapped", []):
            for _p in str(_mt).split("_"):
                if len(_p) > 2:
                    _inj_syn.add(_p.lower())
        _inj_toks = _inj_raw | _inj_syn
        _inj_present = {c.col_id for c in l2_top_k_override}
        if _inj_toks:
            _kw_hits = _rcbnk(list(_inj_toks))
            _kw_new = [
                c for c in _kw_hits
                if c.col_id not in _inj_present
                and (not source_ids or not c.source_id or c.source_id in source_ids)
            ]
            if _kw_new:
                l2_top_k_override = list(l2_top_k_override) + _kw_new
                _inj_present.update(c.col_id for c in _kw_new)
                stats["kw_inject_v2"] = len(_kw_new)

        # B: FK PK injection ───────────────────────────────────────────────
        # incident.id (len=2) can never be keyword-matched; it arrives only via
        # FK edge PK injection.  For every FK edge whose PK (to_col) belongs to
        # a table already in results but isn't in results yet, inject it.
        try:
            _pk_tbl_ids = list(dict.fromkeys(
                r.table_id for r in l2_top_k_override if r.table_id
            ))
            _pk_tbl_set = set(_pk_tbl_ids)
            _pk_edges   = _gfka(_pk_tbl_ids)
            _pk_new: list = []
            for _edge in _pk_edges:
                if (_edge.to_table_id in _pk_tbl_set
                        and _edge.to_col_id
                        and _edge.to_col_id not in _inj_present):
                    _pk_new.append(_RR(
                        col_id        = _edge.to_col_id,
                        col_name      = _edge.to_col_name,
                        table_id      = _edge.to_table_id,
                        table_name    = _edge.to_table_name,
                        semantic_type = "IDENTIFIER",
                        similarity    = 0.0,
                        source_id     = "",
                        embedding     = None,
                    ))
                    _inj_present.add(_edge.to_col_id)
            if _pk_new:
                l2_top_k_override = list(l2_top_k_override) + _pk_new
                stats["pk_inject_v2"] = len(_pk_new)
        except Exception:
            pass

        # C: display col injection ─────────────────────────────────────────
        # documents.name is a display column not reachable by keyword match.
        # get_display_columns returns the registered display col per table.
        try:
            _dc_tbl_ids = list(dict.fromkeys(
                r.table_id for r in l2_top_k_override if r.table_id
            ))
            _dc_info = _gdc(_dc_tbl_ids)
            _dc_new: list = []
            for _tid, _dinfo in _dc_info.items():
                _dcid = _dinfo.get("col_id")
                if _dcid and _dcid not in _inj_present:
                    _dc_new.append(_RR(
                        col_id        = _dcid,
                        col_name      = _dinfo["col_name"],
                        table_id      = _tid,
                        table_name    = _dinfo["table_name"],
                        semantic_type = "IDENTIFIER",
                        similarity    = 0.0,
                        source_id     = "",
                        embedding     = None,
                    ))
                    _inj_present.add(_dcid)
            if _dc_new:
                l2_top_k_override = list(l2_top_k_override) + _dc_new
                stats["display_inject_v2"] = len(_dc_new)
        except Exception:
            pass

        # D: graph supplement ──────────────────────────────────────────────
        # The graph retriever BFS-expands FK edges and surfaces cross-table
        # columns (e.g. state.name via transition→state FK) that the
        # bi-encoder first stage doesn't rank highly. Merge those columns
        # in here so multi-table SQL queries get the right JOIN columns.
        if _graph_result is not None and _graph_helps:
            try:
                from query.graph_retriever import _subgraph_to_retrieval_results
                _graph_rr = _subgraph_to_retrieval_results(_graph_result.columns)
                _g_extra = [
                    c for c in _graph_rr
                    if c.col_id not in _inj_present
                    and c.table_name and c.col_name
                    and (not source_ids or not c.source_id or c.source_id in source_ids)
                ]
                if _g_extra:
                    l2_top_k_override = list(l2_top_k_override) + _g_extra
                    _inj_present.update(c.col_id for c in _g_extra)
                    stats["graph_supplement_v2"] = len(_g_extra)
            except Exception:
                pass

    # ── Gap 4a: value-filter add-back ─────────────────────────────────────
    # The dynamic cutoff (Gap 1) may drop filter columns (e.g. incident_status)
    # because their relevance score to "decision reason" is low.
    # Re-add columns whose SAMPLED VALUES match a query token, regardless of
    # relevance score, so the SLM can emit the correct WHERE clause.
    from config import VALUE_FILTER_ENABLED, VALUE_FILTER_MAX_COLS
    if VALUE_FILTER_ENABLED:
        try:
            from query.value_filter import find_value_filter_columns
            _cand_table_ids = {r.table_id for r in l2_top_k_override if r.table_id}
            # Latent A: expand scope to FK-adjacent tables so the value-filter can find
            # filter columns that belong to a joinable table not yet retrieved.
            _expanded_table_ids = set(_cand_table_ids)
            try:
                from ingestion.vector_store import get_fk_adjacency
                for _e in get_fk_adjacency(list(_cand_table_ids)):
                    _expanded_table_ids.add(getattr(_e, "from_table_id", "") or "")
                    _expanded_table_ids.add(getattr(_e, "to_table_id", "") or "")
                _expanded_table_ids.discard("")
            except Exception:
                pass
            _filter_cols = find_value_filter_columns(
                query, source_ids, candidate_table_ids=_expanded_table_ids
            )
            # Carry over sim=1.0 cols the semantic layer added via expand_query_tokens
            # value injection. When V2 override replaced l2_top_k_override with the
            # reranker output, those cols were lost — merge them back here so the SLM
            # sees filter candidates that the reranker never scored.
            # Only carry cols from tables already in scope to avoid unjoinable tables.
            _fc_ids = {fc.col_id for fc in _filter_cols}
            for _slr in sl.top_k_columns:
                if (float(_slr.similarity) >= 1.0
                        and _slr.col_id not in _fc_ids
                        and _slr.table_id in _expanded_table_ids):
                    _filter_cols = list(_filter_cols) + [_slr]
                    _fc_ids.add(_slr.col_id)
            if _filter_cols:
                # Prepend value_filter cols so they land within TOP_K_TO_LLM
                # positions and reach L3's prompt.  Appending at the tail (old
                # behaviour) placed them at position 16+ which slm_layer[:10]
                # always discarded — the SLM never saw incident_status.
                _new_cols = [fc for fc in _filter_cols[:VALUE_FILTER_MAX_COLS]]
                _present  = {fc.col_id for fc in _new_cols}
                for c in l2_top_k_override:
                    if c.col_id not in _present:
                        _new_cols.append(c)
                        _present.add(c.col_id)
                _added = len([fc for fc in _filter_cols[:VALUE_FILTER_MAX_COLS]
                              if fc.col_id not in {c.col_id for c in l2_top_k_override}])
                # Always apply the reordered list — filter cols may already exist
                # but at a rank beyond TOP_K_TO_LLM where the SLM would never see
                # them.  Prepending unconditionally moves them to the front (B1 fix).
                l2_top_k_override = _new_cols
                if _added:
                    stats["value_filter_added"] = _added
        except Exception as _vfe:
            stats["value_filter_error"] = str(_vfe)

    selected_tables = list(dict.fromkeys(r.table_name for r in l2_top_k_override))

    # When V2 overrides sl.top_k_columns, sl.join_path is stale — it was
    # resolved from the semantic layer's own RRF result (feature, signal_*,
    # organisations …) not from the V2 tables (role, role_permissions,
    # permission …).  Passing stale JOINs to the SLM means it receives 0
    # useful edges and produces un-joined IR.  Recompute from the final
    # column set so the SLM sees the correct role→role_permissions→permission
    # (and analogous) JOIN path.
    if _v2_cols:
        try:
            from query.semantic_layer import _resolve_join_path
            from ingestion.vector_store import get_fk_adjacency as _gfka_jp
            _jp_tids = list(dict.fromkeys(r.table_id for r in l2_top_k_override if r.table_id))
            _jp_fk   = _gfka_jp(_jp_tids)
            sl.join_path = _resolve_join_path(l2_top_k_override, fk_edges=_jp_fk)
            stats["join_path_recomputed"] = len(sl.join_path)
        except Exception:
            pass

    return SelectedRetrieval(
        columns               = l2_top_k_override,
        tables                = selected_tables,
        join_path             = sl.join_path or [],
        short_circuit         = _short_circuit,
        source                = _source,
        semantic_layer_result = sl,
        graph_result          = _graph_result,
        stats                 = stats,
    )
