"""L5 PUBLISH — derived registries + unified graph (atomic activation).

Compiles the relationship graph + semantic registry (fast-path source) and the
unified graph. In the layered model this is the atomic-activate point: everything
is written, then the query tier is told to rehydrate. The registry/graph builds
are pure transforms of the semantic model (LLM-free), so they run in skip_llm too.
"""
from __future__ import annotations

from typing import Dict, List

from ingestion.contracts import SourceContext, StageOutcome


def run(ctx: SourceContext, state: Dict, verbose: bool = False) -> List[StageOutcome]:
    out: List[StageOutcome] = []

    from config import DERIVED_ARTIFACTS_ENABLED

    if DERIVED_ARTIFACTS_ENABLED:
        # relationship graph (join planner / fast path) — non-fatal
        # P0-1/P0-2/P0-4 (2026-09-10): pass ctx (this source, for a per-source path +
        # connector-aware schema fetch) and tables from THIS RUN's own in-memory
        # semantic model / scan result — never re-read from a file another source
        # could have written (see ingestion/relationship_graph.py's docstring).
        try:
            from ingestion.relationship_graph import build_relationship_graph
            _rg_tables = None
            _sm = state.get("semantic_model") or {}
            if _sm.get("tables"):
                _rg_tables = sorted(_sm["tables"].keys())
            elif state.get("scan_result") is not None:
                _rg_tables = sorted(t.table_name for t in state["scan_result"].tables)
            g = build_relationship_graph(tables=_rg_tables, verbose=verbose, ctx=ctx)
            n = len(g.get("edges", [])) if isinstance(g, dict) else 0
            _mode = (g.get("stats") or {}).get("mode", "sql") if isinstance(g, dict) else "sql"
            out.append(StageOutcome("relationship_graph", True, detail=f"{n} edges ({_mode})"))
        except Exception as e:
            out.append(StageOutcome("relationship_graph", False, fatal=False, error=str(e)))

        # semantic registry (fast-path registry, incl. fast-path expansion Q-6) — non-fatal
        try:
            from semantic.compile_semantic_layer import compile_all
            compiled = compile_all(write=True, source_id=ctx.source_id, tenant=ctx.tenant)
            out.append(StageOutcome("semantic_registry", True, detail=(
                f"{len(compiled.get('concepts', {}))} concepts, "
                f"{len(compiled.get('metrics', {}))} metrics")))
        except Exception as e:
            out.append(StageOutcome("semantic_registry", False, fatal=False, error=str(e)))

        # value referents (QSR grounding artifact: value→referent FK closure + per-
        # edge label domains) — consumed by the deterministic planners, typed anchor
        # evidence, the strict Tier-2 qualifier gate and grounded clarifies. Derived
        # from THIS run's column_values (the sampler truncates + rewrites the store,
        # so it holds exactly this source's values here) + the relationship graph
        # built above. Written per (tenant, source); the runtime loader resolves the
        # same scope from the request context. Source conn (when reachable) enables
        # the precise per-edge label closure; otherwise broad closure. Non-fatal.
        try:
            import json as _vjson
            from config import SEMANTIC_MODEL_FILE
            from ingestion.value_referents import write_value_referents
            from ingestion.db_abstraction import (get_client_connection,
                                                  get_internal_connection,
                                                  release_internal_connection)
            try:
                with open(SEMANTIC_MODEL_FILE) as _vf:
                    _vsm = _vjson.load(_vf)
            except Exception:
                _vsm = {}          # no table_type filter → broad (still correct) closure
            try:
                _vsrc = get_client_connection(ctx.source_id)
            except Exception:
                _vsrc = None
            _vconn = get_internal_connection()
            try:
                _vpath = write_value_referents(
                    _vconn, _vsm, source_conn=_vsrc, verbose=verbose,
                    tenant=ctx.tenant, source_id=ctx.source_id)
            finally:
                release_internal_connection(_vconn)
                if _vsrc is not None:
                    try:
                        _vsrc.close()
                    except Exception:
                        pass
            out.append(StageOutcome("value_referents", True, detail=_vpath))
        except Exception as e:
            out.append(StageOutcome("value_referents", False, fatal=False, error=str(e)))

    # Per-source HNSW ef_search (NEW, P7/Q-10) — tune by source size, persist for the
    # activate step to store on SubstrateVersion.hnsw_ef_search — non-fatal.
    try:
        import json as _json
        from config import source_artifact_path
        scan = state.get("scan_result")
        n_tables = int(getattr(scan, "stats", {}).get("total_tables", 0)) if scan else 0
        # larger schema → wider search; clamp to [40, 200]. 40 == shipped default.
        ef = min(200, max(40, 40 + (n_tables // 20) * 20))
        _p = source_artifact_path("veda_hnsw.json", ctx.source_id, ctx.tenant)   # per source (M1)
        import os as _os
        _os.makedirs(_os.path.dirname(_p) or ".", exist_ok=True)
        with open(_p, "w") as _f:
            _json.dump({"hnsw_ef_search": ef, "n_tables": n_tables}, _f)
        state["hnsw_ef_search"] = ef
        out.append(StageOutcome("hnsw_tune", True, detail=f"ef_search={ef} ({n_tables} tables)"))
    except Exception as e:
        out.append(StageOutcome("hnsw_tune", False, fatal=False, error=str(e)))

    # Redis value mirror (NEW, Q-5) — activate-time mirror of column_values — non-fatal
    try:
        from ingestion.value_mirror import mirror_values_to_redis
        vm = mirror_values_to_redis(source_id=ctx.source_id, tenant=ctx.tenant, verbose=verbose)
        out.append(StageOutcome("value_mirror", True, detail=f"{vm.get('values', 0)} keys"))
    except Exception as e:
        out.append(StageOutcome("value_mirror", False, fatal=False, error=str(e)))

    # unified graph (query-time GRAPH_EXPAND) — non-fatal, ALWAYS rebuilt.
    # Previously gated on UNIFIED_GRAPH_ENABLED, which is the master switch for a
    # DIFFERENT system (the Postgres graph_nodes/graph_edges graph). This JSON
    # artifact is consumed by GRAPH_EXPAND and by query-time callers that check
    # neither flag, so gating its BUILD on an unrelated toggle meant turning that
    # toggle off silently froze the artifact while queries kept reading it. It is a
    # pure stdlib transform of files already on disk (seconds), so it now carries
    # the same "regenerated on every ingestion" contract as its DERIVED_ARTIFACTS
    # siblings above. A failure is still non-fatal but is now printed, not swallowed.
    try:
        from ingestion.unified_graph_builder import build_unified_graph, write_unified_graph
        ug = build_unified_graph(ctx.source_id, ctx.tenant)
        path = write_unified_graph(ug, ctx.source_id, ctx.tenant)
        out.append(StageOutcome("unified_graph", True, detail=(
            f"{len(ug.get('nodes', []))} nodes, {len(ug.get('edges', []))} edges → {path}")))
    except Exception as e:
        print(f"  [L5] ⚠ unified_graph rebuild FAILED — query-time graph expansion "
              f"will keep serving the previous (now stale) artifact: "
              f"{type(e).__name__}: {e}")
        out.append(StageOutcome("unified_graph", False, fatal=False, error=str(e)))

    # cross-source join discovery (P4.2/P4.3) — tenant-wide, runs at the END of every
    # ingestion over ALL ready sources so ingesting ANY source (re)links it to the rest
    # via cross_source_fk edges. Cheap (sketch comparisons only). Non-fatal + a no-op
    # until ≥2 sources have sketches.
    try:
        from ingestion.cross_source_graph import discover_and_persist
        stats = discover_and_persist(ctx.tenant, source_ids=None, verbose=verbose)
        out.append(StageOutcome("cross_source_fk", True, detail=(
            f"{stats.get('edges', 0)} edges across {stats.get('sources', 0)} sources "
            f"{stats.get('tiers', {})}")))
    except Exception as e:
        out.append(StageOutcome("cross_source_fk", False, fatal=False, error=str(e)))

    # routing card (Checkpoint B.1) — the compact per-source map the routing SLM reads
    # instead of inferring what a source IS from whichever columns one question
    # happened to surface. Runs LAST in L5 because it reads the cross_source_fk edges
    # the stage immediately above just (re)discovered, plus this run's own semantic
    # model / scan result. Pure transform: no LLM, no new scan, no new embedding — so
    # it runs in skip_llm too, exactly like the registry/graph builds. Non-fatal: a
    # source with no card simply routes the way it did before cards existed.
    try:
        from ingestion.routing_card import write_routing_card
        _cpath = write_routing_card(ctx, state, verbose=verbose)
        import json as _cjson
        with open(_cpath) as _cf:
            _card = _cjson.load(_cf)
        out.append(StageOutcome("routing_card", True, detail=(
            f"{len(_card.get('entities') or [])} entities, "
            f"{len(_card.get('joins_to') or [])} linked sources -> {_cpath}")))
    except Exception as e:
        out.append(StageOutcome("routing_card", False, fatal=False, error=str(e)))

    # entity aliases (2026-09-23) — republish the TRACKED seed into this source's
    # veda_entity_aliases.json. The artifact is hand-curated evidence but lives under
    # the gitignored, reingest-cleared artifact tree, so it was silently destroyed on
    # 2026-09-22 and "property" stopped grounding to assets_asset. The seed is in source
    # control; this stage puts it back on every ingest. Pure file transform (no LLM, no
    # scan) so it runs in skip_llm too; non-fatal — a source with no seed is a no-op.
    try:
        from ingestion.entity_alias_seeder import publish_from_state
        _apath, _n, _dropped = publish_from_state(ctx, state, verbose=verbose)
        if _apath is None:
            out.append(StageOutcome("entity_aliases", True, detail="no seed for this source"))
        else:
            out.append(StageOutcome("entity_aliases", True, detail=(
                f"{_n} aliases -> {_apath}" +
                (f" ({_dropped} dropped: table not in this source)" if _dropped else ""))))
    except Exception as e:
        out.append(StageOutcome("entity_aliases", False, fatal=False, error=str(e)))

    # value aliases (2026-09-23) — the business-PHRASE -> column-VALUE glossary
    # ("currently on the market" -> assets_salelisting.status = APPROVED). Same tracked
    # seed + same reason as the stage above.
    try:
        from ingestion.entity_alias_seeder import publish_values
        _tabs = None
        try:
            _sm = (state or {}).get("semantic_model") or {}
            if _sm.get("tables"):
                _tabs = set(_sm["tables"].keys())
        except Exception:
            _tabs = None
        _vpath, _vn = publish_values(ctx.source_id, getattr(ctx, "tenant", "default"),
                                     known_tables=_tabs)
        out.append(StageOutcome("value_aliases", True, detail=(
            f"{_vn} column(s) -> {_vpath}" if _vpath else "no value seed for this source")))
    except Exception as e:
        out.append(StageOutcome("value_aliases", False, fatal=False, error=str(e)))

    return out
