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

    from config import DERIVED_ARTIFACTS_ENABLED, UNIFIED_GRAPH_ENABLED

    if DERIVED_ARTIFACTS_ENABLED:
        # relationship graph (join planner / fast path) — non-fatal
        try:
            from ingestion.relationship_graph import build_relationship_graph
            g = build_relationship_graph(verbose=verbose)
            n = len(g.get("edges", [])) if isinstance(g, dict) else 0
            out.append(StageOutcome("relationship_graph", True, detail=f"{n} edges"))
        except Exception as e:
            out.append(StageOutcome("relationship_graph", False, fatal=False, error=str(e)))

        # semantic registry (fast-path registry, incl. fast-path expansion Q-6) — non-fatal
        try:
            from semantic.compile_semantic_layer import compile_all
            compiled = compile_all(write=True)
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
        from config import artifact_path
        scan = state.get("scan_result")
        n_tables = int(getattr(scan, "stats", {}).get("total_tables", 0)) if scan else 0
        # larger schema → wider search; clamp to [40, 200]. 40 == shipped default.
        ef = min(200, max(40, 40 + (n_tables // 20) * 20))
        _p = artifact_path("veda_hnsw.json")
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

    # unified graph (query-time GRAPH_EXPAND) — non-fatal
    if UNIFIED_GRAPH_ENABLED:
        try:
            from ingestion.unified_graph_builder import build_unified_graph, write_unified_graph
            ug = build_unified_graph()
            path = write_unified_graph(ug)
            out.append(StageOutcome("unified_graph", True, detail=(
                f"{len(ug.get('nodes', []))} nodes, {len(ug.get('edges', []))} edges → {path}")))
        except Exception as e:
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

    return out
