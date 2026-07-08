# =============================================================================
# ingestion/source_dispatcher.py
# VEDA — Source Dispatcher (Phase 5)
#
# Responsibility:
#   Routes ingestion for any source type through the correct pipeline.
#   Called by main.py once per enabled source in VEDA_SOURCES.
#
# Pipeline per source type:
#   relational — full pipeline (schema → FK → data_graph →
#                semantic inference → metadata → value_sampler → REG →
#                encoder → vector_store → BGE biencoder)
#
#   datalake   — schema-compatible pipeline (same as relational but skips
#                data_graph and value_sampler — both require a psycopg2
#                client connection that datalake sources don't have)
#
#   document   — chunk embedding pipeline only
#                (get_chunks → chunk_embedder → doc_chunks pgvector table)
#
#   nosql      — schema inference pipeline (nosql collections → unifier →
#                schema_scanner → FK → semantic inference → metadata →
#                REG → encoder → vector_store;
#                skips data_graph and value_sampler)
#
# main.py usage after Phase 5:
#   from ingestion.source_dispatcher import dispatch_ingestion
#   for src in get_enabled_sources():
#       dispatch_ingestion(src, verbose=verbose)
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# Output dataclass
# =============================================================================

@dataclass
class DispatchResult:
    """Full ingestion result for one source."""
    source_id:    str
    source_type:  str
    success:      bool
    steps_run:    List[str]
    steps_failed: List[str]
    error:        Optional[str]
    duration_s:   float
    context:      dict = field(default_factory=dict)


# =============================================================================
# Step-level helpers (print progress in the same style as main.py)
# =============================================================================

def _slog(source_id: str, msg: str) -> None:
    print(f"  [{source_id}] {msg}")


def _sok(source_id: str, label: str, elapsed: float) -> None:
    print(f"  [{source_id}] ✓  {label}  ({round(elapsed, 2)}s)")


def _sfail(source_id: str, label: str, err: Exception) -> None:
    print(f"  [{source_id}] ✗  {label} FAILED")
    print(f"  [{source_id}]    {type(err).__name__}: {err}")


# =============================================================================
# Shared schema pipeline — relational, datalake, nosql all flow through this
# =============================================================================

def _run_schema_pipeline(
    raw_schema_dict: dict,
    source_config:   dict,
    run_data_graph:  bool = True,
    run_value_sampler: bool = True,
    verbose:         bool = False,
) -> dict:
    """
    Runs schema_scanner → FK adjacency → (data_graph) → semantic inference →
    table metadata → (value sampler) → REG builder → encoder → vector store →
    synthetic query gen → auto fine-tune.

    Returns a context dict with all step outputs.
    Raises on fatal failures (schema_scanner, encoder, vector_store).
    Non-fatal failures (data_graph, value_sampler) are absorbed — pipeline continues.
    """
    source_id = source_config["id"]
    ctx: dict = {"source_id": source_id}

    # ── Step 1: Schema Scanner ────────────────────────────────────────────────
    t0 = time.time()
    from ingestion.schema_scanner import run_schema_scanner
    scan_result = run_schema_scanner(raw_schema=raw_schema_dict, verbose=verbose)
    ctx["scan_result"] = scan_result
    _sok(source_id,
         f"Schema scanned — {scan_result.stats['total_tables']} tables, "
         f"{scan_result.stats['total_columns']} cols, "
         f"{scan_result.stats['total_fk_edges']} FK edges",
         time.time() - t0)

    if scan_result.stats['total_tables'] == 0:
        _slog(source_id, "⚠  No tables found — skipping remaining pipeline steps")
        return ctx

    # ── Step 2: FK Adjacency Store ────────────────────────────────────────────
    t0 = time.time()
    from ingestion.vector_store import store_fk_adjacency
    fk_result = store_fk_adjacency(scan_result, verbose=verbose)
    ctx["fk_result"] = fk_result
    _sok(source_id,
         f"FK adjacency — {fk_result.edges_written} edges, backend={fk_result.backend}",
         time.time() - t0)

    # ── Step 3: Data Graph (relational only) ──────────────────────────────────
    if run_data_graph:
        t0 = time.time()
        try:
            from ingestion.data_graph import run_data_graph, to_fk_adjacency_rows
            dg_result = run_data_graph(scan_result, source_id=source_id, verbose=verbose)
            ctx["dg_result"] = dg_result
            if dg_result.discovered_edges:
                discovered_rows = to_fk_adjacency_rows(dg_result, include_soft=False)
                if discovered_rows:
                    combined_edges = list(scan_result.fk_edges) + discovered_rows
                    scan_result.fk_edges = combined_edges
                    store_fk_adjacency(scan_result, verbose=False)
            _sok(source_id,
                 f"Data graph — "
                 f"HIGH={dg_result.stats.get('high_certainty', 0)} "
                 f"MEDIUM={dg_result.stats.get('medium_certainty', 0)} edges",
                 time.time() - t0)
        except Exception as e:
            _sfail(source_id, "Data graph", e)
            print(f"  [{source_id}]    Continuing without discovered edges.")

    # ── Step 4: Semantic Type Inference ───────────────────────────────────────
    t0 = time.time()
    from ingestion.semantic_type_inference import run_semantic_type_inference
    inference_result = run_semantic_type_inference(scan_result=scan_result, verbose=verbose)
    ctx["inference_result"] = inference_result
    stats = inference_result.stats
    _sok(source_id,
         f"Semantic types — avg_conf={stats['avg_confidence']}, "
         f"flagged={stats['flagged_count']}",
         time.time() - t0)

    # ── Step 4b: Domain Glossary (one-time, cached) ──────────────────────────
    try:
        from ingestion.domain_glossary import build_glossary
        from config import SLM_OLLAMA_BASE_URL
        _glossary = build_glossary(
            inference_result = inference_result,
            ollama_url       = SLM_OLLAMA_BASE_URL,
            force_rebuild    = False,
        )
        _sok(source_id, f"Domain glossary — {len(_glossary)} terms", 0.0)
    except Exception as _e:
        _slog(source_id, f"Domain glossary failed ({_e}) — continuing without glossary")

    # ── Step 5: Table Metadata Store ──────────────────────────────────────────
    t0 = time.time()
    from ingestion.vector_store import store_table_metadata
    tm_result = store_table_metadata(inference_result, verbose=verbose)
    ctx["tm_result"] = tm_result
    _sok(source_id,
         f"Table metadata — {tm_result.rows_written} tables, backend={tm_result.backend}",
         time.time() - t0)

    # ── Step 6: Value Sampler (relational only) ───────────────────────────────
    if run_value_sampler:
        t0 = time.time()
        try:
            from ingestion.value_sampler import run_value_sampler
            vs_result = run_value_sampler(
                inference_result, source_id=source_id, verbose=verbose
            )
            ctx["vs_result"] = vs_result
            _sok(source_id,
                 f"Value sampler — {vs_result.columns_sampled} cols, "
                 f"{vs_result.total_values} values",
                 time.time() - t0)
        except Exception as e:
            _sfail(source_id, "Value sampler", e)
            print(f"  [{source_id}]    Continuing without value expansion.")

    # ── Step 7: REG Builder ───────────────────────────────────────────────────
    t0 = time.time()
    from ingestion.reg_builder import run_reg_builder
    graph = run_reg_builder(inference_result=inference_result, verbose=verbose)
    ctx["graph"] = graph
    _sok(source_id,
         f"REG builder — {graph.stats['num_table_nodes']} tables, "
         f"{graph.stats['num_column_nodes']} cols, "
         f"{graph.stats['num_fk_to_edges']} FK edges",
         time.time() - t0)

    # ── Step 7b: Unified Graph Persist ───────────────────────────────────────
    from config import UNIFIED_GRAPH_ENABLED, GRAPH_PERSIST_ENABLED
    if UNIFIED_GRAPH_ENABLED and GRAPH_PERSIST_ENABLED:
        t0 = time.time()
        try:
            from ingestion.graph_persist import persist_reg_graph
            gp = persist_reg_graph(
                graph       = graph,
                scan_result = scan_result,
                dg_result   = ctx.get("dg_result"),
                source_id   = source_id,
                verbose     = verbose,
            )
            ctx["graph_persist_result"] = gp
            _sok(source_id,
                 f"Graph persist — {gp.nodes_written} nodes, {gp.edges_written} edges",
                 time.time() - t0)
        except Exception as e:
            _sfail(source_id, "Graph persist", e)

    # ── Step 7c: Unified Graph Embedder ──────────────────────────────────────
    from config import GRAPH_EMBED_ENABLED
    if UNIFIED_GRAPH_ENABLED and GRAPH_EMBED_ENABLED:
        t0 = time.time()
        try:
            from ingestion.graph_embedder import embed_graph_nodes
            ge = embed_graph_nodes(source_id=source_id, verbose=verbose)
            ctx["graph_embed_result"] = ge
            _sok(source_id,
                 f"Graph embedder — {ge.nodes_embedded} nodes embedded",
                 time.time() - t0)
        except Exception as e:
            _sfail(source_id, "Graph embedder", e)

    # Steps 8/9 (ensemble encoder → tfidf/svd pkls + column_embeddings_lt/_hybrid
    # vector store) were removed: the MiniLM/RELGT ensemble retrieval signal is never
    # executed at query time (BGE-only spine), so those artifacts were write-only. The
    # BGE biencoder below is the live retrieval store — same as the primary pipeline.

    # ── Step 9b: BGE Biencoder Ingestion ─────────────────────────────────────
    t0 = time.time()
    try:
        from config import BIENCODER_ENABLED
        if BIENCODER_ENABLED:
            from ingestion.biencoder import run_biencoder_ingestion
            bge_result = run_biencoder_ingestion(
                inference_result, source_id=source_id, verbose=verbose
            )
            ctx["bge_result"] = bge_result
            s = f"{bge_result.cols_embedded} cols" if not bge_result.error else f"warning: {bge_result.error}"
            _sok(source_id, f"BGE biencoder — {s}", time.time() - t0)
    except Exception as e:
        _sfail(source_id, "BGE biencoder", e)

    return ctx


# =============================================================================
# Per-type dispatch functions
# =============================================================================

def _dispatch_relational(source_config: dict, verbose: bool) -> DispatchResult:
    source_id = source_config["id"]
    t_start   = time.time()
    _slog(source_id, f"Relational source  engine={source_config.get('engine', '?')}")

    try:
        from connectors.base import build_connector
        connector = build_connector(source_config)
        status    = connector.connect()
        if not status.ok:
            return DispatchResult(
                source_id=source_id, source_type="relational",
                success=False, steps_run=[], steps_failed=[],
                error=f"Connection failed: {status.message}",
                duration_s=round(time.time() - t_start, 2),
            )

        # Honour the PASSED source (§3.2 / I-2 fix): the ingesting worker injected
        # THIS source's connection into the engine env, so main.run_ingestion (which
        # reads the injected source via config.get_source — the DB Source row is the
        # single source of truth, §3.1) ingests exactly this source. No "primary"
        # re-derivation. _run_schema_pipeline stays as an explicit schema-only opt-in
        # (source_config["use_schema_pipeline"]) for datalake/nosql-style runs.
        use_full_pipeline = not source_config.get("use_schema_pipeline", False)

        # Fetch schema before disconnecting (needed by the schema-pipeline fallback;
        # run_ingestion re-reads the schema itself for the full pipeline).
        raw_schema_dict = connector.get_raw_schema_dict()
        connector.disconnect()

        if use_full_pipeline:
            from main import run_ingestion
            ctx = run_ingestion(verbose=verbose)
            steps = ["schema", "fk", "data_graph", "semantic_types", "metadata",
                     "value_sampler", "reg", "encoder", "vector_store",
                     "semantic_layer", "biencoder", "glossary",
                     "derived_artifacts", "unified_graph"]
        else:
            ctx = _run_schema_pipeline(
                raw_schema_dict  = raw_schema_dict,
                source_config    = source_config,
                run_data_graph   = True,
                run_value_sampler = True,
                verbose          = verbose,
            )
            steps = ["schema", "fk", "data_graph", "semantic", "metadata",
                     "value_sampler", "reg", "encoder", "vector_store"]

        return DispatchResult(
            source_id=source_id, source_type="relational",
            success=True,
            steps_run=steps,
            steps_failed=[],
            error=None,
            duration_s=round(time.time() - t_start, 2),
            context=ctx,
        )
    except Exception as e:
        return DispatchResult(
            source_id=source_id, source_type="relational",
            success=False, steps_run=[], steps_failed=["pipeline"],
            error=f"{type(e).__name__}: {e}",
            duration_s=round(time.time() - t_start, 2),
        )


def _run_tabular_cross_source(connector, source_config: dict, verbose: bool) -> None:
    """Post-schema steps for a file-backed tabular source (Cross-source plan):
    materialize Parquet (P2.2), sketch join keys via the connector (P2.4), and run
    tenant-wide cross_source_fk discovery (P4.2). Each step is best-effort."""
    import os
    source_id = str(source_config["id"])
    tenant = os.environ.get("VEDA_TENANT", "default")
    try:
        from config import ARTIFACT_ROOT
        connector.materialize_parquet(os.path.join(ARTIFACT_ROOT, source_id, "tables"))
    except Exception as e:
        if verbose:
            print(f"  [tabular] materialize skipped: {e}")
    try:
        from ingestion import column_sketches as CS
        n = CS.sketch_columns_via_sampler(source_id, tenant, connector.sample_column_values)
        if verbose:
            print(f"  [tabular] sketched {n} join-key columns")
    except Exception as e:
        if verbose:
            print(f"  [tabular] sketch pass skipped: {e}")
    try:
        from ingestion.cross_source_graph import discover_and_persist
        stats = discover_and_persist(tenant, verbose=verbose)
        if verbose:
            print(f"  [tabular] cross_source_fk: {stats}")
    except Exception as e:
        if verbose:
            print(f"  [tabular] cross_source discovery skipped: {e}")


def _dispatch_datalake(source_config: dict, verbose: bool) -> DispatchResult:
    source_id = source_config["id"]
    t_start   = time.time()
    _slog(source_id, f"Datalake source  engine={source_config.get('engine', '?')}")

    # Cross-source plan P2.1: CSV/Parquet/Excel run through the DuckDB TabularFile
    # connector (deterministic ids) so their columns become REAL tables via the same
    # schema pipeline as relational; then materialize Parquet (P2.2 exec surface),
    # sketch join keys (P2.4) and discover cross-source links (P4.2). Non-tabular
    # datalake engines (delta/iceberg) keep the generic connector + schema pipeline.
    _TABULAR = ("csv", "csv_lake", "parquet", "xlsx", "excel")
    engine = (source_config.get("engine") or "").lower()
    try:
        if engine in _TABULAR:
            from connectors.tabular_files import TabularFileConnector
            connector = TabularFileConnector(source_config)
            status = connector.connect()
            if not status.ok:
                return DispatchResult(
                    source_id=source_id, source_type="datalake",
                    success=False, steps_run=[], steps_failed=[],
                    error=f"Connection failed: {status.message}",
                    duration_s=round(time.time() - t_start, 2))
            raw_schema_dict = connector.get_raw_schema_dict()
        else:
            from connectors.base import build_connector
            from ingestion.schema_unifier import raw_schema_to_dict
            connector = build_connector(source_config)
            status = connector.connect()
            if not status.ok:
                return DispatchResult(
                    source_id=source_id, source_type="datalake",
                    success=False, steps_run=[], steps_failed=[],
                    error=f"Connection failed: {status.message}",
                    duration_s=round(time.time() - t_start, 2))
            raw_schema_dict = raw_schema_to_dict(connector.get_schema())

        ctx = _run_schema_pipeline(
            raw_schema_dict   = raw_schema_dict,
            source_config     = source_config,
            run_data_graph    = engine in _TABULAR,   # value-overlap FK discovery within the files
            run_value_sampler = False,                 # files sampled via the connector below
            verbose           = verbose,
        )

        steps = ["schema", "fk", "semantic", "metadata", "reg", "encoder", "vector_store"]
        if engine in _TABULAR:
            _run_tabular_cross_source(connector, source_config, verbose)
            steps += ["materialize_parquet", "column_sketches", "cross_source_fk"]
        try:
            connector.disconnect()
        except Exception:
            pass

        return DispatchResult(
            source_id=source_id, source_type="datalake",
            success=True, steps_run=steps, steps_failed=[], error=None,
            duration_s=round(time.time() - t_start, 2), context=ctx)
    except Exception as e:
        return DispatchResult(
            source_id=source_id, source_type="datalake",
            success=False, steps_run=[], steps_failed=["pipeline"],
            error=f"{type(e).__name__}: {e}",
            duration_s=round(time.time() - t_start, 2),
        )


def _dispatch_document(source_config: dict, verbose: bool) -> DispatchResult:
    source_id = source_config["id"]
    t_start   = time.time()
    _slog(source_id, f"Document source  engine={source_config.get('engine', '?')}")

    try:
        from connectors.base import build_connector
        from ingestion.chunk_embedder import run_chunk_embedder

        connector = build_connector(source_config)
        status    = connector.connect()
        if not status.ok:
            return DispatchResult(
                source_id=source_id, source_type="document",
                success=False, steps_run=[], steps_failed=[],
                error=f"Connection failed: {status.message}",
                duration_s=round(time.time() - t_start, 2),
            )

        t0     = time.time()
        chunks = list(connector.get_chunks())
        connector.disconnect()

        if not chunks:
            _slog(source_id, "⚠  No chunks extracted — directory empty or no supported files")
            return DispatchResult(
                source_id=source_id, source_type="document",
                success=True, steps_run=["connect"], steps_failed=[],
                error=None,
                duration_s=round(time.time() - t_start, 2),
            )

        result = run_chunk_embedder(chunks, source_id, verbose=verbose)
        _sok(source_id,
             f"Chunk embedder — {result.chunks_embedded} chunks, "
             f"{result.docs_processed} docs, backend={result.backend}",
             time.time() - t0)

        steps = ["connect", "chunk_extract", "chunk_embed"]
        # Cross-source plan P4.1: entity linker bridges chunks → columns (mentions_entity
        # / value_of), carrying cross-source traversal to the tabular side; then P4.2
        # discovery. Best-effort — a failure here never fails the document ingest.
        import os as _os
        _tenant = _os.environ.get("VEDA_TENANT", "default")
        try:
            from ingestion.entity_linker import link_entities
            el = link_entities(chunks, source_id, tenant=_tenant, verbose=verbose)
            _sok(source_id, f"Entity linker — {el.entity_nodes} entities, "
                            f"{el.value_of} value_of, {el.mentions_entity} mentions_entity", 0)
            steps.append("entity_linker")
        except Exception as e:
            _slog(source_id, f"⚠  entity linker skipped: {e}")
        try:
            from ingestion.cross_source_graph import discover_and_persist
            discover_and_persist(_tenant, verbose=verbose)
            steps.append("cross_source_fk")
        except Exception as e:
            _slog(source_id, f"⚠  cross_source discovery skipped: {e}")

        return DispatchResult(
            source_id=source_id, source_type="document",
            success=True,
            steps_run=steps,
            steps_failed=[],
            error=None,
            duration_s=round(time.time() - t_start, 2),
            context={"embed_result": result},
        )
    except Exception as e:
        return DispatchResult(
            source_id=source_id, source_type="document",
            success=False, steps_run=[], steps_failed=["pipeline"],
            error=f"{type(e).__name__}: {e}",
            duration_s=round(time.time() - t_start, 2),
        )


def _dispatch_nosql(source_config: dict, verbose: bool) -> DispatchResult:
    source_id = source_config["id"]
    t_start   = time.time()
    _slog(source_id, f"NoSQL source  engine={source_config.get('engine', '?')}")

    try:
        from connectors.base import build_connector
        from ingestion.schema_unifier import nosql_collections_to_dict

        connector = build_connector(source_config)
        status    = connector.connect()
        if not status.ok:
            return DispatchResult(
                source_id=source_id, source_type="nosql",
                success=False, steps_run=[], steps_failed=[],
                error=f"Connection failed: {status.message}",
                duration_s=round(time.time() - t_start, 2),
            )

        t0          = time.time()
        collections = connector.get_nosql_schema()
        connector.disconnect()

        if not collections:
            _slog(source_id, "⚠  No collections found — skipping schema pipeline")
            return DispatchResult(
                source_id=source_id, source_type="nosql",
                success=True, steps_run=["connect", "schema_sample"], steps_failed=[],
                error=None,
                duration_s=round(time.time() - t_start, 2),
            )

        engine          = source_config.get("engine", "mongodb")
        raw_schema_dict = nosql_collections_to_dict(collections, source_id, engine)
        _sok(source_id,
             f"NoSQL schema — {len(collections)} collections, "
             f"{raw_schema_dict['stats']['total_columns']} fields",
             time.time() - t0)

        ctx = _run_schema_pipeline(
            raw_schema_dict   = raw_schema_dict,
            source_config     = source_config,
            run_data_graph    = False,
            run_value_sampler = False,
            verbose           = verbose,
        )
        ctx["nosql_collections"] = collections
        return DispatchResult(
            source_id=source_id, source_type="nosql",
            success=True,
            steps_run=["connect", "schema_sample", "schema", "fk",
                       "semantic", "metadata", "reg", "encoder",
                       "vector_store"],
            steps_failed=[],
            error=None,
            duration_s=round(time.time() - t_start, 2),
            context=ctx,
        )
    except Exception as e:
        return DispatchResult(
            source_id=source_id, source_type="nosql",
            success=False, steps_run=[], steps_failed=["pipeline"],
            error=f"{type(e).__name__}: {e}",
            duration_s=round(time.time() - t_start, 2),
        )


# =============================================================================
# Public entry point
# =============================================================================

def dispatch_ingestion(
    source_config: dict,
    verbose:       bool = False,
) -> DispatchResult:
    """
    Ingests one source end-to-end based on its type.

    Parameters
    ----------
    source_config : dict
        One entry from VEDA_SOURCES in config.py.
    verbose       : bool
        Print detailed progress per step.

    Returns
    -------
    DispatchResult — always returns; error field set on failure.
    """
    source_type = source_config.get("type", "relational")
    _dispatch = {
        "relational": _dispatch_relational,
        "datalake":   _dispatch_datalake,
        "document":   _dispatch_document,
        "nosql":      _dispatch_nosql,
    }.get(source_type)

    if _dispatch is None:
        return DispatchResult(
            source_id   = source_config.get("id", "?"),
            source_type = source_type,
            success     = False,
            steps_run   = [],
            steps_failed = [],
            error       = f"Unknown source type '{source_type}'",
            duration_s  = 0.0,
        )

    return _dispatch(source_config, verbose)
