# =============================================================================
# main.py
# VEDA POC — Orchestrator
#
# Runs the full POC end-to-end in a single command:
#
#   python main.py
#
# Execution order:
#   1. Schema scanner         (schema/real_schema.py + ingestion/schema_scanner.py)
#   2. FK Adjacency Store     (ingestion/vector_store.py)
#   3. Semantic type inference (ingestion/semantic_type_inference.py)
#   4. REG builder            (ingestion/reg_builder.py)
#   5. Encoder                (ingestion/relgt_encoder.py — mode set in config.py)
#   6. Vector store           (ingestion/vector_store.py)
#   7. Evaluation             (evaluation/evaluator.py — L1 + L2 + L3)
#   8. Report                 (evaluation/report.py)
#
# Optional flags:
#   --ingestion-only    Run steps 1–6 only, skip evaluation
#   --eval-only         Skip ingestion, run evaluation only (requires prior run)
#   --query "..."       Run a single NL query through L1 → L2 → L3
#   --verbose           Print detailed progress for every step
#
# Encoder mode and all parameters are set in config.py (ENCODER_MODE).
# All outputs are written to evaluation/results/
# =============================================================================

import sys
import os
import argparse
import time

# Zero-egress on-prem: force HuggingFace/transformers OFFLINE before any model-loading
# import, so cached BGE/MiniLM models load from disk instead of contacting huggingface.co.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Ensure project root is on the path regardless of where main.py is called from
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger

logger = get_logger(__name__)


def check_ingestion_status() -> dict:
    """Readiness for the HYBRID query engine (what `--query` and the demo actually run).

    The hybrid engine reads exactly three artifacts; ONLY these gate readiness:
      • data/veda_semantic_model.json       — semantic model (routing, display, grounding)
      • column_embeddings_v2 (primary DB)    — BGE retrieval, the live vector store
      • data/veda_relationship_graph.json    — join planner / fast path / graph guard

    Legacy ensemble artifacts (relgt_weights, tfidf/svd, column_embeddings_lt/hybrid) and the
    table_metadata table are NOT read by the hybrid path — display columns come from the
    semantic model + overrides.json (veda.generation._resolve_display_column) — so they must
    never block, or `--query` triggers a needless full re-ingestion every run. Doc-source chunk
    counts are reported for the demo's status display but only matter for RAG on that source, so
    they don't gate SQL/hybrid readiness either. Return shape unchanged: {ready, checks}.
    """
    from config import SEMANTIC_MODEL_FILE, BIENCODER_COL_TABLE, DOC_CHUNKS_TABLE_NAME
    from ingestion.db_abstraction import get_internal_connection

    base = os.path.dirname(os.path.abspath(__file__))

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    def _add_doc_source_checks(cur, checks: dict) -> None:
        """Report per-source doc-chunk presence (informational only — NOT gating). Keyed off
        VEDA_SOURCES so a freshly-enabled doc source is visible in the demo's status panel."""
        try:
            from config import get_enabled_sources
            doc_sources = get_enabled_sources("document")
        except Exception:
            doc_sources = []
        for src in doc_sources:
            sid = src.get("id", "")
            try:
                cur.execute(
                    f'SELECT COUNT(*) FROM {DOC_CHUNKS_TABLE_NAME} WHERE source_id = %s', (sid,))
                checks[f"doc_chunks[{sid}]"] = cur.fetchone()[0] > 0
            except Exception:
                try: cur.connection.rollback()
                except Exception: pass
                checks[f"doc_chunks[{sid}]"] = False

    # ── Core gate: the artifacts the hybrid engine cannot answer without ──
    core = {
        "semantic_model":     os.path.exists(_abs(SEMANTIC_MODEL_FILE)),
        "relationship_graph": os.path.exists(_abs("data/veda_relationship_graph.json")),
    }
    info: dict = {}   # reported for visibility, never gates readiness

    try:
        conn = get_internal_connection()
        cur  = conn.cursor()
        try:
            cur.execute(f"SELECT count(*) FROM {BIENCODER_COL_TABLE}")
            core["column_embeddings_v2"] = cur.fetchone()[0] > 0
        except Exception:
            try: conn.rollback()
            except Exception: pass
            core["column_embeddings_v2"] = False
        _add_doc_source_checks(cur, info)
        cur.close()
        conn.close()
    except Exception:
        core["column_embeddings_v2"] = False

    checks = {**core, **info}
    return {"ready": all(core.values()), "checks": checks}


def _load_query_singletons(verbose: bool = False) -> None:
    """Load in-memory singletons for query pipeline without re-ingestion."""
    # TF-IDF/SVD power only the light-text/ensemble retrieval
    # (query/semantic_layer._encode_light_text), used by the Tier-2 fallback and the
    # evaluator. The hybrid engine retrieves with BGE, so skip the eager load on the
    # V2 path. get_light_text_models() is a lazy cached getter — the Tier-2/eval paths
    # still load it on demand, so nothing is lost.
    try:
        from config import RETRIEVAL_V2_ENABLED
        if not RETRIEVAL_V2_ENABLED:
            from ingestion.relgt_encoder import get_light_text_models
            get_light_text_models()
            if verbose: print("  [Singletons] TF-IDF + SVD loaded ✅")
        elif verbose:
            print("  [Singletons] TF-IDF + SVD skipped (RETRIEVAL_V2 primary path)")
    except Exception as e:
        print(f"  [Singletons] TF-IDF warning: {e}")

    try:
        from config import RETRIEVAL_V2_ENABLED
        if not RETRIEVAL_V2_ENABLED:
            from ingestion.relgt_encoder import _get_minilm_model
            _get_minilm_model()
            if verbose: print("  [Singletons] MiniLM loaded ✅")
        elif verbose:
            print("  [Singletons] MiniLM skipped (RETRIEVAL_V2 primary path)")
    except Exception as e:
        if verbose: print(f"  [Singletons] MiniLM warning: {e}")

    try:
        from ingestion.value_sampler import rebuild_value_index_from_db
        n = rebuild_value_index_from_db()
        if verbose: print(f"  [Singletons] Value index: {n} terms ✅")
    except Exception as e:
        print(f"  [Singletons] Value index warning: {e}")

    try:
        from config import BIENCODER_ENABLED, RETRIEVAL_V2_ENABLED
        if RETRIEVAL_V2_ENABLED and BIENCODER_ENABLED:
            from ingestion.biencoder import _get_biencoder
            _get_biencoder()
            if verbose: print("  [Singletons] BGE loaded ✅")
    except Exception as e:
        print(f"  [Singletons] BGE warning: {e}")


def _header() -> None:
    from config import ENCODER_MODE, POC_LABEL, SLM_MODEL_NAME, SLM_ENABLED
    encoder_line = f"Encoder: {ENCODER_MODE:<10}  L3 SLM: {'on · ' + SLM_MODEL_NAME if SLM_ENABLED else 'off'}"
    # Pad to fit the box (62 chars between ║ and ║)
    inner = f"  {encoder_line}"
    inner = inner[:60].ljust(60)
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  VEDA POC — Natural Language to SQL Pipeline                ║")
    print(f"║  {inner}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def _step(n: int, total: int, label: str) -> None:
    print(f"  [{n}/{total}] {label}")
    print(f"  {'─' * 56}")


def _ok(label: str, duration: float) -> None:
    print(f"  ✓  {label}  ({round(duration, 2)}s)")
    print()


def _fail(label: str, error: Exception) -> None:
    logger.error("Step FAILED: %s — %s: %s", label, type(error).__name__, error, exc_info=True)
    print(f"  ✗  {label} FAILED")
    print(f"     {type(error).__name__}: {error}")
    print()


def _print_query_summary(query: str, l2, l3, l4) -> None:
    """Print a compact per-query metric summary against POC baseline values."""
    from config import TOP_K, POC_LABEL
    from evaluation.report import BASELINE_ESTIMATES
    from evaluation.test_queries import get_all_queries

    W   = 64
    bar = "═" * W
    scored  = [r for r in (l2.top_k_columns if l2 else []) if r.similarity > 0.0]
    avg_sim = sum(r.similarity for r in scored) / len(scored) if scored else 0.0
    bl = BASELINE_ESTIMATES

    # ── Match against test suite for ground-truth metrics ─────────────────
    test_match = next(
        (tq for tq in get_all_queries() if tq.query.strip().lower() == query.strip().lower()),
        None,
    )
    if test_match and l2:
        retrieved = set((r.table_name, r.col_name) for r in l2.top_k_columns)
        expected  = set(test_match.expected_columns)
        matched   = retrieved & expected
        cur_prec  = round(len(matched) / TOP_K,          3)
        cur_rec   = round(len(matched) / len(expected),  3) if expected else 0.0
        cur_hit   = 1.0 if matched else 0.0
    else:
        cur_prec = cur_rec = cur_hit = None

    def _fmt(val, base):
        if val is None:
            return f"{'—':>7}   {'—':>7}    {'—':>7}"
        delta = val - base
        return f"{val:>7.3f}   {base:>7.3f}   {delta:>+.3f}"

    print()
    print(bar)
    print("  VEDA — Query Run Summary")
    print(f"  {POC_LABEL}")
    print(bar)
    print()

    # ── Metrics vs baseline ───────────────────────────────────────────────
    print("  METRICS vs BASELINE")
    if test_match:
        print(f"    (matched test query {test_match.query_id} · {test_match.query_type} · {test_match.difficulty})")
    else:
        print(f"    (no ground truth — query not in test suite)")
    print(f"    {'Metric':<16}  {'Current':>7}   {'Baseline':>7}   {'Δ':>7}")
    print(f"    {'─'*52}")
    print(f"    {'Precision@'+str(TOP_K):<16}  {_fmt(cur_prec, bl['overall_precision'])}")
    print(f"    {'Recall@'+str(TOP_K):<16}  {_fmt(cur_rec,  bl['overall_recall'])}")
    print(f"    {'Hit Rate@'+str(TOP_K):<16}  {_fmt(cur_hit,  bl['overall_hit_rate'])}")
    print()

    # ── L2 detail ─────────────────────────────────────────────────────────
    print("  L2 RETRIEVAL")
    print(f"    Columns retrieved : {len(scored)}/{TOP_K}")
    print(f"    Avg similarity    : {avg_sim:.4f}")
    print(f"    Encoding strategy : {l2.encoding_strategy if l2 else '—'}")
    print(f"    Duration          : {l2.duration_ms if l2 else '—'}ms")
    print()

    # ── L3 ────────────────────────────────────────────────────────────────
    if l3 is not None and not l3.error:
        note = "(≥ avg)" if l3.confidence >= 0.75 else "(< avg)"
        print("  L3 SLM")
        print(f"    Intent      : {l3.intent}")
        print(f"    Complexity  : {l3.complexity}")
        print(f"    Confidence  : {l3.confidence:.3f}  {note}   (POC avg ~0.75)")
        print(f"    Duration    : {l3.duration_ms}ms")
        print()
    else:
        reason = (l3.error if l3 and l3.error else "SLM_ENABLED=False")
        print(f"  L3 SLM             : OFFLINE  ({reason})")
        print()

    # ── L4 ────────────────────────────────────────────────────────────────
    if l4 is not None and not l4.error:
        warn_count  = len(l4.warnings) if l4.warnings else 0
        sql_status  = f"✓   (baseline 100%)" if warn_count == 0 else f"⚠  built with {warn_count} warning(s) — SQL may be incomplete"
        print("  L4 SQL BUILDER")
        print(f"    Query type  : {l4.query_type}")
        print(f"    Params      : {len(l4.params)} bound value(s)")
        print(f"    Duration    : {l4.duration_ms:.1f}ms")
        print(f"    SQL success : {sql_status}")
        print()
    elif l4 is not None and l4.error:
        print(f"  L4 SQL BUILDER     : ✗  {l4.error}")
        print()
    else:
        print("  L4 SQL BUILDER     : N/A (no IR JSON from L3)")
        print()
    print(bar)


# =============================================================================
# Ingestion pipeline
# =============================================================================

def run_ingestion(verbose: bool = False, skip_llm: bool = False) -> dict:
    """
    Runs Steps 1–11: full ingestion pipeline.
    Returns a context dict passed to the evaluation stage.

    skip_llm=True runs the FAST, LLM-free embedding chain only — it skips the three
    Ollama/Qwen steps (9b semantic layer, 10 synthetic-query-gen, 11 fine-tune). The
    biencoder still embeds rich docs if a semantic model already exists on disk from a
    prior run; otherwise it falls back to structural text. This is the old
    `embed_only.py` behaviour, now folded in (see `--embed-only`).
    """
    from config import get_primary_relational_source
    primary_source = get_primary_relational_source()
    source_id      = primary_source["id"]

    total_steps = 11
    context     = {"source_id": source_id}
    t_total     = time.time()
    logger.info("=== Ingestion pipeline started: source_id=%s ===", source_id)

    # Resume (§4.2a/P8-B5): VEDA_RESUME=1 makes each EXPENSIVE stage skip when its persisted
    # output already exists, while the fast prep stages always re-run to rebuild the in-memory
    # context. A resumed job thus continues from the first stage whose output is missing.
    _resume = os.environ.get("VEDA_RESUME") == "1"

    def _table_has_rows(table: str) -> bool:
        try:
            from ingestion.db_abstraction import get_internal_connection, release_internal_connection
            conn = get_internal_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT 1 FROM {table} LIMIT 1")
                    return cur.fetchone() is not None
            finally:
                release_internal_connection(conn)
        except Exception:
            return False

    def _biencoder_embeddings_exist() -> bool:
        from config import BIENCODER_COL_TABLE
        return _table_has_rows(BIENCODER_COL_TABLE)

    # --- Step 1: Schema simulation ---
    _step(1, total_steps, "Schema Scanner (real schema)")
    t0 = time.time()
    try:
        # from schema.simulate_schema import get_simulated_schema
        from schema.real_schema import get_real_schema
        from ingestion.schema_scanner import run_schema_scanner

        # raw_schema  = get_simulated_schema()
        raw_schema = get_real_schema()
        scan_result = run_schema_scanner(raw_schema=raw_schema, verbose=verbose)
        context["scan_result"] = scan_result
        _ok(
            f"Schema scanned — {scan_result.stats['total_tables']} tables, "
            f"{scan_result.stats['total_columns']} columns, "
            f"{scan_result.stats['total_fk_edges']} FK edges",
            time.time() - t0,
        )
    except Exception as e:
        _fail("Schema Scanner", e)
        raise

    # --- Step 1b: FK adjacency store ---
    # Independent of encoder mode — persists FK edges from scan_result
    _step(2, total_steps, "FK Adjacency Store")
    t0 = time.time()
    try:
        from ingestion.vector_store import store_fk_adjacency

        fk_result = store_fk_adjacency(scan_result, verbose=verbose)
        context["fk_result"] = fk_result
        _ok(
            f"FK edges stored — {fk_result.edges_written} edges, "
            f"backend={fk_result.backend}",
            time.time() - t0,
        )
    except Exception as e:
        _fail("FK Adjacency Store", e)
        raise

    # --- Step 2b: Data Graph (undeclared FK discovery) ---
    # Samples actual DB data to find undeclared FK relationships.
    # Results merged into fk_adjacency store.
    # Gracefully skipped if DB unavailable or DATA_GRAPH_ENABLED=False.
    _step(3, total_steps, "Data Graph (undeclared FK discovery)")
    t0 = time.time()
    try:
        from ingestion.data_graph import run_data_graph, to_fk_adjacency_rows
        from ingestion.vector_store import store_fk_adjacency

        dg_result = run_data_graph(scan_result, source_id=source_id, verbose=verbose)
        context["dg_result"] = dg_result

        # Merge HIGH + MEDIUM discovered edges into fk_adjacency
        if dg_result.discovered_edges:
            discovered_rows = to_fk_adjacency_rows(dg_result, include_soft=False)
            if discovered_rows:
                # Append to existing fk_adjacency (re-store with combined edges)
                combined_edges = list(scan_result.fk_edges) + discovered_rows
                scan_result.fk_edges = combined_edges
                store_fk_adjacency(scan_result, verbose=False)

        _ok(
            f"Data graph — "
            f"HIGH={dg_result.stats.get('high_certainty', 0)} "
            f"MEDIUM={dg_result.stats.get('medium_certainty', 0)} "
            f"SOFT={dg_result.stats.get('soft_certainty', 0)} "
            f"discovered edges",
            time.time() - t0,
        )
    except Exception as e:
        _fail("Data Graph", e)
        # Non-fatal — pipeline continues without discovered edges
        print("         Continuing without data graph edges.")

    # --- Step 4: Semantic type inference ---
    _step(4, total_steps, "Semantic Type Inference")
    t0 = time.time()
    try:
        from ingestion.semantic_type_inference import run_semantic_type_inference

        inference_result = run_semantic_type_inference(
            scan_result = scan_result,
            verbose     = verbose,
        )
        context["inference_result"] = inference_result
        stats = inference_result.stats
        _ok(
            f"Types inferred — avg confidence={stats['avg_confidence']}, "
            f"flagged={stats['flagged_count']}, "
            f"LayerA={stats['layer_counts']['A']} "
            f"LayerB={stats['layer_counts']['B']} "
            f"LayerC={stats['layer_counts']['C']}",
            time.time() - t0,
        )
    except Exception as e:
        _fail("Semantic Type Inference", e)
        raise

    # --- Step 4: Table Metadata Store ---
    # Persists display column per table (identified in semantic type inference).
    # Used at query time to inject human-readable display identifiers.
    _step(5, total_steps, "Table Metadata Store (display columns)")
    t0 = time.time()
    try:
        from ingestion.vector_store import store_table_metadata

        tm_result = store_table_metadata(inference_result, verbose=verbose)
        context["tm_result"] = tm_result
        _ok(
            f"Display columns stored — "
            f"{tm_result.rows_written} tables, "
            f"backend={tm_result.backend}",
            time.time() - t0,
        )
    except Exception as e:
        _fail("Table Metadata Store", e)
        raise

    # --- Step 6: Value Sampler ---
    _step(6, total_steps, "Value Sampler (column value indexing)")
    t0 = time.time()
    try:
        from ingestion.value_sampler import run_value_sampler

        vs_result = run_value_sampler(inference_result, source_id=source_id, verbose=verbose)
        context["vs_result"] = vs_result
        _ok(
            f"Values sampled — "
            f"{vs_result.columns_sampled} columns, "
            f"{vs_result.total_values} values, "
            f"backend={vs_result.backend}",
            time.time() - t0,
        )
    except Exception as e:
        _fail("Value Sampler", e)
        print("         Continuing without value expansion.")

    # --- Step 7: REG builder ---
    _step(7, total_steps, "REG Builder (Relational Entity Graph)")
    t0 = time.time()
    try:
        from ingestion.reg_builder import run_reg_builder

        graph = run_reg_builder(
            inference_result = inference_result,
            verbose          = verbose,
        )
        context["graph"] = graph
        _ok(
            f"Graph built — {graph.stats['num_table_nodes']} table nodes, "
            f"{graph.stats['num_column_nodes']} column nodes, "
            f"{graph.stats['num_fk_to_edges']} FK edges, "
            f"torch_geometric={'yes' if graph.stats['torch_geometric'] else 'no (numpy fallback)'}",
            time.time() - t0,
        )
    except Exception as e:
        _fail("REG Builder", e)
        raise

    # --- Step 7b: Unified Graph Persist ---
    from config import UNIFIED_GRAPH_ENABLED, GRAPH_PERSIST_ENABLED
    if UNIFIED_GRAPH_ENABLED and GRAPH_PERSIST_ENABLED:
        print(f"  [7b/{total_steps}] Unified Graph Persist (graph_nodes / graph_edges)")
        print(f"  {'─' * 56}")
        t0 = time.time()
        try:
            from ingestion.graph_persist import persist_reg_graph
            gp = persist_reg_graph(
                graph       = context["graph"],
                scan_result = context["scan_result"],
                dg_result   = context.get("dg_result"),
                source_id   = source_id,
                verbose     = verbose,
            )
            context["graph_persist_result"] = gp
            _ok(
                f"Graph persisted — {gp.nodes_written} nodes, {gp.edges_written} edges, "
                f"backend={gp.backend}",
                time.time() - t0,
            )
        except Exception as e:
            _fail("Unified Graph Persist", e)
            print("         Continuing without persisted graph.")

    # --- Step 7c: Unified Graph Embeddings ---
    from config import GRAPH_EMBED_ENABLED
    if UNIFIED_GRAPH_ENABLED and GRAPH_EMBED_ENABLED:
        print(f"  [7c/{total_steps}] Unified Graph Embedder (graph_node_embeddings)")
        print(f"  {'─' * 56}")
        t0 = time.time()
        try:
            from ingestion.graph_embedder import embed_graph_nodes
            ge = embed_graph_nodes(source_id=source_id, verbose=verbose)
            context["graph_embed_result"] = ge
            _ok(
                f"Graph nodes embedded — {ge.nodes_embedded} nodes, "
                f"dim=384, backend={ge.backend}",
                time.time() - t0,
            )
        except Exception as e:
            _fail("Unified Graph Embedder", e)
            print("         Continuing without unified node embeddings.")

    # --- Step 8: Encoder ---
    _step(8, total_steps, f"Encoder  [mode={__import__('config').ENCODER_MODE}]")
    t0 = time.time()
    try:
        from ingestion.relgt_encoder import run_relgt_encoder, EnsembleEncoderResult

        encoder_result = run_relgt_encoder(
            graph   = graph,
            verbose = verbose,
        )
        context["encoder_result"] = encoder_result

        # Build ok message — ensemble result has different stats shape
        if isinstance(encoder_result, EnsembleEncoderResult):
            _ok(
                f"Embeddings generated (ensemble) — "
                f"{encoder_result.stats['total_embeddings']} columns, "
                f"lt_dim={encoder_result.lt_embedding_dim}, "
                f"hybrid_dim={encoder_result.hybrid_embedding_dim}",
                time.time() - t0,
            )
        else:
            _ok(
                f"Embeddings generated — {encoder_result.stats['total_embeddings']} columns, "
                f"dim={encoder_result.stats['embedding_dim']}, "
                f"encoder={encoder_result.stats['encoder_type']}, "
                f"minilm_used={encoder_result.stats['minilm_used']}",
                time.time() - t0,
            )
    except Exception as e:
        _fail("Encoder", e)
        raise

    # --- Step 9: Vector store ---
    _step(9, total_steps, "Vector Store (pgvector / in-memory fallback)")
    t0 = time.time()
    try:
        from ingestion.vector_store import run_vector_store

        store_result = run_vector_store(
            encoder_result = encoder_result,
            source_id      = context.get("source_id", ""),
            verbose        = verbose,
        )
        context["store_result"] = store_result

        # Build ok message — ensemble result has different shape
        from ingestion.vector_store import EnsembleStoreResult
        if isinstance(store_result, EnsembleStoreResult):
            _ok(
                f"Stored (ensemble) — "
                f"lt={store_result.lt_result.rows_written} rows, "
                f"hybrid={store_result.hybrid_result.rows_written} rows, "
                f"backend={store_result.backend}",
                time.time() - t0,
            )
        else:
            _ok(
                f"Stored — {store_result.rows_written} rows, "
                f"dim={store_result.vector_dim}, "
                f"backend={store_result.backend}, "
                f"index={'created' if store_result.index_created else 'skipped (< 100 rows)'}",
                time.time() - t0,
            )
    except Exception as e:
        _fail("Vector Store", e)
        raise

    # --- Step 9b: Semantic Layer v2 (Qwen) ---
    # MUST run before the biencoder embed below. It builds and SAVES the semantic
    # model (retrieval_documents / glossary / synonyms / concept_graph) to disk.
    # The biencoder step that follows reads retrieval_documents from that file
    # (biencoder._load_retrieval_docs) and embeds the rich NL text into the live
    # _v2 store instead of weak structural text. Without this, the docs never exist
    # and the live store falls back to structural strings. Non-fatal: if Qwen/Ollama
    # is unavailable the biencoder still populates _v2 with structural fallback text.
    from config import SEMANTIC_LAYER_V2_ENABLED, SEMANTIC_MODEL_FILE as _SMF
    # Resume-skip: skip this expensive LLM stage if its output (the semantic model file) exists.
    if _resume and os.path.exists(_SMF):
        print(f"  [9b/{total_steps}] Semantic Layer v2 — SKIPPED (resume: semantic model exists)")
        context["semantic_model"] = None
    elif SEMANTIC_LAYER_V2_ENABLED and not skip_llm:
        print(f"  [9b/{total_steps}] Semantic Layer v2 (retrieval docs / glossary / synonyms / concepts)")
        print(f"  {'─' * 56}")
        t0 = time.time()
        try:
            from schema.real_schema import get_real_schema as _get_real_schema
            from ingestion.semantic_layer_v2 import run_full_semantic_layer, save_semantic_model
            from config import SEMANTIC_MODEL_FILE

            _raw = _get_real_schema()
            schema_dict = {
                t["table_name"]: {"columns": t.get("columns", [])}
                for t in _raw.get("tables", [])
            }
            # force_glossary=True → regenerate data/veda_glossary.json every ingest so
            # the query-time glossary (query_enrichment) never lags the current schema.
            semantic_model = run_full_semantic_layer(
                schema_dict=schema_dict, profiling=None, glossary=None, force_glossary=True,
            )
            save_semantic_model(semantic_model, SEMANTIC_MODEL_FILE)
            context["semantic_model"] = semantic_model
            _ok(
                f"Semantic model built — {len(semantic_model.get('tables', {}))} tables, "
                f"{len(semantic_model.get('retrieval_documents', {}))} retrieval docs, "
                f"{len(semantic_model.get('domain_synonyms', {}))} synonyms, "
                f"{len(semantic_model.get('concept_graph', {}))} concepts",
                time.time() - t0,
            )
        except Exception as e:
            _fail("Semantic Layer v2", e)
            print("         Continuing — biencoder will fall back to structural text.")

    # BGE biencoder ingestion → populates column_embeddings_v2
    # Reads retrieval_documents (written by Step 9b above) when present, so the live
    # store is embedded with rich semantic text per config.EMBED_TEXT_STRATEGY.
    try:
        from config import BIENCODER_ENABLED
        if _resume and _biencoder_embeddings_exist():
            print(f"  [primary_db] BGE biencoder — SKIPPED (resume: embeddings already present)")
        elif BIENCODER_ENABLED:
            from ingestion.biencoder import run_biencoder_ingestion
            bge_result = run_biencoder_ingestion(
                inference_result, source_id=source_id, verbose=verbose)
            if verbose:
                s = f"{bge_result.cols_embedded} cols" if not bge_result.error else f"warning: {bge_result.error}"
                print(f"  [primary_db] ✓  BGE biencoder — {s}")
    except Exception as _e:
        if verbose:
            print(f"  [primary_db] BGE biencoder warning: {_e}")

    # --- Step 12: Derived artifacts (LLM-free) ---
    # Rebuild the files the FAST PATH reads so they never lag the semantic model:
    #   data/veda_relationship_graph.json  → join_planner / fast_path / graph_guard
    #   semantic/{concepts,dimensions,metrics,MANIFEST}.json → fast-path registry
    # Pure transforms of veda_semantic_model.json (+ DB structure) — no Ollama — so
    # they run in --embed-only mode too. Non-fatal: a failure leaves the prior files.
    from config import DERIVED_ARTIFACTS_ENABLED
    if DERIVED_ARTIFACTS_ENABLED:
        print(f"  [12/{total_steps}] Derived artifacts (relationship graph + semantic registry)")
        print(f"  {'─' * 56}")
        t0 = time.time()
        try:
            from ingestion.relationship_graph import build_relationship_graph
            g = build_relationship_graph(verbose=verbose)
            n_edges = len(g.get("edges", [])) if isinstance(g, dict) else 0
            _ok(f"relationship graph rebuilt — {n_edges} edges → data/veda_relationship_graph.json",
                time.time() - t0)
        except Exception as e:
            _fail("Relationship Graph", e)
            print("         Continuing — fast path will use the previous relationship graph.")

        t0 = time.time()
        try:
            from semantic.compile_semantic_layer import compile_all
            compiled = compile_all(write=True)
            _ok(f"semantic registry recompiled — "
                f"{len(compiled.get('concepts', {}))} concepts, "
                f"{len(compiled.get('dimensions', {}))} dimensions, "
                f"{len(compiled.get('metrics', {}))} metrics → semantic/*.json",
                time.time() - t0)
        except Exception as e:
            _fail("Semantic Registry Compile", e)
            print("         Continuing — fast path will use the previous semantic registry.")

    # --- Step 12b: Unified graph (query-time GRAPH_EXPAND) ---
    # Regenerate data/veda_unified_graph.json from the fresh semantic model +
    # relationship graph (Step 12) + concept graph + domain synonyms (Step 9b), so
    # query-time graph expansion (query_graph.py ← veda/pipeline) never reads a
    # stale/missing file. Non-fatal: a failure leaves the prior unified graph.
    from config import UNIFIED_GRAPH_ENABLED as _UG_ENABLED
    if _UG_ENABLED:
        print(f"  [12b/{total_steps}] Unified graph (data/veda_unified_graph.json)")
        print(f"  {'─' * 56}")
        t0 = time.time()
        try:
            from ingestion.unified_graph_builder import build_unified_graph, write_unified_graph
            _ug = build_unified_graph()
            _ug_path = write_unified_graph(_ug)
            _ok(f"unified graph rebuilt — {len(_ug.get('nodes', []))} nodes, "
                f"{len(_ug.get('edges', []))} edges → {_ug_path}",
                time.time() - t0)
        except Exception as e:
            _fail("Unified Graph", e)
            print("         Continuing — graph expansion will use the previous unified graph.")

    ingestion_duration = round(time.time() - t_total, 2)
    logger.info("=== Ingestion pipeline complete in %.2fs ===", ingestion_duration)
    print(f"  ── Ingestion complete in {ingestion_duration}s ──")
    print()

    return context


# =============================================================================
# Document ingestion pipeline (Phase 2)
# =============================================================================

def run_doc_ingestion(verbose: bool = False) -> None:
    """
    Ingests all enabled document sources from VEDA_SOURCES.
    For each source: connects, extracts chunks, embeds, stores in doc_chunks.
    """
    from config import get_enabled_sources
    from connectors.base import build_connector
    from ingestion.chunk_embedder import run_chunk_embedder

    doc_sources = get_enabled_sources("document")
    if not doc_sources:
        print("  ⚠  No enabled document sources found in VEDA_SOURCES.")
        print("     Add a source with type='document' and enabled=True to config.py.")
        return

    for src in doc_sources:
        print(f"\n  Source: '{src['id']}'  path={src.get('path', '?')}")
        t0 = time.time()
        try:
            connector = build_connector(src)
            status    = connector.connect()
            if not status.ok:
                print(f"  ✗ Connection failed: {status.message}")
                continue

            chunks = list(connector.get_chunks())
            connector.disconnect()

            if not chunks:
                print(f"  ⚠  No chunks extracted (directory empty or no supported files)")
                continue

            result = run_chunk_embedder(chunks, src["id"], verbose=verbose)
            print(
                f"  ✓  {result.chunks_embedded} chunks embedded, "
                f"{result.docs_processed} docs, "
                f"backend={result.backend}  ({round(time.time() - t0, 2)}s)"
            )

            from config import UNIFIED_GRAPH_ENABLED, GRAPH_CHUNK_LINKING_ENABLED
            if UNIFIED_GRAPH_ENABLED and GRAPH_CHUNK_LINKING_ENABLED:
                try:
                    from ingestion.chunk_linker import link_chunks_to_graph
                    cl = link_chunks_to_graph(
                        chunks           = chunks,
                        chunk_embeddings = result.embeddings,
                        source_id        = src["id"],
                        verbose          = verbose,
                    )
                    print(
                        f"  ✓  graph links: {cl.chunk_nodes_written} chunk nodes, "
                        f"{cl.link_edges_written} edges {cl.stats}"
                    )

                    from config import GRAPH_EMBED_ENABLED
                    if GRAPH_EMBED_ENABLED:
                        from ingestion.graph_embedder import embed_graph_nodes
                        ge = embed_graph_nodes(source_id=src["id"], verbose=verbose)
                        print(f"  ✓  graph doc embeddings: {ge.nodes_embedded} chunk nodes embedded")
                except Exception as e:
                    print(f"  ⚠  Graph chunk linking failed ({type(e).__name__}: {e}) — continuing")
        except Exception as e:
            print(f"  ✗ '{src['id']}' failed: {type(e).__name__}: {e}")


# =============================================================================
# NoSQL ingestion pipeline (Phase 4)
# =============================================================================

def run_nosql_ingestion(verbose: bool = False) -> None:
    """
    Samples schema from all enabled NoSQL sources in VEDA_SOURCES.
    Schema info is printed/logged; full pgvector ingestion is Phase 5.
    """
    from config import get_enabled_sources
    from connectors.base import build_connector

    nosql_sources = get_enabled_sources("nosql")
    if not nosql_sources:
        print("  ⚠  No enabled NoSQL sources found in VEDA_SOURCES.")
        print("     Add a source with type='nosql' and enabled=True to config.py.")
        return

    for src in nosql_sources:
        print(f"\n  Source: '{src['id']}'  engine={src.get('engine', '?')}")
        t0 = time.time()
        try:
            connector = build_connector(src)
            status    = connector.connect()
            if not status.ok:
                print(f"  ✗ Connection failed: {status.message}")
                continue

            collections = connector.get_nosql_schema()
            connector.disconnect()

            if not collections:
                print(f"  ⚠  No collections found or schema inference returned nothing")
                continue

            total_fields = sum(len(c.inferred_fields) for c in collections)
            print(
                f"  ✓  {len(collections)} collections, "
                f"{total_fields} fields inferred  "
                f"({round(time.time() - t0, 2)}s)"
            )
            if verbose:
                for col in collections:
                    field_names = ", ".join(f["name"] for f in col.inferred_fields[:8])
                    more = f" +{len(col.inferred_fields)-8} more" if len(col.inferred_fields) > 8 else ""
                    print(f"     {col.collection_name}: {col.doc_count} docs — {field_names}{more}")
        except Exception as e:
            print(f"  ✗ '{src['id']}' failed: {type(e).__name__}: {e}")


# =============================================================================
# Unified multi-source ingestion (Phase 5)
# =============================================================================

def run_all_ingestion(verbose: bool = False) -> None:
    """
    Ingests all enabled sources in VEDA_SOURCES via source_dispatcher.
    Routes each source to its correct pipeline automatically:
      relational (primary) → delegates to run_ingestion (the unified 12-step pipeline)
      relational (other)   → shared schema pipeline
      datalake             → schema-compatible pipeline (no data_graph / value_sampler)
      document             → chunk embedding pipeline
      nosql                → schema inference + embedding pipeline
    """
    from config import get_enabled_sources
    from ingestion.source_dispatcher import dispatch_ingestion

    sources = get_enabled_sources()
    if not sources:
        print("  ⚠  No enabled sources found in VEDA_SOURCES.")
        return

    t_total = time.time()
    for src in sources:
        print(f"  ── Source '{src['id']}'  [{src['type']}] ──────────────────────")
        print()
        result = dispatch_ingestion(src, verbose=verbose)
        if result.success:
            print(f"\n  ✓  '{src['id']}' complete in {result.duration_s}s")
        else:
            print(f"\n  ✗  '{src['id']}' FAILED: {result.error}")
        print()

    print(f"  ── All sources ingested in {round(time.time() - t_total, 2)}s ──")
    print()


# =============================================================================
# Evaluation pipeline
# =============================================================================

def run_evaluation_pipeline(verbose: bool = False) -> None:
    """
    Runs evaluation + report generation (2 steps, numbered independently).
    """
    total_steps = 2

    # --- Step 1/2: Evaluator ---
    _step(1, total_steps, "Evaluation — running test suite")
    t0 = time.time()
    try:
        from evaluation.evaluator import run_evaluation

        eval_report = run_evaluation(verbose=verbose)
        _ok(
            f"Evaluation complete — "
            f"P={eval_report.overall_precision}  "
            f"R={eval_report.overall_recall}  "
            f"Hit={eval_report.overall_hit_rate}  "
            f"({eval_report.total_queries} queries)",
            time.time() - t0,
        )
    except Exception as e:
        _fail("Evaluator", e)
        raise

    # --- Step 2/2: Report ---
    _step(2, total_steps, "Report generation")
    t0 = time.time()
    try:
        from evaluation.report import run_report

        run_report(eval_report=eval_report, verbose=verbose)
        _ok("Reports written", time.time() - t0)
    except Exception as e:
        _fail("Report", e)
        raise


# =============================================================================
# Single-query mode — canonical hybrid engine (veda_hybrid)
# =============================================================================

def run_single_query(query: str, verbose: bool = False, debug: bool = False) -> None:
    """
    Run one NL query through the CANONICAL hybrid engine (veda_hybrid.run_hybrid_query):
      • sql    → deterministic engine (+ Tier-2 LLM-IR fallback, graph-guarded)
      • rag    → integrated document retrieval + synthesis
      • hybrid → deterministic SQL rows ⊕ document fusion
      • nosql  → native query builder + execution
      • compound utterances → decomposed into independent sub-queries

    This replaces the legacy L1→L4 SLM→IR path (now removed). It is the SAME engine
    the demo and hybrid suite use, so the CLI, demo, and tests all converge on one
    query layer.

    Requires ingestion to have been run first (vector store + semantic model populated).
    """
    import veda_hybrid

    logger.info("=== Single query (hybrid): %r ===", query)
    print(f"\n  Query: '{query}'")
    print(f"  {'─' * 56}")

    # --debug → full explainability trace (incl. candidate lists). Mirrors
    # veda_hybrid.main(): the flags must be set BEFORE the query so the engine records
    # the trace, then render_trace() prints it AFTER.
    if debug:
        import config as _cfg
        _cfg.EXPLAIN_TRACE_ENABLED = True
        _cfg.EXPLAIN_TRACE_VERBOSE = True

    # L0 — NL Simplifier (parity with the legacy path + the demo backend, which both
    # simplify before retrieval). The hybrid engine itself does not run L0, so apply it
    # here. NOTE: the cleaner long-term home is inside veda_hybrid.run_hybrid_query so
    # every consumer (CLI, demo, hybrid suite) shares it — fold it in when unifying demo.
    try:
        from query.nl_simplifier import run_nl_simplifier
        _l0 = run_nl_simplifier(query, verbose=False)
        if _l0.was_simplified:
            print(f"  [L0] Simplified: '{_l0.simplified_query}' ({_l0.duration_ms}ms)")
        query = _l0.simplified_query
    except Exception:
        pass  # fall back to the original query silently

    result = veda_hybrid.run_hybrid_query(query, verbose=verbose)
    veda_hybrid._render_multi(result)   # compound recap + refusal surfacing (per-head output already streamed)

    if debug:
        from veda.explain import render_trace
        for it in result.items:
            if isinstance(it.result, dict) and it.result.get("trace"):
                print("\n" + render_trace(it.result["trace"]))
    print()


# =============================================================================
# Argument parser
# =============================================================================

def _parse_args():
    from config import ENCODER_MODE
    parser = argparse.ArgumentParser(
        description=f"VEDA POC — NL to SQL pipeline  [encoder={ENCODER_MODE}]",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ingestion-only",
        action="store_true",
        help="Run ingestion pipeline only (steps 1–5). Skip evaluation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only. Requires ingestion to have been run first.",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Fast LLM-free re-embed: run the structural + embedding steps only,\n"
             "skipping the Qwen steps (semantic layer, synthetic-gen, fine-tune).\n"
             "Reuses an existing semantic model on disk if present.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help='Run a single query through the hybrid engine and print results.\nExample: --query "show me total rent by tenant"',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="With --query, print the full explainability trace (candidate lists, scores).",
    )
    parser.add_argument(
        "--ingest-docs",
        action="store_true",
        help="Run document ingestion for all enabled document sources in VEDA_SOURCES.\nEmbeds document chunks into the doc_chunks pgvector table.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-generate HTML report from existing evaluation/results/poc_results.json.\nNo ingestion or evaluation needed.",
    )
    parser.add_argument(
        "--build-glossary",
        action="store_true",
        help="Build (or rebuild) the domain glossary and exit.\nReads schema from DB, generates SLM synonyms via Ollama, saves to glossary/.",
    )
    parser.add_argument(
        "--rebuild-glossary",
        action="store_true",
        help="Force-rebuild glossary even if cache exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress for every pipeline step.",
    )
    return parser.parse_args()


# =============================================================================
# Entry point
# =============================================================================

def main():
    args = _parse_args()
    _header()

    t_start = time.time()

    # ------------------------------------------------------------------
    # Mode: document ingestion (Phase 2)
    # ------------------------------------------------------------------
    if args.ingest_docs:
        print("  Mode: DOCUMENT INGESTION\n")
        run_doc_ingestion(verbose=args.verbose)
        print(f"\n  Total time: {round(time.time() - t_start, 2)}s")
        return

    # ------------------------------------------------------------------
    # Mode: build glossary
    # ------------------------------------------------------------------
    if args.build_glossary or args.rebuild_glossary:
        print("  Mode: BUILD GLOSSARY\n")
        from schema.real_schema import get_real_schema
        from ingestion.schema_scanner import run_schema_scanner
        from ingestion.semantic_type_inference import run_semantic_type_inference
        from ingestion.domain_glossary import build_glossary
        from config import SLM_OLLAMA_BASE_URL
        print("  Step 1/3 — Scanning schema...")
        raw_schema       = get_real_schema()
        scan_result      = run_schema_scanner(raw_schema=raw_schema, verbose=False)
        print(f"  Step 2/3 — Semantic type inference ({scan_result.stats['total_columns']} columns)...")
        inference_result = run_semantic_type_inference(scan_result=scan_result, verbose=False)
        print("  Step 3/3 — Building glossary (Layer C → B → A)...")
        glossary = build_glossary(
            inference_result = inference_result,
            ollama_url       = SLM_OLLAMA_BASE_URL,
            force_rebuild    = args.rebuild_glossary,
        )
        print(f"\n  ✓  Glossary ready: {len(glossary)} terms")
        print(f"  Total time: {round(time.time() - t_start, 2)}s")
        return

    # ------------------------------------------------------------------
    # Mode: report-only (re-generate HTML from saved JSON)
    # ------------------------------------------------------------------
    if args.report_only:
        print("  Mode: REPORT ONLY (from saved JSON)\n")
        json_path = "evaluation/results/poc_results.json"
        if not os.path.exists(json_path):
            print(f"  ✗  No JSON found at {json_path}")
            print("     Run the full pipeline first to generate results.")
            return
        from evaluation.report import load_report_from_json, run_report
        print(f"  Loading results from {json_path} ...")
        eval_report = load_report_from_json(json_path)
        run_report(eval_report=eval_report, verbose=args.verbose)
        print(f"  Total time: {round(time.time() - t_start, 2)}s")
        return

    # ------------------------------------------------------------------
    # Mode: single query
    # ------------------------------------------------------------------
    if args.query:
        print("  Mode: SINGLE QUERY\n")
        status = check_ingestion_status()
        if not status["ready"]:
            missing = [k for k, v in status["checks"].items() if not v]
            print(f"  Artifacts missing: {missing}")
            run_all_ingestion(verbose=args.verbose)
        else:
            print("  Artifacts found — loading singletons...\n")
            _load_query_singletons(verbose=args.verbose)
        run_single_query(args.query, verbose=args.verbose, debug=args.debug)
        return

    # ------------------------------------------------------------------
    # Mode: embed only (fast, LLM-free re-embed of the primary relational source)
    # ------------------------------------------------------------------
    if args.embed_only:
        print("  Mode: EMBED ONLY (fast, no LLM)\n")
        run_ingestion(verbose=args.verbose, skip_llm=True)
        print(f"  Total time: {round(time.time() - t_start, 2)}s")
        return

    # ------------------------------------------------------------------
    # Mode: ingestion only
    # ------------------------------------------------------------------
    if args.ingestion_only:
        print("  Mode: INGESTION ONLY\n")
        run_all_ingestion(verbose=args.verbose)
        print(f"  Total time: {round(time.time() - t_start, 2)}s")
        return

    # ------------------------------------------------------------------
    # Mode: evaluation only (skip ingestion — use existing store)
    # ------------------------------------------------------------------
    if args.eval_only:
        print("  Mode: EVALUATION ONLY\n")
        print("  ⚠  Assuming ingestion was run previously.")
        print("     In-memory store must be populated in this process.")
        print("     If store is empty, run without --eval-only first.\n")
        run_evaluation_pipeline(verbose=args.verbose)
        print(f"  Total time: {round(time.time() - t_start, 2)}s")
        return

    # ------------------------------------------------------------------
    # Mode: full run (default)
    # ------------------------------------------------------------------
    print("  Mode: FULL RUN (ingestion + evaluation)\n")

    print("  ── INGESTION PIPELINE ──────────────────────────────────────")
    print()
    run_all_ingestion(verbose=args.verbose)

    print("  ── EVALUATION PIPELINE ─────────────────────────────────────")
    print()
    run_evaluation_pipeline(verbose=args.verbose)

    total = round(time.time() - t_start, 2)
    print()
    print(f"  ✓  Full POC run complete in {total}s")
    print(f"  ✓  HTML report  → evaluation/results/poc_report.html")
    print(f"  ✓  JSON results → evaluation/results/poc_results.json")
    print()


if __name__ == "__main__":
    main()