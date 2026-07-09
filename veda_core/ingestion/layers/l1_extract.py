"""L1 EXTRACT — the only layer that touches the tenant's source.

After L1 completes, ingestion can finish even if the source goes down. Wraps the
existing schema-scan / FK-adjacency / data-graph / value-sampler stages verbatim.
"""
from __future__ import annotations

import os
from typing import Dict, List

from ingestion.contracts import SourceContext, StageOutcome


def run(ctx: SourceContext, state: Dict, verbose: bool = False) -> List[StageOutcome]:
    out: List[StageOutcome] = []
    source_id = ctx.source_id

    # --- schema scan (real, live introspection) — FATAL on failure -------------
    # Connector-aware (Cross-source plan P2.1): file-backed tabular sources
    # (csv_lake/parquet/xlsx) build their raw schema from the DuckDB tabular
    # connector; relational sources use the live DB introspection. Either way the
    # SAME run_schema_scanner + downstream L1–L5 run, so a CSV column is
    # indistinguishable from a Postgres column.
    try:
        from ingestion.schema_scanner import run_schema_scanner
        engine = (ctx.engine or "").lower()
        if engine in _TABULAR_ENGINES:
            from connectors.tabular_files import TabularFileConnector
            path = (ctx.connection or {}).get("path") or (ctx.connection or {}).get("source_path")
            conn = TabularFileConnector({"id": ctx.source_id, "engine": engine, "path": path})
            st = conn.connect()
            if not st.ok:
                raise RuntimeError(f"tabular connect failed: {st.message}")
            raw_schema = conn.get_raw_schema_dict()
            state["tabular_connector"] = conn      # reused for materialization + sampling
        else:
            from schema.real_schema import get_real_schema
            raw_schema = get_real_schema()
        scan_result = run_schema_scanner(raw_schema=raw_schema, verbose=verbose)
        state["scan_result"] = scan_result
        s = scan_result.stats
        out.append(StageOutcome("schema_scan", True, detail=(
            f"{s['total_tables']} tables, {s['total_columns']} cols, "
            f"{s['total_fk_edges']} FK edges")))
    except Exception as e:
        out.append(StageOutcome("schema_scan", False, fatal=True, error=str(e)))
        return out

    # --- Parquet materialization (tabular sources only, P2.2) — non-fatal --------
    # Write each file table to canonical typed Parquet under the source's artifact
    # dir; this is the Phase-5 federated execution surface (the source file is never
    # re-parsed at query time).
    if (ctx.engine or "").lower() in _TABULAR_ENGINES and state.get("tabular_connector"):
        try:
            from config import artifact_path
            out_dir = artifact_path(os.path.join("tables"))
            written = state["tabular_connector"].materialize_parquet(out_dir)
            state["materialized_parquet"] = written
            out.append(StageOutcome("materialize_parquet", True,
                                    detail=f"{len(written)} tables → {out_dir}"))
        except Exception as e:
            out.append(StageOutcome("materialize_parquet", False, fatal=False, error=str(e)))

    # --- FK adjacency store — FATAL -------------------------------------------
    try:
        from ingestion.vector_store import store_fk_adjacency
        fk_result = store_fk_adjacency(scan_result, verbose=verbose)
        state["fk_result"] = fk_result
        out.append(StageOutcome("fk_adjacency", True, detail=(
            f"{fk_result.edges_written} edges, backend={fk_result.backend}")))
    except Exception as e:
        out.append(StageOutcome("fk_adjacency", False, fatal=True, error=str(e)))
        return out

    # --- data graph (undeclared FK discovery) — non-fatal ----------------------
    try:
        from ingestion.data_graph import run_data_graph, to_fk_adjacency_rows
        from ingestion.vector_store import store_fk_adjacency
        dg_result = run_data_graph(scan_result, source_id=source_id, verbose=verbose)
        state["dg_result"] = dg_result
        if dg_result.discovered_edges:
            rows = to_fk_adjacency_rows(dg_result, include_soft=False)
            if rows:
                scan_result.fk_edges = list(scan_result.fk_edges) + rows
                store_fk_adjacency(scan_result, verbose=False)
        st = dg_result.stats
        out.append(StageOutcome("data_graph", True, detail=(
            f"HIGH={st.get('high_certainty', 0)} MED={st.get('medium_certainty', 0)} "
            f"SOFT={st.get('soft_certainty', 0)}")))
    except Exception as e:
        out.append(StageOutcome("data_graph", False, fatal=False, error=str(e)))

    return out


def run_value_sampling(ctx: SourceContext, state: Dict, verbose: bool = False) -> StageOutcome:
    """Value sampler — L1 extract but sequenced after L2 type inference (needs
    inference_result), so composed by pipeline after l2. Non-fatal (matches today)."""
    try:
        from ingestion.value_sampler import run_value_sampler
        vs = run_value_sampler(state["inference_result"], source_id=ctx.source_id, verbose=verbose)
        state["vs_result"] = vs
        return StageOutcome("value_profiling", True, detail=(
            f"{vs.columns_sampled} cols, {vs.total_values} values, backend={vs.backend}"))
    except Exception as e:
        return StageOutcome("value_profiling", False, fatal=False, error=str(e))


_TABULAR_ENGINES = ("csv", "csv_lake", "parquet", "xlsx", "excel")


def _distinct_sampler(ctx: SourceContext):
    """Return a callable (table, col, n) -> [values] that reads DISTINCT values for
    the sketch pass. Relational → the client DB (read-only SELECT DISTINCT, works on
    PK/FK/id columns the value sampler skips); tabular file → the DuckDB connector."""
    engine = (ctx.engine or "").lower()
    if engine in _TABULAR_ENGINES:
        from connectors.tabular_files import TabularFileConnector
        path = (ctx.connection or {}).get("path") or (ctx.connection or {}).get("source_path")
        conn = TabularFileConnector({"id": ctx.source_id, "engine": engine, "path": path})
        conn.connect()
        return lambda t, c, n: conn.sample_column_values(t, c, n)
    # relational: one client connection reused across columns
    from ingestion.db_abstraction import get_client_connection
    client = get_client_connection(ctx.source_id)
    cur = client.cursor()

    def _sample(table_name, col_name, n):
        try:
            q = lambda s: '"' + str(s).replace('"', "") + '"'
            cur.execute(f"SELECT DISTINCT {q(col_name)} FROM {q(table_name)} "
                        f"WHERE {q(col_name)} IS NOT NULL LIMIT %s", (n,))
            return [r[0] for r in cur.fetchall()]
        except Exception:
            try:
                client.rollback()
            except Exception:
                pass
            return []
    _sample._client = client  # keep a handle so the caller can close it
    return _sample


def run_sketch_pass(ctx: SourceContext, state: Dict, verbose: bool = False) -> StageOutcome:
    """Cross-source plan P2.4/P4.2: MinHash-sketch every join-key-shaped column so
    the tenant-wide cross_source_fk discovery (L5) can find value overlaps ACROSS
    sources. Unlike the value sampler this INCLUDES PK/FK/id columns (the join keys),
    which is exactly what cross-source discovery needs. Non-fatal, and a graceful
    no-op when datasketch is absent (so ingestion never breaks on this stage)."""
    try:
        from ingestion import column_sketches as CS
        from config import CROSS_SOURCE_SKETCH_SAMPLE_SIZE, SENSITIVE_PATTERNS
        if not CS.sketches_available():
            return StageOutcome("column_sketches", True, detail="skipped (datasketch absent)")
        inf = state.get("inference_result")
        if inf is None:
            return StageOutcome("column_sketches", True, detail="skipped (no inference_result)")

        def _shape(tc) -> bool:
            name = (tc.col_name or "").lower()
            if any(p in name for p in SENSITIVE_PATTERNS):
                return False
            return bool(getattr(tc, "is_pk", False) or getattr(tc, "is_fk", False)
                        or (tc.semantic_type or "").upper() in CS.SKETCHABLE_TYPES)

        def _vclass(tc) -> str:
            if getattr(tc, "is_pk", False) or getattr(tc, "is_fk", False):
                return "id"
            return CS.value_class(tc.semantic_type, getattr(tc, "data_type", ""))

        cols = [tc for tc in inf.typed_columns if _shape(tc)]
        sampler = _distinct_sampler(ctx)
        rows = []
        try:
            for tc in cols:
                vals = sampler(tc.table_name, tc.col_name, CROSS_SOURCE_SKETCH_SAMPLE_SIZE)
                sketch, n, vhashes = CS.compute_sketch(vals)
                if sketch is None:
                    continue
                rows.append({"col_id": tc.col_id, "table_name": tc.table_name,
                             "col_name": tc.col_name, "n_distinct": n,
                             "value_class": _vclass(tc), "sketch": sketch,
                             "value_hashes": vhashes})
        finally:
            client = getattr(sampler, "_client", None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        written = CS.persist_sketches(rows, source_id=ctx.source_id, tenant=ctx.tenant)
        return StageOutcome("column_sketches", True,
                            detail=f"{written}/{len(cols)} join-key columns sketched")
    except Exception as e:
        return StageOutcome("column_sketches", False, fatal=False, error=str(e))
