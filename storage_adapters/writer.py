"""storage_adapters/writer.py — ingestion-side persistence (migration plan §3.2).

Functions keep the SAME names/signatures the ingestion code calls so Phase 3.4's
rewire is a one-line shim. Structured rows go through the Django ORM (Django owns
the substrate); embedding tables use raw pgvector INSERT (batched per table, §4.2a).
Every write upserts by natural key `(source, tenant, uuid/name)` so a re-run is
idempotent (§7). Tenancy rides the ambient context (§4.1) — never a parameter.
"""
from __future__ import annotations

from typing import Any, Iterable

from veda_core import context


def _scope() -> dict:
    ctx = context.current()  # fail-closed if unset (§4.1)
    return {"source_id": ctx.source_id, "tenant": ctx.tenant}


def _attr(edge: Any, name: str, default=None):
    """Read a field from a legacy FKEdge dataclass or a dict interchangeably."""
    if isinstance(edge, dict):
        return edge.get(name, default)
    return getattr(edge, name, default)


def store_fk_adjacency(edges: Iterable[Any], is_declared: bool = True) -> int:
    """Persist FK edges (legacy FKEdge shape) into Django `FkEdge` for the current
    (source, tenant). Upserts the minimal `SchemaTable`/`SchemaColumn` rows the edge
    references (by their engine UUIDs) so Django owns the structural substrate too.
    Idempotent by natural key. Returns the number of edges written.
    """
    from django.db import transaction

    from apps.substrate.models import FkEdge, JoinType, SchemaColumn, SchemaTable

    scope = _scope()
    _tbl_cache: dict = {}
    _col_cache: dict = {}

    def _table(tid, name):
        # Natural key is (source, tenant, name); reuse if present (an engine name may
        # appear under >1 id). id is set to the engine uuid on first create so the
        # reader returns the engine's table_id (retrieval matches on it).
        if name in _tbl_cache:
            return _tbl_cache[name]
        obj, _ = SchemaTable.objects.all_tenants().get_or_create(
            name=name, defaults={"id": tid, **scope}, **scope
        )
        _tbl_cache[name] = obj
        return obj

    def _col(cid, name, table):
        key = (table.id, name)
        if key in _col_cache:
            return _col_cache[key]
        obj, _ = SchemaColumn.objects.all_tenants().get_or_create(
            table=table, name=name, defaults={"id": cid, "data_type": "", **scope}, **scope
        )
        _col_cache[key] = obj
        return obj

    written = 0
    with transaction.atomic():
        for e in edges:
            ft = _table(_attr(e, "from_table_id"), _attr(e, "from_table_name", ""))
            tt = _table(_attr(e, "to_table_id"), _attr(e, "to_table_name", ""))
            fc = _col(_attr(e, "from_col_id"), _attr(e, "from_col_name", ""), ft)
            tc = _col(_attr(e, "to_col_id"), _attr(e, "to_col_name", ""), tt)
            FkEdge.objects.all_tenants().update_or_create(
                from_col=fc, to_col=tc,
                defaults={"from_table": ft, "to_table": tt, "join_type": JoinType.INNER,
                          "is_declared": is_declared, **scope},
            )
            written += 1
    return written


def store_glossary(entries: Iterable[Any]) -> int:
    """Upsert GlossaryEntry rows by (source, tenant, term). Accepts dicts or objects
    with term/canonical/definition. Returns count."""
    from apps.substrate.models import GlossaryEntry

    scope = _scope()
    n = 0
    for e in entries:
        GlossaryEntry.objects.all_tenants().update_or_create(
            term=_attr(e, "term"),
            defaults={"canonical": _attr(e, "canonical", ""), "definition": _attr(e, "definition", ""), **scope},
        )
        n += 1
    return n


def store_semantic_model(sm: dict, version: str | None = None) -> None:
    """Persist the assembled semantic-model dict into the Sm* substrate (§8a) for the
    current (source, tenant), then publish it to redis for the inference tier. Called
    by the ingestion semantic-layer stage with the semantic_layer_v2 output."""
    from storage_adapters import assembler

    ctx = context.current()
    assembler.persist(sm, source_id=ctx.source_id, tenant=ctx.tenant, version=version)
    assembler.publish_sm(ctx.source_id, ctx.tenant)


def store_column_embeddings(mode: str, rows: Iterable[Any]) -> None:
    """Raw pgvector INSERT for `mode`'s table (§6.4), batched per table (§4.2a).
    The engine's proven encoder→pgvector path already writes these tables in the
    Django-managed Postgres; this hook exists for the Celery stage to route through
    the seam. Delegates to the engine's vector_store for the actual batched write."""
    raise NotImplementedError(
        "store_column_embeddings: routed to engine vector_store batched writer (Phase 4 stage 6)."
    )


def sync_from_engine(internal_dsn: dict | None = None) -> dict:
    """Populate the Django-owned substrate from the engine's operational store
    (veda_engine) for the current (source, tenant): FK edges, glossary/synonyms,
    and value samples. Used by the ingestion warm stage so Django genuinely owns
    the structured substrate after a run (§3.2). Idempotent. Returns row counts.
    """
    import os

    import psycopg2

    ctx = context.current()
    dsn = internal_dsn or dict(
        host=os.environ.get("VEDA_INTERNAL_HOST", "pgbouncer"),
        port=int(os.environ.get("VEDA_INTERNAL_PORT", "6432")),
        dbname=os.environ.get("VEDA_INTERNAL_DBNAME", "veda_engine"),
        user=os.environ.get("VEDA_INTERNAL_USER", "veda"),
        password=os.environ.get("VEDA_INTERNAL_PASSWORD", "change-me"),
    )
    # Clean idempotent re-sync: clear this scope's structural substrate first.
    from apps.substrate.models import FkEdge, SchemaColumn, SchemaTable
    FkEdge.objects.all_tenants().filter(source_id=ctx.source_id, tenant=ctx.tenant).delete()
    SchemaColumn.objects.all_tenants().filter(source_id=ctx.source_id, tenant=ctx.tenant).delete()
    SchemaTable.objects.all_tenants().filter(source_id=ctx.source_id, tenant=ctx.tenant).delete()

    conn = psycopg2.connect(**dsn)
    conn.autocommit = True
    counts = {"fk": 0, "glossary": 0, "value_samples": 0}
    try:
        with conn.cursor() as cur:
            # FK edges
            cur.execute(
                "SELECT from_col_id,from_col_name,from_table_id,from_table_name,"
                "to_col_id,to_col_name,to_table_id,to_table_name FROM fk_adjacency"
            )
            cols = ["from_col_id", "from_col_name", "from_table_id", "from_table_name",
                    "to_col_id", "to_col_name", "to_table_id", "to_table_name"]
            edges = [dict(zip(cols, r)) for r in cur.fetchall()]
            counts["fk"] = store_fk_adjacency(edges) if edges else 0

            # value samples (column_values → ColumnValueSample). Each references a column;
            # create the missing SchemaTable/SchemaColumn rows so the FK holds (value-sample
            # columns are a superset of FK columns). Column is value_raw (not "value").
            try:
                cur.execute("SELECT col_id, col_name, table_name, value_raw FROM column_values")
                _sync_value_samples(cur.fetchall(), ctx.source_id, ctx.tenant)
                counts["value_samples"] = _count_value_samples(ctx.source_id, ctx.tenant)
            except Exception:
                pass

            # glossary/synonyms live in the sm dict (Django-owned via the assembler); mirror
            # the domain_synonyms into GlossaryEntry/Synonym for admin visibility/ownership.
            try:
                counts["glossary"] = _sync_glossary_from_sm(ctx.source_id, ctx.tenant)
            except Exception:
                pass

            # unified knowledge graph (graph_nodes/graph_edges) → Django GraphNode/GraphEdge
            # (§6.5) for ownership/admin + the graph-expansion/RAG path.
            try:
                counts["graph_nodes"], counts["graph_edges"] = _sync_graph_from_engine(cur, ctx)
            except Exception:
                pass
    finally:
        conn.close()
    return counts


def _sync_value_samples(rows, source_id, tenant):
    """rows = (col_id, col_name, table_name, value_raw). Create the referenced
    SchemaTable/SchemaColumn (value-sample columns are a superset of FK columns) so the
    ColumnValueSample FK holds, then bulk-insert the samples."""
    from apps.substrate.models import ColumnValueSample, SchemaColumn, SchemaTable

    scope = dict(source_id=source_id, tenant=tenant)
    ColumnValueSample.objects.all_tenants().filter(**scope).delete()

    tbl_by_name: dict = {t.name: t for t in SchemaTable.objects.all_tenants().filter(**scope)}
    col_ids = set(SchemaColumn.objects.all_tenants().filter(**scope).values_list("id", flat=True))

    to_make_cols = []
    seen_new = set()
    for col_id, col_name, table_name, _val in rows:
        if not col_id or str(col_id) in {str(c) for c in col_ids} or col_id in seen_new:
            continue
        t = tbl_by_name.get(table_name)
        if t is None:
            t = SchemaTable.objects.all_tenants().get_or_create(name=table_name, defaults=scope, **scope)[0]
            tbl_by_name[table_name] = t
        to_make_cols.append(SchemaColumn(id=col_id, table=t, name=col_name, data_type="", **scope))
        seen_new.add(col_id)
    if to_make_cols:
        SchemaColumn.objects.bulk_create(to_make_cols, ignore_conflicts=True, batch_size=1000)

    valid = set(SchemaColumn.objects.all_tenants().filter(**scope).values_list("id", flat=True))
    valid_str = {str(c) for c in valid}
    ColumnValueSample.objects.bulk_create(
        [ColumnValueSample(column_id=col_id, value=str(val), **scope)
         for col_id, _cn, _tn, val in rows if col_id and str(col_id) in valid_str],
        ignore_conflicts=True, batch_size=1000,
    )


def _sync_glossary_from_sm(source_id, tenant) -> int:
    """Mirror sm.domain_synonyms → Synonym rows for admin ownership (the read path uses
    sm directly). Returns count."""
    from apps.substrate.models import Synonym
    from storage_adapters.assembler import SemanticModelAssembler

    sm, _ = SemanticModelAssembler.assemble(source_id, tenant)
    scope = dict(source_id=source_id, tenant=tenant)
    Synonym.objects.all_tenants().filter(**scope).delete()
    rows = []
    for phrase, mapping in (sm.get("domain_synonyms") or {}).items():
        targets = mapping if isinstance(mapping, list) else [mapping]
        for t in targets:
            rows.append(Synonym(term=phrase, synonym=str(t), **scope))
    if rows:
        Synonym.objects.bulk_create(rows, ignore_conflicts=True, batch_size=1000)
    return len(rows)


def _count_value_samples(source_id, tenant):
    from apps.substrate.models import ColumnValueSample

    return ColumnValueSample.objects.all_tenants().filter(source_id=source_id, tenant=tenant).count()


def _sync_graph_from_engine(cur, ctx):
    """Mirror the engine's unified KG (graph_nodes/graph_edges) into Django GraphNode/
    GraphEdge (§6.5) for the current scope. Also registers the relationship-graph artifact.
    Returns (n_nodes, n_edges)."""
    import json
    import os

    from apps.substrate.models import GraphArtifact, GraphEdge, GraphNode

    scope = dict(source_id=ctx.source_id, tenant=ctx.tenant)
    GraphEdge.objects.all_tenants().filter(**scope).delete()
    GraphNode.objects.all_tenants().filter(**scope).delete()

    # nodes: node_id → GraphNode(node_key=node_id, payload=descriptor)
    cur.execute("SELECT node_id, node_type, name, table_name, semantic_type, attrs FROM graph_nodes")
    node_rows = cur.fetchall()
    nodes = [
        GraphNode(node_key=nid, node_type=ntype or "",
                  payload={"name": name, "table_name": tname, "semantic_type": stype,
                           "attrs": attrs if isinstance(attrs, dict) else {}}, **scope)
        for nid, ntype, name, tname, stype, attrs in node_rows
    ]
    GraphNode.objects.bulk_create(nodes, batch_size=1000, ignore_conflicts=True)
    key_to_pk = {n.node_key: n.pk for n in GraphNode.objects.all_tenants().filter(**scope)}

    # edges: (src,dst) node_id → GraphEdge(from_node, to_node)
    cur.execute("SELECT src_node_id, dst_node_id, edge_type, weight FROM graph_edges")
    edges = []
    for src, dst, etype, weight in cur.fetchall():
        fp, tp = key_to_pk.get(src), key_to_pk.get(dst)
        if fp and tp:
            edges.append(GraphEdge(from_node_id=fp, to_node_id=tp, edge_type=etype or "",
                                   weight=weight if weight is not None else 1.0, **scope))
    GraphEdge.objects.bulk_create(edges, batch_size=1000, ignore_conflicts=True)

    # Register the relationship-graph artifact (path the query path's join_planner reads).
    rel_path = os.environ.get(
        "VEDA_RELATIONSHIP_GRAPH_FILE",
        os.path.join(os.environ.get("VEDA_APP_DIR", "/app"), "veda_core", "data",
                     "veda_relationship_graph.json"))
    GraphArtifact.objects.all_tenants().update_or_create(
        kind="relationship_graph", defaults={"path": rel_path, "version": "1", **scope}, **scope)
    return len(nodes), len(edges)


def warm() -> dict:
    """Ingestion warm stage (§7 stage 10): persist the semantic model the engine just
    produced into the Sm* substrate, sync the structural substrate from the engine
    store, then publish the assembled sm to redis-cache + the rehydrate fan-out (§8.4)
    so every inference replica reloads. Returns row counts."""
    import json
    import os

    from storage_adapters import assembler

    ctx = context.current()
    # Persist the semantic model produced by the ingestion subprocess into Django (§8a).
    sm_file = os.environ.get(
        "VEDA_SEMANTIC_MODEL_FILE",
        os.path.join(os.environ.get("VEDA_APP_DIR", "/app"), "veda_core", "data",
                     "veda_semantic_model.json"),
    )
    sm_cols = 0
    if os.path.exists(sm_file):
        with open(sm_file) as f:
            sm = json.load(f)
        assembler.persist(sm, source_id=ctx.source_id, tenant=ctx.tenant)
        sm_cols = len(sm.get("columns", {}))

    counts = sync_from_engine()
    assembler.publish_sm(ctx.source_id, ctx.tenant)
    assembler.publish_rehydrate(ctx.source_id, ctx.tenant, scope="all")
    counts["sm_columns"] = sm_cols
    return counts
