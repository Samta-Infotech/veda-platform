# =============================================================================
# ingestion/relationship_graph.py
# VEDA V1.5 — deterministic relationship graph (the foundation for join planning)
#
# Builds veda_relationship_graph.json at ingestion time. Runtime NEVER infers
# relationships — it only reads this file.
#
# Edge sources (in priority):
#   1. declared FK            (schema is_fk / fk_ref_table / fk_ref_col)
#   2. polymorphic            (object_id + object_type/model_name) — resolved by
#                             DATA CORRELATION, not string-matching, because:
#                               • the join key may be a business key, not the PK
#                                 (annotation_record.object_id = counterparty_details.counterparty_id)
#                               • some discriminator values are categories, not
#                                 entity pointers (e.g. 'SAR', 'Level 1') → non-joinable
#
# Each edge carries: type, weight (deterministic), cardinality (inferred),
# polymorphic flag, requires_predicate, discovery, confidence.
# =============================================================================

import sys
import os
import json
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import get_primary_relational_source, source_artifact_path, artifact_path
from utils.logger import get_logger
import psycopg2
from schema.real_schema import get_real_schema

logger = get_logger(__name__)

# Legacy flat default — kept ONLY for a ctx-less call (dev CLI / `__main__` below).
# Every real ingestion run now persists per-(tenant, source) via
# config.source_artifact_path() (P0-1, 2026-09-10): before this fix, EVERY source
# wrote here, so source B's publish silently overwrote source A's graph and both
# read whichever ran last — see docs/backlog/query-engine-open-items.md.
RELATIONSHIP_GRAPH_FILE = artifact_path("veda_relationship_graph.json")

# Mirrors ingestion/dispatcher.py's _TABULAR_ENGINES / ingestion/layers/l1_extract.py:
# file-backed sources run the full L1-L5 pipeline over a DuckDB-backed connector, not
# a live SQL server — there is no pg_index/INFORMATION_SCHEMA to introspect for
# cardinality or PKs (P0-2).
_TABULAR_ENGINES = ("csv", "csv_lake", "parquet", "xlsx", "excel")

# deterministic weights by edge type (lower = preferred path)
_WEIGHT = {
    "business_core": 1, "bridge": 1, "reference": 2, "lookup": 2,
    "polymorphic": 2, "data_inferred": 3, "audit": 10, "history": 10,
}
_AUDIT_TABLE_RE = re.compile(r"(_history|_log|_audit|_archive)$", re.I)
# Ownership / "who touched this row" FK columns. These are a near-universal naming
# convention (NOT db-specific), and they all point at the users table — which turns
# `user` into a hub that falsely connects every table to every other table via
# "edited by the same person". Classifying them as audit (weight 10) keeps the edge
# usable for a DIRECT "X and its <owner>" join but stops the planner routing THROUGH
# user to bridge two unrelated business tables (10+10 ≫ any real path).
_AUDIT_COL_RE = re.compile(r"_(by|by_id|by_group)$|^(owned_by|assigned_to)", re.I)
_POLY_VALUE_SAMPLE = 200      # object_id values sampled per discriminator value
_MATCH_FLOOR = 0.80           # min correlation to accept a polymorphic/inferred edge


def _engine_of(ctx) -> str:
    """Normalized engine name for a ctx (or the legacy ctx-less Postgres default)."""
    return ((ctx.engine if ctx is not None else None) or "postgresql").lower()


# Dialect-specific identifier quoting / value-list / text-cast helpers (P0-2 follow-up,
# 2026-09-15). Everything below this point used to be written ONLY against Postgres
# syntax (double-quoted identifiers, `::text` casts, `= ANY(%s)` array binds) even
# though `_table_meta`'s information_schema query was already ANSI-portable — these
# three were the actual remaining Postgres-only dependencies. Verified live against a
# throwaway MySQL 8 container (see docs/backlog/query-engine-open-items.md) — mysql's
# `information_schema.table_constraints`/`key_column_usage` join matches Postgres's
# shape exactly, so only quoting/casting/set-membership syntax needed to change.
_MYSQL_ENGINES = ("mysql",)


def _ident(engine: str, name: str) -> str:
    """Quote a bare identifier for the given engine dialect."""
    if engine in _MYSQL_ENGINES:
        return f"`{name.replace(chr(96), '')}`"
    return f'"{name}"'


def _cast_text(engine: str, quoted_ident: str) -> str:
    """Cast an already-quoted identifier to a text/string type, per dialect."""
    if engine in _MYSQL_ENGINES:
        return f"CAST({quoted_ident} AS CHAR)"
    return f"{quoted_ident}::text"


def _in_clause(n: int) -> str:
    """A dialect-neutral `(%s, %s, ...)` for `col IN (...)` with n bound params —
    avoids Postgres-only `= ANY(%s)` array binding, which mysql-connector-python
    (and most non-psycopg2 DB-API drivers) don't support."""
    return "(" + ",".join(["%s"] * n) + ")"


def _conn(ctx=None):
    """A DB-API connection to the source being ingested, dialect-dispatched on
    ``ctx.engine`` (P0-2 follow-up, 2026-09-15 — was unconditionally psycopg2;
    ``_can_sql_introspect`` gated every non-Postgres engine to declared-FK-only mode
    specifically BECAUSE this function couldn't connect to anything else yet).

    ``ctx`` (an ``ingestion.contracts.SourceContext``) carries the ingesting
    source's OWN connection dict — passed in by ``layers/l5_publish.py`` (P0-2,
    2026-09-10). Falls back to ``get_primary_relational_source()`` (the
    currently env-injected source) only for a ctx-less call — the dev-CLI
    ``__main__`` path below, and callers on older call sites that haven't been
    updated to pass ``ctx`` yet. In this codebase's single-source-per-process
    ingestion model that fallback is byte-identical to passing ``ctx``: the
    injected source IS the one being ingested in that process either way."""
    cfg = (ctx.connection if ctx is not None and ctx.connection else None) or get_primary_relational_source()
    engine = _engine_of(ctx)

    if engine in _MYSQL_ENGINES:
        import mysql.connector
        return mysql.connector.connect(
            host=cfg["host"], port=cfg.get("port", 3306), database=cfg["dbname"],
            user=cfg["user"], password=cfg.get("password", ""))

    conn = psycopg2.connect(host=cfg["host"], port=cfg["port"], dbname=cfg["dbname"],
                            user=cfg["user"], password=cfg.get("password", ""))
    # P0-2: the unqualified "{table}" identifiers in _polymorphic_edges/_cardinality below
    # rely on search_path, not the DEFAULT "public" the rest of this module used to
    # assume — a source declaring a non-public schema (ctx.schema_filter, or a bare
    # ctx-less call's cfg["schema"]) previously read/joined against the wrong tables
    # (or none) whenever its schema wasn't literally "public". Same idiom veda/runtime.py
    # already uses for the query-time connection.
    schema = (ctx.schema_filter if ctx is not None else None) or cfg.get("schema")
    if schema and schema != "public":
        try:
            from psycopg2 import sql as _sql
            with conn.cursor() as _cur:
                _cur.execute(_sql.SQL("SET search_path TO {}, public").format(_sql.Identifier(schema)))
            conn.commit()
        except Exception:
            logger.warning("relationship_graph: could not set search_path to %r; "
                           "continuing on the connection default", schema)
    return conn


def _raw_schema_for(ctx=None):
    """The raw schema dict for the source being ingested — connector-aware, mirroring
    ``ingestion/layers/l1_extract.py``'s own dispatch (P0-2): a file-backed tabular
    source builds its schema from the DuckDB connector directly (``get_real_schema()``
    always assumes a live relational connection and raises for anything else);
    everything else uses the legacy relational shim, which is correct here because
    this process was launched FOR ``ctx.source_id`` (single-source-per-process)."""
    engine = ((ctx.engine if ctx is not None else "") or "").lower()
    if ctx is not None and engine in _TABULAR_ENGINES:
        from connectors.tabular_files import TabularFileConnector
        path = (ctx.connection or {}).get("path") or (ctx.connection or {}).get("source_path")
        conn = TabularFileConnector({"id": ctx.source_id, "engine": engine, "path": path})
        st = conn.connect()
        if not st.ok:
            raise RuntimeError(f"relationship_graph: tabular connect failed: {st.message}")
        try:
            return conn.get_raw_schema_dict()
        finally:
            conn.disconnect()
    return get_real_schema()


def _q(cur, sql, args=None):
    cur.execute(sql, args or [])
    return cur.fetchall()


def _table_meta(cur, tables, schema="public", engine="postgresql"):
    """Per table: columns, PK, unique 'key-like' columns, and column data types.
    ``schema`` (P0-2) is the SOURCE's declared schema, not a hardcoded 'public' —
    a Postgres source using a non-default schema previously got empty metadata for
    every table (the information_schema filter never matched). ``engine`` (P0-2
    follow-up) only affects identifier quoting below — the INFORMATION_SCHEMA query
    itself is unchanged, verified ANSI-portable to MySQL's `table_constraints`/
    `key_column_usage` (same join shape, same PK marker `constraint_type='PRIMARY
    KEY'`)."""
    meta = {}
    for t in tables:
        qt = _ident(engine, t)
        rows = _q(cur,
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema=%s AND table_name=%s", [schema, t])
        cols = [r[0] for r in rows]
        dtypes = {r[0]: r[1] for r in rows}
        # PK via INFORMATION_SCHEMA (ANSI-standard: table_constraints + key_column_usage)
        # instead of pg_index/pg_attribute (P0-2) — those Postgres system catalogs were
        # the one remaining Postgres-only dependency in this function; INFORMATION_SCHEMA
        # works identically on Postgres and is the same query a future MySQL/SQL Server
        # source would need, so there is now only one PK-detection code path to trust.
        pks = [r[0] for r in _q(cur,
            "SELECT kcu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON kcu.constraint_name = tc.constraint_name "
            " AND kcu.table_schema = tc.table_schema "
            "WHERE tc.constraint_type = 'PRIMARY KEY' "
            "  AND tc.table_schema = %s AND tc.table_name = %s", [schema, t])]
        rowcount = _q(cur, f"SELECT count(*) FROM {qt}")[0][0]
        # key-like = PK + columns that are unique (candidate business keys)
        key_cols = list(pks)
        for c in cols:
            if c in key_cols:
                continue
            # cheap uniqueness check only for id/key/code/no-suffixed columns
            if re.search(r"(_id|_no|_code|_key|_number)$", c) or c.endswith("id"):
                d = _q(cur, f"SELECT count(DISTINCT {_ident(engine, c)}) FROM {qt}")[0][0]
                if rowcount and d >= rowcount * 0.95:
                    key_cols.append(c)
        meta[t] = {"columns": cols, "pk": pks, "rowcount": rowcount,
                   "key_cols": key_cols, "dtypes": dtypes}
    return meta


_STRING_TYPES = ("character varying", "varchar", "text", "char", "uuid", "citext")


def _name_affinity(disc_value, target_table):
    """Does the discriminator value relate to the target table name?
    e.g. 'counterparty' ↔ counterparty_details ✓ ; 'User' ↔ counterparty_details ✗"""
    try:
        from retrieval.query_enrichment import _singularize
    except Exception:
        def _singularize(w): return w.rstrip("s")
    v = _singularize(re.sub(r"[^a-z]", "", str(disc_value).lower()))
    if len(v) < 3:
        return False
    tbl_tokens = {_singularize(tok) for tok in target_table.lower().split("_")}
    return v in tbl_tokens or any(v in tok or tok in v for tok in tbl_tokens)


def _cardinality(cur, child_t, child_col, parent_t, parent_col, engine="postgresql"):
    """1:1 / N:1 / 1:N from distinctness on each side."""
    try:
        qct, qcc = _ident(engine, child_t), _ident(engine, child_col)
        qpt, qpc = _ident(engine, parent_t), _ident(engine, parent_col)
        ch_rows = _q(cur, f"SELECT count(*) FROM {qct}")[0][0]
        ch_distinct = _q(cur, f"SELECT count(DISTINCT {qcc}) FROM {qct}")[0][0]
        pa_distinct = _q(cur, f"SELECT count(DISTINCT {qpc}) FROM {qpt}")[0][0]
        pa_rows = _q(cur, f"SELECT count(*) FROM {qpt}")[0][0]
        child_unique = ch_rows and ch_distinct >= ch_rows * 0.95
        parent_unique = pa_rows and pa_distinct >= pa_rows * 0.95
        if child_unique and parent_unique:
            return "1:1"
        if parent_unique:
            return "N:1"      # many child rows → one parent
        return "N:M"
    except Exception:
        return "unknown"


def _declared_fk_edges(schema_tables):
    edges = []
    for t in schema_tables:
        tname = t["table_name"]
        for c in t.get("columns", []):
            if c.get("is_fk") and c.get("fk_ref_table"):
                edges.append({
                    "source_table": tname, "source_column": c["col_name"],
                    "target_table": c["fk_ref_table"], "target_column": c.get("fk_ref_col") or "id",
                    "relationship_type": "audit" if (_AUDIT_TABLE_RE.search(tname)
                        or _AUDIT_COL_RE.search(c["col_name"])) else "business_core",
                    "discovery": "declared_fk", "polymorphic": False,
                    "requires_predicate": None, "confidence": 1.0,
                })
    return edges


def _discovered_fk_edges(tables, ctx=None):
    """Undeclared FK edges the data graph DISCOVERED (value-overlap correlation,
    ingestion/data_graph.py) for THIS source's tables, read back from the engine store's
    `fk_adjacency` (M1 close-out, 2026-09-15). Until now discovered edges only ever fed
    that store (retrieval Signals 3/4) — never this graph, so the join planner, the
    firewall's join check and the fast path saw declared FKs only. A file-backed
    (csv/parquet) source has NO declared FKs at all, so its graph was always empty even
    when the data proved a join (source 4: maintenance.ticket_id ↔ vendors.ticket_id,
    100% overlap, found and then dropped on the floor).

    `fk_adjacency` carries no source_id column; rows are matched on BOTH the table name
    (must be one of this source's tables) AND the table id, taken from the engine store's
    own `graph_nodes` rows for THIS source (always source-scoped) — so a same-named table
    in another source can't leak in. No graph_nodes for the source → no discovered edges."""
    tset = set(tables)
    if not tset or ctx is None:
        return []
    sid = str(ctx.source_id)
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT table_id FROM graph_nodes WHERE source_id = %s "
                            "AND node_type = 'table' AND table_id IS NOT NULL", [sid])
                expected_ids = {str(r[0]) for r in cur.fetchall()}
                if not expected_ids:
                    return []
                ph = ",".join(["%s"] * len(tset))
                cur.execute(
                    "SELECT from_table_name, from_col_name, to_table_name, to_col_name, "
                    "from_table_id, to_table_id FROM fk_adjacency "
                    f"WHERE from_table_name IN ({ph}) AND to_table_name IN ({ph})",
                    list(tset) + list(tset))
                rows = cur.fetchall()
        finally:
            release_internal_connection(conn)
    except Exception as e:
        logger.warning("relationship_graph: could not read discovered FK edges (%s: %s) — "
                       "declared edges only", type(e).__name__, str(e)[:120])
        return []
    edges = []
    seen_pairs = set()
    for ft, fc, tt, tc, fid, tid in rows:
        if str(fid) not in expected_ids or str(tid) not in expected_ids:
            continue                                   # a same-named table from another source
        # The data graph records a 100%-overlap pair in BOTH directions; the join planner
        # then sees "two keys" and refuses as ambiguous ("which key: ticket_id, ticket_id?").
        # Keep one direction per unordered column pair (first seen, i.e. store order).
        pair = frozenset(((ft, fc), (tt, tc)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        edges.append({
            "source_table": ft, "source_column": fc,
            "target_table": tt, "target_column": tc,
            "relationship_type": "audit" if (_AUDIT_TABLE_RE.search(ft) or _AUDIT_COL_RE.search(fc))
                                 else "data_inferred",
            "discovery": "data_inferred", "polymorphic": False,
            "requires_predicate": None, "confidence": 0.9,
        })
    return edges


def _merge_edges(declared, discovered):
    """Declared edges win; a discovered edge is added only for a (source col → target col)
    pair no declared edge already covers."""
    seen = {(e["source_table"], e["source_column"], e["target_table"], e["target_column"])
            for e in declared}
    out = list(declared)
    for e in discovered:
        k = (e["source_table"], e["source_column"], e["target_table"], e["target_column"])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def _polymorphic_edges(cur, tables, meta, engine="postgresql"):
    """Detect *_id + (*_type|model_name) pairs and resolve each discriminator value
    to a target table.column by DATA CORRELATION (not string matching)."""
    edges = []
    for t in tables:
        cols = meta[t]["columns"]
        id_cols = [c for c in cols if c == "object_id" or c.endswith("_object_id")]
        disc_cols = [c for c in cols if c in ("object_type", "model_name") or c.endswith("_type")]
        if not id_cols or not disc_cols:
            continue
        id_col, disc_col = id_cols[0], disc_cols[0]
        qt, qid, qdisc = _ident(engine, t), _ident(engine, id_col), _ident(engine, disc_col)

        for (val,) in _q(cur, f"SELECT DISTINCT {qdisc} FROM {qt} "
                              f"WHERE {qdisc} IS NOT NULL"):
            sample = [r[0] for r in _q(cur,
                f"SELECT {qid} FROM {qt} WHERE {qdisc}=%s "
                f"AND {qid} IS NOT NULL LIMIT {_POLY_VALUE_SAMPLE}", [val])]
            if not sample:
                continue
            sample = [str(x) for x in sample]

            best = None
            for cand_t in tables:
                if cand_t == t:
                    continue
                for kc in meta[cand_t]["key_cols"]:
                    qct, qkc = _ident(engine, cand_t), _cast_text(engine, _ident(engine, kc))
                    # dialect-neutral `IN (...)` (P0-2 follow-up) — Postgres's `= ANY(%s)`
                    # array bind isn't supported by mysql-connector-python or most other
                    # non-psycopg2 DB-API drivers.
                    present = _q(cur,
                        f"SELECT count(DISTINCT {qkc}) FROM {qct} "
                        f"WHERE {qkc} IN {_in_clause(len(sample))}", sample)[0][0]
                    rate = present / len(set(sample))
                    if rate < _MATCH_FLOOR:
                        continue
                    # Collision guard: a numeric surrogate-PK match is only trustworthy
                    # if the discriminator value also NAMES the target table. String/
                    # business-key matches are collision-resistant on their own.
                    is_string_key = meta[cand_t]["dtypes"].get(kc, "") in _STRING_TYPES
                    affinity = _name_affinity(val, cand_t)
                    if not (is_string_key or affinity):
                        continue   # reject numeric-overlap-without-name-affinity (User→cp.id)
                    score = rate + (0.5 if affinity else 0) + (0.3 if is_string_key else 0)
                    if best is None or score > best["_score"]:
                        best = {"target_table": cand_t, "target_column": kc,
                                "confidence": round(rate, 3), "_score": score}
            if best:
                best.pop("_score", None)

            if best:
                edges.append({
                    "source_table": t, "source_column": id_col,
                    "target_table": best["target_table"], "target_column": best["target_column"],
                    "relationship_type": "polymorphic", "discovery": "data_inferred",
                    "polymorphic": True,
                    "requires_predicate": f"{t}.{disc_col} = '{val}'",
                    "confidence": best["confidence"],
                })
            # else: discriminator value is categorical (e.g. 'SAR','Level 1') → no edge
    return edges


_SQL_INTROSPECTABLE_ENGINES = ("postgresql", "postgres") + _MYSQL_ENGINES


def _can_sql_introspect(ctx) -> bool:
    """True when this source has a live, dialect-known SQL connection this module
    knows how to introspect for cardinality/PK (Postgres and, as of the P0-2
    follow-up 2026-09-15, MySQL — live-verified against a throwaway MySQL 8
    container, see docs/backlog/query-engine-open-items.md). False for a
    file-backed tabular source (no SQL server at all) or a relational engine this
    module STILL hasn't been verified against (e.g. SQL Server, Oracle — no live
    instance available to verify dialect-specific SQL against) — both fall back to
    declared-FK-only edges rather than crashing or guessing at unverified SQL."""
    if ctx is None:
        return True   # legacy ctx-less call: byte-identical to pre-fix behaviour
    engine = (ctx.engine or "postgresql").lower()
    if engine in _TABULAR_ENGINES:
        return False
    return ctx.type == "relational" and engine in _SQL_INTROSPECTABLE_ENGINES


def build_relationship_graph(tables=None, verbose=False, ctx=None):
    """Build and persist the relationship graph for ONE source.

    ``ctx`` (``ingestion.contracts.SourceContext``) is the source being ingested —
    passed by ``layers/l5_publish.py`` (P0-2/P0-1, 2026-09-10). It is optional and
    defaults to the legacy ctx-less behaviour (currently-injected source, flat
    ``RELATIONSHIP_GRAPH_FILE``) for the ``__main__`` dev-CLI entry point below.

    ``tables`` should come from the CALLER's own in-memory scan/semantic-model for
    this run (``state["semantic_model"]`` / ``state["scan_result"]``) — never
    re-derived from a file on disk here, which is what let one source's stale or
    foreign semantic-model file silently produce an empty (or wrong-source) graph
    for another source (P0-4). ``None`` still falls back to reading
    ``SEMANTIC_MODEL_FILE`` for the ctx-less dev-CLI path only.
    """
    source_id = ctx.source_id if ctx is not None else None
    tenant = (ctx.tenant if ctx is not None else None) or "default"
    sql_mode = _can_sql_introspect(ctx)

    raw = _raw_schema_for(ctx)
    schema_tables = raw.get("tables", [])
    by_name = {t["table_name"]: t for t in schema_tables}

    if tables is None:
        # ctx-less dev-CLI fallback only (real ingestion runs always pass tables
        # explicitly — see docstring). Default to whatever the semantic model covers.
        try:
            from config import SEMANTIC_MODEL_FILE
            sm = json.load(open(SEMANTIC_MODEL_FILE))
            tables = sorted(sm.get("tables", {}).keys())
        except Exception:
            tables = list(by_name.keys())
    tables = [t for t in tables if t in by_name]

    # P0-4: a schema scan that found tables but ended up with an empty scoped list
    # (e.g. a foreign/stale tables argument that shares no names with this source's
    # OWN schema) is a bug upstream, not "this source has no tables" — refuse to
    # overwrite whatever graph (if any) is already on disk for it. An honestly-empty
    # source (0 tables in its own schema too) still gets an empty graph, correctly.
    if not tables and by_name:
        raise RuntimeError(
            f"relationship_graph: refusing to write an empty graph for source "
            f"{source_id!r} — schema scan found {len(by_name)} table(s) but none "
            f"matched the requested `tables` scope (stale/foreign semantic model?)")

    if sql_mode:
        engine = _engine_of(ctx)
        # MySQL has no "public" schema — information_schema.*'s "schema" IS the
        # database name there (mirrors connectors/relational.py::MySQLConnector's
        # own `db = schema or self._config.get("dbname", "")`); Postgres keeps its
        # existing "public" default unchanged.
        default_schema = (ctx.connection.get("dbname") if ctx is not None and ctx.connection
                          and engine in _MYSQL_ENGINES else None) or "public"
        schema_name = (ctx.schema_filter if ctx is not None else None) \
            or (ctx.connection.get("schema") if ctx is not None and ctx.connection else None) \
            or default_schema
        conn = _conn(ctx)
        cur = conn.cursor()
        try:
            meta = _table_meta(cur, tables, schema=schema_name, engine=engine)

            edges = _declared_fk_edges([by_name[t] for t in tables])
            edges = [e for e in edges if e["target_table"] in tables]
            edges = _merge_edges(edges, _discovered_fk_edges(tables, ctx))   # M1 close-out
            edges += _polymorphic_edges(cur, tables, meta, engine=engine)

            for e in edges:
                e["cardinality"] = _cardinality(cur, e["source_table"], e["source_column"],
                                                e["target_table"], e["target_column"],
                                                engine=engine)
                e["weight"] = _WEIGHT.get(e["relationship_type"], 3)
        finally:
            cur.close(); conn.close()
    else:
        # Tabular source, or a relational engine this module can't safely introspect
        # yet (P0-2): declared/inferred FK edges only — no live SQL connection, so no
        # cardinality/rowcount/polymorphic-value correlation. Real edges, just a
        # coarser confidence than the full SQL-introspected mode; a source that used
        # to get NO graph at all (ValueError → non-fatal stage failure) now gets one.
        edges = _declared_fk_edges([by_name[t] for t in tables])
        edges = [e for e in edges if e["target_table"] in tables]
        edges = _merge_edges(edges, _discovered_fk_edges(tables, ctx))   # M1 close-out
        for e in edges:
            e["cardinality"] = "unknown"
            e["weight"] = _WEIGHT.get(e["relationship_type"], 3)
        if verbose:
            logger.info("relationship_graph: declared-FK-only mode for source %r "
                       "(engine=%r, type=%r) — no live SQL introspection",
                       source_id, getattr(ctx, "engine", None), getattr(ctx, "type", None))

    graph = {
        "tables": tables,
        "edges": edges,
        "stats": {
            "num_tables": len(tables), "num_edges": len(edges),
            "declared": sum(1 for e in edges if e["discovery"] == "declared_fk"),
            "polymorphic": sum(1 for e in edges if e["polymorphic"]),
            "mode": "sql" if sql_mode else "declared_fk_only",
        },
    }

    out_path = source_artifact_path("veda_relationship_graph.json", source_id, tenant) \
        if source_id is not None else RELATIONSHIP_GRAPH_FILE
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(graph, open(out_path, "w"), indent=2)
    if verbose:
        logger.info("%s", json.dumps(graph["stats"], indent=2))

    # P0-6: drop this source's cached graph so the next read (this process or, via
    # rehydrate, an inference worker) picks up what was just written instead of a
    # stale in-memory copy.
    try:
        from veda.runtime import invalidate_graph_cache
        invalidate_graph_cache(source_id)
    except Exception:
        pass

    return graph
    return graph


if __name__ == "__main__":
    g = build_relationship_graph(verbose=True)
    print(f"\n✓ wrote {RELATIONSHIP_GRAPH_FILE}: {g['stats']}")
    for e in g["edges"]:
        pred = f"  [{e['requires_predicate']}]" if e["requires_predicate"] else ""
        print(f"  {e['source_table']}.{e['source_column']} → "
              f"{e['target_table']}.{e['target_column']}  "
              f"({e['relationship_type']}, {e['cardinality']}, w={e['weight']}, "
              f"conf={e['confidence']}){pred}")
