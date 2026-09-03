# =============================================================================
# query/cross_source_composer.py
# VEDA — Hybrid cross-source answer composition (Cross-source plan, Phase 5.2)
#
# When a query's selected subgraph spans > 1 source, the answer is composed from
# two halves with provenance:
#   - tabular subgraph (columns/tables + join path, possibly multi-source) →
#     federated SQL executed through query.federated_executor
#   - chunk nodes → evidence set (already PPR-scored; cross-encoder reranks top-k)
#
# The NL SQL itself is produced by the existing deterministic SQL head over the
# MERGED semantic model (Phase 1.4 source-qualified names), so this module does NOT
# reimplement SQL generation — it decides WHEN to federate, partitions the selected
# subgraph, resolves per-source execution surfaces, runs the federated executor, and
# assembles a provenance-tagged payload the UI can render as "how this was assembled".
#
# Deterministic parts (should_federate / partition / provenance assembly) are pure
# and unit-tested; surface resolution + execution reuse verified components.
# =============================================================================

from __future__ import annotations

import os
from typing import Dict, List, Optional

from query.federated_executor import (
    FederatedExecutor, SourceSurface, catalog_name, FederatedError, FED_MAX_SOURCES,
)


def selected_source_ids(selected_columns) -> List[str]:
    """Distinct, non-empty source ids among the selected columns (order-stable)."""
    out: List[str] = []
    for c in selected_columns or []:
        sid = str(getattr(c, "source_id", "") or (c.get("source_id") if isinstance(c, dict) else ""))
        if sid and sid not in out:
            out.append(sid)
    return out


def should_federate(selected_columns) -> bool:
    """True when the selected tabular subgraph spans ≥ 2 sources — the routing
    decision (not a feature flag) that sends a plan to the federated path. Single-
    source plans keep the existing direct execution path."""
    return len(selected_source_ids(selected_columns)) >= 2


def partition_subgraph(selected_columns, selected_chunks) -> Dict:
    """Split the PPR-selected subgraph into the SQL half (tabular columns grouped by
    source) and the evidence half (chunk nodes). Returns
    {tabular: {source_id: [cols]}, evidence: [chunks], sources: [ids]}."""
    tabular: Dict[str, list] = {}
    for c in selected_columns or []:
        sid = str(getattr(c, "source_id", "") or "")
        if not sid:
            continue
        tabular.setdefault(sid, []).append(c)
    return {"tabular": tabular, "evidence": list(selected_chunks or []),
            "sources": list(tabular.keys())}


def resolve_surface(source_id: str, tenant: str) -> Optional[SourceSurface]:
    """Build one source's federated execution surface, resolving credentials/paths
    SERVER-SIDE (never into generated SQL):
      relational → postgres_scanner DSN from the Source registry row
      tabular    → parquet views from the source's materialized artifact dir
    Returns None when the surface can't be resolved (skipped, logged by caller)."""
    kind, dsn, tables = _resolve_source_kind(source_id)
    if kind == "postgres":
        return SourceSurface(source_id=source_id, kind="postgres", pg_dsn=dsn)
    if kind == "parquet":
        return SourceSurface(source_id=source_id, kind="parquet", tables=tables)
    return None


def _resolve_source_kind(source_id: str):
    """(kind, pg_dsn, {table: parquet_path}) for a source. Reads the Source registry
    (dialect → relational vs tabular) and, for tabular, the materialized parquet dir.
    Import-light + fail-soft: returns ("", None, {}) when unresolved."""
    try:
        from storage_adapters import reader
        # relational sources resolve a live connection; tabular resolve a parquet dir.
        # We piggyback on the registry query used by source_connection.
        conn = reader._connection()
        with conn.cursor() as cur:
            cur.execute("SELECT dialect, host, port, dbname, db_user, password_env, "
                        "password_inline, source_path FROM sources_source WHERE id = %s",
                        [source_id])
            row = cur.fetchone()
    except Exception:
        return "", None, {}
    if not row:
        return "", None, {}
    dialect, host, port, dbname, db_user, pw_env, pw_inline, source_path = row
    if dialect in ("csv_lake", "parquet", "xlsx", "excel", "csv"):
        return "parquet", None, _materialized_parquet(source_id, source_path)
    # relational: server-side DSN for postgres_scanner
    pw = os.environ.get(pw_env, "") if pw_env else (pw_inline or "")
    dsn = f"host={host} port={port or 5432} dbname={dbname} user={db_user} password={pw}"
    return "postgres", dsn, {}


_PG_SCHEMA_CACHE: Dict[str, str] = {}


def resolve_pg_schema(source_id: str) -> str:
    """The postgres SCHEMA a relational source's tables actually live under. DuckDB's postgres ATTACH
    exposes them at ``<catalog>.<schema>.<table>``; the catalog qualifier historically ASSUMED
    ``public``, which is wrong when a source uses a named schema (homzhub's tables live under the
    ``homzhub`` schema — the ``public`` assumption made every federated query to it fail with a catalog
    error). When FEDERATED_SCHEMA_DISCOVERY_ENABLED, discover it from the source's OWN DB — the
    non-system schema holding the most base tables — data-driven, no hardcoded schema name. Cached per
    source_id; any failure, a non-postgres source, or the flag OFF → ``public`` (byte-identical)."""
    sid = str(source_id)
    try:
        from config import FEDERATED_SCHEMA_DISCOVERY_ENABLED as _on
    except Exception:
        _on = False
    if not _on:
        return "public"
    if sid in _PG_SCHEMA_CACHE:
        return _PG_SCHEMA_CACHE[sid]
    schema = "public"
    try:
        kind, dsn, _ = _resolve_source_kind(sid)
        if kind == "postgres" and dsn:
            import psycopg2
            conn = psycopg2.connect(dsn)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT table_schema FROM information_schema.tables "
                        "WHERE table_type = 'BASE TABLE' "
                        "AND table_schema NOT IN ('pg_catalog', 'information_schema') "
                        "GROUP BY table_schema ORDER BY count(*) DESC LIMIT 1")
                    row = cur.fetchone()
                    if row and row[0]:
                        schema = row[0]
            finally:
                conn.close()
    except Exception:
        schema = "public"
    _PG_SCHEMA_CACHE[sid] = schema
    return schema


def _materialized_parquet(source_id: str, source_path: str) -> Dict[str, str]:
    """{table_name: parquet_path} for a tabular source's materialized artifacts
    (written at L1, P2.2). Looks under the source's artifact tables/ dir."""
    from config import ARTIFACT_ROOT
    tables: Dict[str, str] = {}
    for base in (os.path.join(ARTIFACT_ROOT, str(source_id), "tables"),
                 os.path.join(ARTIFACT_ROOT, "tables")):
        if os.path.isdir(base):
            for f in os.listdir(base):
                if f.endswith(".parquet"):
                    tables[f[:-len(".parquet")]] = os.path.join(base, f)
            if tables:
                break
    return tables


def build_provenance(federated_result: Optional[dict], evidence: List) -> List[dict]:
    """Structured provenance array the UI renders as "how this answer was assembled":
    one entry per source/table used and per evidence doc, with join edges + tiers."""
    prov: List[dict] = []
    if federated_result:
        for cat in federated_result.get("catalogs", []):
            prov.append({"kind": "sql", "catalog": cat})
    for ev in evidence or []:
        prov.append({
            "kind": "evidence",
            "doc": (getattr(ev, "attrs", {}) or {}).get("doc_id", "") if not isinstance(ev, dict)
                   else ev.get("doc_id", ""),
            "section": (getattr(ev, "attrs", {}) or {}).get("section_path", "") if not isinstance(ev, dict)
                       else ev.get("section_path", ""),
        })
    return prov


def _federated_rbac_enabled() -> bool:
    try:
        import config as _cfg
        return bool(getattr(_cfg, "FEDERATED_RBAC_ENFORCE_ENABLED", True))
    except Exception:
        return True


def _federated_rbac_block(sql: str) -> Optional[list]:
    """Apply the RBAC data-scope to a federated SQL BEFORE it executes — the gate the single-source head
    runs (narrow_allowed / restricted_names) that the cross-source composer otherwise bypasses. Returns a
    list of RBAC-restricted table/column names the SQL references (→ the caller refuses), or None to allow.
    A no-op (None) when the request carries no data-scope, so unrestricted users are byte-identical."""
    if not _federated_rbac_enabled():
        return None
    try:
        from veda_core.context import try_current
        ctx = try_current()
        if ctx is None or getattr(ctx, "allowed_resources", None) is None:
            return None                      # no RBAC scope on this request → nothing to enforce
        from veda_hybrid import _load_semantic_model
        sm, _ = _load_semantic_model()
        from veda.rbac_filter import restricted_names
        rn = restricted_names(sm, ctx) or {}
        # Match on restricted TABLE names only. Column names are intentionally NOT matched here: RBAC
        # restricts columns per (table, column), but restricted_names flattens to bare column names, so a
        # common name like "id"/"name" that is restricted on some table would over-block a legitimate
        # query on an ALLOWED table that also has that column (a false-deny). Table names are unambiguous.
        # (Finer per-column narrowing WITHIN an allowed table on the federated path is a follow-up; the
        # single-source head already does it via narrow_allowed.)
        restricted = {str(x).lower() for x in (rn.get("tables") or [])}
        if not restricted:
            return None
        # AST table names when the SQL parses, PLUS a raw identifier token scan (also covers a plan dict
        # serialized to text) — UNIONED so a restricted table caught by either wins (fail-safe).
        names = set()
        try:
            import sqlglot
            from sqlglot import expressions as _exp
            parsed = sqlglot.parse_one(sql)
            names |= {t.name.lower() for t in parsed.find_all(_exp.Table) if t.name}
        except Exception:
            pass
        import re as _re
        names |= {t.lower() for t in _re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql or "")}
        hit = sorted(names & restricted)
        return hit or None
    except Exception:
        # A scope EXISTS but we could not evaluate it → fail closed (refuse) rather than leak.
        return ["__rbac_eval_error__"]


def compose_federated(query: str, sql: str, selected_columns, selected_chunks,
                      tenant: str = "default", params: Optional[list] = None) -> Dict:
    """Execute a federated plan and assemble a composed, cited payload. ``sql`` is the
    already-generated source-qualified SQL from the deterministic head. Returns
    {status, sql, result, evidence, provenance, sources} or a structured refusal when
    the firewall blocks the plan (never a silent source drop)."""
    sources = selected_source_ids(selected_columns)
    surfaces: List[SourceSurface] = []
    for sid in sources:
        surf = resolve_surface(sid, tenant)
        if surf is not None:
            surfaces.append(surf)
    if len(surfaces) < 2:
        return {"status": "not_federated", "reason": "fewer than 2 resolvable sources",
                "sources": sources}
    _blocked = _federated_rbac_block(sql)
    if _blocked:
        return {"status": "refused_rbac", "reason": "access denied to restricted resource(s)",
                "sql": sql, "sources": sources}
    try:
        fx = FederatedExecutor(surfaces)
        result = fx.execute(sql, params=params)
    except FederatedError as e:
        # Refusal path explains the block (e.g. ungrounded join / out-of-scope) rather
        # than silently dropping a source (Phase 5.2 §4).
        return {"status": "refused_federated", "reason": str(e), "sql": sql,
                "sources": sources}
    except Exception as e:
        # Execution errors (e.g. a DuckDB feature the generated SQL used, unknown column)
        # must NOT propagate — otherwise the caller's generic guard silently falls back to
        # the (slow, single-source) deterministic head. Return a structured refusal with the
        # engine error so the caller can retry or surface it.
        return {"status": "exec_error_federated", "reason": str(e), "sql": sql,
                "sources": sources}
    provenance = build_provenance(result, selected_chunks)
    return {"status": "ok", "sql": sql, "result": result,
            "evidence": list(selected_chunks or []), "provenance": provenance,
            "sources": sources}


def compose_federated_plan(query: str, plan: dict, selected_columns, selected_chunks,
                           tenant: str = "default") -> Dict:
    """Aggregate-pushdown variant of compose_federated: executes a per-metric plan through
    FederatedExecutor.execute_plan (each metric aggregated independently then joined on the
    group key) — correct for multi-metric cross-source queries (no join fan-out). Same
    surface resolution + refusal contract as compose_federated."""
    sources = selected_source_ids(selected_columns)
    surfaces: List[SourceSurface] = []
    for sid in sources:
        surf = resolve_surface(sid, tenant)
        if surf is not None:
            surfaces.append(surf)
    if len(surfaces) < 2:
        return {"status": "not_federated", "reason": "fewer than 2 resolvable sources",
                "sources": sources}
    # RBAC data-scope on the aggregate-pushdown plan: scan the plan's own SQL/column text for any
    # restricted name (coarse but fail-closed — never executes restricted data cross-source).
    import json as _json
    if _federated_rbac_block(_json.dumps(plan, default=str)):
        return {"status": "refused_rbac", "reason": "access denied to restricted resource(s)",
                "plan": plan, "sources": sources}
    try:
        fx = FederatedExecutor(surfaces)
        result = fx.execute_plan(plan)
    except FederatedError as e:
        return {"status": "refused_federated", "reason": str(e), "plan": plan,
                "sources": sources}
    except Exception as e:
        return {"status": "exec_error_federated", "reason": str(e), "plan": plan,
                "sources": sources}
    provenance = build_provenance(result, selected_chunks)
    return {"status": "ok", "plan": plan, "result": result,
            "evidence": list(selected_chunks or []), "provenance": provenance,
            "sources": sources}
