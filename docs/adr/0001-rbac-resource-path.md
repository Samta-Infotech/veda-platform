# ADR-0001 — RBAC resource paths

**Status:** accepted · **Applies to:** `apps/access_management`, `apps/query/scope.py`,
`veda_core/veda/rbac_filter.py`

> This ADR was reconstructed on 2026-09-09 from the code that cites it
> (`apps/access_management/resource_path.py`, `services/resolver.py`,
> `services/data_scope.py`, `models/grants.py`, `models/catalog.py`) after the original
> file was found missing. The section numbers below match the `§N` references still in
> those source files. If the original resurfaces, reconcile.

## Context

RBAC needs one way to name the thing a permission is granted on — a whole source, a
table, a column, a document tree. That name is written into every `RolePermission` row,
returned in every resolver result, used in every gate decision, and will key the
permission cache. Its shape is effectively a schema for all of those.

The obvious alternative — a foreign key from `RolePermission` to a substrate row — was
rejected: the substrate is **deleted and recreated on every re-ingestion**, so an FK
would cascade-delete or dangle every grant each time a source is rebuilt. Grants must
outlive re-ingestion.

## Decision

### §3.1 Grammar

A resource is named by a colon-separated **canonical path**:

```
<kind>:<source>[:<segment>]*

db:crm_postgres                     the whole source
db:crm_postgres:employee            one table
db:crm_postgres:employee:salary     one column
```

- Minimum 2 segments (`<kind>:<source>` — a bare kind is never a resource), maximum 8,
  maximum 512 characters (matches `CatalogResource.path`).
- Segments are lowercased and trimmed; the allowed charset is
  `[a-z0-9_.\-]` — the three separators real table and file names use. Anything else
  (whitespace, unicode, a nested `:`) is **rejected at write time**, never escaped
  (§3.6): an escaping scheme would make every resolver comparison and every cache key
  subtly wrong.
- The single entry point for untrusted input is `resource_path.validate()`; a stored or
  submitted path is canonicalised through it before being compared against anything.

### §3.2 Kinds are derived from `Source.dialect`, not invented alongside it

`kind ∈ {db, nosql, files, lake}`, mapped from the dialect
(`postgres/mysql/sqlite/oracle/sqlserver/duckdb → db`, `mongo/es/dynamo → nosql`,
`filesystem/s3_docs → files`, `delta/parquet/csv_lake/iceberg → lake`). An unmapped
dialect raises `UnknownDialect` and **fails closed** — an unaddressable source cannot
have anything granted on it. A test asserts `KIND_BY_DIALECT` stays in step with
`sources.Source.Dialect` so the decoupling cannot silently drift.

`resource_path.py` is **deliberately pure** — no Django imports, no I/O, total functions
over strings — which is what lets the models, the discovery service, the resolver, and
the gates all share one definition without an import cycle, and makes it exhaustively
testable without a database.

### §3.4 Prefix inheritance, segment-wise

A grant on an ancestor path covers every descendant. `prefixes("db:crm:employee:salary")`
→ `["db:crm", "db:crm:employee", "db:crm:employee:salary"]` — the set the resolver
matches grants against, broadest first.

Matching is **segment-wise, never string-wise**. `is_prefix_of("db:crm", "db:crm_postgres")`
is `False`. A naive `startswith` would grant every source whose name merely begins with
another's — the classic prefix-authorization bug.

A blank path (`resource_path=""`) has zero segments and is **never** a prefix of a real
path.

### §3.5 Resolution rules — deny-wins + strict hierarchy

For a permission on a resource:

1. Collect every grant (from an active user → active role → active permission chain)
   whose path is a prefix-or-equal of the requested resource.
2. If **any** matched grant has `effect = deny` → **DENY**. Unpierceable at any depth.
3. Else, if the **source-level ancestor** (the 2-segment `db:<source>` prefix) is itself
   explicitly `allow` → **ALLOW**.
4. Else → **DENY**.

**Strict hierarchy** (added 2026-08): the source is the gate. Rule 3 means an ALLOW on a
table or column with no source-level ALLOW above it grants **nothing**. The model is
"allow the source, then refine **down** with denies", not "allow-list individual tables
from an ungranted source".

The recommended grant pattern is therefore: whole-source ALLOW + narrower DENY carve-outs
(e.g. `ALLOW db:crm` + `DENY db:crm:employee:salary`).

`resolver.allows()` carries the full reasoning; `apps/query/scope.py::permitted_source_ids`
(the coarse source gate) and `services/catalog.py::CatalogService._resolve_effect` (the
admin tree overlay) **mirror it and MUST change with it** — a comment on each says so, and
a live drift bug (parent-DENY + child-ALLOW showing green in the tree) was fixed in 2026-08.

### §3.4 (cont.) Global grants do not cover resources

`resource_path=""` means "this permission is not resource-scoped" (`user.manage`,
`role.manage`), **not** "every resource". A check for `data.read` on `db:crm:employee` is
**not** satisfied by a blank-path grant. Fail-closed reading: an admin who grants
`data.read` with no resource must not silently open every table.

### §7 Single-tenant — the path carries no tenant

There is no tenant or scope segment on a resource path, and no tenant column on any RBAC
model. `data_scope.py` and `catalog.py` read the substrate with `all_tenants()` precisely
because a resource path has no tenant and there is no ambient request context in those
services. Multi-tenant RBAC is deferred; when it lands it will most likely be a prefix
segment or a separate scoping table, not a change to this grammar.

## Consequences

- Grants survive re-ingestion (the substrate churns; the string doesn't).
- `CatalogResource.substrate_id` is a **plain UUID column, not an FK**, for the same
  reason — it's a best-effort back-pointer, reconciled by `CatalogDiscoveryService`
  (upsert / reactivate / deactivate-never-delete).
- The resolver is **one query per resolution** + O(depth) per check. A permission cache
  (`PermissionVersion` counter) is designed but not built — noted rather than half-built.
- Enforcement is gated by `VEDA_RBAC_MODE` (`off` / `shadow` / `enforce`), default `off`.
  See [`../RBAC.md`](../RBAC.md).
