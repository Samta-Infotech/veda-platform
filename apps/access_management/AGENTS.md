# apps/access_management/ — RBAC: who may exist, and do what

61 files. The RBAC data model + resolver + the two enforcement gates + admin CRUD.
Enforcement is gated by `VEDA_RBAC_MODE` (`off` / `shadow` / `enforce`), **default `off`**.
Full reference: [../../docs/RBAC.md](../../docs/RBAC.md). Grammar:
[../../docs/adr/0001-rbac-resource-path.md](../../docs/adr/0001-rbac-resource-path.md).

## Package roots
| File | Role |
|------|------|
| `gate.py` | **Gate 2** — `RequiresPermission(BasePermission)` + `rbac_mode()`. off → no-op; shadow → always allow but log `WOULD DENY`; enforce → honour. Fail-closed if a view opts in with no `required_permission`. Resolver result cached on `request._veda_effective_permissions`. |
| `codes.py` | `PermissionCode` constants — `query.execute`, `data.read`, `source.manage`, `ingestion.run`, `evaluation.run`, `user.manage`, `role.manage`. **No `permission.read`** (removed by migration 0010). |
| `resource_path.py` | Pure string module implementing the ADR-0001 canonical path `<kind>:<source>[:seg]*`. `validate` / `build` / `segments` / `prefixes` / `is_prefix_of` (segment-wise, not string-wise) / `kind_for_dialect`. No Django imports. |
| `admin.py` | `Role` editable (bootstrap path); `Permission` read-mostly; `CatalogResource` / `UserRole` / `RolePermission` read-only. |

## `models/`
| File | Model |
|------|-------|
| `roles.py` | `Role` — CI-unique `name`, `is_active`, `deleted_at` (retire-not-delete). |
| `permissions.py` | `Permission` — CI-unique `code`, code-defined, seeded by migration, read-only API. |
| `grants.py` | `Effect` (allow/deny). `UserRole` (user CASCADE, role PROTECT, `granted_by` SET_NULL; unique `(user,role)`). `RolePermission` (role CASCADE, permission PROTECT, **`resource_path` CharField not FK**, `effect`, `granted_by`; unique `(role,permission,resource_path)` — effect NOT in the key, so re-grant flips in place). |
| `catalog.py` | `CatalogResource` — unique `path`, `kind`, indexed `parent_path`, `source` FK PROTECT, **`substrate_id` plain UUID column not FK** (substrate is deleted+recreated every re-ingestion), `is_active`. |
| `profile.py` | `UserProfile` — `OneToOneField(AUTH_USER_MODEL)`, `deleted_at`. |

## `services/` (13)
| File | Role |
|------|------|
| `resolver.py` | **`PermissionResolver`** + immutable `EffectivePermissions`. `resolve(user)` = one query. `allows(code, path)` = **deny-wins + strict hierarchy** (a table/column ALLOW grants nothing without the 2-segment `db:<source>` ancestor also ALLOWed; a blank-path grant never satisfies a resource check). Anonymous/inactive → `NO_PERMISSIONS`. |
| `data_scope.py` | **Gate 1** payload. `resolve_effective_permissions(user)` — the single per-request "is this subject to RBAC" decision (`is_staff` is **not** a bypass). `compute_data_scope(user, source_ids, effective)` → `{source_id: SourceDataScope(open, tables)}`. `serialize_data_scope()` → the `X-Veda-Data-Scope` wire payload. Resolves `substrate_id` → real table/column names via `all_tenants()`. |
| `permissions.py` | `PermissionService` — read-only list/get (drops `data.read` from the picker). |
| `roles.py` | `RoleService` — create/list/get/update + `_sync_grants` (desired-state replace). `AdminRoleProtected`. |
| `grants.py` | `UserRoleService` (assign/revoke, idempotent), `RolePermissionService` (grant/revoke, `_canonical_path`), `role_stats()`. |
| `users.py` | `UserService` — create/list/get/update. `is_admin` → `is_superuser`. Deactivation writes `UserProfile.deleted_at` + `revoke_all_refresh_tokens`. |
| `catalog.py` | `CatalogDiscoveryService.sync_source/sync_all` — reconcile from substrate (upsert / reactivate / **deactivate-never-delete**). `CatalogService.get_tree` — `_resolve_effect` MUST mirror `resolver.allows` (live drift bug fixed 2026-08). |
| `admin_guard.py` | "≥ 1 active admin, always" invariant. `active_admins()`, `is_last_active_admin()`. |
| `bootstrap.py` | `AdminBootstrapService.bootstrap()` — first user only, race-safe via `select_for_update()`. Not an HTTP route. |
| `base.py` | error hierarchy (`NotFoundError` 404 / `ConflictError` 409) + `paginate()`. |

## `serializers/` (8) — INPUT-only, never render responses
`base.py` (`PaginatedListSerializer`), `users.py` (`PRIVILEGED_FIELDS` denylist),
`roles.py` (`permission_ids` + `resource_grants`), `grants.py` (`_ResourcePathField`
canonicalises on input), `permissions.py`, `catalog.py`, `resolver.py`.

## `views/` (8) — thin DRF `APIView`, all extend `AdminView`
`base.py` (`AdminView` = `[IsAdminUser, RequiresPermission]`, `required_permission=None`
fail-closed), `users.py` / `roles.py` / `grants.py` (all declare a `required_permission`),
`permissions.py` / `catalog.py` (`[IsAdminUser]` only — RBAC gate **dropped**, not blanked,
after migration 0010), `resolver.py` (`EffectivePermissionsView`, GET).

## `urls/` (7)
`urls/__init__.py` concatenates per-domain modules. Convention `<resource>/<action>`;
mutating = POST, read-only = **GET** (breaking change 2026-08-09). Mounted under `/api/v1/`.

## `management/commands/`
`bootstrap_admin.py` (the only way to create the first admin), `sync_catalog.py`.

## `migrations/` (11)
`0001` partial CI-unique index on `auth_user.email`; `0004` seeds 8 permissions incl.
`permission.read`; `0007` seeds `Role("Admin")` + every permission globally;
`0010` removes `permission.read` (a fresh DB seeds it then removes it).

## Gotchas
- **No tenant column anywhere** — single-tenant (ADR §7); `data_scope`/`catalog` read the
  substrate with `all_tenants()`.
- **`granted_by` is the only audit record** — no history/event table (`AUTH_ISSUES_BACKLOG.md`
  M8, referenced but missing).
- The deny-wins + strict-hierarchy rule is reimplemented in **three** places
  (`resolver.allows`, `apps/query/scope.permitted_source_ids`, `catalog._resolve_effect`) —
  they MUST change together.
- `AdminPrivilegesRequired` is defined **twice** in `../authentication/services.py`
  (harmless dup).
- `select_for_update` is a Postgres-only guarantee; SQLite (local tests) silently no-ops it.
