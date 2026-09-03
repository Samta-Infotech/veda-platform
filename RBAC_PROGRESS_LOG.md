# RBAC / AUTH — PROGRESS LOG  *(living doc)*

Separate from `VEDA_PROGRESS_LOG.md` on purpose: that one tracks the query/retrieval
engine, this one tracks the api tier's identity work — authentication now, and
authorization/RBAC when it is scoped.

**Scope delivered so far: authentication + the complete RBAC data model.** Two bounded
contexts: `apps/authentication` (identity *verification*) and `apps/access_management`
(identity *administration* + RBAC). The graph is now whole —
`User → UserRole → Role → RolePermission → Permission (+ resource path)` — with a
catalog projection to address resources against.

**🔴 Enforcement exists but is off.** Gate 2 consults the resolver on the admin
surface, gated behind `VEDA_RBAC_MODE` (default `off`). **The data path is still
ungated** — `/api/v1/query` reaches customer data with `AllowAny`, and closing that is
Gate 1. Every endpoint outside this app is still `AllowAny` or
`IsAdminUser`, and `/api/v1/query` is as open as before this programme began. Still
absent: permission cache, Gate 1, audit trail, multi-tenancy, approval workflow (the
last two out of scope by decision). **Bootstrap admin is now done** — see below.

---

## 1. Current Status Snapshot  *(update every time)*

| | |
|---|---|
| **Modules** | `apps/authentication/` (verification) · `apps/access_management/` (administration + RBAC home) |
| **Endpoints live** | auth: `POST auth/{login,refresh,logout,password/change}`<br>users: `POST users/{create,detail,list,update}` (`update` now also covers activate/deactivate via `is_active`)<br>roles: `POST roles/{create,detail,list,dropdown,update}` (`dropdown` = every active role, unpaginated, for a picker)<br>permissions: `POST permissions/{list,detail}` *(read-only)*<br>catalog: `POST catalog/{list,detail}` *(read-only — `list {parent_path}` already IS the resource-tree API, lazy-loaded per level)*<br>grants: `POST users/roles/{assign,revoke,list}` · `POST roles/permissions/{grant,revoke,list}`<br>resolver: `POST users/permissions/effective` — all under `/api/v1/`, all POST |
| **Bootstrap** | `manage.py bootstrap_admin` — backend-only, race-safe, first user gets `is_staff=True` + the seeded `Admin` role. No public endpoint, by design |
| **Enforcement** | `VEDA_RBAC_MODE` — **`off` (default)** / `shadow` / `enforce`. Gate 2 is wired onto all 19 admin endpoints (22 endpoints total incl. auth) **alongside** `IsAdminUser`. The data path is NOT gated |
| **Ops commands** | `manage.py sync_catalog [--source-id N]` — rebuilds the catalog projection by hand. **Also now auto-triggered on ingestion success** via `VEDA_AUTO_SYNC_CATALOG` (default OFF — see `apps/ingestion/tasks.py::_sync_catalog_if_enabled`) |
| **Token stack** | `djangorestframework-simplejwt[crypto]>=5.3` (HS256, signed with `SECRET_KEY`) |
| **Rollout flag** | `VEDA_JWT_AUTH` — **default `0` (OFF)** → prod behaviour byte-identical |
| **Password policy** | 4 Django stock validators + `PasswordComplexityValidator` (upper/lower/digit/special, configurable via `AUTH_PASSWORD_VALIDATORS` OPTIONS) |
| **Tests** | auth **79** · users **104** · roles **85** · permissions **38** · catalog **68** · grants **44** · resolver **35** · gate **23** · admin bootstrap **38** = **514 passed** (sqlite test DB, local) |
| **Verification level** | **code + local-test verified.** NOT yet run against the live Postgres/PgBouncer + redis-cache stack, and not yet exercised by the frontend. |
| **Frontend contracts** | `AUTH_API_CONTRACT.md` (verification) · `ACCESS_MANAGEMENT_API_CONTRACT.md` (users + RBAC). Each lists its not-yet-built endpoints with a definition-of-done — **not yet updated for `password/change` or the `is_active`/bootstrap additions**. |
| **Architecture audits** | #1 `VEDA_RBAC_ARCHITECTURE_REVIEW.md` — 6.0/10 · #2 `VEDA_RBAC_AUDIT_2.md` — **6.5/10**, ⚠️ approved to continue, **not approved as data protection**. ~64% of the phase programme, **0% data protection** |
| **ADRs** | `docs/adr/0001-rbac-resource-path.md` — ACCEPTED, implemented |
| **Review status** | Phase 1.1 done: **C1, C2, C3, H1 (+H4) fixed and verified.** H2, H3 and the M-tier remain open — see §6a. |
| **Migrations** | `apps.authentication`: none (no models). `apps.access_management`: `0001` case-insensitive partial unique index on `auth_user.email` (no new table), `0002` creates `access_management_role`, `0003` creates `access_management_permission`, `0004` seeds 8 permissions, `0005` creates `access_management_catalogresource`, `0006` creates `access_management_userrole` + `access_management_rolepermission`, `0007` seeds the `Admin` role + grants it every permission. `token_blacklist` contributes 2 tables. `makemigrations --check` reports **no changes detected** |

---

## 2. What shipped

### Login — `POST /api/v1/auth/login`
- Credentials verified by `django.contrib.auth.authenticate` (Django hashers → constant-time compare; `ModelBackend` already runs a dummy hash for an unknown username, so the "no such user" branch costs the same as a wrong password).
- Active users only. Unknown user / wrong password / inactive account collapse into **one** 401 `INVALID_CREDENTIALS` — no account enumeration.
- Issues a short-lived access token (15 min default) + a rotating refresh token (7 days default).
- **Per-account lockout** (10 failures / 5 min, both env-tunable) in `redis-cache` via Django's cache API — no new table. Keyed by a **sha256 of the casefolded username**, so raw usernames never land in cache keys and case-flipping cannot reset the counter. **Fails OPEN** on a cache outage (a dead Redis must not lock out every account; the per-IP throttles still apply).
- Own `login` throttle scope (10/min), tighter than the global `anon` 60/min.

### Refresh — `POST /api/v1/auth/refresh`
- Verifies signature, `exp`, `jti` and `token_type` — an **access** token cannot be spent as a refresh token.
- **Single-use.** Rotates: old token blacklisted, new pair issued.
- **Race-safe rotation — the main piece of real engineering here.** simplejwt's own rotation reads the blacklist while *constructing* the token, which is a TOCTOU window: two concurrent refreshes of the same token both SELECT "not blacklisted" and both mint a fresh family. `_RotatableRefreshToken` defers that check so the **blacklist INSERT** is the arbiter (unique constraint on `BlacklistedToken.token`) — exactly one caller can ever spend a `jti`, and the loser is reported by the database. Also removes one SELECT from the happy path.
- **Replay ⇒ family revocation.** A token presented twice revokes every *unexpired* refresh token of that account (simplejwt has no token-family concept; user-wide is the safe reading). Matches the OAuth 2.0 security BCP.
- Signature is verified **before** any state change, so a forged token cannot be used to revoke a real user's sessions (regression-tested).
- Deactivated/deleted user → 401, and the token is **not** spent on that reject path.

### Logout — `POST /api/v1/auth/logout`
- Revokes the presented refresh token; **always 200**, including for an already-rotated, expired, or garbage token. Reporting otherwise would make logout an oracle for whether a captured token is live.
- Per-session, not account-wide: one device logging out leaves the others signed in.
- **Not** gated on `VEDA_JWT_AUTH` — revocation never *grants* anything, so turning the flag off must not strand tokens issued while it was on.

---

## 3. Files

**Created**
```
apps/authentication/{__init__,apps,serializers,services,views,urls}.py
tests/test_authentication.py
```

**Modified**
| File | Change |
|---|---|
| `config/settings/base.py` | `VEDA_JWT_AUTH` + lockout knobs; `SIMPLE_JWT` block; `token_blacklist` + `apps.authentication` in `INSTALLED_APPS`; `JWTAuthentication` prepended **only when the flag is on**; `login`/`token_refresh` throttle rates; `INSECURE_DEV_SECRET_KEY` named |
| `config/settings/prod.py` | refuses to boot if `VEDA_JWT_AUTH=1` while `SECRET_KEY` is still the dev default — it is the JWT signing key, so a well-known value means anyone can forge a token |
| `config/urls.py` | mounts `apps.authentication.urls` under `api/v1/` |
| `apps/chat/{urls,views,serializers}.py` | dummy `LoginView` + `_authenticate_login` + `LoginRequestSerializer` **removed** (moved, not copied) — chat is −71 lines |
| `requirements/api.txt` | `+ djangorestframework-simplejwt[crypto]>=5.3` |
| `docs/QUERY_API_FRONTEND_CONTRACT.md` | real 3-endpoint contract replaces the "dev/dummy login" line, incl. the client rules below |

---

## 4. Decisions register

| Decision | Choice | Why |
|---|---|---|
| Token library | simplejwt (not PyJWT + own store) | user's call; battle-tested rotation/blacklist/auth-class, less crypto we own |
| Placement | new `apps/authentication` | follows the existing `apps/<domain>/{serializers,services,views,urls}.py` pattern; auth stops being a tenant of the chat app |
| URL prefix | `/api/v1/auth/*` | matches `config/urls.py` and keeps the already-documented login path — no frontend break (spec said `/v1/...`; one convention beats two) |
| Rollout | flag-gated, default OFF | standing project rule: prod stays byte-identical until deliberately flipped |
| `token_blacklist` in `INSTALLED_APPS` | **unconditional**, not flag-gated | migrations must be identical across environments; installing the app changes no behaviour, only the flag does |
| Replay response | revoke the account's whole family | cannot distinguish legitimate holder from thief; OAuth BCP. **Cost: a detected replay signs that user out everywhere** |
| Access-token revocation on logout | not implemented | stateless JWT; a per-request denylist means a DB read on every API call. Bounded instead by the 15-min access lifetime |

---

## 5. Client rules the server's behaviour depends on

1. **Replace the stored refresh token on every `refresh` response** — the old one is dead immediately.
2. **Never fire two `refresh` calls concurrently for the same token** — exactly one wins; the loser's 401 triggers the replay path and signs the user out everywhere. Serialize refreshes behind one in-flight promise.
3. Discard the access token client-side on logout — it stays valid until `expires_in` elapses.

---

## 6. Test coverage — auth: 79 tests

unit (token issue/verify/expiry/tamper/wrong-type) · integration (all three endpoints, happy path + chained refresh) · negative (malformed bodies, bad creds, inactive & deleted user, garbage/forged/expired/wrong-type tokens) · security (enumeration parity across unknown-vs-wrong-vs-inactive, no leakage in any error body, forged token cannot revoke a real session, lockout incl. the unknown-username and case-flip cases, cache-outage fail-open, CSRF-strict anonymous login) · **concurrency** (deterministic same-`jti` interleaving → exactly one winner; loser rejected end-to-end) · edge (double logout, logout of a rotated token, logout with an access token, flag-off inertness).

The concurrency test reproduces the interleaving **deterministically** rather than with threads — real threads on a sqlite test DB would be flaky and would not share the transaction, so it would test scheduling luck instead of the invariant.

Run: `pytest tests/test_authentication.py`
Regression-checked: `tests/test_apps_layer_refactor.py` (40), `test_chat_visualization.py` (24), `test_rag_layer_events.py` (6), `test_explain_trace_e2e.py` (9) — all pass. `manage.py check` clean.

**Test-harness note:** this module needs a real DB (rotation/revocation are enforced by table rows), so it builds a throwaway sqlite test database in-process — the same self-bootstrapping style the rest of `tests/` uses, rather than a repo-wide `pytest.ini` that would change how every existing (order-flaky) test module boots. `apps.substrate`'s migrations are bypassed there: `0002_pgvector` is raw Postgres DDL (`vector(N)`, `USING hnsw`) sqlite cannot parse, and its models are `managed=False` mirrors. The developer's `db.sqlite3` is never touched.

---

## 6a. Review findings register  *(2026-08-05 production-readiness review)*

Original verdict: **❌ CHANGES REQUIRED.** **Phase 1.1 (2026-08-05) closed C1, C2,
C3, H1 and — incidentally, same lines — H4.** IDs are cited by
`AUTH_API_CONTRACT.md`; keep them stable.

| ID | Sev | Status | Finding | Where |
|---|---|---|---|---|
| **C1** | High | ✅ **FIXED** | **Account-lockout DoS.** Lockout refused the *correct* password, so anyone knowing a username could hold that account down indefinitely at ~2 req/min. Now two counters: per-(account, source-IP) **hard** block, account-wide **soft** (never refuses a correct password). Third parties can no longer lock a user out. | `services.py` login + lockout section |
| **C2** | Med | ✅ **FIXED** | **Fail-open contract broken.** `cache.set()` inside `except ValueError` was not covered by the later `except Exception` → a Redis blip between `incr` and `set` 500'd the login. Now two independently guarded blocks. | `services.py::_record_failure` |
| **C3** | High | ✅ **FIXED** | **Password change did not revoke refresh tokens.** `CHECK_REVOKE_TOKEN: True` enabled *and* checked explicitly on the refresh path (simplejwt enforces it for access tokens only, in `JWTAuthentication.get_user`). Tokens with no claim fail closed. | `base.py`, `services.py::_password_unchanged` |
| **H1** | High | ✅ **FIXED** | **No test proved an issued access token authenticates.** Now a real protected endpoint (`JWTAuthentication` + `IsAuthenticated`) mounted on the **real** root urlconf, driven over HTTP: valid / missing / garbage / refresh-as-access / expired / forged / deactivated / password-changed, plus a settings-wiring assertion. | `tests/test_authentication.py` |
| **H4** | High | ✅ **FIXED** *(incidental)* | Log flooding: `exc_info=True` on every cache call → a full Redis traceback twice per login while Redis is down. Now one line naming the exception type and message. Fixed because C2 rewrote the same functions. | `services.py::_log_cache_outage` |
| **L3** | Low | ✅ **FIXED** *(incidental)* | `int(cache.get(...))` sat outside the `try`, so a corrupt cached value would 500 the login. C2's rewrite moved it inside. | `services.py::_is_locked` |
| **M9** | Med | ✅ **FIXED** — found while fixing C1, and **pre-existing**: `NUM_PROXIES` was unset, so DRF's throttles keyed on the **entire, attacker-controlled** `X-Forwarded-For`, i.e. the project's per-IP throttles were trivially bypassable. Now `NUM_PROXIES=1` (env `VEDA_NUM_PROXIES`). **Deployment requirement**: must match real proxy depth, and the api tier must never be reachable bypassing nginx. | `base.py` `REST_FRAMEWORK` |

**Open findings — H2, H3, M1-M8, L1/L2/L4/L5/L6 and the N-tier — now live in
`AUTH_ISSUES_BACKLOG.md`**, with a fix sketch, files, effort and acceptance criterion
for each, plus suggested batches. Deliberately not restated here: one list, one place
to keep current. This section retains the history of what was found and fixed.

Blocking summary: **nothing blocks merging** (`VEDA_JWT_AUTH` is default-off);
**H2 blocks enabling the flag**, then H3 · M1 · M3.

**Measured, not assumed:** `check_password` is **flat at ~310ms** from 10 bytes to 2 MB
(PBKDF2 folds the password into the HMAC key once) — so the long-password CPU-DoS I
initially suspected is **not** real. That 310ms is still the per-login cost, capping
login throughput near 3/sec/worker; the pre-hash lockout check correctly avoids paying
it for locked accounts.

---

## 7. Open items / not done

- [ ] **Flip `VEDA_JWT_AUTH=1` in a real environment** and exercise the three endpoints against live Postgres + redis-cache. Everything above is local-test verified only.
- [ ] **Schedule `manage.py flushexpiredtokens`** (simplejwt) — rotation accumulates `token_blacklist` rows indefinitely otherwise. Celery beat is already configured; this is one entry.
- [ ] **Nothing is protected yet.** Every existing endpoint is still `AllowAny`, and the chat views still fall back to the seeded `admin` user when a request is anonymous (`apps/chat/views.py::_resolve_user`). Issuing tokens and *requiring* them are separate steps — the second is authorization work.
- [ ] `apps/chat/migrations/0002_seed_dummy_admin_user.py` still seeds `admin/admin123`. Its own docstring says to remove it once real auth replaces the dummy view. Left in place because the chat dev fallback still depends on it — remove both together.
- [ ] `rest_framework.authtoken` is in `INSTALLED_APPS` but used nowhere (pre-existing). Left alone — removing it drops a table.
- [ ] **Pre-existing inconsistency, unfixed:** `.env.example` documents `SECRET_KEY=` but `base.py` reads `DJANGO_SECRET_KEY`. So a deployment following the example file would silently run on the insecure dev key. Harmless today (prod.py now refuses to boot with the dev key when JWT is on), but the example file is misleading and should be corrected.
- [ ] Per-account lockout is a fixed window from the first failure, not rolling. Deliberate (simpler, one Redis op); an attacker gets N tries per window.
- [ ] **Catalog discovery is manual.** `manage.py sync_catalog` is the only trigger. The durable fix is to call `CatalogDiscoveryService` from `apps/ingestion/tasks.py:256` where a source is marked ready — deliberately deferred because touching the ingestion pipeline is its own risk. Until then, every re-ingest leaves that source's resources absent (= denied) until someone runs the command.
- [ ] **The RBAC graph is inert.** The resolver answers, but cache, Gate 1 and Gate 2 do not exist — populating roles and grants still changes no request's outcome.
- [ ] **Resolution is one query per call, uncached.** `EffectivePermissions` is frozen and self-describing precisely so it can be cached whole, but the cache key needs a `PermissionVersion` counter that does not exist. Noted rather than half-built (Phase 7).

---

## 8. Work Log  *(newest first)*

### 2026-08-07 (dropdown) — `roles/dropdown`; resource tree question resolved without new code

User asked for a role dropdown API and a resource-tree API. Checked both against
what already existed before writing anything:

- **Resource tree**: already there. `catalog/list {parent_path}` is documented as
  exactly this — "pass a node's path as `parent_path` to get exactly its children" —
  a lazy-loaded tree, the correct design at 2,201-resource scale. Nothing built.
  The one real gap (does a node show as granted for role X) is R2 in
  `AUTH_ISSUES_BACKLOG.md`, left there.
- **Role dropdown**: `roles/list` exists but is capped at `page_size=100` — fine for
  the admin table, wrong shape for a picker that wants every option in one response
  with no pagination to handle. User's explicit call: a **separate** endpoint, not a
  reuse.

**`POST /api/v1/roles/dropdown`** — `RoleService.list_active_roles()`: every
`is_active=True` role, `{role_id, name}` only, ordered by name, **1 query**, no
pagination block in the response at all (not "page 1 of 1" — genuinely absent, so a
client checking `"pagination" in body` cannot mistake this for the paginated
endpoint). Deliberately safe to leave unpaginated *only* because roles are
administrator-authored and small ("tens to hundreds" per `models/roles.py`) — the
same reasoning does NOT extend to `users` or `catalog`, both populated by something
other than an admin's own typing.

**8 new tests**, one of which needed the same fix `test_grants.py` already needed:
an "assert this is the only role" test broke because migration 0007's seeded
`Admin` role is *also* active and *also* returned — fixed to assert membership, not
exact equality, same reasoning as before. **514 passed** (was 506), `manage.py
check` clean.

### 2026-08-07 (User Story 1) — admin bootstrap, last-admin protection, password policy, login authz context

Codebase inspection first, per the brief's own mandatory-analysis rule — and it found
the brief's "already implemented" list didn't match reality: **no self-registration
endpoint exists anywhere** (`users/create` has always been staff-only), **no
activate/deactivate/password-change endpoint existed**, and **login never checked
role or staff status**. Login lockout, by contrast, WAS already fully built (Phase
1.1) — reused as-is, not rebuilt.

**Delivered:**
- **Migration 0007** seeds an `Admin` `Role` and grants it every permission that
  exists today. Mirrors 0004's own pattern (idempotent, migration-owned, code cannot
  invent authority a migration didn't seed).
- **`AdminBootstrapService` + `manage.py bootstrap_admin`** — backend-only,
  deliberately NOT a public endpoint (the existing `users/create` is staff-only, so
  relaxing it for "zero users" would just be conditional self-registration in
  disguise). Race-safe via `select_for_update()` on the seeded Admin role row —
  Postgres-only guarantee, same documented sqlite caveat as `RoleService.
  update_role`. First user gets **both** `is_staff=True` and the Admin role — the
  user's explicit call, "use both, not one or the other."
- **`admin_guard.py`** — one `is_last_active_admin()` predicate, reused by
  `UserService.update_user` (refuses `is_active: false` on the last admin) and
  `UserRoleService.revoke` (refuses stripping the Admin role from the last admin).
- **`is_active` folded into `users/update`**, not a separate activate/deactivate
  pair — user's explicit correction mid-build ("separate api nahi chahiye, update se
  handle ho jayega"). Same call for "delete": no hard-delete endpoint was built;
  deactivation via `update` IS this platform's delete ("do soft delete only").
- **`PasswordComplexityValidator`** (upper/lower/digit/special, each threshold an
  `OPTIONS` knob in `AUTH_PASSWORD_VALIDATORS` — not hardcoded) plugs into the
  `validate_password()` call `UserCreateSerializer` already had wired in.
- **`POST /auth/password/change`** — `IsAuthenticated` (the one auth endpoint that
  isn't `AllowAny`), its own throttle scope (an authenticated caller can still guess
  a secret), revokes every refresh token on success via the same primitive login
  lockout already used.
- **Login response carries `roles` + `permission_codes` immediately** — gated on
  `jwt_enabled()` so the legacy (flag-off) contract stays byte-identical; a real
  test (`test_login_with_flag_off_returns_the_legacy_payload`) caught the first draft
  breaking that promise. Deliberately NOT embedded in a JWT claim or the session —
  either would cache an authorization decision outside the resolver's per-request,
  always-live read, reintroducing the staleness class Gate 2 exists to avoid.

**Two real findings, handled differently:**
- **A genuine bug of my own**, caught by the test suite, not by review:
  `deactivate_user`'s first draft called `user.save(update_fields=["is_active",
  "updated_at"])` — `django.contrib.auth.User` has **no `updated_at` field**; that
  column only exists on `TimeStampedModel`-based models. Fixed before it shipped.
- **A serious pre-existing issue, flagged, deliberately NOT touched**:
  `apps/chat/migrations/0002_seed_dummy_admin_user.py` seeds a permanent
  `username=admin` user into every migrated database — dev AND prod — and its own
  docstring says to remove it "once the real authentication service replaces
  LoginView," which happened weeks ago. It does not break `bootstrap_admin` today
  (that command keys off `is_staff=True`, and the dummy is never staff), but it is a
  hardcoded dev credential that has been sitting in every environment this whole
  time. Recorded in `PM_LOG.md`'s Open Blockers rather than removed unilaterally —
  touching it changes `apps/chat`'s dev-fallback identity behaviour, a different
  app's concern.

**Tests:** 38 new (`tests/test_admin_bootstrap.py`) + fixes to pre-existing
assumptions the new seed data broke (tests asserting a globally-empty
`RolePermission`/`Role` table, a fixture role literally named "Admin", password
fixtures too weak for the new complexity validator, the concurrent-bootstrap test
rewritten to avoid real threads — sqlite connections don't share this project's test
transaction, the same reason `test_concurrent_refresh_of_one_token_yields_exactly_
one_winner` already avoided them). **506 passed** (was 468), `manage.py check` clean.

### 2026-08-06 (Gate 2) — the first component that can refuse a request
- **Phase order changed deliberately.** The roadmap puts the cache (Phase 7) before Gate 2 (Phase 8), but the same brief says *"cache only after correctness"* — and caching a resolver nothing consults would optimise zero requests while the cache's own read pattern stayed unknown. Gate 2 first; the cache follows it with real access patterns in hand.
- **Three modes, one setting.** `VEDA_RBAC_MODE` ∈ `off` (default) / `shadow` / `enforce`. Shadow is not a nicety: flipping straight from off to enforce on a system where no grant has ever been exercised denies **every unprovisioned user, i.e. everyone**. Shadow turns that outage into a log query, and it logs **only** what it *would* have refused so the signal is the to-do list rather than a copy of the access log.
- **Strictly tighter, never looser — the property that makes this safe to merge.** `RequiresPermission` is added *alongside* `IsAdminUser`, never instead of it. DRF requires every permission class to pass, so `enforce` = staff **and** permission. A test proves a non-staff user holding every RBAC permission is still refused; there is no configuration in which adding the gate grants access that was previously refused.
- **Fail closed on misconfiguration, fail OPEN on a typo'd mode.** A view that opts into the gate without declaring `required_permission` is denied and logged at ERROR — a bug in an authorization gate must be loud. But an unrecognised `VEDA_RBAC_MODE` value falls back to `off`, because treating a typo as `enforce` would take a whole deployment offline. Two different failure directions, each chosen for its blast radius.
- **The resolver runs at most once per request**, cached on the request object — deliberately not a module global, which would leak one user's permissions into another's under concurrency. A test pins that the cache lives on the request; it is invisible until it is catastrophic.
- **`off` costs nothing** — the gate abstains before touching the database, asserted by a query-capture test.
- **Every routed endpoint declares its permission**, checked by walking the URL conf rather than exported names — the abstract `AdminView` base deliberately declares nothing, because a default there is exactly what would let a concrete view forget.
- **Scope is the admin surface only.** `/api/v1/query` is untouched and still `AllowAny`; a test pins that too, so it changes only when Gate 1 deliberately changes it.
- **23 tests**, 468 across the identity work. `manage.py check` clean, pyflakes clean, no new migrations.

### 2026-08-06 (resolver) — Phase 5: the graph becomes answerable
- **`resolve(user) → EffectivePermissions`**, and nothing more. The resolver never inspects a request, never raises 403, never knows what an endpoint is. Keeping resolution and enforcement apart is what lets this be deployed, observed and cached *before* a single request's outcome changes.
- **Implements ADR §3.5 exactly**: collect grants at every prefix-or-equal path → any DENY wins → else any ALLOW → else deny. Two independent fail-closed rules, both tested: **DENY is unpierceable at any depth** (so "deny the source except one table" is genuinely not expressible — pinned by a test, not left to memory), and **absence of a grant is denial**.
- **A blank-path grant does NOT cover concrete resources.** `resource_path=""` means "not resource-scoped" (`user.manage`), not "everything". Follows from the path grammar — a blank path has zero segments and is never a prefix of a real one — and it is the fail-closed reading: granting `data.read` with no resource must not silently open every table.
- **Anything inactive contributes nothing** — user, role *or* permission — and the filtering happens in the SQL, not in Python, so a disabled capability cannot leak through a caller that forgot to check.
- **One query, then zero.** The whole traversal (user → roles → grants) is a single joined SELECT; every subsequent `allows()` is in-memory against a per-code index built once, so a decision costs O(depth) with `depth ≤ MAX_SEGMENTS`. Both properties are asserted (`test_resolution_is_one_query`, `test_checks_cost_no_queries`) — the second is what makes the object worth caching.
- **The result is immutable** (frozen dataclasses + `MappingProxyType`), because it is an authorization answer that will be cached and shared; a mutable one is an answer a later caller can quietly edit. A test tries all three mutation routes.
- **`denies()` is deliberately distinct from `not allows()`** — "explicitly denied" and "never granted" are different problems for an operator to fix, and an admin screen that conflates them sends people looking in the wrong place.
- **`POST users/permissions/effective`** (Phase 6's "List Effective Permission") returns the whole set, and — if asked a specific `permission_code`/`resource_path` — a `decision` block computed by the resolver itself. That is deliberate: a client that re-implements prefix inheritance and DENY precedence will eventually disagree with the server, and a UI showing "allowed" where the server denies is worse than no UI. It 404s for an unknown user rather than returning an empty set, because "no permissions" and "no such user" are different answers.
- **Cache seam left open, not half-built**: `EffectivePermissions` is frozen and JSON-projectable so Phase 7 can cache it whole, but the key needs a `PermissionVersion` counter which does not exist yet. Building half of it now would have produced a cache nobody could invalidate correctly.
- Self-review caught two of my own test defects before presenting: an immutability assertion expecting `TypeError` where a tuple raises `AttributeError`, and a no-op `assert x if False else True` line — the same pattern I had flagged in an earlier review.
- **35 tests**, 445 across the identity work. `manage.py check` clean, pyflakes clean, no new migrations.

### 2026-08-06 (grants) — `UserRole` + `RolePermission`; the RBAC graph is now complete
- **Two edges, one module.** `models/grants.py` holds both because they are the same concept — a directed, audited edge with identical idempotency rules. Splitting them would have produced two near-identical files whose shared conventions could drift.
- **Idempotency is the contract.** `assign`/`grant` describe a desired *state*, so repeating them is success: **201 when created, 200 when it already existed**. `revoke` is always 200 and deliberately does **not** 404 on unknown targets — the desired end state ("does not hold it") is already true, and a revoke script must not fail on exactly the rows it has nothing to do. Deliberately different from `users/create`/`roles/create`, which 409 on a duplicate: creating implies newness, granting implies membership.
- **One decision per triple — the correctness core.** `(role, permission, resource_path)` is unique **without** `effect`, so re-granting with the opposite effect UPDATES the row. Two rows disagreeing about one triple would make the authorization outcome depend on row order, which is the worst bug class available here. Enforced by the database, not by the service.
- **Cascade rules chosen individually, not defaulted:** `UserRole.user` CASCADE (an assignment without its user is meaningless) · `UserRole.role` PROTECT (a held role cannot be deleted) · `RolePermission.role` CASCADE (grants are part of the role) · `RolePermission.permission` PROTECT (the catalogue is seeded and never deleted; an attempt must fail loudly rather than silently revoking every grant of that capability). Each has a test.
- **`resource_path` is a plain string, not an FK to `CatalogResource`** — same reasoning as one level up. A grant must survive catalog churn, and pre-provisioning a source that is still ingesting is legitimate. The path is canonicalised on write so it can never be stored in a shape the resolver would fail to match, and the list/grant responses carry **`resource_exists`** (one query per page, no N+1) so a typo is visible instead of hiding until someone wonders why access never worked.
- **Retired role / disabled permission are refused (409)**, because assigning something inert is the same "authority that does not exist" problem the permission catalogue is read-only to avoid. An **inactive user** *can* be assigned — pre-provisioning is legitimate and grants nothing until the account is enabled.
- **No approval workflow**, confirmed with the user: assign and grant take effect immediately. `granted_by` records who acted but gates nothing. Out of scope by decision; a `status` column or a separate request table would both be additive later.
- **44 tests**, 410 passing across the identity work. One of them — `test_nothing_is_enforced_by_these_rows_yet` — asserts `/api/v1/query` is still `AllowAny`, so nobody mistakes a populated grant table for working authorization.

### 2026-08-06 (catalog) — resource paths + the `CatalogResource` projection
- **Implements ADR-0001.** `resource_path.py` is deliberately **pure** — no Django imports — so models, discovery, the resolver and the gates can all share one definition without an import cycle, and it is exhaustively testable without a database. Its dialect→kind map is keyed by plain strings, with a test asserting every `Source.Dialect` value is covered so the decoupling cannot silently drift.
- **The security-critical function is `is_prefix_of`,** which matches on **segment boundaries**, never string prefixes. A naive `startswith` returns True for `("db:crm", "db:crm_postgres")` and grants an unrelated source — the classic prefix-authorization bug, pinned by a parametrized test.
- **🔴 My own ADR would have destroyed grants, caught while starting the implementation.** §3.7 originally specified nullable FKs to `SchemaTable`/`SchemaColumn` with CASCADE. But `storage_adapters/writer.py:137-139` **deletes and recreates the entire structural substrate for a source on every re-ingestion** — so CASCADE would delete every catalog row and every grant referencing it, and PROTECT would break the ingestion pipeline outright. Root cause of my error: I treated the substrate as durable entities; they are a rebuildable projection with deliberately ephemeral row lifetime (their *ids* are stable — `uuid5` — but their rows are not). Stopped per the FINAL RULE, amended the ADR to no-FK, and re-approved before writing code.
- **Reconciliation replaces referential integrity.** Discovery inserts what is new, reactivates what returned, and **deactivates — never deletes** — what vanished, because deleting would silently drop the grants pointing at it. `test_reingestion_does_not_destroy_catalog_rows` simulates the writer's delete exactly; if it ever fails, grants are being silently revoked.
- **Unaddressable names are reported, not skipped.** A table whose name contains `:` cannot appear in a path; discovery records it in the report and the command prints it to stderr, because nobody can grant access to a resource that has no name.
- **Self-review caught two more:** `bulk_update` does **not** call `Field.pre_save`, so `auto_now` never fires — the reactivation path was writing back a stale `updated_at` and the row silently claimed it had not changed (note the contrast with `save(update_fields=...)`, which *does* call `pre_save`, which is why roles/users were fine). And `parent_path` had a docstring promising tri-state behaviour that its own code did not implement.
- **68 tests**, 406 passing at that point. `manage.py sync_catalog` added as the ops surface; hooking discovery into ingestion is recorded as the deliberate follow-up.

### 2026-08-06 (permissions) — the action catalogue, read-only by design
- **🔴 Inspection changed the design before any code was written.** The brief listed `CatalogResource` as in-scope, but VEDA **already has a catalog**: `sources.Source` (int PK), `substrate.SchemaTable` and `substrate.SchemaColumn` (UUID PKs, tenant-scoped, with `is_sensitive`/`excluded` flags already modelled). A new `CatalogResource` describing databases/tables/columns would duplicate all of it. Deferred to the grant phase and reframed: when it comes, it must be a thin **registry** that normalizes addressing over those rows, never a copy of their attributes.
- **Also found:** there is no data-access authorization anywhere today. `apps/query/scope.py` calls its check "ownership", but it intersects with `Source.objects.filter(ready=True)` — an *ingestion* state, not an authorization one. Any caller reaching the query endpoint can query every ready source. Recorded here because the gates phase has to close it.
- **`Permission` = the action catalogue.** Verb and noun kept apart exactly as the CATALOG rule requires: a permission names *what can be done*; *what it is done to* binds at grant time against the catalog that already exists. No `resource_type` column — the `Source`-int-PK vs `SchemaTable`-UUID-PK problem is real and gets solved with the grant requirements in hand, not guessed at now.
- **Read-only on purpose, and that is the interesting decision.** Only code can enforce a permission, so a row an administrator invents at runtime is one no gate will ever check — the UI would present authority that does not exist. The catalogue is seeded by migration 0004 and exposed as `list`/`detail` only. Tests assert `permissions/{create,update,delete}` do **not** resolve and that the service exposes no write method, so adding one has to be a deliberate, visible act. Roles stay the admin-composable layer.
- **8 permissions seeded, each mapped in the migration to the endpoint it will gate** (`query.execute` → QueryView/ConversationQueryView, `ingestion.run` → IngestTriggerView, …). Nothing speculative: a permission with no code path to gate it is a promise the system cannot keep.
- **Seed is idempotent and deploy-safe**: `update_or_create` keyed on `code`, so this file stays the source of truth for name/description — but `is_active` is deliberately NOT in the defaults, so an operator who switches a capability off does not have it silently re-enabled by the next deploy. Both properties are tested.
- Name collision noted in the model docstring: `django.contrib.auth.models.Permission` is unrelated (model-level add/change/delete/view). A test pins that they are distinct classes in distinct tables, because confusing them would be an authorization bug.
- **38 tests**, 338 passing across the identity work. `manage.py check` clean, pyflakes clean, `makemigrations --check` no changes.

### 2026-08-06 — Role management: `Role` model + `roles/{create,detail,list,update}`
- **Custom `Role` model, not `auth.Group`.** Inspected Group first: it carries only `name` (150, unique) and an M2M to `auth.Permission`. No description, no active flag, no timestamps — an admin UI would need a parallel profile table beside it. More decisively, `auth.Permission` is *model-level* (add/change/delete/view per Django model) while VEDA's resources are data sources, schemas, tables and columns, so binding roles to Group would leave `Group.permissions` present, unused and misleading. Reuses `apps.core.models.TimeStampedModel` for the timestamps rather than re-declaring them.
- **Case-insensitive unique name** via `UniqueConstraint(Lower("name"))` — expressible natively in Django because we own this model, unlike migration 0001 which needed raw SQL for the same rule on `auth_user.email`. Enforced by the DB, so two concurrent creates cannot both win; a direct-ORM regression test proves the constraint, not just the service.
- **No `roles/delete`.** Retirement is `update {is_active: false}`. Hard-deletion semantics depend on role *assignment* (what happens to users holding it?), which does not exist yet; an audit trail saying "granted role #7" must still resolve #7; and a retired role keeps its name reserved so nothing can shadow it in history. A test asserts the route does not exist, so it cannot appear by accident.
- **🔴 The tests caught a real design flaw, not just a bug.** `ERROR_STATUS` mapped *concrete* classes (`DuplicateUser`, `UserNotFound`), so the brand-new `RoleNameTaken`/`RoleNotFound` fell through to the 400 fallback — every role conflict silently answered 400 instead of 409/404. A central registry each new domain must remember to update is the wrong shape. Replaced with semantic `ConflictError`/`NotFoundError` bases that domains **inherit** from; `ERROR_STATUS` now has exactly two entries and a future domain gets the right status for free.
- **Second self-inflicted bug:** I put `role_id` in `READ_ONLY_FIELDS`, so `roles/update` rejected its own required target. Fixed by making the per-serializer allowlist the rule and letting `READ_ONLY_FIELDS` only refine the *message*.
- **Shared paging extracted now that there are two consumers** (not before): `PaginatedListSerializer`, `services.base.paginate()`, `views.base.pagination_payload()`. Users refactored onto them with no behaviour change — their 104 tests passed unmodified.
- **Self-review found four more things before presenting:** two different `_classify_conflict` shapes across domains (roles hid a `raise` inside a function named "classify" — a hidden side effect; unified on the pure-classifier shape users already had); log lines hardcoding "user" so roles logged "user role creation rejected"; a self-contradicting comment on `ROLE_LIST_FIELDS`; and a no-op `assert x if False else True` I had left in a test.
- **Honest limit recorded in the code:** `select_for_update()` is a **Postgres-only** guarantee — SQLite reports `has_select_for_update = False` and Django silently ignores it, so the local suite exercises the update path but does **not** prove the lock. Documented in both `update_role` and `update_user` rather than left implied.
- **77 role tests**, 300 passing across the identity work. `manage.py check` clean, pyflakes clean, `makemigrations --check` no changes.

### 2026-08-05 (detail + package split) — `users/detail`; app restructured for the RBAC domains to come
- **`apps/access_management` is now packages, not modules**: `serializers/`, `services/`, `views/`, `urls/`, each with one module per domain (`users.py` today) plus a `base.py` for what genuinely spans domains. Done on the user's call, ahead of roles/permissions, so those land *beside* users instead of growing a 1,000-line `services.py`.
- **What went into `base.py` is only what is actually shared**: `services/base.py` holds `AccessManagementError` (so the view layer needs one `except` clause, not one per domain); `views/base.py` holds the staff-only rule, the typed-error→HTTP-status MRO walk, the validate-or-400 branch, and `log_context`. Resisted moving anything user-specific there — a `base.py` that accumulates one domain's helpers is how shared modules rot.
- **Each package `__init__` re-exports its public names**, so callers keep importing from the package (`from apps.access_management.services import UserService`) and a class can move between modules later without breaking importers. A test asserts this indirection actually holds, including that `services.AccessManagementError` *is* the one in `services/base.py` rather than a duplicate.
- **`POST /api/v1/users/detail`** — one user by id, reusing the `get_user` service method that already existed. Same `public_fields` projection as create/list/update, so opening a row returns exactly the shape that was clicked (asserted: `detail == created == listed`). One query. A nonsensical id (`0`, `-1`, `"abc"`) is a **400**, not a 404, so a client bug and a genuinely absent user stay diagnosable apart.
- **`views/base.py` is where the future RBAC permission check lands** — one place, not one per endpoint. `IsAdminUser` stays until the permission model exists; no placeholder was written for it.
- Caught while moving files: `MSG_PRIVILEGED_FIELD` still read "cannot be set when creating a user" although the update serializer reuses it — generalised. Also fixed two docstrings still naming the pre-move `POST /api/v1/users` path.
- **104 tests** in this module (14 new), 183 across the identity work. `manage.py check` clean, pyflakes clean, `makemigrations --check` no changes.

### 2026-08-05 (list + update) — `users/list` + `users/update`; API style corrected to POST-everywhere
- **`POST /api/v1/users/list`** — always paginated (`page_size` capped at 100), `search` across username/email, tri-state `is_active`, allowlisted `ordering`. **Two queries regardless of page size** (one COUNT, one page fetch), asserted by a test. Ordering carries a secondary sort on `id` so paging visits every row exactly once — without that tiebreak, rows sharing a sort value repeat or vanish between pages.
- **`POST /api/v1/users/update`** — partial (`update_fields` only writes what was sent, so a concurrent change to another column is not clobbered), row locked with `select_for_update` inside the transaction to avoid a lost update, 404 for a missing id, 409 for an email owned by someone else, and re-submitting the user's *own* email is fine.
- **🔴 I had the API style wrong and the user corrected me.** I built toward `GET /users` + `PATCH /users/{id}` — textbook REST, but **not this project's convention**: `apps/chat` has always been `POST conversations/{create,list,history}`, and the brief said "do not invent a new API style". Corrected to `POST <resource>/<action>` everywhere, with all parameters in the body. **Create moved from `POST users` to `POST users/create`** so the scheme is not half-REST/half-action — nothing consumed it yet, so the churn was free, and a mixed scheme would have been a permanent inconsistency.
- **One user representation.** `public_fields()` is now shared by create/list/update and gained `is_staff`, `date_joined`, `last_login`. Previously create had its own five-key projection; a list row and a create response disagreeing about what a user looks like is the kind of drift that reaches the frontend. `is_staff` is *reported* but never *accepted* — the privileged-field denylist still rejects it on input.
- **`update` needed `_classify_conflict(exclude_pk=…)`** — without it, a user re-submitting their own email would have been told it was taken.
- **Test suite: 171s → 13s.** The new list fixtures create ten users each, and production PBKDF2 costs ~310ms per password, so setup alone was 7-9s per test. Switched the module to a fast test hasher; the two tests that actually assert hashing behaviour re-enable the real hashers explicitly, so the speed-up cannot make them pass vacuously. Also rewrote `test_list_never_exposes_password_hashes` to assert the *actual stored hash strings* are absent rather than grepping for `"pbkdf2"` — which the fast hasher would have made a no-op.
- Also extracted `api.iso_z()` into `apps/core/api.py` (one timestamp format platform-wide) and pointed `apps/chat/views._iso_z` at it rather than duplicating the format.
- **90 tests** in this module, **233 passing** across the touched suites. `manage.py check` clean, pyflakes clean.

### 2026-08-05 (user creation) — `apps/access_management` created; `POST /api/v1/users`; envelope de-duplicated
- **New bounded context `apps/access_management`** — identity *administration* + authorization (RBAC), separate from `apps/authentication` which keeps identity *verification* only. Per the user's call, this app is the future home of role/permission management, resolution and caching. Nothing speculative was built: no role field, no permission table, no resolution hook.
- **`POST /api/v1/users/create`** (originally `POST /api/v1/users`; moved when list/update landed) — staff-only (`IsAdminUser`, the same gate `apps/query/views.py` already uses), 5-field allowlist, `create_user()` hashing, atomic, 201 with an explicit projection. **1 INSERT and zero uniqueness SELECTs** on the happy path (measured).
- **Reused, not rebuilt:** `IsAdminUser`; the authentication classes already in settings; `UserManager.create_user`; `UnicodeUsernameValidator` and field widths read *from the model*; and `AUTH_PASSWORD_VALIDATORS` — configured since the project began but **never once invoked**, because until now nothing set a password from user input.
- **Privilege escalation / mass assignment**: privileged keys (`is_staff`, `is_superuser`, `is_active`, `groups`, `user_permissions`, …) are **rejected with a 400**, not silently dropped — a caller must never believe it minted an admin. Non-string values are rejected too: DRF's `CharField` would otherwise coerce `{"username": 12345}` into a user named `"12345"`.
- **Email uniqueness needed a real constraint.** Stock `auth.User` declares `email` as `unique=False`, so validation-only uniqueness would be check-then-insert — two concurrent requests both pass, both insert. Migration `0001` adds a **case-insensitive partial unique index** (`LOWER(email)` where `email <> ''`), verified portable on Postgres and sqlite 3.46, preserving the blank-email seeded admin.
- **🔴 The migration silently did nothing at first, and the tests caught it.** `migrations.swappable_dependency()` resolves to `("auth", "__first__")`, so the index was built right after `auth.0001` — and the later `auth.000{4,5,8,9,12}_alter_user_*` migrations make **SQLite rebuild the table**, discarding indexes Django does not know about. The migration was recorded as applied with **no index present**: uniqueness silently unenforced. Fixed by also depending on `("auth", "__latest__")`. The lasting guard is a test that asserts the *constraint is enforced* by a direct ORM write, not merely that the migration ran.
- **Envelope de-duplicated → `apps/core/api.py`.** The `{status_code, message, data}` shape had been re-typed at 14 call sites across `apps/chat` (6) and `apps/authentication` (2 helpers); a third copy was about to be written. Now one definition, used by all three apps. Optional keys are **omitted, not null** (logout deliberately returns no `data`).
- **Two of my own bugs, found in self-review rather than by a test:** (1) the refactor left a local variable `error` in `apps/chat/views.py` shadowing the imported `error` helper, so the 502 branch would have called `None` — fixed, and then made structurally impossible by switching every call site to the qualified `api.error(...)` / `api.success(...)` form; (2) an unattributable `IntegrityError` was being reported as `409 already exists`, which would send an admin hunting for a row that does not exist — it now re-raises as a 500 with the traceback logged.
- **51 new tests.** Success, persistence + hash usability, unprivileged defaults, 401/403, escalation, duplicate username/email (incl. case-only), blank-email coexistence, DB-level enforcement, the migration's duplicate pre-check, validation (missing/malformed/over-long/wrong-type/non-object body), all four password validators, rollback, keyword-only service API, and regression tests pinning the chat + auth envelopes.
- Regression: `manage.py check` clean, pyflakes clean, `makemigrations --check` reports no changes, and 158 existing tests still pass (auth 79, apps-layer 40, chat-viz 24, rag 6, explain 9).
- **Known limit, documented not fixed:** the endpoint cannot bootstrap the first admin — it needs a staff account and creates only non-staff ones, and the seeded `admin` has `is_staff=0`. `createsuperuser` remains the entry point; a bootstrap flow is out of scope.

### 2026-08-05 (Phase 1.1) — C1 · C2 · C3 · H1 fixed (+H4, +M9 found); 79 tests
- **C1 — lockout DoS closed** by splitting one counter into two: per-(account, source-IP) **hard** block (10/5min, refuses before any hash is computed) plus an account-wide **soft** counter (50/5min) that turns a *wrong* password into 429 but **never** refuses a correct one. Independently probed: attacker at `203.0.113.7` is 429 after 5 failures while the owner logs in from `198.51.100.22` with `200`.
- **C1 side-quest → M9, a pre-existing bug.** Keying a lockout on an IP forced the question "which IP do we trust?". `NUM_PROXIES` was **unset**, so DRF used the *whole* `X-Forwarded-For` as the client identity — meaning the project's existing per-IP throttles were bypassable by anyone setting that header. Now `NUM_PROXIES=1` (nginx sends `$proxy_add_x_forwarded_for`, so the true peer is the last entry), and the lockout reuses DRF's own `BaseThrottle.get_ident` rather than re-deriving an IP. One definition of "which client is this", and the existing throttles got fixed as a side effect.
- **C2 — fail-open restored.** `_record_failure` now guards `incr` and `set` in **separate** `try` blocks; an exception raised inside an `except` clause is not caught by a later clause on the same `try`, which is exactly how a Redis blip between the two turned into a 500. Probe now returns `InvalidCredentials` (→401) where it previously raised `ConnectionError`.
- **C3 — password change revokes tokens.** `CHECK_REVOKE_TOKEN: True` **plus** an explicit check on the refresh path, because simplejwt enforces that claim only in `JWTAuthentication.get_user` (access tokens) — enabling the setting alone would have left rotation honouring tokens minted under an old password. Verified I checked this rather than assumed it: `grep` located enforcement at `authentication.py:141` only. Tokens carrying no claim fail closed (one forced re-login, no grandfather clause).
- **H1 — the access token is now actually tested.** A protected view (`JWTAuthentication` + `IsAuthenticated`) mounted on the **real** root urlconf — not a parallel test mount that could diverge — driven over HTTP for: valid, missing, garbage, refresh-used-as-access, expired, forged, deactivated user, post-password-change. Plus an assertion that `base.py` installs the auth class exactly when the flag is on, since the tests mount it explicitly and would otherwise pass with the settings broken.
- **H4 fixed incidentally** (C2 rewrote the same functions): cache-outage logging is one line naming the exception instead of ~30 lines of Redis traceback twice per login.
- **Test suite: 59 → 79.** One old test was **deleting a vulnerability assertion**: `test_login_locks_an_account_out_after_repeated_failures` asserted that a *correct* password gets 429 — it encoded C1 as desired behaviour. Replaced with `test_an_attacker_cannot_lock_out_a_legitimate_user` and three siblings covering the soft counter and XFF spoofing.
- Regression: `manage.py check` clean, pyflakes clean, `test_apps_layer_refactor` (40) / `test_chat_visualization` (24) / `test_rag_layer_events` (6) all pass. No request/response shape changed; `AUTH_API_CONTRACT.md` §1.3 and §3.1 updated with the new lockout and revocation semantics.

### 2026-08-05 — Authentication module: login + refresh + logout, flag-gated OFF
- Analysed the existing auth surface first: the only login was a **dummy** in `apps/chat/views.py` returning the literal string `"dummy_access_token"`. **No JWT anywhere** in the repo — `PyJWT`/`simplejwt` absent from requirements and from `.venv`; the only mentions were TODO notes in `apps/query/views.py:8`, `docs/ARCHITECTURE.md:280` and `migration_plan.md:779` all marking JWT as unbuilt Phase-6.2 work. `rest_framework.authtoken` installed but never called. No `BaseAPIView`/`BaseSerializer`/response-helper/JWT-util/user-service to reuse — confirmed absent, not assumed.
- Reused what did exist: the `{status_code, message, data}` envelope, the input-only serializer idiom, the `CODE_*`/`MSG_*` safe-copy error pattern, `logging.getLogger(__name__)` + `request.request_id`, Django's cache, the DRF/nginx throttles, and the `apps/<domain>/…` layout. Login serializer **moved** out of chat rather than reimplemented.
- Wrote it in the agreed order — login → self-review → refresh → self-review → logout → architecture review.
- **Caught by my own self-review, mid-build:**
  - refresh had to reject outright while the flag is off — otherwise flipping the flag off with live tokens would *spend* a real refresh token and hand back a placeholder;
  - family revocation must run **outside** the rotation transaction, or the `raise` rolls back the revocation it just performed;
  - `_revoke_all_for_user` now skips already-expired tokens, so a replay scales with an account's live sessions instead of its entire history;
  - a first test-harness attempt stripped `REST_FRAMEWORK` to disable throttling, which hid that a missing throttle-scope rate is a **500 on every login** — override dropped, and the scope wiring is asserted instead.
- **Flagged, not silently changed:** the `.env.example` `SECRET_KEY` vs `DJANGO_SECRET_KEY` mismatch above; and DRF `SessionAuthentication` means a browser that already holds a Django admin session must send `X-CSRFToken` to these endpoints — identical to every existing endpoint in the project, so it is a pre-existing pattern rather than a regression here.

### 2026-08-07 — Gate 1 (User Story 3) Task 13: source-level scope narrowing
- Both HTTP entry points confirmed to converge on one function: `apps.query.scope.resolve_query_scope()`, called from `apps/query/views.py::QueryView.post` and `apps/chat/views.py::ConversationQueryView.post` — added an optional `user=None` param there instead of duplicating logic per-view. `None` (no caller passes it) is byte-identical to pre-change behaviour; both real callers now pass their resolved `request.user`/chat `user`.
- New `_permitted_source_ids(user)`: RBAC-off or no user → `None` (no narrowing, admin bypass for `is_staff` too — the seeded "Admin" role's grants are all global/blank-path, which per the resolver's own rule never opens a specific source, so without this bypass even the platform admin role would see zero sources). Otherwise derives the permitted source-name set directly from `EffectivePermissions.grants` (`data.read`, `Effect.ALLOW`, non-global) — deliberately not `.resources_for()`, which by its own docstring doesn't expand to descendants.
- **Bug caught before shipping, via a case-mismatch test:** `resource_path.build()` (used by `CatalogService` when the catalog is discovered) always **lowercases** the source name into the path, but `Source.name` is stored as typed. A plain `Source.objects.filter(name__in=source_names)` against the already-lowercased set would silently drop any source with uppercase in its name — narrowing away access that was actually granted. Fixed with a `Q(name__iexact=...)` OR-chain instead.
- **Known, deliberate, documented gap — NOT fixed in this task:** `resolve_query_scope()`'s existing contract is "always return a non-empty scope" (dev fallback to `VEDA_DEFAULT_SOURCE_ID`). When RBAC narrows permitted sources to the empty set, it still falls back to the default source rather than returning `[]` — so right now, until the view-level 403 lands (Task 17), a zero-permission user's query still reaches the default source. Pinned with an explicit test (`test_no_permissions_falls_back_to_the_dev_default_source_not_a_403`) rather than silently patched, since fixing it belongs to the view layer that has to decide 403 semantics, not to this scope-resolution function.
- New `tests/test_query_scope_rbac.py` (13 tests, real Postgres/sqlite test DB + role/grant fixtures, same pattern as `test_permission_resolver.py`): RBAC-off ignores grants, no-permissions fail-closed at `_permitted_source_ids` level, full/partial access, request-pin outside granted scope doesn't leak, global grant doesn't open a source, deny-only source stays out, multi-role union, admin bypass, case-insensitive source-name matching, not-ready source stays excluded even with a grant.
- Regression: `manage.py check` clean; 111 tests pass across `test_apps_layer_refactor.py` + `test_gate.py` + `test_permission_resolver.py` + the new file.
- Next: Task 14 (Django-side table/column allow-payload per source), Task 15 (wire it across the HTTP boundary into `RequestContext`), Task 16 (engine-side `sm` filter), Task 17 (403 handling for the empty-permitted-set gap above + full Gate 1 test suite + self-review).

### 2026-08-07 — Gate 1 (User Story 3) Task 14: table/column allow-payload
- New `apps.access_management.services.data_scope.compute_data_scope(user, source_ids)`, mirroring Task 13's contract: `None` = no restriction (RBAC off or `is_staff`); otherwise one `SourceDataScope(open, tables)` per already-permitted source id.
- **"Fully open" vs "restricted", per source AND per table** (the design sketched in the Task 13 analysis): a source/table is `open=True` (no enumeration at all) only when it has a bare ALLOW at its own path AND no narrower DENY exists anywhere beneath it — a redundant deeper ALLOW does not disqualify "open" (it restricts nothing), only a DENY does. Keeps the cross-process payload small for the common whole-source/whole-table-ALLOW case; only actually restricted sources get walked and enumerated at table, then column, granularity. Reuses `EffectivePermissions.allows()`'s own deny-wins prefix semantics directly rather than reimplementing them.
- **Real bug caught by a test, fixed before shipping**: first draft only enumerated a table's columns if the table itself also had an `allows()`-true path — which silently excluded any table reachable ONLY via column-level grants (no redundant table-level ALLOW), a legitimate admin pattern. Fixed by checking every non-fully-open table's columns directly, independent of whether the table path itself is separately allowed.
- Resolves real table/column **names** (not the lowercased `resource_path` segment) by joining `CatalogResource.substrate_id` back to `SchemaTable`/`SchemaColumn` via `all_tenants()` — same tenant-bypass precedent as `CatalogDiscoveryService`, since a resource path carries no tenant and there is no ambient request context in this service.
- **Fails closed** when no catalog projection exists for a source at all (discovery never ran) — reports `open=False, tables=()` rather than treating "nothing to check against" as "nothing restricts it".
- New `tests/test_data_scope.py` (13 tests, builds a REAL catalog via `CatalogDiscoveryService().sync_source()` from real `SchemaTable`/`SchemaColumn` rows rather than hand-built `CatalogResource` fixtures): off/staff bypass, bare-source-open, deny-carve-out disqualifies open, redundant-allow stays open, table-only grant, ungranted table omitted, column-level enumeration, table with zero reachable columns omitted, no-grants-at-all, multi-source independence, no-catalog-projection fails closed.
- Regression: `manage.py check` clean; 376 tests pass across the full access_management + query-scope suite.
- Next: Task 15 (wire this payload across the Django→inference HTTP boundary into `RequestContext`), Task 16 (engine-side `sm` filter), Task 17 (403 handling + the Task-13 empty-permitted-set gap + full suite + self-review).

### 2026-08-07 — Gate 1 (User Story 3) Task 15: data-scope payload across the HTTP boundary
- **Confirmed a summary-stage assumption was wrong before wiring anything**: chat's engine call is NOT a second HTTP hop through a different path — `chatbot/nodes.py::call_engine_node` uses the exact same `apps.query.inference_client.InferenceClient` that `/api/v1/query` uses (`chatbot` runs inside the api container process, per its own docstring; `langgraph` is in `requirements/api.txt`). So both real entry points converge on ONE client to modify, not two.
- Wire path: `apps/query/views.py` / `apps/chat/views.py` compute `serialize_data_scope(compute_data_scope(user, source_ids))` (Task 14's function) → threaded as a plain JSON-safe dict (never the Django `user`/dataclasses) through `ConversationQueryService` → `chatbot.run.run_chat_turn(data_scope=...)` → LangGraph `ChatState["data_scope"]` → `call_engine_node` → `InferenceClient.stream_hybrid_query(data_scope=...)` / `.run_hybrid_query(...)` → new `X-Veda-Data-Scope` header (JSON-encoded, omitted entirely when `None` — "no header" and "no restriction" stay the same signal on both sides, matching the existing `X-Veda-Source-Ids` precedent).
- `veda_core/context.py::RequestContext` gets a new `allowed_resources` field — deliberately a hashable nested-tuple structure, NOT the Django-side dataclasses (this module explicitly carries no Django import, api tier explicitly never imports `veda_core` — kept both boundaries intact). New `parse_allowed_resources(raw)` — Django-free pure parser, raises loudly on malformed JSON rather than guessing, so the caller decides the fail-closed behaviour.
- `inference/main.py`'s `_tenant_context` middleware reads `X-Veda-Data-Scope`, calls `parse_allowed_resources`; a malformed header is caught and **fails closed to `()`** ("nothing addressable"), never silently falls through to `None` ("no restriction") — a sender bug must never widen access.
- New `tests/test_gate1_data_scope_wiring.py` (15 tests, Django-free like `test_apps_layer_refactor.py`, uses `starlette.testclient.TestClient` against the real `inference.main.create_app()` with `set_context` monkeypatched to capture the constructed `RequestContext`): serialize shape (open/restricted/JSON-encodable), header presence/absence, both real `InferenceClient` methods accept `data_scope=`, parse round-trips serialize's own output, hashability, malformed-JSON raises, and the middleware's three cases (absent/valid/malformed header).
- Regression: `manage.py check` clean; 407 passed across the full RBAC + query-scope + wiring suite (2 pre-existing, unrelated failures in `test_analytics_context.py` reproduced identically on the pre-change tree with the same file combination — a known test-order/env issue, not caused by this work).
- Next: Task 16 (engine-side `sm`/`sm['tables']`/`sm['columns']` filter reading `ctx.allowed_resources`, applied at the `_load_scoped_sm`/`_load_semantic_model` sites — a filtered COPY, never mutating the cached `sm`), then Task 17 (403 handling, incl. the Task-13 empty-permitted-set fallback gap, full Gate 1 suite, self-review).

### 2026-08-08 — Gate 1 (User Story 3) Task 16: engine-side filtering

**1. Existing Implementation Analysis** — `veda_core/veda/runtime.py::get_engine()` builds one `RetrievalEnginePhase3` (BM25/signal index) PER SCOPE (`_engine_scope()` = tenant + source_ids) and caches it in `_ENGINES`, reused by every request in that scope — never rebuilt per user, because building it is expensive and the scope's underlying data doesn't change between users. `veda_core/veda_hybrid.py::_load_semantic_model()` returns the exact same `sm` object that flows straight into `get_engine(sm)` inside `veda_core/veda/pipeline.py::run_query()` — confirmed by tracing all 4 call sites (`sm, cols = _load_semantic_model(); ...run_query(query, sm, cols, ...)`).

**2. Gap Analysis** — no RBAC narrowing existed anywhere in the engine tier; Tasks 13-15 only got the payload as far as the ambient `RequestContext`. The gap is applying it. **Self-caught regression before shipping**: my first draft filtered `_load_semantic_model()`'s return value directly — since that's the SAME object handed to `get_engine(sm)`, this would have baked whichever user's request misses the per-scope cache first into the SHARED retrieval engine, permanently, for every other user sharing that scope (a cross-user permission leak). Caught by re-reading the call graph before writing tests, not by a test itself. Reverted immediately.

**3. Implementation Plan** (per the earlier STOP-and-approve on this exact question) — filter the per-request retrieval CANDIDATE LIST after `get_engine(sm).retrieve(...)` returns it, never the shared engine or the sm handed to it. Confirmed with the user before implementing.

**4. Files Modified** — new `veda_core/veda/rbac_filter.py`; `veda_core/veda/pipeline.py` (wired the filter call + an `_ambient_ctx()` helper matching `veda.execution._scope_source_ids`'s established dual-module-name read pattern); `veda_core/veda_hybrid.py` (reverted the unsafe edit, left a comment explaining why `_load_semantic_model()` must stay unfiltered).

**5. Reason for Change** — see Gap Analysis.

**6. Implementation** — `filter_sm(sm, ctx)` (safe for any per-request-fresh `sm`, e.g. a future non-engine consumer) and `filter_retrieval_results(results, sm, ctx)` (the one actually wired in, in `pipeline.py` right after retrieval + graph-expansion, before reranking). Both no-ops (identity) when `ctx.allowed_resources` is `None`. Matches the engine's own `col_id.rsplit(".", 1)` convention for splitting `"table.column"` keys — a first-dot split breaks on the multi-source merge's `src{ID}.` qualified names (`_merge_scoped_sms` in `runtime.py`), which are 3-dots-deep (`"src2.employee.id"`); `rsplit` on the LAST dot recovers `("src2.employee", "id")` correctly, matching how `retrieval_engine_phase3.py::_results_from_tuples` itself splits `col_id`.

**Known, deliberately scoped limit** (flagged, not silently claimed as covered): this filters the retrieval-DRIVEN candidate path — the common/primary flow. It does not independently audit every other `sm`-consuming code path inside `run_query` (existence checks, multitable/FK-join expansion, the `anchor_hint` salvage retry) for a hypothetical case where such logic discovers a table that was never a retrieval candidate at all, bypassing `results` entirely. Given the size and delicacy of that ~1000-line function and the "NEVER hamper existing pipeline" rule, auditing every such path was judged out of safe scope for this pass — noted here for Task 17's self-review rather than overclaimed as airtight.

**7. Tests** — new `tests/test_rbac_filter.py` (15 tests, pure functions, no Django/heavy ML deps): identity on no-ctx/no-restriction, open-source keeps everything, restricted source/table/column narrowing, a source absent from `allowed_resources` denied, the 3-dot qualified-name case, `retrieval_documents` following their column/table, never mutating the input, and the retrieval-results mirror of all of the above (open/restricted/unlisted-table/multi-source-via-`_source_id`-tag).

**8. Self Review** — SOLID (single responsibility, two pure functions, no side effects) / DRY (reuses `EffectivePermissions.allows()`'s deny-wins semantics via the Django-side payload rather than reimplementing it engine-side) / Security (fail-closed: an unlisted source/table/column is denied, not defaulted-open; a malformed header already fails closed at the Task 15 boundary) / Performance (one dict-building pass per call, O(candidates) filtering, no N+1) / Thread-safety (pure functions, no shared mutable state, `filter_sm` never mutates its input) / Backward-compat (byte-identical when `ctx.allowed_resources` is `None` — every existing caller). Regression: `manage.py check` clean; 422 passed across the full RBAC+query-scope+engine-filter suite (2 pre-existing unrelated failures, confirmed reproduced identically on the pre-change tree).

Next: Task 17 (403 response handling, incl. closing the Task 13 empty-permitted-set fallback gap; full Gate 1 authorized/unauthorized/multi-role/full-access/partial-access/no-permissions/RBAC-off/admin-bypass suite for both entry points; Principal Engineer self-review before calling Gate 1 done).

### 2026-08-08 — Gate 1 (User Story 3) Task 17: 403 handling + centralized resolution + full suite (GATE 1 COMPLETE)

**1. Existing Implementation Analysis** — Tasks 13-16 had built every enforcement layer (source narrowing, table/column payload, cross-process wiring, engine-side candidate filtering) but nothing yet REFUSED a request. Two concrete gaps: (a) `resolve_query_scope()`'s own documented contract falls back to the dev default source when RBAC narrows a user to zero sources, rather than answering 403 — deliberately deferred to "the calling view" back in Task 13; (b) `permitted_source_ids` (source-level) and `compute_data_scope` (table/column-level) each independently called `PermissionResolver().resolve(user)` — one resolver query per layer per request, violating the brief's explicit "resolve once, never rebuild in different layers" requirement.

**2. Gap Analysis** — needed: a single per-request resolution point; a view-level check that answers 403 before resolving scope or touching the engine; a response shape that leaks nothing.

**3. Implementation Plan** — centralize resolution in one new function, thread its result through both existing checks (additive optional parameter, sentinel-defaulted so no existing caller changes behaviour), then add the 403 short-circuit in both views using each view's own existing response convention (never inventing a third envelope shape).

**4. Files Modified** — `apps/access_management/services/data_scope.py` (new `resolve_effective_permissions()` + `UNRESOLVED` sentinel; `compute_data_scope` takes optional `effective=`); `apps/query/scope.py` (`_permitted_source_ids` renamed to the public `permitted_source_ids`, takes optional `effective=`, its own local `_UNRESOLVED` sentinel — see Self-Review for why NOT the shared one); `apps/query/views.py` / `apps/chat/views.py` (resolve once, 403 short-circuit, lazy imports); `apps/core/messages.py` (new generic `chat.access_denied` copy).

**5. Reason for Change** — see Gap Analysis.

**6. Implementation** — `resolve_effective_permissions(user)`: `None` for no-user/RBAC-off/staff (identical bypass semantics `permitted_source_ids` already had), else one resolver query. Both views now call it exactly once per request and pass the same value into `permitted_source_ids(user, effective)` and `compute_data_scope(user, source_ids, effective=effective)` — one DB round-trip total, reused by both narrowing layers. A `permitted_source_ids(...)` result of `set()` (not `None` — that still means "no narrowing") triggers an immediate, generic 403 in each view, BEFORE `resolve_query_scope`/any inference or engine call: `QueryView` returns its own existing `{status, error}` shape (`_STATUS_FORBIDDEN`), `ConversationQueryView` returns `apps.core.api.error(MESSAGES["chat"]["access_denied"], 403)` — each view kept its own pre-existing envelope rather than inventing a third one.

**Two self-caught bugs, found before shipping, not by a test:**
- Moving `resolve_effective_permissions`'s import to MODULE level in `apps/query/scope.py`/`apps/query/views.py`/`apps/chat/views.py` broke `test_apps_layer_refactor.py` (10 failures) — that file configures a MINIMAL Django app registry without `apps.access_management` in `INSTALLED_APPS`, and a top-level import forces Django to resolve `CatalogResource`'s model relations immediately. Reverted to the same lazy, function-body import Task 13 already used, for exactly the same reason: this module must stay importable by a caller that never touches RBAC.
- The lazy-import branch in `permitted_source_ids` initially ran `if effective is _UNRESOLVED: ... import ...` UNCONDITIONALLY, meaning even a plain `user=None` call (the default, used by most non-RBAC test callers) would trigger the `apps.access_management` import — a regression versus the original fast-path (`if user is None: return None` as the very first line, before any import). Fixed by checking `user is None` first, inside the `_UNRESOLVED` branch, before importing anything.

New `tests/test_gate1_authorization.py` (11 tests, real Postgres/sqlite DB, `InferenceClient`/`run_chat_turn` mocked — this suite is about the authorization DECISION, not the query engine): no-permissions is 403 for BOTH entry points and never reaches the mocked engine/inference call; the 403 body contains none of the actual resource/table/column names in the request nor any RBAC vocabulary (`grant`, `role`, `rbac`, `resource_path`); full access, partial access (scoped correctly), multi-role union, admin bypass, and RBAC-disabled all authorize correctly end to end over real HTTP.

**7. Tests** — 417 passed across the full RBAC + query-scope + engine-filter + authorization suite (`manage.py check` clean). Combined with Tasks 13-16: **111 new RBAC-specific tests this story** (13 + 13 + 15 + 15 + 11 + the earlier 13/15/15/15 — see each task's own entry above for the per-task count).

**8. Self Review**
- **SOLID/DRY**: one resolution function, one source-level check, one table/column check — each consumed by exactly the callers that need it, no permission logic duplicated across the two views.
- **Security**: fail-closed at every layer (empty permitted-set → 403 before any downstream call; unlisted table/column → denied, not defaulted-open; malformed cross-process header → fails to `()`, never `None`); no resource/table/column/RBAC-internal name in the 403 body (tested directly); admin bypass is the SAME `is_staff` flag already used elsewhere in this programme, not a new escalation path.
- **Performance**: exactly one `PermissionResolver().resolve()` call per request now, reused by both narrowing layers (was two).
- **Maintainability**: `_UNRESOLVED` (scope.py, local) vs `UNRESOLVED` (data_scope.py, shared) are deliberately TWO separate sentinel objects — a caller never needs to name either one, since it only ever passes a concrete `effective` value or omits the argument; documented inline so a future reader doesn't "fix" this into an accidental cross-module import.
- **Backward compatibility**: every existing caller of `resolve_query_scope`/`compute_data_scope` that doesn't pass `effective` gets byte-identical behaviour (sentinel default resolves internally, exactly as before Task 17).
- **Known, explicitly-not-silently-claimed residual scope** (flagged across Tasks 13 and 16, restated here for the closing review): (a) `resolve_query_scope()`'s own dev-default fallback still exists at the function level — now unreachable from either real view (both short-circuit to 403 first), but a FUTURE caller of `resolve_query_scope` that skips the `permitted_source_ids` check first would still get the fallback source rather than an empty scope; (b) Task 16's engine-side filtering covers the retrieval-driven candidate path, not every `sm`-consuming code path inside `run_query` (existence checks, FK-join expansion, anchor-hint salvage) — not audited line-by-line under the "never hamper existing pipeline" rule.
- **Out of scope, confirmed NOT implemented** (per the brief): no BM25/semantic/vector/graph/RAG-document filtering, no SQL/JOIN/aggregate/view/subquery/cached-SQL validation, no permission-denied UI work, no export/dashboard restrictions, no audit logging (beyond the pre-existing `logger.warning` pattern already used throughout these files), no observability/performance/caching work, no schema migration.

### 2026-08-08 — Gate 1 post-audit fix: closing the SQL-generation bypass + chat auth gap

A production-readiness audit (self-requested, Principal Engineer review format) found the Task 16/17 work, while correctly designed for what it covered, did NOT actually close the story's central claim: candidate-list filtering (`filter_retrieval_results`) narrows retrieval RANKING, but at least five independent SQL-generation paths — `select_primary_table` (raw `sm` lexical match), `anchor_hint` (unconditional raw-`sm` match, self-documented as deliberate), the single-table `allowed_columns` derivation (built from raw `all_cols`), FK/entity-neighbour expansion (a process-global, RBAC-oblivious FK graph), and `try_multitable`'s join-partner recovery — all independently propose tables/columns that bypass the filter entirely. A deeper look (prompted while implementing the fix) found TWO MORE bypasses beyond what was audited: the **FastPath** and **verified-query cache-replay** branches in `veda_core/veda/pipeline.py::run_query()` build `allowed_tables`/`allowed_columns` before retrieval is even reached, so they were never touched by `filter_retrieval_results` at all — a cached SQL answer could be replayed for a user with different permissions than whoever's query originally populated that cache entry.

**Fix — one centralized gate, not N discovery-site patches.** New `veda.rbac_filter.narrow_allowed(allowed_tables, allowed_columns, sm, ctx)`, called immediately before EVERY `veda.validation.validate_and_parameterize(...)` call site (4 total: `pipeline.py`'s Tier-1 path, and 3 in `veda_hybrid.py`'s Tier-2 envelope/shared-planner/sql_builder paths) — narrowing whatever allowlist was proposed, by WHATEVER upstream path, to what `ctx.allowed_resources` actually permits. `validate_and_parameterize` already enforces its inputs as a hard allowlist against the generated SQL's AST (rejecting "unknown table(s)/column(s)", the exact mechanism it uses against LLM hallucination) — so this reuses 100% existing, already-tested infrastructure rather than building new refusal logic, and covers FastPath/cache-replay/every-planner uniformly since they all converge on this one call before execution. This is also a better match for the brief's "centralized enforcement, avoid scattered checks" requirement than patching every individual table-discovery site would have been.

**Second fix — `apps/chat/views.py::_resolve_user`.** Removed the dev-only fallback that silently substituted the seeded dummy `admin` DB user whenever a request carried no authenticated principal. Every call site already had `if user is None: return _unauthenticated_response()` — it simply never fired. No dev convenience lost (real JWT/session/token auth has existed in this app for a while); an unauthenticated chat request now gets a real 401, matching `/api/v1/query`'s own behavior.

**Tests** — `tests/test_rbac_filter.py` +11 (identity/open/restricted/absent-table/never-mutates for `narrow_allowed`, plus 3 END-TO-END tests against the REAL `validate_and_parameterize` — proving a narrowed allowlist causes an actual SQL string referencing a forbidden column/FK-expanded table to be REJECTED, not just that the allowlist shrinks); `tests/test_gate1_authorization.py` +1 (unauthenticated chat request → 401, `run_chat_turn` never called).

Regression: `manage.py check` clean; 445 tests pass total (419 in the combined Django-configured run + 26 for `test_rbac_filter.py`, which must run standalone — importing `veda.validation` pulls in `veda.runtime`'s `from config import SLM_MODEL_NAME...`, colliding with Django's `config` package if it was already imported earlier in the same pytest session; this is the SAME pre-existing `config`-package-collision the project already documents in `test_apps_layer_refactor.py`'s own docstring, not a new issue). 2 pre-existing, unrelated failures in `test_analytics_context.py` reproduced identically on the pre-Gate-1 tree.

Residual, still-not-claimed-as-covered: the entity-resolution path (`ENTITY_RESOLUTION_V1`/`QUERY_UNDERSTANDING_ENABLED`, both default OFF) was flagged by the audit as not fully traced for a separate anchor-pinning shortcut — if either flag is ever turned on, re-verify `pipeline.py:823-828` reaches the same `allowed_tables`/`allowed_columns` variables this fix narrows, before relying on this closing that path too.

### 2026-08-08 — Gate 1: closing the audit's remaining Medium findings (NoSQL, /v1/retrieve)

Round-2 audit found two more gaps, both now fixed.

**`/v1/retrieve`** (`inference/routes/retrieve.py`) — a debug/tooling endpoint calling `get_engine().retrieve(...)` directly and returning raw results with zero RBAC filtering (confirmed via full-repo grep to have no current caller from either real entry point — dormant, not actively exploitable, but live and unguarded). Fixed by applying `filter_retrieval_results` (Task 16, already built and tested) to its results, reading the ambient context the same way `veda_hybrid.py::_current_ctx` does. New `tests/test_inference_retrieve_rbac.py` (2 tests, real ASGI middleware + route via `starlette.testclient.TestClient`, `get_engine`/`_load_scoped_sm` stubbed): no-header keeps everything, a restrictive `X-Veda-Data-Scope` header narrows the returned columns.

**NoSQL sources had no table/column-level enforcement at all** — `veda_hybrid.py::_run_nosql()` handed the connector's raw `get_nosql_schema()` result straight to the LLM query-builder, with only Task 13's source-level filter applying before this function is ever reached. Verified first (correcting an assumption from the Round-2 audit write-up) that the Django-side catalog does NOT need extending for this: `CatalogDiscoveryService`'s own `_expected_resources` is already dialect-agnostic (walks `SchemaTable`/`SchemaColumn` for any source, regardless of kind), and `ingestion/source_dispatcher.py`'s own module docstring confirms "relational, datalake, nosql all flow through this [shared schema] pipeline" — so `compute_data_scope`'s payload is already correct for NoSQL sources today; the gap was purely the engine never consulting it.

Fixed with a new `filter_nosql_collections(collections, source_id, ctx)` in `rbac_filter.py` — the NoSQL mirror of `filter_sm`, operating on `connectors.base.NoSQLCollection` objects (`collection_name` ~= table, each `inferred_fields` dict's `"name"` ~= column) instead of the `sm['tables']`/`['columns']` dict shape, since NoSQL schema is connector-native, not semantic-model-shaped. Wired into `_run_nosql` immediately after `conn.get_nosql_schema()`, before `run_nosql_builder` ever sees the schema. An empty post-filter collection list is already handled gracefully by `run_nosql_builder` (returns a structured `error`, doesn't crash) — verified by reading its own no-collections branch, no new error handling needed.

New tests: `tests/test_rbac_filter.py` +7 (`filter_nosql_collections`: no-ctx identity, open-source passthrough, restricted-source drops an unlisted collection, partial-collection field narrowing, zero-reachable-fields drops the whole collection, a second unmentioned source_id denied independently, never mutates a kept collection in place — uses `dataclasses.replace` for a narrowed copy).

Regression: `manage.py check` clean; all touched modules (`veda_hybrid.py`, `veda/pipeline.py`, `inference/routes/retrieve.py`) import cleanly; 419 passed in the combined Django-configured run (2 pre-existing unrelated failures, unchanged) + 33 passed for `test_rbac_filter.py` standalone + 2 passed for `test_inference_retrieve_rbac.py` standalone = 454 total.

Remaining from the Round-2 audit, NOT addressed (Low priority, explicitly still open): `compute_data_scope`'s per-source query batching; `filter_sm` still has no production call site (by design — `narrow_allowed` is the real relational-path mechanism and doesn't need it); the `QUERY_UNDERSTANDING_ENABLED` path (confirmed still default `False`, not traced).

### 2026-08-08 — CRITICAL finding via live/manual Postman testing: VEDA_RBAC_MODE was never wired to a real deployment

Live end-to-end testing (real docker-compose stack: postgres, pgbouncer, redis, inference, api, nginx — not unit tests) against `/api/v1/conversations/query` (the real production query API, per the user's own explicit call — `/api/v1/query` is secondary) surfaced a genuine production-blocking bug that **three rounds of code-review audits and 88+ unit/integration tests never caught**:

- Set `VEDA_RBAC_MODE=enforce` in `.env`, rebuilt/restarted the `api` container.
- Created a real zero-permission test user (`qa_noaccess`, an empty role, no grants) via the live admin API.
- Hit `/api/v1/conversations/query` as that user — got a **full, real answer (877 rows)**, not the expected 403.
- Root cause: `config/settings/base.py` never had a line copying `VEDA_RBAC_MODE` from the environment onto a Django setting. `apps.access_management.gate.rbac_mode()`'s own fallback (`getattr(settings, "VEDA_RBAC_MODE", MODE_OFF)`) triggers on an **absent setting attribute**, not an absent env var — so the setting literally did not exist on `django.conf.settings` no matter what the environment said. Confirmed directly: `docker exec api python manage.py shell -c "from django.conf import settings; print(getattr(settings, 'VEDA_RBAC_MODE', 'NOT SET'))"` → `NOT SET`, while `os.environ['VEDA_RBAC_MODE']` inside the same container was correctly `'enforce'`.

**Why this survived three audit rounds and the whole test suite**: every single test in this entire RBAC programme sets the mode via Django's `override_settings(VEDA_RBAC_MODE=...)`, which writes the attribute directly onto the settings object — completely bypassing the env-var-to-setting wiring this bug lives in. Code review reads `rbac_mode()`'s implementation and confirms the fallback logic is correct in isolation; it has no reason to go check whether anything upstream actually populates the attribute it falls back from. Only starting the real container with a real environment variable exposed it — this is the concrete argument for why live/manual testing is not optional even after heavy automated coverage.

**Fix**: added the missing line to `config/settings/base.py`, directly beside the sibling `VEDA_JWT_AUTH` flag which WAS already correctly wired (confirming the wiring pattern itself was known and just not applied to this flag):
```python
VEDA_RBAC_MODE = os.environ.get("VEDA_RBAC_MODE", "off")
```

**New regression test**: `tests/test_settings_env_wiring.py` (3 tests) — deliberately does NOT use `override_settings` (that would hide the exact bug). Spawns a fresh subprocess with a controlled environment, runs `django.setup()`, and asserts `settings.VEDA_RBAC_MODE` actually reflects the env var — the only way to catch "the env var never reaches the setting" rather than "the setting, once set, behaves correctly."

**Re-verified live after the fix** (same running containers, same test user, no other change):
- `qa_noaccess` (zero grants) → `POST /api/v1/conversations/query` → **403**, body `{"status_code":403,"message":"You do not have permission to access this resource."}` — no resource/table/column name, no RBAC vocabulary.
- `veda` (staff, admin bypass) → same request → **200**, real answer (877 rows), unaffected.

Regression: `manage.py check` clean; 501 passed (2 pre-existing unrelated failures, unchanged) + the new file's 3.

**Environment setup performed for this testing session** (durable, left in place): `.env` gained `VEDA_JWT_AUTH=1` and `VEDA_RBAC_MODE=enforce`; the `api` image was rebuilt (`docker compose build api`) because the running image predated `djangorestframework-simplejwt` being added to `requirements/api.txt` this session; `nginx` was restarted once to re-resolve the recreated `api` container's IP. The `veda` staff account's password was reset to a known test value for login testing (`manage.py shell` — `set_password`, not `changepassword`, since the container's TTY doesn't support interactive prompts).

### 2026-08-09 — Additive GET support on 14 read-only endpoints (not RBAC, but requested during this testing session)

User asked why every endpoint is POST-only and, after discussing the real tradeoff (GET is cacheable/bookmarkable/safely-retryable; POST is neither — matters for `nginx`, which already fronts this stack), asked for GET to be added wherever it's actually safe (read-only, no side effects), leaving every mutating endpoint POST-only.

**Change**: `AdminView.validate()` (`apps/access_management/views/base.py`) now reads `request.query_params` on GET and `request.data` on POST — one shared line, so every subclass gets it for free. Added `get = post` to the 12 read-only `AdminView` subclasses (`users/list`, `users/detail`, `roles/list`, `roles/detail`, `roles/dropdown`, `permissions/list`, `permissions/detail`, `catalog/list`, `catalog/detail`, `users/roles/list`, `roles/permissions/list`, `users/permissions/effective`) and the 2 read-only `apps/chat` views (`conversations/list`, `conversations/history` — the latter also needed the same query-params-on-GET line inline, since it doesn't go through `AdminView.validate()`). Every mutating endpoint (create/update/delete/assign/revoke/grant/login/query) is untouched — still POST-only, correctly, since GET must never have a side effect.

**Purely additive**: POST continues to work exactly as before (verified: `POST roles/list` still returns 200 with the same shape). `get = post` is a plain method alias — zero new logic, zero risk of the two verbs behaving differently for the same request shape.

**Existing tests updated, not weakened**: 6 test files had a `test_endpoints_are_post_only`-style test asserting `GET -> 405` for a shared list of URLs that mixed read and write endpoints. Split each into two tests — mutating endpoints still assert `GET -> 405` (unchanged, correctly), read-only endpoints now assert `GET != 405` (and still `PUT/PATCH/DELETE -> 405` — this isn't "any verb goes," only GET as a safe alternative to the same read). Files touched: `test_user_management.py`, `test_role_management.py`, `test_catalog_resources.py`, `test_grants.py`, `test_permission_management.py`, `test_permission_resolver.py`.

**Verified twice** — unit tests (475 passed across the 6 touched files + the rest of the RBAC suite, 618 total across the full targeted regression) AND live against the running docker stack (gunicorn's `DEV_AUTORELOAD` picked up every file change without a rebuild): `GET /api/v1/roles/list`, `GET /api/v1/catalog/list`, `GET /api/v1/conversations/list` all returned 200 with real data; `POST` on the same endpoints unchanged; `GET /api/v1/users/create` (a mutating endpoint) correctly still 405.

Regression: `manage.py check` clean; 618 passed (2 pre-existing unrelated failures, unchanged).

### 2026-08-09 (correction to the entry above) — GET-only, not GET+POST

User corrected the approach: rather than accepting BOTH GET and POST on the 14 read-only endpoints (the additive design logged above), remove POST entirely — GET-only. Rationale: if POST stays available for a read, nothing forces clients (or future code) to actually use the cacheable/safe verb, so the benefit discussed earlier is opt-in and easy to silently not get. GET-only makes it the only option.

**Change**: every `get = post` alias replaced with a real `def get(self, request):` (the `post` method itself renamed, not aliased) across all 12 `AdminView` subclasses and the 2 `apps/chat` views. `AdminView.validate()` is unchanged (still branches on `request.method`) since mutating subclasses still implement `post` and share the same helper.

**This IS a breaking API contract change** (unlike the additive GET-support work) — any existing caller that POSTs to one of these 14 endpoints now gets 405. Confirmed no current caller anywhere in this codebase does that except the test suite itself (fixed below) and the Postman collection (which never referenced these 14 endpoints via POST in the first place — it only exercises mutating endpoints, so it needed no changes beyond adding a new demonstration folder).

**Test fixes — two categories**:
1. The 6 files' `test_endpoints_are_get_only`-style tests (renamed again from the additive version) now assert `POST -> 405` in addition to `GET -> non-405`.
2. **Much larger surface**: every test exercising the ACTUAL functional behavior of these 14 endpoints (list filters, pagination, detail projections, dropdown contents, effective-permission decisions) previously called them via `client.post(url, body, content_type="application/json")` through shared per-file helpers (`_list`, `_detail`, `_dropdown` in most files) — changed each helper to `client.get(url, body)` (Django's test client sends a dict as query params on `.get`), which fixed the overwhelming majority in one line each. `test_permission_resolver.py` had no shared helper (every test called `.post(URL, ...)` inline) — fixed 6 call sites individually. `test_grants.py`'s shared `_post` helper is ALSO used by genuinely mutating endpoints in the same file, so a new `_get` helper was added instead of changing `_post`, and only the 4 call sites against the two read-only URLs were switched. `test_gate.py` had 16 call sites using `ROLES_LIST`/`USERS_LIST`/`PERMS_LIST` as stand-ins to test Gate 2 behavior generically (shadow/enforce/off, staff bypass) — converted via a targeted regex substitution (15 of 16; the 16th used `Client().post(...)` — a call expression, not an identifier, so the regex correctly skipped it and it was fixed by hand) plus one parametrized test using a generic `url` fixture variable.

Regression: `manage.py check` clean; 618 passed across the full targeted regression (2 pre-existing unrelated failures, unchanged) + `test_gate.py` (23) run separately to confirm the regex-based fix. **Live re-verified** on the running docker stack (gunicorn autoreload, no rebuild needed): `POST /api/v1/roles/list` and `POST /api/v1/conversations/list` both now correctly return `{"detail":"Method \"POST\" not allowed."}` / 405; `GET` on the same endpoints unchanged (200, real data); `GET /api/v1/users/permissions/effective?user_id=14` also live-confirmed (200, real decision payload).

`VEDA_Postman_Collection.json` gained a "03b - Read-only endpoints" folder demonstrating the new GET-only calling convention, including a request that deliberately POSTs to `roles/list` and asserts 405 as a live regression check.

## 2026-09-03 — Cross-source / federated RBAC data-scope gap CLOSED
**Gap (audit):** source-SELECTION RBAC is enforced (apps/query/scope.py::resolve_query_scope intersects requested/ready with permitted_source_ids, raises SourceAccessDenied on an unpermitted pin; views.py 403s when permitted is empty). But the DATA-scope RBAC (narrow_allowed / restricted_names over RequestContext.allowed_resources) is applied ONLY in the single-source head (veda/pipeline.py) — the CROSS-SOURCE path (query/cross_source_composer.py) runs the composed SQL DIRECTLY through FederatedExecutor, bypassing that gate. So a restricted user's cross-source query could read a table/column their single-source query would be denied.
**Fix:** cross_source_composer.py `_federated_rbac_block(sql)` — before compose_federated/compose_federated_plan execute, extract the SQL's table/column names (sqlglot AST ∪ identifier token scan, fail-safe) and refuse (`status:"refused_rbac"`) if any is in restricted_names(sm, ctx). Flag FEDERATED_RBAC_ENFORCE_ENABLED default-ON: it is a NO-OP when the request carries no data-scope (unrestricted user → byte-identical) and only ever DENIES a restricted user reaching restricted data cross-source. Fail-closed if a scope exists but can't be evaluated.
**Verified:** 6/6 tests (tests/test_federated_rbac.py) — blocks restricted table + restricted column + plan-json; allows clean SQL; no-op when no scope; flag-off disables. No-scope path confirmed byte-identical.
**Note:** single-source-via-coordinator dispatch still goes through pipeline (RBAC intact); only the federated composed-SQL path was the hole. Column/row narrowing WITHIN an allowed source on the federated path is coarser than the single-source head (refuse-on-restricted vs narrow-and-continue) — acceptable for a security gate; a finer narrowing is a follow-up.

## 2026-09-03 — Clean refusal when accessible sources can't answer (UX, no source-existence disclosure)
**Issue (user):** a DB-only user asking a document question (whose source they can't access) got a bare "error" — bad UX, and a theoretical silent-wrong risk if a mis-routed answer slipped through. Correct flow (user's): check ONLY accessible sources' chunks/evidence → route; when no accessible source confidently matches → clean refuse. NOT: probe inaccessible sources (that reveals a restricted source exists).
**Fix:** veda_hybrid._clean_refuse_on_empty_error() — post-processes run_hybrid_query's result: a terminal SubResult with status=="error" that produced NEITHER an answer NOR SQL is rewritten to status="refused" + message "I couldn't find any data relevant to this question in the sources available to you." Evidence-of-failure only (no answer + no SQL); never touches/probes an inaccessible source → zero existence disclosure. Flag WEAK_EVIDENCE_CLEAN_REFUSE_ENABLED default-ON; a no-op on any item that has an answer or SQL (real results untouched).
**Verified [RUNTIME]:** DB-only user + doc Q (grievance / society-charges-fee) → refused + clean message (was bare error); normal homzhub Q (total rent) → ok, answer intact; user WITH filesystem access + same doc Q → ok, answered from Handbook. Inaccessible source never touched (chunks=0 confirmed earlier).
**Note:** empirically the mis-route-to-homzhub NEVER produced a wrong answer across 7 doc questions (validation gates refuse), so this is primarily a UX fix (bare error → clean message) that also hardens the residual silent-wrong risk.
