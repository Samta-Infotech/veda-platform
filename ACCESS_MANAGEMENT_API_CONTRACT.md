# Access Management API Contract — Frontend Integration Reference

Endpoints served by `apps/access_management`: identity **administration** and
authorization (RBAC). Sibling document to `AUTH_API_CONTRACT.md`, which owns identity
**verification** (login/refresh/logout) — obtain your token there, use it here.

**Living document.** §12 lists the RBAC endpoints that do not exist yet. Each one
moves out of §12 into a real section in the same change that implements it.

| | |
|---|---|
| **Base** | `/api/v1/` |
| **Implemented now** | **users**: `create` · `detail` · `list` · `update`<br>**roles**: `create` · `detail` · `list` · `update`<br>**permissions**: `list` · `detail` *(read-only)*<br>**catalog**: `list` · `detail` *(read-only)*<br>**grants**: `users/roles/{assign,revoke,list}` · `roles/permissions/{grant,revoke,list}` |
| **Not built yet** | everything in §12 — the resolver and the gates, permission grants, the resolver |
| **Status** | code + local-test verified (410 tests). Not yet run against the live stack or by a frontend. |

---

> ## ⚠️ Nothing here is enforced yet
>
> The full RBAC graph can be built through these endpoints — users hold roles, roles
> grant permissions on resources. **No request in VEDA is authorized differently as a
> result.** There is no resolver (nothing traverses the graph) and no gate (nothing
> acts on it); `/api/v1/query` is as open as it was before any of this existed.
>
> Build admin screens against these endpoints. Do **not** build a user-facing
> experience that implies a permission is being honoured — it is not, yet.


---

## 0. Conventions

**Every endpoint is `POST <resource>/<action>`**, and every parameter travels in the
JSON body — never the query string or the path. This is the convention `apps/chat`
already uses (`conversations/create`, `conversations/list`, `conversations/history`),
kept deliberately rather than switching to REST verbs: one style across the platform
beats a textbook style in one app. Any other method returns `405`.

Identical envelope to the rest of the platform (one implementation:
`apps/core/api.py`):

```ts
{ status_code: number; message: string; data?: object }                    // success
{ status_code: 400; message: "Invalid request data."; errors: {...} }      // bad body
{ status_code: number; message: string; code: ErrorCode }                  // failure
```

Read the HTTP status; the body's `status_code` mirrors it for convenience. **Branch on
`code`, never on `message`.**

### The user object

One representation, returned identically by every endpoint below:

```ts
interface User {
  user_id: number;
  username: string;
  email: string;
  display_name: string;      // first_name if set, else username
  is_active: boolean;
  is_staff: boolean;         // reported, never accepted as input
  date_joined: string;       // "YYYY-MM-DDTHH:MM:SSZ"
  last_login: string | null; // null until the first login
}
```

Exactly these eight keys, from create, detail, list and update alike. It is an explicit projection, so the password hash and
internal columns cannot appear — including after a future migration adds one.

---

## 1. `POST /api/v1/users/create`

Creates one active, unprivileged user.

**Requires an authenticated staff account** (`is_staff`). Send the access token from
`AUTH_API_CONTRACT.md`:

```
Authorization: Bearer <access_token>
```

> Session authentication also works (a browser logged into `/admin/`), which is how
> this endpoint is reachable today while `VEDA_JWT_AUTH` is still default-off.

### Request

```json
{
  "username": "jdoe",
  "email": "j.doe@example.com",
  "password": "…",
  "first_name": "Jane",
  "last_name": "Doe"
}
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `username` | string | yes | ≤150 chars; letters, digits and `@ . + - _` only (Django's `UnicodeUsernameValidator` — the same rule the database column uses). Unique. |
| `email` | string | yes | Valid address, ≤254 chars. Unique, **case-insensitively**. |
| `password` | string | yes | Must pass the project's four configured validators — see §9. Whitespace is **not** trimmed. |
| `first_name` | string | no | ≤150 chars, defaults to `""` |
| `last_name` | string | no | ≤150 chars, defaults to `""` |

**Only these five fields are accepted, and sending any privileged field is an
error — not silently ignored.** `is_staff`, `is_superuser`, `is_active`, `groups`,
`user_permissions`, `last_login`, `date_joined`, `id`, `pk` all return 400. A caller
must never believe it created an administrator when it did not. Every user created
here is active and unprivileged; granting anything more is role assignment (§8).

Numbers and booleans are rejected where a string is expected — `{"username": 12345}`
is a 400, not a user named `"12345"`.

### Success — `201`

```json
{
  "status_code": 201,
  "message": "User created successfully.",
  "data": {
    "user_id": 7,
    "username": "jdoe",
    "email": "j.doe@example.com",
    "display_name": "Jane",
    "is_active": true,
    "is_staff": false,
    "date_joined": "2026-08-05T11:04:22Z",
    "last_login": null
  }
}
```

`data` is the shared `User` object from §0.

### Failures

| HTTP | `code` | When |
|---|---|---|
| 400 | — (`errors` instead) | missing/blank field, bad email, bad username charset, over-long value, wrong type, **privileged field present**, or password policy failure |
| 401 | — | no credentials, or invalid/expired token |
| 403 | — | authenticated but **not staff** |
| 409 | `USERNAME_TAKEN` | that username exists |
| 409 | `EMAIL_TAKEN` | that email exists (compared case-insensitively) |

409 rather than 400 for conflicts, so a client can tell "your request was malformed"
from "your request was fine, but the name is gone" — the second is worth a retry with
a different value, the first is not.

Field errors in a 400 describe **your submission only**. They never report on stored
state, which is why a duplicate is a 409 with a `code` rather than a field error.

### Client notes

- **Email uniqueness is case-insensitive and enforced by a database index**, so it
  holds under concurrent requests. `j.doe@example.com` and `J.Doe@EXAMPLE.com` are
  the same address → 409.
- **The stored address is normalized, so it may differ from what you sent.** Django's
  `UserManager.create_user` lowercases the **domain** part and leaves the local part
  alone: `Priya@Example.com` is stored — and returned in `data.email` — as
  `Priya@example.com`. Display the value from the response rather than echoing your
  own input, or the two will disagree.
- Users with a **blank** email are permitted (the uniqueness rule applies only to
  non-blank addresses) — but this endpoint requires one, so only accounts created by
  other means can have a blank email.
- Creation is atomic. A 4xx/5xx means no user row exists; there is no partial state.

---

## 2. `POST /api/v1/users/detail`

Staff-only. One user by id.

### Request

```json
{ "user_id": 7 }
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `user_id` | int | yes | ≥ 1. In the body, not the path |

### Success — `200`

```json
{ "status_code": 200, "message": "User retrieved successfully.", "data": { /* User, §0 */ } }
```

Same `User` object as list and create, so an admin UI can open a row without
reconciling two shapes. No extra fields: roles, permissions and sessions do not exist
yet, and placeholders for them would be a contract we would have to break.

### Failures

| HTTP | `code` | When |
|---|---|---|
| 400 | — (`errors`) | missing `user_id`, or not a positive integer |
| 401 / 403 | — | not authenticated / not staff |
| 404 | `USER_NOT_FOUND` | no user with that id |

A nonsensical id (`0`, `-1`, `"abc"`) is a **400**, not a 404 — a client bug and a
genuinely absent user stay diagnosable apart.

---

## 3. `POST /api/v1/users/list`

Staff-only. Always paginated.

### Request — every field optional

```json
{ "page": 1, "page_size": 25, "search": "pri", "is_active": true, "ordering": "username" }
```

| Field | Type | Default | Rules |
|---|---|---|---|
| `page` | int | `1` | ≥ 1. A page past the end is an empty list, not an error |
| `page_size` | int | `25` | 1–**100**. Above 100 is a 400 — one caller cannot pull the whole table |
| `search` | string | `""` | Case-insensitive substring, matched against username **or** email |
| `is_active` | bool \| null | `null` | Tri-state: **omit for "all"**. `false` means "only inactive" |
| `ordering` | string | `"username"` | One of `id`, `username`, `email`, `date_joined`, `last_login`, optionally `-` prefixed. Anything else is a 400 |

### Success — `200`

```json
{
  "status_code": 200,
  "message": "Users retrieved successfully.",
  "data": {
    "users": [ /* User objects, §0 */ ],
    "pagination": {
      "page": 1, "page_size": 25, "total": 137,
      "total_pages": 6, "has_next": true, "has_previous": false
    }
  }
}
```

### Client notes

- **Paging is stable.** Results are ordered by your key *and then by `id`*, so paging
  through the whole list visits every user exactly once — without that tiebreak, rows
  sharing a sort value can repeat on the next page or be skipped.
- Drive the pager from `pagination`, not from `users.length`.
- `total` is a real COUNT over the filtered set, so it reflects `search`/`is_active`.
- Failures: `400` (invalid paging/ordering), `401` (no credentials), `403` (not staff).

---

## 4. `POST /api/v1/users/update`

Staff-only. Partial update of profile fields.

### Request

```json
{ "user_id": 7, "email": "janet@example.com", "first_name": "Janet", "last_name": "Doherty" }
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `user_id` | int | **yes** | Target user. In the body, not the path |
| `email` | string | no | Valid address, unique case-insensitively |
| `first_name` | string | no | ≤150 chars, may be blank |
| `last_name` | string | no | ≤150 chars, may be blank |

**Only the fields you send are changed** — omitting `last_name` leaves it as it was,
it does not blank it. A body carrying only `user_id` is a **400**, not a silent
success: a client that sent no changes almost certainly meant to send some.

**These are rejected with a 400, each because it belongs elsewhere:**

| Field | Why not here |
|---|---|
| `username` | Renaming an identity is its own operation (audit, cache invalidation), not a profile edit |
| `password` | Password lifecycle lives in `apps/authentication` — and a change there revokes existing tokens (`AUTH_API_CONTRACT.md` §3.1). A second way to set a credential would be a second thing to get wrong |
| `is_active` | Deactivation/reactivation is its own endpoint (§12) |
| `is_staff`, `is_superuser`, `groups`, `user_permissions` | Privilege granting is role assignment (§8) |

Any unrecognised key is also a 400 rather than being ignored, so a client cannot
believe a change took effect when it did not.

### Success — `200`

```json
{ "status_code": 200, "message": "User updated successfully.", "data": { /* User, §0 */ } }
```

### Failures

| HTTP | `code` | When |
|---|---|---|
| 400 | — (`errors`) | no updatable field, unknown/forbidden field, invalid email, over-long value |
| 401 / 403 | — | not authenticated / not staff |
| 404 | `USER_NOT_FOUND` | no user with that `user_id` |
| 409 | `EMAIL_TAKEN` | that email belongs to a **different** user |

Re-submitting the user's **own** current email is fine — a form that posts every field
back is not treated as conflicting with itself.

---

## 5. Roles

Staff-only, same envelope and same conventions as the user endpoints. A role is a
named bundle of authority; **it grants nothing yet** — permissions attach to roles,
and users to roles, in later phases (§10).

### The role object

Returned identically by all four role endpoints:

```ts
interface Role {
  role_id: number;
  name: string;
  description: string;
  is_active: boolean;
  created_at: string;   // "YYYY-MM-DDTHH:MM:SSZ"
  updated_at: string;
}
```

### 5.1 `POST /api/v1/roles/create`

```json
{ "name": "Data Analyst", "description": "Reads dashboards." }
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `name` | string | yes | ≤150 chars, non-blank after trimming. Unique **case-insensitively**. Leading/trailing whitespace is stripped before storing |
| `description` | string | no | Free text, defaults to `""` |

Roles are always created **active**. `is_active` is not accepted here — "create a role
that is already retired" is not a thing an administrator means; retiring is an update.
Any other key (including `id`, `created_at`, `updated_at`) is a **400**, not silently
ignored.

**201** → `{ "status_code": 201, "message": "Role created successfully.", "data": Role }`

| HTTP | `code` | When |
|---|---|---|
| 400 | — (`errors`) | missing/blank name, over-long name, unknown or read-only field |
| 401 / 403 | — | not authenticated / not staff |
| 409 | `ROLE_NAME_TAKEN` | a role with that name exists (compared case-insensitively) |

### 5.2 `POST /api/v1/roles/detail`

```json
{ "role_id": 3 }
```

**200** → `{ ..., "data": Role }` · **404** `ROLE_NOT_FOUND` · **400** for a
non-positive or non-integer `role_id` (a client bug and a genuinely absent role stay
diagnosable apart).

### 5.3 `POST /api/v1/roles/list`

Same paging contract as `users/list` — identical field names, identical `pagination`
block, identical caps:

```json
{ "page": 1, "page_size": 25, "search": "analyst", "is_active": true, "ordering": "name" }
```

| Field | Default | Notes |
|---|---|---|
| `page` / `page_size` | `1` / `25` | `page_size` capped at **100** |
| `search` | `""` | Case-insensitive substring over name **or description** |
| `is_active` | *(omitted)* | Tri-state: **omit for "all"**; `false` lists retired roles |
| `ordering` | `"name"` | One of `id`, `name`, `created_at`, `updated_at`, optionally `-` prefixed |

**200** → `{ ..., "data": { "roles": Role[], "pagination": Pagination } }`

Paging is stable (secondary sort on `role_id`), so paging through visits every role
exactly once.

### 5.4 `POST /api/v1/roles/update`

```json
{ "role_id": 3, "name": "Senior Analyst", "description": "...", "is_active": false }
```

Partial — only the fields you send change. A body carrying only `role_id` is a
**400**. Unknown or read-only keys are a 400.

**200** → `{ ..., "data": Role }` · **404** `ROLE_NOT_FOUND` · **409**
`ROLE_NAME_TAKEN` when the new name belongs to a *different* role. Re-submitting the
role's own current name is fine.

### 5.5 There is no `roles/delete`

**`update {is_active: false}` is how a role is retired.** Deliberate:

- Hard-deletion semantics depend entirely on role *assignment* — what happens to the
  users holding the role? — and assignment does not exist yet. Deciding it now would
  be a guess.
- An audit trail that records "granted role #7" must still be able to resolve #7.
- A retired role **keeps its name reserved**, so a new role cannot silently shadow it
  in history.

Retired roles are excluded from `list` only when you pass `is_active: true`.

---

## 6. Permissions — read-only catalogue

A permission names an **action** the platform can authorize (`data.read`,
`ingestion.run`). It is the *verb*; the *noun* — which source, which table — is bound
when grants arrive (§9). Nothing is enforced by them yet.

### 6.1 There is no create, update or delete — by design

The catalogue is **code-defined** and seeded by migration. Only code can enforce a
permission: a row an administrator invents at runtime is one no gate will ever check,
so the UI would offer authority that does not exist. Adding a permission is a code
change — a seed entry plus the gate that checks it — and that friction is deliberate.

**Roles are the layer you compose freely** (§5). Permissions are the fixed vocabulary
roles are built from.

### 6.2 The permission object

```ts
interface Permission {
  permission_id: number;
  code: string;          // stable machine key, e.g. "data.read"
  name: string;          // human label
  description: string;   // what granting it actually allows
  is_active: boolean;    // a capability switched off platform-wide
  created_at: string;
  updated_at: string;
}
```

**Branch on `code`, never on `permission_id`** — ids differ between environments,
codes do not.

### 6.3 The seeded catalogue

| `code` | Allows |
|---|---|
| `query.execute` | Run analytical queries and conversational turns |
| `data.read` | Read data from granted sources *(the one grants will scope per source/table)* |
| `source.manage` | Register, configure and retire data sources |
| `ingestion.run` | Trigger ingestion of a data source |
| `evaluation.run` | Trigger evaluation runs and read results |
| `user.manage` | Create, view and update users |
| `role.manage` | Create, view, update and retire roles |

Treat it as **data, not a constant**: fetch it with `permissions/list` and render what
comes back. Hard-coding this table client-side guarantees drift the first time an
entry is added.

### 6.4 `POST /api/v1/permissions/list`

```json
{ "page": 1, "page_size": 25, "search": "ingest", "is_active": true, "ordering": "code" }
```

Same paging contract as every other list endpoint. `search` matches **code or name**
— deliberately not `description`, which is prose and would make a search for a common
word return most of the catalogue. `ordering`: `id`, `code`, `name` (`-` prefixed for
descending).

**200** → `{ ..., "data": { "permissions": Permission[], "pagination": Pagination } }`

### 6.5 `POST /api/v1/permissions/detail`

```json
{ "permission_id": 4 }
```

**200** → `{ ..., "data": Permission }` · **404** `PERMISSION_NOT_FOUND` · **400** for
a non-positive or non-integer id.

Both endpoints are staff-only (**401** anonymous, **403** non-staff) and POST-only.

---

## 7. Catalog resources — read-only

The addressable inventory grants point at. Rows are produced by **discovery**
(`manage.py sync_catalog`) from VEDA's real catalog — never authored through the API.

### 7.1 Resource paths

Full specification: `docs/adr/0001-rbac-resource-path.md`.

```
<kind>:<source>[:<segment>]*

db:crm_postgres                     the whole source
db:crm_postgres:employee            one table
db:crm_postgres:employee:salary     one column
files:contracts_s3:msa_2024.pdf     one document
```

- `kind` ∈ `db` · `nosql` · `files` · `lake` — derived from the source's dialect.
- `source` is the **source name** (unique), not a dialect. There is **no schema
  segment**: a VEDA source already *is* one schema.
- Paths are lowercased and canonicalised on write. `DB:CRM` and `db:crm` are one path.
- Segment charset `[a-z0-9_.-]`. A name containing `:` is unaddressable and is
  reported by discovery rather than silently skipped.

**Prefix inheritance** works on whole segments, never string prefixes:

```
db:crm_postgres            covers  db:crm_postgres:employee:salary
db:crm_postgres            does NOT cover  db:crm_postgres_replica     ← different source
```

### 7.2 The resource object

```ts
interface CatalogResource {
  path: string;
  kind: "db" | "nosql" | "files" | "lake";
  parent_path: string;          // "" for a source-level resource
  source_id: number;
  substrate_id: string | null;  // the underlying table/column id, if any
  is_active: boolean;           // false = discovery no longer finds it upstream
  created_at: string;
  updated_at: string;
}
```

### 7.3 `POST /api/v1/catalog/list`

```json
{ "page": 1, "page_size": 25, "parent_path": "db:crm_postgres", "kind": "db",
  "source_id": 3, "search": "employee", "is_active": true, "ordering": "path" }
```

`parent_path` is **tri-state** and is the tree-navigation control:

| `parent_path` | Returns |
|---|---|
| omitted | every resource |
| `""` | the source-level roots |
| a path | exactly that node's children |

A lazy admin tree loads one level per call. `ordering`: `id`, `path`, `kind`.

**200** → `{ ..., "data": { "resources": CatalogResource[], "pagination": Pagination } }`

### 7.4 `POST /api/v1/catalog/detail`

```json
{ "path": "db:crm_postgres:employee" }
```

**200** → `{ ..., "data": CatalogResource }` · **404** `RESOURCE_NOT_FOUND`.

A path that is not even expressible returns **404**, not 400 — an unaddressable string
names nothing, and shaping it as a validation error would let a caller probe which
paths are well-formed.

### 7.5 Staleness is a real state

Discovery is not yet triggered by ingestion (`manage.py sync_catalog` is the current
surface). Between "a source finished re-ingesting" and "discovery ran", its resources
are absent from the catalog. A vanished resource is marked `is_active: false`, **never
deleted** — deleting would silently drop the grants pointing at it.

---

## 8. Grants — who holds what, and what it allows

Two edges complete the RBAC graph:

```
User ──users/roles/assign──> Role ──roles/permissions/grant──> Permission
                                            └──> resource_path
```

### 8.1 Both operations are idempotent

`assign` and `grant` describe a **desired state**, not an event. Repeating them is
success, so "make sure alice is an analyst" is safe to run twice:

| Outcome | Status |
|---|---|
| edge created | **201** |
| edge already existed | **200** |

`revoke` is always **200**, even for an edge that never existed — the desired end
state ("does not hold it") is already true. `data.removed` says which happened.

This differs from `users/create` and `roles/create`, which **409** on a duplicate:
creating implies newness, granting implies membership.

### 8.2 `POST /api/v1/users/roles/{assign,revoke}`

```json
{ "user_id": 7, "role_id": 3 }
```

**Assign** → `{ "user_id", "role_id", "granted_by", "created_at" }`

| HTTP | `code` | When |
|---|---|---|
| 404 | `USER_NOT_FOUND` / `ROLE_NOT_FOUND` | unknown target *(assign only)* |
| 409 | `ROLE_INACTIVE` | the role is retired — assigning it would confer authority that is switched off |

An **inactive user** *can* be assigned a role: pre-provisioning is legitimate, and the
assignment grants nothing until the account is active.

`revoke` deliberately does **not** 404 on unknown targets, so a revoke script does not
fail on exactly the rows it has nothing to do.

### 8.3 `POST /api/v1/users/roles/list`

```json
{ "user_id": 7, "role_id": 3, "page": 1, "page_size": 25 }
```

Both filters optional — `user_id` answers "what does this user hold", `role_id`
answers "who holds this role", neither returns everything.

### 8.4 `POST /api/v1/roles/permissions/grant`

```json
{ "role_id": 3, "permission_id": 2,
  "resource_path": "db:crm_postgres:employee", "effect": "allow" }
```

| Field | Required | Notes |
|---|---|---|
| `role_id` | yes | must be **active** |
| `permission_id` | only without a `resource_path` | must be **active**. Omit it when granting a resource and `data.read` is implied — see below |
| `resource_path` | no | canonical path, or `""` for a permission that is not resource-scoped (`user.manage`) |
| `effect` | no | `allow` (default) or `deny` |

**Granting a resource does not need a `permission_id`.** Resource access is always
`data.read` ON that path — the caller picks a resource in the catalog tree, never a
permission, and `data.read` is deliberately absent from `permissions/dropdown` for that
reason. Send the path alone and the server fills it in:

```json
{ "role_id": 3, "resource_path": "db:crm_postgres:employee", "effect": "allow" }
```

An explicit `permission_id` is never overridden — a body carrying one behaves exactly as
it always did, with or without a path. Sending **neither** still 400s: a blank-path
`data.read` grant covers no resource, so it would look real and do nothing.

`revoke` accepts the same omission, so a client can revoke exactly what it granted.

**Re-granting the same triple with the opposite effect UPDATES it (200), it does not
add a second row.** `(role, permission, resource_path)` is unique *without* `effect` —
two rows disagreeing about one triple would make the outcome depend on row order.

Response adds **`resource_exists`**: whether that path is currently a live catalog
resource. Granting on an undiscovered path is **allowed** (pre-provisioning a source
that is still ingesting), so this flag is how a typo becomes visible instead of hiding
until someone wonders why access never worked.

| HTTP | `code` | When |
|---|---|---|
| 400 | — | `resource_path` not canonical |
| 404 | `ROLE_NOT_FOUND` / `PERMISSION_NOT_FOUND` | unknown target |
| 409 | `ROLE_INACTIVE` / `PERMISSION_INACTIVE` | target switched off |

### 8.5 `POST /api/v1/roles/permissions/{revoke,list}`

`revoke` takes `{role_id, permission_id, resource_path?}` — no `effect`, because there
is only ever one decision per triple. **Removing a DENY does not create an ALLOW**:
with nothing matching, default-deny applies.

`list` filters by `role_id`, `permission_id`, `resource_path` and reports
`resource_exists` for the whole page in one query.

### 8.6 How the decision *will* be made (not built)

Recorded so client-side gating logic matches the server when it arrives:

```
1. Collect every grant whose path is a prefix-or-equal of the requested resource.
2. If ANY is deny  -> DENY.
3. Else if ANY is allow -> ALLOW.
4. Else -> DENY.                    (absence of a grant is a denial)
```

**DENY wins globally at any depth.** Consequence: "deny the whole source except one
table" is not expressible — grant only that table instead.

### 8.7 There is no approval workflow

Assign and grant take effect **immediately**. There is no pending state, no requester,
no approver, no approve/reject endpoint. `granted_by` records who acted, but nothing
gates it. Deliberate, and out of scope.

---

## 9. Password policy

Enforced by the project's `AUTH_PASSWORD_VALIDATORS` (`config/settings/base.py`) —
reused as configured, not reimplemented here:

| Validator | Rejects |
|---|---|
| `MinimumLengthValidator` | shorter than 8 characters |
| `CommonPasswordValidator` | passwords on Django's common-password list (e.g. `password`) |
| `NumericPasswordValidator` | entirely numeric |
| `UserAttributeSimilarityValidator` | too similar to the submitted username/email/name |

Failures come back as a 400 with the validators' own messages under
`errors.password` — a list, since several can fail at once. Render them verbatim;
they are written for end users. The submitted password is never logged.

Changing the policy is a settings change and needs no code change here, so **do not
mirror these rules in client-side validation** — you will drift. Validate presence
client-side and let the server rule on strength.

---

## 10. TypeScript types

```ts
type AccessErrorCode =
  | "USERNAME_TAKEN" | "EMAIL_TAKEN" | "USER_CONFLICT" | "USER_NOT_FOUND"
  | "ROLE_NAME_TAKEN" | "ROLE_NOT_FOUND"
  | "PERMISSION_NOT_FOUND" | "PERMISSION_INACTIVE"
  | "ROLE_INACTIVE" | "RESOURCE_NOT_FOUND" | "INVALID_RESOURCE_PATH";

// The shared user object — see §0.
interface User {
  user_id: number;
  username: string;
  email: string;
  display_name: string;
  is_active: boolean;
  is_staff: boolean;
  date_joined: string;
  last_login: string | null;
}

interface CreateUserRequest {
  username: string;
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
}

interface UserDetailRequest {
  user_id: number;
}

interface ListUsersRequest {
  page?: number;
  page_size?: number;                  // 1..100
  search?: string;
  is_active?: boolean | null;          // omit for "all"
  ordering?: "id" | "username" | "email" | "date_joined" | "last_login"
           | "-id" | "-username" | "-email" | "-date_joined" | "-last_login";
}

interface Pagination {
  page: number; page_size: number; total: number;
  total_pages: number; has_next: boolean; has_previous: boolean;
}

interface ListUsersSuccess {
  status_code: 200; message: string;
  data: { users: User[]; pagination: Pagination };
}

interface UpdateUserRequest {
  user_id: number;
  email?: string;
  first_name?: string;
  last_name?: string;
}

interface CreateUserSuccess {
  status_code: 201; message: string; data: User;
}
interface UpdateUserSuccess {
  status_code: 200; message: string; data: User;
}
interface UserDetailSuccess {
  status_code: 200; message: string; data: User;
}

interface Role {
  role_id: number;
  name: string;
  description: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface CreateRoleRequest  { name: string; description?: string }
interface RoleDetailRequest  { role_id: number }
interface UpdateRoleRequest  {
  role_id: number; name?: string; description?: string; is_active?: boolean;
}
interface ListRolesRequest {
  page?: number; page_size?: number; search?: string;
  is_active?: boolean | null;
  ordering?: "id" | "name" | "created_at" | "updated_at"
           | "-id" | "-name" | "-created_at" | "-updated_at";
}
interface ListRolesSuccess {
  status_code: 200; message: string;
  data: { roles: Role[]; pagination: Pagination };
}

interface Permission {
  permission_id: number;
  code: string;
  name: string;
  description: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface PermissionDetailRequest { permission_id: number }
interface ListPermissionsRequest {
  page?: number; page_size?: number; search?: string;
  is_active?: boolean | null;
  ordering?: "id" | "code" | "name" | "-id" | "-code" | "-name";
}
interface ListPermissionsSuccess {
  status_code: 200; message: string;
  data: { permissions: Permission[]; pagination: Pagination };
}

type ResourceKind = "db" | "nosql" | "files" | "lake";

interface CatalogResource {
  path: string;
  kind: ResourceKind;
  parent_path: string;
  source_id: number;
  substrate_id: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface ListCatalogRequest {
  page?: number; page_size?: number; search?: string;
  is_active?: boolean | null;
  ordering?: "id" | "path" | "kind" | "-id" | "-path" | "-kind";
  parent_path?: string | null;   // null = all, "" = roots, path = children
  kind?: ResourceKind | "";
  source_id?: number | null;
}
interface CatalogDetailRequest { path: string }

interface RoleAssignment {
  user_id: number;
  role_id: number;
  granted_by: number | null;
  created_at: string;
}

interface PermissionGrant {
  role_id: number;
  permission_id: number;
  resource_path: string;          // "" = not resource-scoped
  effect: "allow" | "deny";
  resource_exists: boolean;       // false = granted on a path the catalog lacks
  granted_by: number | null;
  created_at: string;
  updated_at: string;
}

interface AssignRoleRequest { user_id: number; role_id: number }
interface GrantPermissionRequest {
  role_id: number;
  permission_id: number;
  resource_path?: string;
  effect?: "allow" | "deny";
}
//  assign/grant -> 201 when created, 200 when it already existed
//  revoke       -> 200 always, data.removed says which
interface RevokeSuccess {
  status_code: 200; message: string; data: { removed: boolean };
}
interface CreateUserConflict {
  status_code: 409; message: string; code: AccessErrorCode;
}
interface ValidationFailure {
  status_code: 400; message: "Invalid request data.";
  errors: Record<string, string[]>;
}
```

---

## 11. Operational note — there is no bootstrap path

This endpoint requires an existing **staff** account, and creates only
non-staff accounts. It therefore cannot produce the first administrator.

The seeded `admin` account (chat migration 0002) has `is_staff = 0`, so **it cannot
call this endpoint**. The first staff user must come from `manage.py createsuperuser`
or the Django admin. A bootstrap-admin flow is deliberately out of scope for this
phase.

---

## 12. Not built yet

Nothing below exists. Do not code against these shapes — this is a scope list, not a
contract. There is still **no authorization layer**: no roles, no permissions, and no
endpoint outside this app requires a token.

| Endpoint | Purpose |
|---|---|
| `POST users/deactivate` · `users/reactivate` | account lifecycle (`is_active`) |
| `POST users/rename` | change `username` — deliberately excluded from `users/update` |
| `POST users/roles/assign` · `users/roles/revoke` | role assignment |
| — | permission resolution + caching, resource registry, authorization gates |

When role assignment lands it will join the **same transaction** as user creation, so
a user is never left half-provisioned. That is the only forward-looking property the
current implementation deliberately preserves; nothing else about the RBAC model is
assumed or pre-built.

### Definition of done for each endpoint above

1. Replace its row with a full section here (request, success, every failure `code`,
   client rules).
2. Add new codes to §0/§1 and to the `AccessErrorCode` union in §3.
3. Update `RBAC_PROGRESS_LOG.md` and `PM_LOG.md`.
4. Add a §6 revision row.

Expect **403** responses to become common once permission checks exist, and the
access token to start carrying role/permission claims (making it larger, and stale
for up to `expires_in` after a role change).

---

## 13. Revision history

| Date | Change |
|---|---|
| 2026-08-05 | Initial contract: `POST /api/v1/users`. Split out of `AUTH_API_CONTRACT.md` §7, which now points here for user management. |
| 2026-08-06 (grants) | Added §8 — `users/roles/{assign,revoke,list}` and `roles/permissions/{grant,revoke,list}`. Idempotent (201 new / 200 existing), one decision per `(role, permission, resource)` triple, no approval workflow. The RBAC graph is now complete — and still unenforced. |
| 2026-08-06 (catalog) | Added §7 — resource paths (ADR-0001) and the read-only `catalog/{list,detail}` projection, populated by `manage.py sync_catalog`. |
| 2026-08-06 (permissions) | Added §6 — the read-only permission catalogue on a new `access_management_permission` table, seeded by migration 0004. No write endpoints: only code can enforce a permission. Roles remain the admin-composable layer. |
| 2026-08-06 (roles) | Added §5 — role management (`create`/`detail`/`list`/`update`) on a new `access_management_role` table. No `roles/delete`: retirement is `update {is_active:false}`. Paging contract now shared with `users/list`, so both answer identically. |
| 2026-08-05 (detail) | Added §2 `users/detail`. `apps/access_management` restructured into `serializers/`, `services/`, `views/`, `urls/` packages (one module per domain) so roles and permissions can land beside users — no endpoint behaviour changed. |
| 2026-08-05 (list + update) | **Create moved to `POST users/create`** (was `POST users`) so all three endpoints follow the platform's `POST <resource>/<action>` convention. Added §2 `users/list` and §3 `users/update`. **The user object gained `is_staff`, `date_joined`, `last_login`** and is now defined once in §0, shared by all three endpoints. |
