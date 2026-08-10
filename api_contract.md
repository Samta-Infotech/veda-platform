# VEDA Admin and Chatbot RBAC — API Contract

**Version:** 1.0

**Status:** Proposed implementation contract

**Audience:** VEDA Admin frontend, Admin backend, and Chatbot backend teams
**Scope:** Data Sources, Roles and permissions, Users and role assignment, and runtime authorization
a
This document is the authoritative V1 contract for the VEDA RBAC flow.

```text
Admin connects a data source
  → Admin creates a role and selects permitted resources
  → Admin creates a user and assigns exactly one role
  → User logs into Chatbot
  → Backend permits or rejects protected-data access using the current role grants
```

The frontend manages forms and renders backend results. Authorization decisions, validation against connected resources, dependency enforcement, and safe error generation belong to the backend. The backend's internal database, cache, transaction, archival, and token-revocation implementation is outside this contract.

---

## 1. V1 decisions

The following decisions are fixed for V1:

1. A user has exactly one assigned role.
2. A role may exist with no permissions; empty permissions mean no data access.
3. Permissions are binary. V1 does not expose `READ`, `QUERY`, `DOWNLOAD`, or deny actions.
4. A grant applies either to one exact resource or to a resource and its descendants.
5. Resource IDs are stable, opaque backend IDs. Names and paths are display values only.
6. Resource trees load lazily and support pagination.
7. V1 currently allows one connected source per source type, while collection responses and permission payloads support multiple sources for future expansion.
8. Role creation/update and user creation/update are atomic from the API consumer's perspective.
9. User passwords created by an Admin are permanent initial passwords.
10. Users support `ACTIVE` and `INACTIVE` status.
11. V1 does not require idempotency keys, optimistic-lock versions, `If-Match`, or catalog-version tokens.
12. Submit buttons are disabled while their mutation is in progress.
13. Admin accounts are provisioned by backend/DevOps; the Admin Portal does not create other Admins.
14. Chatbot users log in with the `username` and permanent password created through User Management.

---

## 2. Common API conventions

### 2.1 Authentication

The rewritten APIs use the authentication endpoints defined below and VEDA's existing tenant-context behavior. No new tenant header is introduced.

```http
Authorization: Bearer <access-token>
Content-Type: application/json
```

Request bodies MUST NOT accept client-provided ownership or audit fields such as `tenant_id`, `created_by`, or `updated_by`.

Admin and data-access roles are separate concepts:

```text
Platform privilege:
  Admin account → may use Admin APIs

Data-access role:
  Chatbot user → exactly one role containing source/resource grants
```

V1 Admin provisioning rules:

- Initial and additional Admin accounts are provisioned by backend/DevOps.
- Public self-registration is not supported.
- `POST /v1/admin/users` creates Chatbot users only.
- User create/update requests MUST NOT accept `is_admin`, `account_type`, or equivalent privilege fields.
- Admin endpoints independently verify that the authenticated account has Admin privileges.

### 2.2 Naming and formats

- JSON fields use `snake_case`.
- IDs use descriptive names: `user_id`, `role_id`, `source_id`, and `resource_id`.
- IDs are opaque. Clients MUST NOT construct IDs from display names or paths.
- Timestamps use UTC ISO-8601, for example `2026-07-31T10:30:00Z`.
- Enums use uppercase values.
- Unknown request fields SHOULD be rejected.

### 2.3 Single-resource success

The HTTP status is authoritative. Responses do not duplicate it as `status_code`.

```json
{
  "status_code": 200,
  "message": "Human-readable success message.",
  "data": {}
}
```

### 2.4 Collection success

```json
{
  "status_code": 200,
  "message": "Human-readable success message.",
  "data": {
    "items": [],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total": 0,
      "total_pages": 0,
      "has_next": false,
      "has_previous": false
    }
  }
}
```

Page-based collection endpoints accept:

```text
page=1
page_size=20
search=<optional-search>
```

`page` defaults to `1`. `page_size` defaults to `20` and has a maximum of `100`.

### 2.5 Cursor-paginated hierarchy success

```json
{
  "status_code": 200,
  "message": "Human-readable success message.",
  "data": {
    "items": [],
    "pagination": {
      "page_size": 100,
      "next_cursor": null,
      "has_more": false
    }
  }
}
```

### 2.6 Error response

```json
{
  "status_code": 401,
  "message": "Human-readable message.",
  "code": "ERROR_CODE",
  "data": {}
}
```

- `code` is stable and machine-readable.
- `message` is safe to display unless an endpoint states otherwise.
- `details` is optional.
- Errors MUST NOT expose credentials, raw driver errors, stack traces, restricted resource names, SQL, or protected values.

Validation errors may include field errors:

```json
{
  "status_code": 400,
  "message": "Invalid request data.",
  "errors": {
    "email": [
      "Enter a valid email address."
    ]
  }
}
```

### 2.7 HTTP status usage

| HTTP | Meaning |
|---:|---|
| `200` | Successful read or update. |
| `201` | Resource created. |
| `204` | Successful deletion; response has no body. |
| `400` | Malformed request. |
| `401` | Authentication missing or invalid. |
| `403` | Authenticated caller is not allowed. |
| `404` | Requested resource does not exist. |
| `409` | Duplicate or dependency/state conflict. |
| `422` | Request fields or selected resources are invalid. |
| `503` | Connected external source is temporarily unavailable. |

### 2.8 Secret rules

Passwords, account keys, tokens, and equivalent credentials are write-only.

- They are sent only over HTTPS.
- They are never returned by an API.
- They are never included in logs, error details, or audit metadata.
- Omitted secrets remain unchanged during `PATCH`.
- Empty secret strings do not erase stored secrets; the frontend omits unchanged secrets.

### 2.9 Authentication API summary

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/auth/login` | Authenticate an Admin or Chatbot user. |
| `POST` | `/api/v1/auth/refresh` | Obtain a new access token using a refresh token. |
| `POST` | `/api/v1/auth/logout` | End the current refresh session. |
| `POST` | `/api/v1/auth/password/change` | Change password for the authenticated user. |

There is no public registration endpoint in V1.

### 2.10 Login

```http
POST /v1/auth/login
```

The Chatbot user enters the same `username` and permanent password created by the Admin:

```json
{
  "username": "alice",
  "password": "permanent-password"
}
```

Before issuing tokens, the backend verifies:

1. username/password credentials are valid;
2. the account is `ACTIVE`; and
3. a non-Admin Chatbot user has a valid assigned role.

Admin accounts do not require a data-access role to enter the Admin Portal.

```http
200 OK
```

Chatbot user response:

```json
{
  "status_code": 200,
  "message": "Logged in successfully.",
  "data": {
    "access_token": "access-token",
    "refresh_token": "refresh-token",
    "token_type": "Bearer",
    "user": {
      "user_id": 101,
      "username": "alice",
      "email": "alice@example.com",
      "is_admin": false,
      "status": "ACTIVE",
      "role": {
        "role_id": 1,
        "role_name": "Database Analyst"
      }
    }
  }
}
```

Admin response:

```json
{
  "status_code": 200,
  "message": "Logged in successfully.",
  "data": {
    "access_token": "access-token",
    "refresh_token": "refresh-token",
    "token_type": "Bearer",
    "user": {
      "user_id": 1,
      "username": "admin",
      "email": "admin@example.com",
      "is_admin": true,
      "status": "ACTIVE",
      "role": null
    }
  }
}
```

The returned role is a display/session summary. Full permission trees are not returned to or enforced by the frontend. Protected backend operations evaluate the current saved grants.

Invalid username or password returns the same generic response:

```http
401 Unauthorized
```

```json
{
  "status_code": 401,
  "message": "Invalid username or password.",
  "code": "INVALID_CREDENTIALS"
}
```

The response MUST NOT reveal whether the username exists.

Inactive account:

```http
403 Forbidden
```

```json
{
  "status_code": 403,
  "message": "Your account has been deactivated. Please contact your admin.",
  "code": "ACCOUNT_INACTIVE"
}
```

Chatbot user without an assigned role:

```http
403 Forbidden
```

```json
{
  "status_code": 403,
  "message": "No role has been assigned to your account. Please contact your admin.",
  "code": "ROLE_NOT_ASSIGNED"
}
```

### 2.11 Refresh access token

```http
POST /v1/auth/refresh
```

```json
{
  "refresh_token": "refresh-token"
}
```

```json
{
  "status_code": 200,
  "message": "Token refreshed.",
  "data": {
    "access_token": "new-access-token",
    "token_type": "Bearer"
  }
}
```

An invalid, expired, or unusable refresh token returns `401 INVALID_REFRESH_TOKEN` and the frontend clears local authentication state.

### 2.12 Logout

```http
POST /v1/auth/logout
Authorization: Bearer <access-token>
```

```json
{
  "refresh_token": "refresh-token"
}
```

```http
204 No Content
```

The frontend clears local authentication state after logout completion. A subsequent refresh using the ended session returns `401 INVALID_REFRESH_TOKEN`.

---

## 3. Resource and permission model

### 3.1 Source types and hierarchy

```text
DATABASE
  database → schema → table → column

DATALAKE
  container → folder → object/file

FILE_SYSTEM
  folder → file
```

A resource node has this common shape:

```json
{
  "resource_id": "table_monthly_revenue",
  "resource_type": "TABLE",
  "name": "monthly_revenue",
  "parent_id": "schema_reporting",
  "has_children": true
}
```

`resource_type`, `name`, and `parent_id` help render the UI. Only `resource_id` is permission identity.

### 3.2 Grant scopes

```json
{
  "resource_id": "table_monthly_revenue",
  "scope": "SELF_AND_DESCENDANTS"
}
```

| Scope | Meaning |
|---|---|
| `SELF` | Only the exact resource. |
| `SELF_AND_DESCENDANTS` | The resource and all current and future descendants. |

Examples:

```text
Table + SELF_AND_DESCENDANTS → table and all its columns
Column + SELF                → only that column
Folder + SELF_AND_DESCENDANTS → folder and all nested folders/files
File + SELF                  → only that file
```

Access to a leaf implicitly permits traversal through its ancestors to locate that resource. It does not grant sibling resources or the other contents of an ancestor.

### 3.3 Checkbox behavior and canonical payload

The checkbox UI and submitted grants are intentionally different concepts:

1. Selecting a table selects all its columns.
2. Selecting a column automatically marks its table and higher ancestors as selected/partial in the UI.
3. If only some columns are selected, the parent appears checked/indeterminate but is not submitted as a full-table grant.
4. A partial selection submits one `SELF` grant for each selected leaf.
5. When every child is selected and the frontend knows it has the complete child set, it may normalize the selection into one parent `SELF_AND_DESCENDANTS` grant. Selecting the parent directly always expresses this full-subtree intent without loading every child.
6. Unchecking one column from a fully selected table converts the full-table grant into `SELF` grants for the columns that remain selected.
7. V1 does not support “everything below this parent except this one child.”
8. Grants already covered by an ancestor `SELF_AND_DESCENDANTS` grant MUST NOT also be submitted.

Only one column selected:

```json
{
  "source_id": "source_db_01",
  "grants": [
    {
      "resource_id": "column_amount",
      "scope": "SELF"
    }
  ]
}
```

Whole table selected:

```json
{
  "source_id": "source_db_01",
  "grants": [
    {
      "resource_id": "table_monthly_revenue",
      "scope": "SELF_AND_DESCENDANTS"
    }
  ]
}
```

### 3.4 Empty permissions

```json
{
  "permissions": []
}
```

An empty permission list is valid and always means no protected-data access. It never means full access.

### 3.5 Missing resources

If a granted resource is later removed from the connected source:

- the saved grant remains visible when editing the role;
- its status becomes `UNAVAILABLE`;
- runtime access through that grant is denied; and
- it is never remapped by display name or path.

---

## 4. Endpoint summary

All endpoints in the platform use an RPC-style `<resource>/<action>` pattern and are mounted under `/api/v1/`.
All requests use the `POST` HTTP method. Parameters (including pagination and filters) travel in the JSON body rather than the URL path or query string.

### Authentication (`apps.authentication`)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/auth/login` | Authenticate an Admin or Chatbot user. |
| `POST` | `/api/v1/auth/refresh` | Refresh an access token. |
| `POST` | `/api/v1/auth/logout` | End the current refresh session. |
| `POST` | `/api/v1/auth/password/change`| Change password for the current user. |

### Catalog (`apps.access_management.urls.catalog`)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/catalog/detail` | Load catalog item details. |
| `POST` | `/api/v1/catalog/list` | List catalog items (replaces old data-sources list). |

### Roles (`apps.access_management.urls.roles`)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/roles/create` | Create a role. |
| `POST` | `/api/v1/roles/detail` | Load role details. |
| `POST` | `/api/v1/roles/list` | List roles. |
| `POST` | `/api/v1/roles/dropdown` | List roles available for user assignment. |
| `POST` | `/api/v1/roles/update` | Update a role. |
| `POST` | `/api/v1/roles/delete` | Soft delete an unassigned role. |

### Permissions & Grants (`apps.access_management.urls.*`)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/permissions/detail` | Load permission details. |
| `POST` | `/api/v1/permissions/list` | List available permissions. |
| `POST` | `/api/v1/roles/permissions/grant` | Grant a permission to a role. |
| `POST` | `/api/v1/roles/permissions/revoke`| Revoke a permission from a role. |
| `POST` | `/api/v1/roles/permissions/list`  | List permissions for a role. |
| `POST` | `/api/v1/users/roles/assign`      | Assign a role to a user. |
| `POST` | `/api/v1/users/roles/revoke`      | Revoke a role from a user. |
| `POST` | `/api/v1/users/roles/list`        | List roles assigned to a user. |
| `POST` | `/api/v1/users/permissions/effective` | Resolve effective permissions for a user. |

### Users (`apps.access_management.urls.users`)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/users/create` | Create a user. |
| `POST` | `/api/v1/users/detail` | Load user details. |
| `POST` | `/api/v1/users/list` | List users. |
| `POST` | `/api/v1/users/update` | Update user details. |
| `POST` | `/api/v1/users/delete` | Delete a user. |

### Chat (`apps.chat`)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/conversations/query` | Submit a query in a conversation. |
| `POST` | `/api/v1/conversations/create`| Create a new conversation. |
| `POST` | `/api/v1/conversations/list`  | List conversations. |
| `POST` | `/api/v1/conversations/history`| Load conversation history. |

### Query Engine (`apps.query`)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/query` | Submits a query. |
| `POST` | `/api/v1/admin/ingest` | Trigger data ingestion (staff-only). |
| `POST` | `/api/v1/admin/eval` | Trigger evaluation (staff-only). |

---

## 5. Data Sources APIs

### 5.1 V1 cardinality and future compatibility

The API models sources as a collection and all permission payloads use `source_id`. V1 may enforce one connected source for each type. This limit can be removed later without changing response or permission shapes.

If the V1 limit is reached:

```http
409 Conflict
```

```json
{
  "status_code": 409,
  "message": "Only one Database source can currently be connected.",
  "code": "SOURCE_TYPE_LIMIT_REACHED",
  "data": {
    "source_type": "DATABASE"
  }
}
```

The frontend stores sources as an array even when only one item of each type exists.

### 5.2 List sources

```http
POST /api/v1/data-sources/list
```

```json
{
  "source_type": "DATABASE",
  "status": "CONNECTED",
  "page": 1,
  "page_size": 20,
  "search": "finance"
}
```

`source_type`, `status`, and `search` are optional.

```json
{
  "status_code": 200,
  "message": "Data sources retrieved successfully.",
  "data": {
    "items": [
      {
        "source_id": "source_db_01",
        "source_type": "DATABASE",
        "name": "Finance PostgreSQL",
        "status": "CONNECTED",
        "metadata": {
          "db_type": "POSTGRESQL",
          "host": "db.example.com",
          "port": 5432,
          "database": "finance",
          "username": "veda_reader"
        },
        "last_checked_at": "2026-07-31T10:30:00Z",
        "status_message": null,
        "created_at": "2026-07-31T09:00:00Z",
        "updated_at": "2026-07-31T09:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total": 1,
      "total_pages": 1,
      "has_next": false,
      "has_previous": false
    }
  }
}
```

Supported statuses:

| Status | Meaning |
|---|---|
| `CONNECTED` | Last connection validation succeeded. |
| `ERROR` | Source is configured but is currently unavailable. |

An `ERROR` source remains listed. `status_message` contains only a safe message such as `The connection could not be established.` Existing role grants remain stored but cannot provide runtime access while the source is unavailable.

### 5.3 Connect Database

```http
POST /api/v1/data-sources/connect
```

```json
{
  "source_type": "DATABASE",
  "name": "Finance PostgreSQL",
  "config": {
    "db_type": "POSTGRESQL",
    "host": "db.example.com",
    "port": 5432,
    "database": "finance",
    "username": "veda_reader",
    "password": "write-only-password"
  }
}
```

### 5.4 Connect Datalake

The request matches the existing Admin form. `account_key` is write-only.

```json
{
  "source_type": "DATALAKE",
  "name": "Finance Data Lake",
  "config": {
    "data_lake_type": "AZURE_DATA_LAKE",
    "account_name": "financeaccount",
    "tenant_container": "finance",
    "root_path": "/",
    "account_key": "write-only-account-key"
  }
}
```

Supported V1 `data_lake_type` values:

```text
AZURE_DATA_LAKE
AWS_DATA_LAKE
GOOGLE_CLOUD_DATA_LAKE
```

### 5.5 Connect File System

```json
{
  "source_type": "FILE_SYSTEM",
  "name": "Finance Shared Files",
  "config": {
    "root_path": "/shared/finance"
  }
}
```

### 5.6 Connect success and failure

The backend validates the connection before reporting success. A failed initial validation does not create a source.

```http
201 Created
```

```json
{
  "status_code": 201,
  "message": "Source connected successfully.",
  "data": {
    "source_id": "source_db_01",
    "source_type": "DATABASE",
    "name": "Finance PostgreSQL",
    "status": "CONNECTED",
    "metadata": {
      "db_type": "POSTGRESQL",
      "host": "db.example.com",
      "port": 5432,
      "database": "finance",
      "username": "veda_reader"
    },
    "created_at": "2026-07-31T09:00:00Z"
  }
}
```

```http
422 Unprocessable Entity
```

```json
{
  "status_code": 422,
  "message": "The data source connection could not be established.",
  "code": "SOURCE_CONNECTION_FAILED"
}
```

### 5.7 Lazily browse resources

Root resources:

```http
POST /api/v1/data-sources/resources
```

```json
{
  "source_id": "source_db_01",
  "page_size": 100
}
```

Children of one resource:

```http
POST /api/v1/data-sources/resources
```

```json
{
  "source_id": "source_db_01",
  "parent_id": "schema_reporting",
  "page_size": 100,
  "cursor": "cursor_abc"
}
```

```json
{
  "status_code": 200,
  "message": "Resources retrieved successfully.",
  "data": {
    "items": [
      {
        "resource_id": "table_monthly_revenue",
        "resource_type": "TABLE",
        "name": "monthly_revenue",
        "parent_id": "schema_reporting",
        "has_children": true
      }
    ],
    "pagination": {
      "page_size": 100,
      "next_cursor": null,
      "has_more": false
    }
  }
}
```

The Data Sources page renders this hierarchy read-only. The Roles page uses the same resource endpoint with checkboxes. This endpoint never saves permissions.

If the source is unavailable:

```http
503 Service Unavailable
```

```json
{
  "status_code": 503,
  "message": "The data source is currently unavailable.",
  "code": "SOURCE_UNAVAILABLE"
}
```

### 5.8 Edit a source

```http
POST /api/v1/data-sources/update
```

`POST` to `/update` acts as a partial update. Supplied fields change and omitted fields remain unchanged.

```json
{
  "source_id": "source_db_01",
  "name": "Finance PostgreSQL Production",
  "config": {
    "host": "new-db.example.com",
    "port": 5432
  }
}
```

Safe edits that retain `source_id` include:

- display-name changes;
- credential changes;
- port/connection-option changes; and
- host migration or failover for the same logical data source.

V1 treats these connection identity fields as immutable:

| Source type | Immutable identity fields |
|---|---|
| `DATABASE` | `source_type`, `db_type`, `database` |
| `DATALAKE` | `source_type`, `data_lake_type`, `account_name`, `tenant_container`, `root_path` |
| `FILE_SYSTEM` | `source_type`, `root_path` |

Identity-changing edits are not accepted as ordinary edits:

- changing the database to a different logical database;
- changing the Datalake account/container to unrelated data;
- changing the File System root to unrelated data; or
- changing `source_type`.

For those changes, the Admin creates a new source and explicitly assigns its resources to roles.

```http
409 Conflict
```

```json
{
  "error": {
    "code": "SOURCE_IDENTITY_CHANGE_REQUIRED",
    "message": "This change points to a different data source. Create a new source and update its role permissions.",
    "details": {},
    "request_id": "req_01J..."
  }
}
```

The backend validates the changed connection before reporting success. A failed validation leaves the previous working configuration unchanged.

### 5.9 Delete a source

```http
DELETE /v1/admin/data-sources/{source_id}
```

If no role references the source:

```http
204 No Content
```

After success, the source disappears from the list, its resources cannot be browsed, and its secrets are no longer usable.

If any role references it:

```http
409 Conflict
```

```json
{
  "error": {
    "code": "SOURCE_USED_BY_ROLE",
    "message": "Remove this source from all roles before deleting it.",
    "details": {
      "roles_count": 3
    },
    "request_id": "req_01J..."
  }
}
```

The frontend shows the backend error. How the backend guarantees a safe dependency check is outside the frontend contract.

### 5.10 Data Source validation and errors

- `name` is trimmed and 2–100 characters.
- Ports are integers in the supported range.
- Source-specific required fields must be present during creation.
- Paths are normalized and validated by the backend.
- Secret fields are write-only.
- The configured source must be reachable before create/update success is returned.

| HTTP | Code | Meaning |
|---:|---|---|
| `404` | `SOURCE_NOT_FOUND` | Source does not exist. |
| `409` | `SOURCE_TYPE_LIMIT_REACHED` | V1 already has a source of this type. |
| `409` | `SOURCE_IDENTITY_CHANGE_REQUIRED` | Edit attempts to point at different data. |
| `409` | `SOURCE_USED_BY_ROLE` | Source cannot be deleted while referenced. |
| `422` | `SOURCE_CONNECTION_FAILED` | Supplied connection cannot be validated. |
| `503` | `SOURCE_UNAVAILABLE` | Existing source is temporarily unavailable. |

---

## 6. Roles APIs

All role endpoints require `IsAdminUser` + `RequiresPermission(ROLE_MANAGE)`.

### 6.1 List roles

```http
GET /api/v1/roles/list?page=1&page_size=25&search=analyst&is_active=true&ordering=name
```

All query params are optional. `search` matches name or description (case-insensitive). `is_active` is tri-state: omitted = no filter, `true` = active only, `false` = inactive only. `ordering` accepts: `id`, `name`, `created_at`, `updated_at` (prefix with `-` for descending). Defaults: `page=1`, `page_size=25` (max 100), `ordering=name`.

```json
{
  "status_code": 200,
  "message": "Roles retrieved successfully.",
  "data": {
    "roles": [
      {
        "role_id": 1,
        "name": "Database Analyst",
        "role_name": "Database Analyst",
        "description": "Access to finance databases.",
        "is_active": true,
        "created_at": "2026-07-31T09:00:00Z",
        "updated_at": "2026-07-31T10:00:00Z",
        "deleted_at": null,
        "users_count": 4,
        "connected_sources": ["Database"],
        "last_updated": "Jul 31, 2026"
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 25,
      "total": 1,
      "total_pages": 1,
      "has_next": false,
      "has_previous": false
    }
  }
}
```

Each role in the list carries enriched fields from `role_stats()`:
- `users_count`: number of users holding this role (from `UserRole`).
- `connected_sources`: list of human-readable source kind labels (e.g. `"Database"`, `"File System"`, `"Datalake"`, `"NoSQL"`) derived from the resource paths in this role's permission grants. Global grants (empty `resource_path`) do not contribute.
- `role_name`: duplicate of `name`, included for backward compatibility.
- `last_updated`: human-formatted date string (e.g. `"Jul 31, 2026"`), not ISO-8601.

### 6.2 Roles dropdown (for user assignment selector)

```http
GET /api/v1/roles/dropdown
```

No query params. Returns every active role, **unpaginated** — safe because roles are administrator-authored (tens to hundreds).

```json
{
  "status_code": 200,
  "message": "Roles retrieved successfully.",
  "data": {
    "roles": [
      {
        "role_id": 1,
        "name": "Database Analyst"
      }
    ]
  }
}
```

Retired (`is_active=false`) roles are excluded.

### 6.3 Create role

```http
POST /api/v1/roles/create
```

```json
{
  "name": "Database Analyst",
  "description": "Access to finance databases."
}
```

- `name` is required, max length from model, trimmed, must not be blank after trimming.
- `description` is optional, defaults to `""`.
- Unknown fields are rejected (400), not silently ignored.
- Server-owned fields (`id`, `created_at`, `updated_at`) are rejected with `"This field is read-only."`.

```http
201 Created
```

```json
{
  "status_code": 201,
  "message": "Role created successfully."
}
```

No `data` in the response — the created role's id is not returned. Permissions are granted separately via `POST /api/v1/roles/permissions/grant`.

**Errors:**

| HTTP | Code | Meaning |
|---:|---|---|
| `400` | — | Malformed body, blank name, unknown field, read-only field. |
| `409` | `ROLE_NAME_TAKEN` | A role with that name already exists (case-insensitive). |

### 6.4 Get role details

```http
GET /api/v1/roles/detail?role_id=1
```

```json
{
  "status_code": 200,
  "message": "Role retrieved successfully.",
  "data": {
    "role_id": 1,
    "name": "Database Analyst",
    "description": "Access to finance databases.",
    "is_active": true,
    "created_at": "2026-07-31T09:00:00Z",
    "updated_at": "2026-07-31T10:00:00Z",
    "deleted_at": null
  }
}
```

Same projection as create/list/update — the `public_fields()` shape. `deleted_at` is the timestamp when `is_active` was set to `false` (or `null` if still active).

Permissions are loaded separately via `GET /api/v1/roles/permissions/list?role_id=1`.

**Errors:**

| HTTP | Code | Meaning |
|---:|---|---|
| `400` | — | `role_id` missing, not an integer, or < 1. |
| `404` | `ROLE_NOT_FOUND` | No role with that id. |

### 6.5 Update role

```http
POST /api/v1/roles/update
```

```json
{
  "role_id": 1,
  "name": "Senior Database Analyst",
  "description": "Updated description.",
  "is_active": false
}
```

**Partial update** — only the fields present (besides `role_id`) are written. At least one updatable field must be provided or the request is rejected.

Updatable fields: `name`, `description`, `is_active`.

Setting `is_active` to `false` is how a role is **retired**. When that happens, `deleted_at` is automatically stamped. If `is_active` flips back to `true`, `deleted_at` is cleared.

The row is locked (`SELECT ... FOR UPDATE`) for the duration so concurrent writes cannot clobber each other.

```json
{
  "status_code": 200,
  "message": "Role updated successfully."
}
```

No `data` in the response.

**Errors:**

| HTTP | Code | Meaning |
|---:|---|---|
| `400` | — | No updatable fields provided, unknown fields, read-only fields. |
| `404` | `ROLE_NOT_FOUND` | No role with that id. |
| `409` | `ROLE_NAME_TAKEN` | The new name belongs to a different role. |

### 6.6 Delete role (soft)

```http
POST /api/v1/roles/delete
```

```json
{
  "role_id": 1
}
```

```json
{
  "status_code": 200,
  "message": "Role deleted successfully."
}
```

A convenience wrapper — internally calls `RoleService.update_role(role_id, is_active=False)`. No row is ever removed. The role stays queryable but is no longer grantable.

**Errors:** same as update — `ROLE_NOT_FOUND` (404).

### 6.7 Role validation summary

| HTTP | Code | Meaning |
|---:|---|---|
| `400` | — | Malformed body, blank name, unknown or read-only field. |
| `404` | `ROLE_NOT_FOUND` | Role does not exist. |
| `409` | `ROLE_NAME_TAKEN` | Name already used (case-insensitive). |

---

## 7. Users APIs

All user endpoints require `IsAdminUser` + `RequiresPermission(USER_MANAGE)`.

### 7.1 User rules

- Roles are assigned via the separate `/api/v1/users/roles/assign` endpoint, not during user creation.
- Permissions live on roles, never on user records.
- `is_active` controls login and Chatbot access. An inactive user remains manageable in Admin.
- Password changes use `POST /api/v1/auth/password/change` (authenticated endpoint in `apps.authentication`).
- The platform must always have at least one active admin — deactivating the last one is refused.

### 7.2 List users

```http
GET /api/v1/users/list?page=1&page_size=25&search=alice&is_active=true&ordering=username
```

All query params are optional. `search` matches username or email (case-insensitive). `is_active` is tri-state: omitted = no filter. `ordering` accepts: `id`, `username`, `email`, `date_joined`, `last_login` (prefix with `-` for descending). Defaults: `page=1`, `page_size=25` (max 100), `ordering=username`.

```json
{
  "status_code": 200,
  "message": "Users retrieved successfully.",
  "data": {
    "users": [
      {
        "user_id": 101,
        "username": "alice",
        "email": "alice@example.com",
        "display_name": "Alice",
        "is_active": true,
        "is_staff": false,
        "date_joined": "2026-07-31T10:00:00Z",
        "last_login": "2026-08-01T08:00:00Z",
        "roles": ["Database Analyst"],
        "created_at": "Jul 31, 2026",
        "updated_at": "Jul 31, 2026",
        "deleted_at": null
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 25,
      "total": 1,
      "total_pages": 1,
      "has_next": false,
      "has_previous": false
    }
  }
}
```

Each user in the list carries enriched fields:
- `display_name`: `first_name` if set, otherwise `username`.
- `is_staff`: whether the account has admin privileges (reported but never writable through these endpoints).
- `roles`: list of role name strings assigned to this user (from `UserRole`). May be empty.
- `created_at` / `updated_at`: human-formatted date strings (e.g. `"Jul 31, 2026"`), not ISO-8601. `updated_at` comes from `UserProfile`, may be `""` for users created before the profile migration.
- `deleted_at`: human-formatted date from `UserProfile`, or `null` if the user was never soft-deleted.

### 7.3 Create user

```http
POST /api/v1/users/create
```

```json
{
  "username": "alice",
  "email": "alice@example.com",
  "password": "permanent-initial-password",
  "first_name": "Alice",
  "last_name": "Smith"
}
```

- `username` is required, max length from model, validated with Django's `UnicodeUsernameValidator`.
- `email` is required, validated as a proper email address.
- `password` is required, write-only, validated against `AUTH_PASSWORD_VALIDATORS` (including `UserAttributeSimilarityValidator` against the submitted username/email).
- `first_name` and `last_name` are optional, default to `""`.
- Non-string values for any field (e.g. `{"username": 12345}`) are rejected.
- Privileged fields (`is_staff`, `is_superuser`, `is_active`, `groups`, `user_permissions`, `last_login`, `date_joined`, `password_hash`, `id`, `pk`) are rejected with `"This field cannot be set through this endpoint."`.

Roles are assigned separately via `POST /api/v1/users/roles/assign`.

```http
201 Created
```

```json
{
  "status_code": 201,
  "message": "User created successfully."
}
```

No `data` in the response. The password is never returned.

**Errors:**

| HTTP | Code | Meaning |
|---:|---|---|
| `400` | — | Malformed body, non-string field, privileged field, password policy violation. |
| `409` | `USERNAME_TAKEN` | A user with that username already exists. |
| `409` | `EMAIL_TAKEN` | A user with that email already exists (case-insensitive). |

### 7.4 Get user details

```http
GET /api/v1/users/detail?user_id=101
```

```json
{
  "status_code": 200,
  "message": "User retrieved successfully.",
  "data": {
    "user_id": 101,
    "username": "alice",
    "email": "alice@example.com",
    "display_name": "Alice",
    "is_active": true,
    "is_staff": false,
    "date_joined": "2026-07-31T10:00:00Z",
    "last_login": "2026-08-01T08:00:00Z"
  }
}
```

Same projection as list — the `public_fields()` shape. The response never contains a password or password hash.

Role assignments are loaded separately via `GET /api/v1/users/roles/list?user_id=101`.

**Errors:**

| HTTP | Code | Meaning |
|---:|---|---|
| `400` | — | `user_id` missing, not an integer, or < 1. |
| `404` | `USER_NOT_FOUND` | No user with that id. |

### 7.5 Update user details

```http
POST /api/v1/users/update
```

```json
{
  "user_id": 101,
  "email": "alice.new@example.com",
  "first_name": "Alice New",
  "last_name": "Smith",
  "is_active": false
}
```

**Partial update** — only the fields present (besides `user_id`) are written. At least one updatable field must be provided.

Updatable fields: `email`, `first_name`, `last_name`, `is_active`.

Deliberately excluded (and rejected if submitted):
- `username` — renaming an identity is a separate concern.
- `password` — password lifecycle lives in `apps.authentication`.
- `is_staff` / `is_superuser` — privilege granting is role assignment.

Setting `is_active` to `false`:
- Stamps `deleted_at` on the user's `UserProfile`.
- Revokes all live refresh tokens (the user cannot mint new access tokens).
- Is refused if this is the platform's **last active admin** (409 `LAST_ADMIN_PROTECTED`).

The row is locked (`SELECT ... FOR UPDATE`) for the duration.

```json
{
  "status_code": 200,
  "message": "User updated successfully."
}
```

No `data` in the response.

**Errors:**

| HTTP | Code | Meaning |
|---:|---|---|
| `400` | — | No updatable fields, unknown fields, privileged fields. |
| `404` | `USER_NOT_FOUND` | No user with that id. |
| `409` | `EMAIL_TAKEN` | The new email belongs to someone else. |
| `409` | `LAST_ADMIN_PROTECTED` | Cannot deactivate the platform's last active admin. |

### 7.6 Delete user (soft)

```http
POST /api/v1/users/delete
```

```json
{
  "user_id": 101
}
```

```json
{
  "status_code": 200,
  "message": "User deleted successfully."
}
```

A convenience wrapper — internally calls `UserService.update_user(user_id, is_active=False)`. Same last-admin guard, same token revocation. Idempotent: calling it twice just means the account stays inactive.

No row is ever removed. `deleted_at` on `UserProfile` records when this happened; `is_active` stays the one flag every access check keys off.

**Errors:** same as update — `USER_NOT_FOUND` (404), `LAST_ADMIN_PROTECTED` (409).

### 7.7 User validation summary

| HTTP | Code | Meaning |
|---:|---|---|
| `400` | — | Malformed body, non-string fields, privileged fields, password policy violation. |
| `404` | `USER_NOT_FOUND` | User does not exist. |
| `409` | `USERNAME_TAKEN` | Username already in use. |
| `409` | `EMAIL_TAKEN` | Email already in use (case-insensitive). |
| `409` | `LAST_ADMIN_PROTECTED` | Cannot deactivate the last active admin. |

---

## 8. Chatbot runtime authorization boundary

The Chatbot frontend does not calculate, cache, or override authorization.

```text
User submits a question
  → Backend authenticates the user
  → Backend evaluates current user status and assigned role
  → Backend identifies every protected resource required by the operation
  → Backend evaluates the saved grants
  → All resources allowed: perform operation
  → Any resource denied/unavailable: do not fetch restricted data; return safe error
```

### 8.1 Submit message

```http
POST /v1/chat/sessions/{session_id}/messages
```

```json
{
  "message": "What was the revenue last month?",
  "client_message_id": "client_msg_01J..."
}
```

The backend obtains user and role context from the authenticated request. The request does not accept trusted `user_id`, `role_id`, or permission data.

### 8.2 Allowed response

```http
200 OK
```

```json
{
  "data": {
    "message_id": "assistant_msg_01J...",
    "type": "ANSWER",
    "answer": "The revenue last month was $125,000."
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

### 8.3 Restricted response

If any required resource is outside the assigned grants, unavailable, or belongs to an unavailable source:

```http
403 Forbidden
```

```json
{
  "error": {
    "code": "DATA_ACCESS_DENIED",
    "message": "You don't have permission to access this data. Please contact your admin for more details.",
    "details": {},
    "request_id": "req_01J..."
  }
}
```

The error MUST NOT reveal restricted resource names, paths, schemas, SQL, metadata, or data values.

An authenticated but inactive account receives:

```http
403 Forbidden
```

```json
{
  "error": {
    "code": "ACCOUNT_INACTIVE",
    "message": "Your account has been deactivated. Please contact your admin.",
    "details": {},
    "request_id": "req_01J..."
  }
}
```

A deleted account is handled by the existing authentication flow and cannot establish or continue an authenticated application session.

### 8.4 No-access cases

The backend denies protected-data access when any of the following is true:

- user is `INACTIVE` or deleted;
- user has no usable assigned role;
- role permissions are empty;
- a required resource is not covered by a grant;
- a saved grant is `UNAVAILABLE`; or
- the required source has `ERROR` status.

If one question requires both allowed and restricted resources, V1 rejects the complete question. It does not return a partial answer.

### 8.5 Grant evaluation examples

```text
Grant: table_monthly_revenue + SELF_AND_DESCENDANTS
Request: column_amount under that table
Result: allowed
```

```text
Grant: column_amount + SELF
Request: sibling column_employee_salary
Result: denied
```

```text
Grant: folder_finance + SELF_AND_DESCENDANTS
Request: nested file folder_finance/2026/report.pdf
Result: allowed
```

---

## 9. Admin UI flows

### 9.1 Connect source

```text
Open source form
→ enter source-specific fields
→ disable Connect while saving
→ POST source
→ success: close and refresh source list
→ failure: keep form open and show safe backend message
```

### 9.2 Create role

```text
Open Create Role
→ enter role name
→ load configured sources
→ lazily load hierarchy branches as expanded
→ select parent subtrees and/or individual leaves
→ normalize selection into minimal grants
→ disable Create while saving
→ POST role once with complete permissions
→ success: close and refresh roles
```

No connected source is required; `permissions: []` remains valid.

### 9.3 Edit role

```text
GET role details
→ show saved available and unavailable grants
→ lazily load required hierarchy branches
→ preselect grants
→ Admin changes name/selections
→ normalize selection into minimal complete grant set
→ PUT role once
→ success: close and refresh roles
```

### 9.4 Create user

```text
Open Create User
→ load /roles/available
→ enter email, username, permanent password and confirmation
→ select exactly one role
→ disable Create while saving
→ POST user
→ clear password fields
→ success: close and refresh users
```

If no role is available, the form displays `Create a role before creating a user.`

### 9.5 Edit/deactivate/delete user

```text
Edit details/role → PUT user → refresh users
Activate/deactivate → PATCH status → update row
Delete → confirm → DELETE user → refresh users
```

---

## 10. Acceptance criteria

### Common

- [ ] All endpoints use the same `data/meta` success and `error` failure envelopes.
- [ ] Lists use `data.items` and the agreed pagination fields.
- [ ] APIs use consistent `user_id`, `role_id`, `source_id`, and `resource_id` names.
- [ ] Secrets never appear in responses or errors.
- [ ] The shared frontend API client parses the common envelope and handles `204 No Content` centrally.

### Authentication

- [ ] Admin accounts are provisioned only by backend/DevOps in V1.
- [ ] There is no public registration endpoint.
- [ ] Admin User Management cannot grant Admin privileges.
- [ ] Chatbot users log in with the username/password created by an Admin.
- [ ] Invalid credentials use one generic error.
- [ ] Inactive accounts and Chatbot users without roles receive the defined errors and no token.
- [ ] Login, refresh, and logout use the common response/error conventions.

### Data Sources

- [ ] Database, Datalake, and File System connections can be created with their current form fields.
- [ ] Failed initial connection does not create a source.
- [ ] Resource hierarchy loads lazily and is cursor-paginated.
- [ ] Data Sources renders hierarchy read-only; Roles adds checkboxes.
- [ ] `PATCH` changes only supplied fields and preserves omitted secrets.
- [ ] Identity-changing edits require a new source.
- [ ] An unavailable source remains visible with `ERROR` status.
- [ ] A source referenced by a role returns `409 SOURCE_USED_BY_ROLE` on delete.

### Roles and permissions

- [ ] Roles can be created with empty permissions.
- [ ] Empty permissions always mean no protected-data access.
- [ ] Selecting a child marks its ancestors selected/partial in the UI.
- [ ] Partial parent selection submits only selected child `SELF` grants.
- [ ] Complete parent selection submits one `SELF_AND_DESCENDANTS` grant.
- [ ] Create and update accept role data and the complete permission set in one request.
- [ ] Update removes grants omitted from the request.
- [ ] Missing resources appear as `UNAVAILABLE` and never grant access.
- [ ] Assigned roles return `409 ROLE_ASSIGNED` on delete.

### Users

- [ ] User creation requires exactly one assignable role.
- [ ] Password is permanent, write-only, and never returned.
- [ ] Updating a user and replacing their role is one request.
- [ ] Users can be activated and deactivated.
- [ ] Inactive/deleted users cannot use protected Chatbot endpoints.

### Runtime

- [ ] Authorization is enforced by the backend, not the Chatbot frontend or LLM.
- [ ] All protected resources required by an operation must be allowed.
- [ ] One denied resource rejects the complete operation.
- [ ] Restricted responses do not leak resource or data details.

---

## 11. Explicitly deferred from V1

- Multiple roles per user.
- Multiple active sources per source type in the UI; API shapes remain future-compatible.
- Action-level permissions such as `READ`, `QUERY`, and `DOWNLOAD`.
- Explicit deny grants and “subtree except child” rules.
- Idempotency keys.
- Optimistic-lock versions and `If-Match`.
- Catalog snapshot/version tokens.
- Frontend-defined session, cache, transaction, archival, or token-revocation mechanisms.
- Denied-access audit logging and Admin audit-log APIs.
