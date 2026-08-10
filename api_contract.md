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
  "data": {},
  "meta": {
    "request_id": "req_01J..."
  }
}
```

### 2.4 Collection success

```json
{
  "data": {
    "items": []
  },
  "meta": {
    "request_id": "req_01J...",
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 0,
      "total_pages": 0,
      "has_next_page": false,
      "has_previous_page": false
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
  "data": {
    "items": []
  },
  "meta": {
    "request_id": "req_01J...",
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
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message.",
    "details": {},
    "request_id": "req_01J..."
  }
}
```

- `code` is stable and machine-readable.
- `message` is safe to display unless an endpoint states otherwise.
- `details` is optional.
- Errors MUST NOT expose credentials, raw driver errors, stack traces, restricted resource names, SQL, or protected values.

Validation errors may include field errors:

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "One or more fields are invalid.",
    "details": {
      "field_errors": [
        {
          "field": "email",
          "code": "INVALID_EMAIL",
          "message": "Enter a valid email address."
        }
      ]
    },
    "request_id": "req_01J..."
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
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

Admin response:

```json
{
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
  },
  "meta": {
    "request_id": "req_01J..."
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
  "error": {
    "code": "INVALID_CREDENTIALS",
    "message": "Invalid username or password.",
    "details": {},
    "request_id": "req_01J..."
  }
}
```

The response MUST NOT reveal whether the username exists.

Inactive account:

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

Chatbot user without an assigned role:

```http
403 Forbidden
```

```json
{
  "error": {
    "code": "ROLE_NOT_ASSIGNED",
    "message": "No role has been assigned to your account. Please contact your admin.",
    "details": {},
    "request_id": "req_01J..."
  }
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
  "data": {
    "access_token": "new-access-token",
    "token_type": "Bearer"
  },
  "meta": {
    "request_id": "req_01J..."
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
  "error": {
    "code": "SOURCE_TYPE_LIMIT_REACHED",
    "message": "Only one Database source can currently be connected.",
    "details": {
      "source_type": "DATABASE"
    },
    "request_id": "req_01J..."
  }
}
```

The frontend stores sources as an array even when only one item of each type exists.

### 5.2 List sources

```http
GET /v1/admin/data-sources?source_type=DATABASE&status=CONNECTED&page=1&page_size=20&search=finance
```

`source_type`, `status`, and `search` are optional.

```json
{
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
    ]
  },
  "meta": {
    "request_id": "req_01J...",
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 1,
      "total_pages": 1,
      "has_next_page": false,
      "has_previous_page": false
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
POST /v1/admin/data-sources
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
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

```http
422 Unprocessable Entity
```

```json
{
  "error": {
    "code": "SOURCE_CONNECTION_FAILED",
    "message": "The data source connection could not be established.",
    "details": {},
    "request_id": "req_01J..."
  }
}
```

### 5.7 Lazily browse resources

Root resources:

```http
GET /v1/admin/data-sources/{source_id}/resources?page_size=100
```

Children of one resource:

```http
GET /v1/admin/data-sources/{source_id}/resources?parent_id=schema_reporting&page_size=100&cursor=cursor_abc
```

```json
{
  "data": {
    "items": [
      {
        "resource_id": "table_monthly_revenue",
        "resource_type": "TABLE",
        "name": "monthly_revenue",
        "parent_id": "schema_reporting",
        "has_children": true
      }
    ]
  },
  "meta": {
    "request_id": "req_01J...",
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
  "error": {
    "code": "SOURCE_UNAVAILABLE",
    "message": "The data source is currently unavailable.",
    "details": {},
    "request_id": "req_01J..."
  }
}
```

### 5.8 Edit a source

```http
PATCH /v1/admin/data-sources/{source_id}
```

`PATCH` uses merge semantics. Supplied fields change and omitted fields remain unchanged.

```json
{
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

### 6.1 List roles

```http
GET /v1/admin/roles?page=1&page_size=20&search=analyst
```

```json
{
  "data": {
    "items": [
      {
        "role_id": 1,
        "role_name": "Database Analyst",
        "users_count": 4,
        "connected_sources": [
          {
            "source_id": "source_db_01",
            "source_name": "Finance PostgreSQL",
            "source_type": "DATABASE"
          }
        ],
        "created_at": "2026-07-31T09:00:00Z",
        "updated_at": "2026-07-31T10:00:00Z"
      }
    ]
  },
  "meta": {
    "request_id": "req_01J...",
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 1,
      "total_pages": 1,
      "has_next_page": false,
      "has_previous_page": false
    }
  }
}
```

`connected_sources` is display information only, not an authorization policy.

### 6.2 Available roles for User selector

```http
GET /v1/admin/roles/available?page=1&page_size=20&search=analyst
```

```json
{
  "data": {
    "items": [
      {
        "role_id": 1,
        "role_name": "Database Analyst"
      }
    ]
  },
  "meta": {
    "request_id": "req_01J...",
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 1,
      "total_pages": 1,
      "has_next_page": false,
      "has_previous_page": false
    }
  }
}
```

Deleted or otherwise unusable roles are not returned.

### 6.3 Create role and permissions

```http
POST /v1/admin/roles
```

```json
{
  "role_name": "Database Analyst",
  "permissions": [
    {
      "source_id": "source_db_01",
      "grants": [
        {
          "resource_id": "table_monthly_revenue",
          "scope": "SELF_AND_DESCENDANTS"
        }
      ]
    }
  ]
}
```

A role without data access is valid:

```json
{
  "role_name": "New Analyst",
  "permissions": []
}
```

```http
201 Created
```

```json
{
  "data": {
    "role_id": 3,
    "role_name": "New Analyst",
    "permissions_count": 0,
    "users_count": 0,
    "created_at": "2026-07-31T10:00:00Z",
    "updated_at": "2026-07-31T10:00:00Z"
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

The backend validates all submitted sources, resources, scopes, and ownership before reporting success. Failure creates neither a role nor partial grants.

### 6.4 Get role details

```http
GET /v1/admin/roles/{role_id}
```

```json
{
  "data": {
    "role_id": 1,
    "role_name": "Database Analyst",
    "permissions": [
      {
        "source_id": "source_db_01",
        "source_name": "Finance PostgreSQL",
        "source_status": "CONNECTED",
        "grants": [
          {
            "resource_id": "table_monthly_revenue",
            "resource_name": "monthly_revenue",
            "scope": "SELF_AND_DESCENDANTS",
            "status": "AVAILABLE"
          },
          {
            "resource_id": "column_old_metric",
            "resource_name": "old_metric",
            "scope": "SELF",
            "status": "UNAVAILABLE"
          }
        ]
      }
    ],
    "users_count": 4,
    "created_at": "2026-07-31T09:00:00Z",
    "updated_at": "2026-07-31T10:00:00Z"
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

The Edit UI preselects available grants and visibly lists unavailable grants. It does not silently discard or remap them.

### 6.5 Replace role name and permissions

```http
PUT /v1/admin/roles/{role_id}
```

```json
{
  "role_name": "Senior Database Analyst",
  "permissions": [
    {
      "source_id": "source_db_01",
      "grants": [
        {
          "resource_id": "column_month",
          "scope": "SELF"
        },
        {
          "resource_id": "column_amount",
          "scope": "SELF"
        }
      ]
    }
  ]
}
```

This request replaces the complete name and permission set. Omitted grants are removed. `permissions: []` removes all data access while keeping the role.

If any submitted source or resource is invalid, the prior role name and complete permission set remain unchanged.

```json
{
  "data": {
    "role_id": 1,
    "role_name": "Senior Database Analyst",
    "permissions_count": 2,
    "updated_at": "2026-07-31T11:00:00Z"
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

### 6.6 Delete role

```http
DELETE /v1/admin/roles/{role_id}
```

If no user is assigned:

```http
204 No Content
```

The role then disappears from normal lists and can no longer grant access. Physical deletion versus internal archival is a backend decision.

If users are assigned:

```http
409 Conflict
```

```json
{
  "error": {
    "code": "ROLE_ASSIGNED",
    "message": "Reassign the users before deleting this role.",
    "details": {
      "users_count": 4
    },
    "request_id": "req_01J..."
  }
}
```

### 6.7 Role validation

- `role_name` is trimmed, 2–100 characters, and case-insensitively unique in the applicable context.
- Every `source_id` must exist.
- Every `resource_id` must belong to the submitted source.
- `scope` must be `SELF` or `SELF_AND_DESCENDANTS` and be valid for the resource.
- Duplicate or redundant overlapping grants are rejected.

| HTTP | Code | Meaning |
|---:|---|---|
| `404` | `ROLE_NOT_FOUND` | Role does not exist. |
| `404` | `SOURCE_NOT_FOUND` | Source does not exist. |
| `409` | `DUPLICATE_ROLE_NAME` | Role name already exists. |
| `409` | `ROLE_ASSIGNED` | Assigned role cannot be deleted. |
| `422` | `INVALID_PERMISSION_RESOURCE` | Resource is missing or belongs to another source. |
| `422` | `INVALID_PERMISSION_SCOPE` | Scope is invalid for the resource. |
| `422` | `REDUNDANT_PERMISSION_GRANT` | Payload contains overlapping grants. |

---

## 7. Users APIs

### 7.1 User rules

- V1 assigns exactly one role to each user.
- A role is required during creation and update.
- Permissions are stored only on the role, never copied to the user record.
- A user must be `ACTIVE` to use protected Chatbot APIs.
- An `INACTIVE` user remains manageable in Admin but cannot log in or access protected Chatbot APIs.
- Password changes/resets are outside these page endpoints and use a separate authentication flow.

### 7.2 List users

```http
GET /v1/admin/users?page=1&page_size=20&search=alice&status=ACTIVE
```

```json
{
  "data": {
    "items": [
      {
        "user_id": 101,
        "username": "alice",
        "email": "alice@example.com",
        "role": {
          "role_id": 1,
          "role_name": "Database Analyst"
        },
        "status": "ACTIVE",
        "created_at": "2026-07-31T10:00:00Z",
        "updated_at": "2026-07-31T10:00:00Z"
      }
    ]
  },
  "meta": {
    "request_id": "req_01J...",
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 1,
      "total_pages": 1,
      "has_next_page": false,
      "has_previous_page": false
    }
  }
}
```

### 7.3 Create user with permanent password and role

```http
POST /v1/admin/users
```

```json
{
  "email": "alice@example.com",
  "username": "alice",
  "password": "permanent-initial-password",
  "role_id": 1
}
```

Observable behavior:

1. Backend validates user fields and the selected role.
2. On success, the user and role assignment both exist.
3. On failure, neither a partial user nor assignment is reported as created.
4. The password is never returned.

```http
201 Created
```

```json
{
  "data": {
    "user_id": 101,
    "email": "alice@example.com",
    "username": "alice",
    "role": {
      "role_id": 1,
      "role_name": "Database Analyst"
    },
    "status": "ACTIVE",
    "created_at": "2026-07-31T10:00:00Z"
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

The frontend:

- validates against the same password policy as the backend;
- asks the Admin to confirm the password;
- disables Create while saving; and
- clears password fields after completion or modal closure.

### 7.4 Get user details

```http
GET /v1/admin/users/{user_id}
```

```json
{
  "data": {
    "user_id": 101,
    "email": "alice@example.com",
    "username": "alice",
    "role": {
      "role_id": 1,
      "role_name": "Database Analyst"
    },
    "status": "ACTIVE",
    "created_at": "2026-07-31T10:00:00Z",
    "updated_at": "2026-07-31T10:00:00Z"
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

The response never contains a password or password hash.

### 7.5 Update user details and role

```http
PUT /v1/admin/users/{user_id}
```

```json
{
  "email": "alice.new@example.com",
  "username": "alice_new",
  "role_id": 2
}
```

The request replaces editable details and the assigned role. It does not change the password or status. If the selected role is invalid, the prior user details and role assignment remain unchanged.

```json
{
  "data": {
    "user_id": 101,
    "email": "alice.new@example.com",
    "username": "alice_new",
    "role": {
      "role_id": 2,
      "role_name": "File Viewer"
    },
    "status": "ACTIVE",
    "updated_at": "2026-07-31T11:00:00Z"
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

### 7.6 Activate or deactivate user

```http
PATCH /v1/admin/users/{user_id}/status
```

```json
{
  "status": "INACTIVE"
}
```

```json
{
  "data": {
    "user_id": 101,
    "status": "INACTIVE",
    "updated_at": "2026-07-31T11:30:00Z"
  },
  "meta": {
    "request_id": "req_01J..."
  }
}
```

After success, the frontend reflects the returned status. The backend is responsible for rejecting subsequent protected requests from an inactive user.

### 7.7 Delete user

```http
DELETE /v1/admin/users/{user_id}
```

```http
204 No Content
```

After success, the user disappears from the normal users list and can no longer use the account. Physical deletion versus internal archival is a backend decision.

### 7.8 User validation

- Email is trimmed, lowercased, valid, and unique in the applicable context.
- Username is trimmed, 2–50 characters, and case-insensitively unique.
- Password follows the existing VEDA password policy.
- `role_id` is required and must refer to a role available for assignment.
- Status is `ACTIVE` or `INACTIVE`.

| HTTP | Code | Meaning |
|---:|---|---|
| `404` | `USER_NOT_FOUND` | User does not exist. |
| `404` | `ROLE_NOT_FOUND` | Selected role does not exist. |
| `409` | `DUPLICATE_EMAIL` | Email is already used. |
| `409` | `DUPLICATE_USERNAME` | Username is already used. |
| `422` | `ROLE_REQUIRED` | A role was not supplied. |
| `422` | `ROLE_NOT_ASSIGNABLE` | Role cannot be assigned. |
| `422` | `INVALID_USER_STATUS` | Status is unsupported. |

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
