# Auth + RBAC API Contract — Frontend Integration Reference

Covers the authentication endpoints served by `apps/authentication`. This is the
**single source of truth** for how the frontend authenticates; `CHAT_API_CONTRACT.md`
and `docs/QUERY_API_FRONTEND_CONTRACT.md` point here rather than restating any of it.

**Living document.** §7 lists the RBAC endpoints that are *not built yet*. As each
one is picked up, its full request/response shape moves out of §7 and into a real
section here in the same turn it is implemented — an RBAC endpoint that ships
without updating this file is an incomplete change. §8 tracks the revision history.

| | |
|---|---|
| **Base** | `/api/v1/` (nginx → Django api tier) |
| **Implemented now** | `POST auth/login` · `POST auth/refresh` · `POST auth/logout` |
| **Not built yet** | everything in §7 (roles, permissions, user management, `me`) |
| **Rollout flag** | `VEDA_JWT_AUTH` — **default `0`** → login returns the legacy placeholder token (§1.4) |
| **Status** | code + local-test verified (59 tests). **Not yet exercised against the live stack or by a frontend.** See `RBAC_PROGRESS_LOG.md` §7 for open items and known defects. |

---

## 0. Conventions

Every response uses the envelope already used across this api:

```ts
// success
{ status_code: number; message: string; data?: object }
// validation failure (malformed body)
{ status_code: 400; message: "Invalid request data."; errors: { [field: string]: string[] } }
// auth failure
{ status_code: number; message: string; code: ErrorCode }
```

`status_code` is duplicated inside the body **and** set as the real HTTP status —
read the HTTP status; the body field is for logging convenience.

**Always branch on `code`, never on `message`.** Copy will change; codes will not.

```ts
type ErrorCode = "INVALID_CREDENTIALS" | "ACCOUNT_LOCKED" | "INVALID_TOKEN";
```

| `code` | HTTP | Meaning |
|---|---|---|
| `INVALID_CREDENTIALS` | 401 | Wrong password, unknown user, **or** disabled account — deliberately indistinguishable |
| `ACCOUNT_LOCKED` | 429 | Per-account failure threshold hit (§1.3) |
| `INVALID_TOKEN` | 401 | Refresh token malformed, wrongly signed, expired, wrong type, already spent, or replayed — deliberately indistinguishable |

All three endpoints are **`POST` only**. A `GET` returns `405`.

---

## 1. `POST /api/v1/auth/login`

### Request

```json
{ "username": "admin", "password": "admin123" }
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | string | yes | non-blank; whitespace is trimmed |
| `password` | string | yes | non-blank; never logged |

### 1.1 Success — `200`

```json
{
  "status_code": 200,
  "message": "Login successful.",
  "data": {
    "user_id": 1,
    "username": "admin",
    "display_name": "Administrator",
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 900
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `user_id` | int | Django user pk |
| `username` | string | canonical username as stored |
| `display_name` | string | first name if set, else `username` |
| `access_token` | string | JWT. Send as `Authorization: Bearer <access_token>` |
| `refresh_token` | string | JWT. **Single-use** — see §2 |
| `token_type` | `"Bearer"` | constant |
| `expires_in` | int | **access** token lifetime in seconds (900 = 15 min default). No `refresh_expires_in` is sent; the refresh lifetime is 7 days by default |

### 1.2 Failures

| HTTP | `code` | When |
|---|---|---|
| 400 | — | missing/blank `username` or `password` (`errors` present) |
| 401 | `INVALID_CREDENTIALS` | wrong password · unknown username · **inactive account** |
| 429 | `ACCOUNT_LOCKED` | too many failures from your address, **or** an account-wide flood *and* a wrong password — see §1.3 |
| 429 | — (DRF throttle body) | more than **10 login requests/min from one IP**. This is a *different* 429 with no `code` field — treat a 429 without `code` as "slow down", not as a lockout |

The three `INVALID_CREDENTIALS` causes return **byte-identical** bodies by design.
Do not attempt to render "no such user" vs "wrong password" — the server will not
tell you, and asking for it would make login an account-enumeration oracle.

### 1.3 Account lockout — behaviour the UI must handle

Two counters, both keyed on the username case-insensitively, both over a rolling
**5-minute** window (`VEDA_AUTH_LOGIN_LOCKOUT_SECONDS`):

| Counter | Threshold | Effect |
|---|---|---|
| per **(account, your IP)** | 10 failures (`VEDA_AUTH_LOGIN_MAX_FAILURES`) | **Hard.** Every further attempt from that address is `429`, including one with the correct password. |
| **account-wide**, all IPs | 50 failures (`VEDA_AUTH_LOGIN_ACCOUNT_MAX_FAILURES`) | **Soft.** A *wrong* password returns `429` instead of `401`. A **correct password is always accepted.** |

The practical consequences for the UI:

- A user who mistypes 10 times from their own device gets `429` for 5 minutes. Show
  "too many attempts, try again in a few minutes". **No `Retry-After` header is
  sent**, so do not render a countdown.
- A successful login clears both counters immediately.
- Somebody else attacking the account **cannot lock the real user out** — the
  blocking counter is per source address, and the account-wide one never refuses a
  correct password. A `429` therefore always means *this client* (or this account,
  under a distributed flood) should back off, never that the account is frozen.

`ACCOUNT_LOCKED` is returned for both cases and is not distinguishable by the
client — deliberately, since telling them apart would report on other clients'
traffic.

### 1.4 While `VEDA_JWT_AUTH=0` (the current default)

Login returns the **pre-JWT payload**, unchanged from the previous dummy view:

```json
{
  "status_code": 200,
  "message": "Login successful.",
  "data": {
    "user_id": 1, "username": "admin", "display_name": "Administrator",
    "access_token": "dummy_access_token", "token_type": "Bearer"
  }
}
```

No `refresh_token`, no `expires_in`, and the access token is a placeholder that
authenticates nothing. `auth/refresh` returns `401 INVALID_TOKEN` in this mode;
`auth/logout` returns `200`.

**Client rule:** treat `refresh_token`/`expires_in` as optional and branch on their
presence, so the same build works before and after the flag is flipped.

---

## 2. `POST /api/v1/auth/refresh`

Exchanges a refresh token for a **new pair**. Requires no `Authorization` header —
the refresh token *is* the credential (your access token is expected to be expired
by the time you call this).

### Request

```json
{ "refresh_token": "eyJhbGciOiJIUzI1NiIs..." }
```

### Success — `200`

```json
{
  "status_code": 200,
  "message": "Token refreshed successfully.",
  "data": {
    "user_id": 1, "username": "admin", "display_name": "Administrator",
    "access_token": "<new>", "refresh_token": "<new>",
    "token_type": "Bearer", "expires_in": 900
  }
}
```

Identity fields are included so a page reload can rehydrate the session without a
separate `me` call (which does not exist yet — §7).

### Failures

| HTTP | `code` | When |
|---|---|---|
| 400 | — | missing/blank `refresh_token` |
| 401 | `INVALID_TOKEN` | malformed · bad signature · expired · **access token passed instead** · already spent · replayed · account disabled or deleted |
| 429 | — | more than 60 refresh requests/min from one IP |

### 2.1 Three rules the client MUST follow

**Rule 1 — replace the stored refresh token on every success.** Rotation is strict:
the token you just sent is dead the instant the server accepts it, regardless of
whether you received the response.

**Rule 2 — never fire two `refresh` calls concurrently for the same token.**
Exactly one can win. Serialize behind a single in-flight promise:

```ts
let inFlight: Promise<Tokens> | null = null;

function refreshOnce(): Promise<Tokens> {
  if (!inFlight) {
    inFlight = doRefresh(store.refreshToken)
      .then(t => { store.set(t); return t; })          // Rule 1
      .finally(() => { inFlight = null; });
  }
  return inFlight;                                      // all callers await the same call
}
```

**Rule 3 — on `401 INVALID_TOKEN` from `refresh`, go straight to the login screen.**
Do not retry, and do not retry with the old token: a second presentation is what
triggers §2.2.

### 2.2 Replay ⇒ every session of that account is revoked

Presenting the same refresh token twice is treated as a captured token. The server
cannot tell the legitimate holder from the thief, so **all** of that user's refresh
tokens are revoked — including the one just issued and any on other devices.

```
send token T  → 200, get T'          (T is now dead)
send token T  → 401 INVALID_TOKEN    → T' and every other device's token also die
```

This is the OAuth 2.0 security BCP behaviour and it is intentional. The practical
consequence: **a client that violates Rule 1 or Rule 2 will log the user out of
every device.** A dropped response followed by a blind retry does the same, so
retry logic around `refresh` must be absent, not merely careful.

---

## 3. `POST /api/v1/auth/logout`

### Request

```json
{ "refresh_token": "eyJhbGciOiJIUzI1NiIs..." }
```

### Response — `200`, always

```json
{ "status_code": 200, "message": "Logout successful." }
```

No `data`. **Idempotent and unconditional**: an already-rotated, expired, revoked,
or entirely garbage token still returns `200`. Reporting otherwise would tell a
caller whether the token it holds is live. The only non-200 is `400` for a
missing/blank `refresh_token`.

Scope: **this session only.** Other devices stay signed in (unlike the replay path,
§2.2). There is no "log out everywhere" endpoint — see §7.

### 3.1 What logout does *not* do

It revokes the refresh token, so no new access token can be obtained. The **access
token already in the client's hands is not revoked server-side** and remains
accepted until it expires (≤ `expires_in`, 15 min default). A per-request denylist
would mean a DB read on every API call and was deliberately not built.

**Client rule:** discard both tokens locally on logout. Do not rely on the server
rejecting the old access token immediately — it will not.

Two related revocations that **are** immediate, both enforced per request on access
tokens and again on refresh:

| Event | Effect on existing tokens |
|---|---|
| User deactivated (`is_active = False`) | Access token rejected on the next request; refresh rejected |
| **Password changed** | Every token minted under the old password is rejected — access and refresh alike |

So a password change is an effective "sign me out everywhere". Once a
password-change screen exists (§7), expect the user to be logged out of all devices
by it, and clear local tokens on success.

---

## 4. Using the access token

```
Authorization: Bearer <access_token>
```

Only active while `VEDA_JWT_AUTH=1`. **No endpoint requires it yet** — every
existing endpoint is still `AllowAny`, and the chat endpoints fall back to a seeded
`admin` user for anonymous requests. Sending the header early is harmless and is
the right thing to build against.

### 4.1 Recommended interceptor

```
request  → attach Authorization if a token is stored
response → 401 from a NON-auth endpoint      → refreshOnce(), retry the request ONCE
         → 401 INVALID_TOKEN from auth/refresh → clear tokens, redirect to login
         → 429 with    code=ACCOUNT_LOCKED     → "too many attempts, try again shortly"
         → 429 without code                    → generic rate-limit backoff
```

Retry **once** only, and never retry `auth/refresh` itself (§2.1 Rule 3).

### 4.2 CSRF

These endpoints are CSRF-exempt for anonymous callers, so a normal login flow needs
no CSRF token. One edge case: a browser that already holds a **Django admin session
cookie** will have DRF's `SessionAuthentication` engage and demand `X-CSRFToken`,
yielding `403`. This affects developers testing while logged into `/admin/` and is a
pre-existing project-wide pattern, not specific to auth.

### 4.3 Token storage

Tokens are returned in the JSON body only; the server sets no cookies. The refresh
token is therefore whatever the client makes it — and JS-readable storage means XSS
exposure. `httpOnly; Secure; SameSite` cookie delivery is **not** offered today
(tracked as M4). Prefer memory + a short-lived mechanism over `localStorage` if the
threat model includes XSS.

---

## 5. Error handling summary

| HTTP | Body has `code` | Meaning | Client action |
|---|---|---|---|
| 200 | — | success | proceed |
| 400 | no (`errors` instead) | malformed body | fix the request; show field errors |
| 401 | `INVALID_CREDENTIALS` | login rejected | show one generic credential error |
| 401 | `INVALID_TOKEN` | refresh token unusable | clear tokens → login screen |
| 429 | `ACCOUNT_LOCKED` | per-account lockout | "try again in a few minutes" |
| 429 | no | per-IP throttle | backoff |
| 5xx | — | server fault | generic error; **never** shown as a credential problem |

No error response ever contains a stack trace, a secret, a token internal, or a hint
about which half of a credential pair was wrong. If you see one, that is a bug —
report it.

---

## 6. TypeScript types

```ts
type TokenType = "Bearer";
type AuthErrorCode = "INVALID_CREDENTIALS" | "ACCOUNT_LOCKED" | "INVALID_TOKEN";

interface LoginRequest  { username: string; password: string }
interface RefreshRequest { refresh_token: string }
interface LogoutRequest  { refresh_token: string }

interface AuthData {
  user_id: number;
  username: string;
  display_name: string;
  access_token: string;
  token_type: TokenType;
  refresh_token?: string;   // absent while VEDA_JWT_AUTH=0 (§1.4)
  expires_in?: number;      // absent while VEDA_JWT_AUTH=0
}

interface AuthSuccess { status_code: number; message: string; data: AuthData }
interface LogoutSuccess { status_code: 200; message: string }          // no `data`
interface AuthFailure { status_code: number; message: string; code: AuthErrorCode }
interface ValidationFailure {
  status_code: 400; message: "Invalid request data.";
  errors: Record<string, string[]>;
}
```

---

## 7. Still to come

**Identity administration and RBAC have moved to their own bounded context** —
`apps/access_management`, documented in **`ACCESS_MANAGEMENT_API_CONTRACT.md`**.
`POST /api/v1/users` is implemented and specified there; roles, permissions and the
rest of the RBAC scope are listed there as not-built.

This document covers identity **verification** only. What remains unbuilt here:

| Endpoint | Purpose | Status |
|---|---|---|
| `GET  auth/me` | current principal (+ roles/permissions once RBAC exists) for UI gating | ❌ not started |
| `POST auth/logout-all` | revoke every session of the account | ❌ not started |
| `POST auth/password/change` | change password. Already handled server-side: a password change **does** revoke existing tokens (§3.1) | ❌ not started |
| `POST auth/password/reset` · `.../confirm` | forgotten-password flow | ❌ not started |
| tenant scoping | today the tenant is derived from `request.user.username` (`apps/query/views.py::_resolve_tenant`) with a `"default"` dev fallback | ❌ not designed |

There is still **no authorization layer**, and no endpoint outside
`apps/access_management` requires a token.

### Definition of done for each RBAC endpoint

An RBAC endpoint is not complete until, in the same change:

1. its row above is replaced by a full section here (request, success, every failure `code`, and any client rule);
2. new error codes are added to §0 and §5 and to the `AuthErrorCode` union in §6;
3. `RBAC_PROGRESS_LOG.md` §1 snapshot and §8 work log are updated;
4. §8 below gets a revision row.

Two things that will change *existing* sections when RBAC lands and must be
called out here at that time: the access token will start carrying role/permission
claims (making it larger, and stale for up to `expires_in` after a role change), and
endpoints will begin returning **`403`** — a status this contract does not use today.

---

## 8. Revision history

| Date | Change |
|---|---|
| 2026-08-05 | Initial contract: `auth/login`, `auth/refresh`, `auth/logout`. Extracted from `docs/QUERY_API_FRONTEND_CONTRACT.md`, which now links here. RBAC scope listed in §7 as not-built. |
| 2026-08-05 (user creation) | §7 rewritten: user management + RBAC split into `ACCESS_MANAGEMENT_API_CONTRACT.md` (separate bounded context). No auth endpoint changed. |
| 2026-08-05 (Phase 1.1) | **§1.3 lockout semantics changed** — two counters; a correct password is no longer refused account-wide, and a third party can no longer lock a user out (C1). **§3.1**: a password change now revokes existing access *and* refresh tokens (C3). §1.2 failure table clarified. No request/response shape changed. |
