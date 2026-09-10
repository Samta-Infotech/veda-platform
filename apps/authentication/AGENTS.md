# apps/authentication/ — identity verification

7 files, no models. Login / refresh / logout / password-change. JWT via
`rest_framework_simplejwt`, behind `VEDA_JWT_AUTH` (default **off**). Full reference:
[../../docs/RBAC.md](../../docs/RBAC.md) §Authentication.

| File | Role |
|------|------|
| `views.py` | `LoginView` / `TokenRefreshView` / `LogoutView` (`AllowAny` — the credential is the token), `PasswordChangeView` (`IsAuthenticated`). All POST. `_ERROR_STATUS` maps `AuthError` subclasses → HTTP status; only `exc.message` / `exc.code` reach the client. Scoped throttles (`login`, `token_refresh`, `password_change`). |
| `services.py` | `AuthService` — login / refresh / logout / change_password. `jwt_enabled()`. Two-tier Redis lockout. `_RotatableRefreshToken` (defers the blacklist check so the INSERT is the race arbiter). `_authorization_context` (roles + permission_codes in the login response, JWT-on only, not in a claim — resolver is read live). Full `AuthError` taxonomy (`AccountLocked`, `AccountInactive`, `InvalidCredentials`, `NoRoleAssigned`, `AdminPrivilegesRequired`, `InvalidRefreshToken`, `CurrentPasswordIncorrect`). **`AdminPrivilegesRequired` is defined twice** (harmless dup). |
| `serializers.py` | `LoginRequestSerializer` (username/password/optional `is_admin`), `RefreshTokenRequestSerializer` (refresh + logout share it), `PasswordChangeRequestSerializer` (runs `validate_password`). Shallow validation only. |
| `password_validators.py` | `PasswordComplexityValidator` — min upper/lower/digit/special, all via `OPTIONS` in `AUTH_PASSWORD_VALIDATORS`. Policy is edited in that dict, never in code. |
| `urls.py` | `auth/login`, `auth/refresh`, `auth/logout`, `auth/password/change`. |
| `apps.py` / `__init__.py` | config + marker. |

## Flows (see RBAC.md for the state machines)
- **Login**: per-(account,IP) lockout checked before any hash compare → `authenticate()` →
  correct-password-but-inactive probe → failure counters (per-IP hard `VEDA_AUTH_LOGIN_MAX_FAILURES`
  default 10; account-wide **soft** `..._ACCOUNT_MAX_FAILURES` default 50 — never refuses a
  correct password, can't DoS a real user) → `is_admin` vs `is_superuser` → `is_staff`-or-has-role.
- **Refresh**: parse+verify (defer blacklist) → load active user → password-hash claim check
  → `_spend` (blacklist INSERT = race arbiter) → on `created is False` → **replay** →
  `revoke_all_refresh_tokens(user)` → 401.
- **Logout**: always 200 (idempotent; not gated on `jwt_enabled`).
- **Password change**: `check_password` (constant-time) → `set_password` → `revoke_all_refresh_tokens`.

## Gotchas
- **`VEDA_JWT_AUTH` off (default)**: login returns `{"access_token": "dummy_access_token"}`,
  refresh always 401s. Real JWTs only mint with the flag on.
- **Lockout is Redis-only, fail-open** — no `is_locked` / `password_attempts` /
  `last_failed_attempt` model fields exist (a snippet with those belongs to a different
  codebase). `_is_locked()` is the Redis method.
- Stock `django.contrib.auth.User` — not swapped. `token_blacklist` app provides
  `OutstandingToken` / `BlacklistedToken`.
- Shares only `apps/core/token_revocation.revoke_all_refresh_tokens` with
  `access_management` — neither app imports the other.
