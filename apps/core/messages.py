"""Every API success/failure message, in one nested dict — no copy typed twice.

``MESSAGES["role"]["created"]`` rather than a class-per-domain: a wrong key raises
``KeyError`` immediately, the same typo-safety a class attribute gives, and the
whole set of user-facing copy is one JSON-shaped object a non-engineer could read
or a future i18n layer could load from a file instead of a Python literal.

Covers BOTH sides of every endpoint: the ``api.success(...)`` copy typed in each
``views.py``, and the ``MSG_*`` constants each ``services.py`` attaches to its typed
exceptions (``RoleNotFound.message``, ``InvalidCredentials.message``, etc.) — those
modules import specific values from here rather than re-typing them, so the
exception classes keep their own names (readable at the raise site) while the copy
itself has exactly one home.
"""
from __future__ import annotations

MESSAGES = {
    "auth": {
        "login_success": "Login successful.",
        "token_refreshed": "Token refreshed successfully.",
        "logout_success": "Logout successful.",
        "invalid_credentials": "Invalid username or password.",
        "account_locked": "Too many failed login attempts. Please try again later.",
        "invalid_token": "Invalid or expired token.",
        "password_changed": "Password changed successfully.",
        "current_password_incorrect": "The current password is incorrect.",
        "already_bootstrapped": (
            "An administrator already exists; bootstrap can only run on an "
            "empty user table."),
    },
    "user": {
        "created": "User created successfully.",
        "retrieved": "User retrieved successfully.",
        "list": "Users retrieved successfully.",
        "updated": "User updated successfully.",
        "username_taken": "A user with that username already exists.",
        "email_taken": "A user with that email address already exists.",
        "conflict": "A user with those details already exists.",
        "not_found": "No such user.",
        "last_admin_protected": (
            "This is the last active administrator; the platform must always "
            "have at least one."),
        "deleted": "User deleted successfully.",
    },
    "role": {
        "created": "Role created successfully.",
        "retrieved": "Role retrieved successfully.",
        "list": "Roles retrieved successfully.",
        "dropdown": "Roles retrieved successfully.",
        "updated": "Role updated successfully.",
        "name_taken": "A role with that name already exists.",
        "not_found": "No such role.",
        "deleted": "Role deleted successfully.",
        "invalid_grant": ("Some of the permissions or resource grants in this "
                          "request are invalid."),
    },
    "permission": {
        "retrieved": "Permission retrieved successfully.",
        "list": "Permissions retrieved successfully.",
        "not_found": "No such permission.",
    },
    "catalog": {
        "retrieved": "Catalog resource retrieved successfully.",
        "list": "Catalog resources retrieved successfully.",
        "not_found": "No such catalog resource.",
    },
    "user_role": {
        "assigned": "Role assigned successfully.",
        "already_assigned": "Role was already assigned.",
        "revoked": "Role revoked successfully.",
        "not_assigned": "Role was not assigned.",
        "list": "Role assignments retrieved successfully.",
    },
    "role_permission": {
        "granted": "Permission granted successfully.",
        "updated": "Grant updated successfully.",
        "revoked": "Permission revoked successfully.",
        "not_granted": "Permission was not granted.",
        "list": "Permission grants retrieved successfully.",
    },
    "grant": {
        "role_inactive": "That role is retired and cannot be assigned.",
        "permission_inactive": "That permission is disabled and cannot be granted.",
        "invalid_resource": "That resource path is not valid.",
        "last_admin_role_protected": (
            "This is the last active administrator; the Admin role cannot be "
            "removed from them."),
    },
    "resolver": {
        "resolved": "Effective permissions resolved successfully.",
    },
    "conversation": {
        "created": "Conversation created successfully.",
        "retrieved": "Conversation retrieved successfully.",
        "list": "Conversations retrieved successfully.",
        "not_found": "Conversation not found.",
        "query_processed": "Query processed successfully.",
    },
    "chat": {
        "not_found": "Chat not found.",
        "auth_required": "Authentication required.",
        # Gate 1 (User Story 3, Task 17): generic on purpose — never names a
        # resource, table, column, or any internal RBAC detail.
        "access_denied": "You do not have permission to access this resource.",
        "llm_unavailable": ("The AI assistant is temporarily unavailable. "
                             "Please try again in a moment."),
        "model_error": "Something went wrong while generating a response. Please try again.",
    },
}
