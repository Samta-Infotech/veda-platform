"""Effective-permission read endpoint.

Phase 6's "List Effective Permission". Answers what the RBAC graph *would* decide —
**nothing is enforced by it**. This is the endpoint that makes the graph observable
before any gate exists, and the one an admin screen uses to explain why someone can or
cannot do something.
"""
from __future__ import annotations

from apps.core import api
from apps.core.messages import MESSAGES

from ..codes import PermissionCode
from ..serializers import EffectivePermissionsSerializer
from ..services import AccessManagementError, PermissionResolver, UserService
from .base import AdminView


class EffectivePermissionsView(AdminView):
    """GET /api/v1/users/permissions/effective?user_id=&permission_code=&resource_path=

    Without ``permission_code``: the user's whole effective set.
    With it: the same, plus a ``decision`` block answering that exact question the way
    the resolver would — so a client never has to re-implement prefix inheritance or
    DENY precedence and drift from the server.
    """

    serializer_class = EffectivePermissionsSerializer
    action = "effective permissions"
    required_permission = PermissionCode.USER_MANAGE

    def get(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            # Confirms the user exists (404 otherwise) rather than returning an empty
            # set for a nonexistent id — "no permissions" and "no such user" are very
            # different answers to an administrator.
            user = UserService(request).get_user(data["user_id"])
        except AccessManagementError as exc:
            return self.failure(request, exc)

        effective = PermissionResolver(request).resolve_for_user_id(user.pk)
        payload = {
            **effective.as_dict(),
            "username": user.username,
            "is_active": user.is_active,
            "permission_codes": list(effective.permission_codes),
        }

        code = data.get("permission_code")
        if code:
            path = data.get("resource_path", "")
            payload["decision"] = {
                "permission_code": code,
                "resource_path": path,
                "allowed": effective.allows(code, path),
                # Distinct from `not allowed`: an explicit DENY and "never granted"
                # look identical otherwise, and they are not the same thing to fix.
                "explicitly_denied": effective.denies(code, path),
                "granted_on": list(effective.resources_for(code)),
            }

        return api.success(MESSAGES["resolver"]["resolved"], payload)
