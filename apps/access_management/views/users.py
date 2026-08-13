"""Thin DRF views for user administration.

Each view validates a body with a serializer, hands the validated values to
``UserService``, and renders the outcome via ``apps.core.api``. No creation logic, no
uniqueness handling and no hashing here.

Access control, the service-error-to-status mapping and the validate-or-400 branch
all come from ``views/base.py``, so every endpoint in this app enforces them the same
way and a future RBAC permission check has one place to land.
"""
from __future__ import annotations

from rest_framework import status
from ..models import UserRole

from apps.core import api
from apps.core.messages import MESSAGES

from ..codes import PermissionCode
from ..serializers import (
    UserCreateSerializer,
    UserDetailSerializer,
    UserListSerializer,
    UserUpdateSerializer,
)
from ..models import UserProfile
from ..services import AccessManagementError, UserRoleService, UserService
from .base import AdminView, pagination_payload


def public_fields(user) -> dict:
    """What a caller may see of a user — the ONE projection for every user endpoint.

    An explicit projection, not a model dump: that is what guarantees the password
    hash and internal columns cannot leak, including after a future migration adds
    one. Shared by create/detail/list/update so a user has exactly one representation
    across the API. ``display_name`` follows the rule login already uses, so the name
    shown after creating a user matches the name shown after they sign in.

    ``is_staff`` is included because an admin screen needs to see who is privileged;
    every caller here is already staff. Nothing writable-by-privilege appears — the
    flag is reported, never accepted (see ``serializers.PRIVILEGED_FIELDS``).
    """
    return {
        "user_id": user.pk,
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "display_name": user.first_name or user.username,
        "is_active": user.is_active,
        "is_staff": user.is_staff,
        "date_joined": api.iso_z(user.date_joined),
        "last_login": api.iso_z(user.last_login),
    }


class UserCreateView(AdminView):
    """POST /api/v1/users/create {username, email, password, first_name?, last_name?}.

    Returns 201 with the created user. A uniqueness conflict is 409 (the resource
    already exists) rather than a 400, so a client can distinguish "your request was
    malformed" from "your request was fine but the username is gone". The password is
    never echoed in any form.
    """

    serializer_class = UserCreateSerializer
    action = "user creation"
    required_permission = PermissionCode.USER_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            user = UserService(request).create_user(**data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["user"]["created"],
                           status_code=status.HTTP_201_CREATED)


class UserDetailView(AdminView):
    """GET /api/v1/users/detail?user_id= -> one user.

    Same projection as list, so an admin UI can open a row without reconciling two
    shapes. Deliberately no extra fields: the interesting additions (roles,
    permissions, sessions) do not exist yet, and inventing placeholders for them now
    would be a contract we would have to break.
    """

    serializer_class = UserDetailSerializer
    action = "user detail"
    required_permission = PermissionCode.USER_MANAGE

    def get(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            user = UserService(request).get_user(data["user_id"])
        except AccessManagementError as exc:
            return self.failure(request, exc)

        role_ids = list(UserRole.objects.filter(user=user).values_list("role_id", flat=True))
        return api.success(MESSAGES["user"]["retrieved"], {
            **public_fields(user),
            "role_ids": role_ids,
            # Which frontend app this account was created for — set once at
            # creation (see UserCreateSerializer), shown only on the detail page.
            "is_admin": user.is_superuser,
        })


class UserListView(AdminView):
    """GET /api/v1/users/list?page=&page_size=&search=&is_active=&ordering=

    Always paginated — an unbounded list endpoint is a production incident waiting for
    the user table to grow. ``page_size`` is capped by the serializer, and the
    response carries the totals a client needs to render a pager.
    """

    serializer_class = UserListSerializer
    action = "user list"
    required_permission = PermissionCode.USER_MANAGE

    def get(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        users, total = UserService(request).list_users(**data)
        page, page_size = data["page"], data["page_size"]

        # Two more queries for the whole page — see UserRoleService.roles_for_users.
        roles_by_user = UserRoleService.roles_for_users([user.pk for user in users])
        profiles_by_user = {
            profile.user_id: profile
            for profile in UserProfile.objects.filter(user_id__in=[u.pk for u in users])
        }

        return api.success(MESSAGES["user"]["list"], {
            "users": [
                {
                    **public_fields(user),
                    "roles": roles_by_user.get(user.pk, []),
                    "created_at": api.human_date(user.date_joined),
                    # "" / None for a user created before migration 0008 backfilled
                    # a profile row — not invented, genuinely never recorded.
                    "updated_at": (api.human_date(profiles_by_user[user.pk].updated_at)
                                  if user.pk in profiles_by_user else ""),
                    "deleted_at": (api.human_date(profiles_by_user[user.pk].deleted_at)
                                  if user.pk in profiles_by_user
                                  and profiles_by_user[user.pk].deleted_at else None),
                }
                for user in users
            ],
            "pagination": pagination_payload(page, page_size, total),
        })


class UserUpdateView(AdminView):
    """POST /api/v1/users/update {user_id, email?, first_name?, last_name?, is_active?, is_superuser?}.

    Partial: only the fields present are written. ``username``, ``password`` and
    ``is_staff`` are deliberately not updatable here — see
    ``serializers.UserUpdateSerializer`` for why each belongs elsewhere.
    ``is_active`` IS updatable here — deactivating a user is a profile edit, not a
    separate action, and ``UserService.update_user`` carries the last-admin guard
    and the token-revocation-on-deactivate step for it. ``is_superuser`` is also
    updatable — this platform's "which frontend app" flag, checked at login.

    404 when the id does not exist, 409 when the new email belongs to someone else
    or when ``is_active: false`` targets the platform's last active admin.
    """

    serializer_class = UserUpdateSerializer
    action = "user update"
    required_permission = PermissionCode.USER_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        user_id = data.pop("user_id")
        try:
            user = UserService(request).update_user(user_id, **data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["user"]["updated"])


class UserDeleteView(AdminView):
    """POST /api/v1/users/delete {user_id} -> 200.

    A SOFT delete — a named convenience over ``UserService.update_user(user_id,
    is_active=False)``, not a second code path. Same guard (refused for the
    platform's last active admin), same token revocation, same idempotency: calling
    it twice is still just "this account is inactive", not an error.

    No row is ever removed — ``deleted_at`` (``UserProfile``) records when this
    happened, ``is_active`` stays the one flag every access check keys off.
    """

    serializer_class = UserDetailSerializer
    action = "user deletion"
    required_permission = PermissionCode.USER_MANAGE

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            user = UserService(request).update_user(data["user_id"], is_active=False)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success(MESSAGES["user"]["deleted"])
