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

from apps.core import api

from ..serializers import (
    UserCreateSerializer,
    UserDetailSerializer,
    UserListSerializer,
    UserUpdateSerializer,
)
from ..services import AccessManagementError, UserService
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
    required_permission = "user.manage"

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            user = UserService(request).create_user(**data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success("User created successfully.", public_fields(user),
                           status_code=status.HTTP_201_CREATED)


class UserDetailView(AdminView):
    """POST /api/v1/users/detail {user_id} -> one user.

    Same projection as list, so an admin UI can open a row without reconciling two
    shapes. Deliberately no extra fields: the interesting additions (roles,
    permissions, sessions) do not exist yet, and inventing placeholders for them now
    would be a contract we would have to break.
    """

    serializer_class = UserDetailSerializer
    action = "user detail"
    required_permission = "user.manage"

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        try:
            user = UserService(request).get_user(data["user_id"])
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success("User retrieved successfully.", public_fields(user))


class UserListView(AdminView):
    """POST /api/v1/users/list {page?, page_size?, search?, is_active?, ordering?}.

    Always paginated — an unbounded list endpoint is a production incident waiting for
    the user table to grow. ``page_size`` is capped by the serializer, and the
    response carries the totals a client needs to render a pager.
    """

    serializer_class = UserListSerializer
    action = "user list"
    required_permission = "user.manage"

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        users, total = UserService(request).list_users(**data)
        page, page_size = data["page"], data["page_size"]

        return api.success("Users retrieved successfully.", {
            "users": [public_fields(user) for user in users],
            "pagination": pagination_payload(page, page_size, total),
        })


class UserUpdateView(AdminView):
    """POST /api/v1/users/update {user_id, email?, first_name?, last_name?}.

    Partial: only the fields present are written. ``username``, ``password`` and the
    privilege flags are deliberately not updatable here — see
    ``serializers.UserUpdateSerializer`` for why each belongs elsewhere.

    404 when the id does not exist, 409 when the new email belongs to someone else.
    """

    serializer_class = UserUpdateSerializer
    action = "user update"
    required_permission = "user.manage"

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure

        user_id = data.pop("user_id")
        try:
            user = UserService(request).update_user(user_id, **data)
        except AccessManagementError as exc:
            return self.failure(request, exc)

        return api.success("User updated successfully.", public_fields(user))
