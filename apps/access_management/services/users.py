"""User administration business logic.

Views stay thin: validate a body, call in here, render the outcome. Everything that
decides whether a user can be created, and what the resulting row looks like, lives
in this module.

Nothing here implements hashing or password policy. ``UserManager.create_user``
hashes via ``set_password`` (project-configured hashers), and the policy is the
four ``AUTH_PASSWORD_VALIDATORS`` already applied by the serializer. This module
owns exactly one thing the framework does not give us: turning a database
uniqueness violation into a precise, safe API answer without a check-then-insert
race.

Error contract mirrors ``apps.authentication.services``: expected failures are
typed exceptions carrying a stable ``code`` and safe ``message``; the view maps the
class to an HTTP status. Unexpected failures propagate, so DRF returns a 500 and
the traceback lands in the log rather than on the wire.
"""
from __future__ import annotations

import logging

from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.utils import timezone
from ..models import Role, UserRole

from apps.core.messages import MESSAGES
from apps.core.token_revocation import revoke_all_refresh_tokens

from ..models import UserProfile
from .admin_guard import (
    ADMIN_ROLE_NAME,
    LastAdminProtected,
    LastAdminRoleProtected,
    is_last_active_admin,
)
from .base import AccessManagementError, ConflictError, NotFoundError, paginate

logger = logging.getLogger(__name__)

CODE_USERNAME_TAKEN = "USERNAME_TAKEN"
CODE_EMAIL_TAKEN = "EMAIL_TAKEN"
CODE_USER_CONFLICT = "USER_CONFLICT"
CODE_USER_NOT_FOUND = "USER_NOT_FOUND"
CODE_INVALID_ROLE = "INVALID_ROLE"

# Columns every user-facing projection needs. Applied via .only() so a list page
# never drags the password hash and permission bitfields across the wire from the
# database — the projection in views/users.py drops them, but they should not be fetched
# in the first place.
USER_LIST_FIELDS = (
    "id", "username", "email", "first_name", "is_active", "is_staff",
    "date_joined", "last_login",
)


class UserNotFound(NotFoundError):
    """No user with that id. Inherits its 404 from ``NotFoundError``."""

    code = CODE_USER_NOT_FOUND
    message = MESSAGES["user"]["not_found"]


class InvalidRole(AccessManagementError):
    """One or more specified role IDs do not exist or are inactive."""

    code = CODE_INVALID_ROLE
    message = MESSAGES["role"]["invalid_role"]


class DuplicateUser(ConflictError):
    """A uniqueness constraint rejected the new user.

    Not raised directly — it groups the two concrete conflicts below, which inherit
    their 409 from ``ConflictError``. Its ``code``/``message`` exist only as the safe
    default a future subclass would inherit if it forgot to set its own.
    """

    code = CODE_USER_CONFLICT
    message = MESSAGES["user"]["conflict"]


class UsernameTaken(DuplicateUser):
    code = CODE_USERNAME_TAKEN
    message = MESSAGES["user"]["username_taken"]


class EmailTaken(DuplicateUser):
    code = CODE_EMAIL_TAKEN
    message = MESSAGES["user"]["email_taken"]


class UserService:
    """User administration for one request.

    ``request`` is optional so the service is usable (and testable) outside HTTP; it
    is read only to tag log lines with the ambient request id and the acting admin.
    """

    def __init__(self, request=None):
        self._request = request

    def create_user(self, *, username: str, email: str, password: str,
                    first_name: str = "", last_name: str = "",
                    role_ids: list[int] | None = None):
        """Create one active, unprivileged user.

        Keyword-only by design: these are five same-typed strings, and a positional
        call that transposed ``username`` and ``email`` would be accepted silently.
        """
        user_model = get_user_model()
        try:
            with transaction.atomic():
                user = user_model.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                    first_name=first_name,
                    last_name=last_name,
                )
                UserProfile.objects.create(user=user)
                if role_ids is not None:
                    self._sync_roles(user, role_ids)
        except ValueError as exc:
            raise InvalidRole() from exc
        except IntegrityError as exc:
            # The transaction is already rolled back here; classify and re-raise.
            conflict = self._classify_conflict(username, email)
            if conflict is None:
                # Some other constraint failed — a NOT NULL, a column added by a
                # later migration, a genuine bug. Reporting that as "already exists"
                # would be a lie that sends an admin looking for a user who is not
                # there, so let it surface as a 500 with the traceback in the log.
                logger.exception("user creation failed on an unattributable integrity "
                                 "error username=%s %s", username, self._log_context())
                raise
            raise conflict from exc

        logger.info("user created user_id=%s username=%s %s",
                    user.pk, user.username, self._log_context())
        return user

    def list_users(self, *, page: int, page_size: int, search: str = "",
                   is_active=None, ordering: str = "username") -> tuple[list, int]:
        """One page of users, plus the total matching count.

        Args:
            page: 1-based page number, already validated.
            page_size: Already capped by the serializer — an uncapped value would
                let one caller pull the whole table in a single response.
            search: Case-insensitive substring matched against username OR email.
            is_active: Tri-state — None means "no filter", not "False".
            ordering: Already allowlisted by the serializer, so it is safe to hand
                to ``order_by``.

        Returns:
            ``(users, total)``. Exactly two queries — see ``base.paginate``.

        This method owns only what "search" and "active" MEAN for a user; the paging
        mechanics are shared with every other list endpoint in the app.
        """
        queryset = get_user_model().objects.all()
        if search:
            queryset = queryset.filter(
                Q(username__icontains=search) | Q(email__icontains=search))
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)

        return paginate(queryset, page=page, page_size=page_size, ordering=ordering,
                        only_fields=USER_LIST_FIELDS)

    def update_user(self, user_id: int, **fields):
        """Apply profile changes to one user — including, now, ``is_active`` and ``role_ids``."""
        role_ids = fields.pop("role_ids", None)
        if not fields and role_ids is None:  # the serializer rejects this; belt-and-braces for direct calls
            raise ValueError("update_user requires at least one field")

        user_model = get_user_model()
        deactivating = False
        activity_changed = False
        try:
            with transaction.atomic():
                user = user_model.objects.select_for_update().filter(pk=user_id).first()
                if user is None:
                    raise UserNotFound()
                if "is_active" in fields and fields["is_active"] != user.is_active:
                    activity_changed = True
                    deactivating = fields["is_active"] is False
                    if deactivating and is_last_active_admin(user):
                        raise LastAdminProtected()
                if fields:
                    for name, value in fields.items():
                        setattr(user, name, value)
                    user.save(update_fields=list(fields))

                if activity_changed:
                    profile, _ = UserProfile.objects.get_or_create(user=user)
                    profile.deleted_at = timezone.now() if deactivating else None
                    profile.save(update_fields=["deleted_at", "updated_at"])

                if role_ids is not None:
                    self._sync_roles(user, role_ids)
        except ValueError as exc:
            raise InvalidRole() from exc
        except IntegrityError as exc:
            # Only email carries a uniqueness rule among the updatable fields.
            conflict = self._classify_conflict(
                username="", email=fields.get("email", ""), exclude_pk=user_id)
            if conflict is None:
                logger.exception("user update failed on an unattributable integrity "
                                 "error user_id=%s %s", user_id, self._log_context())
                raise
            raise conflict from exc

        if deactivating:
            revoked = revoke_all_refresh_tokens(user_id)
            logger.info("user deactivated via update user_id=%s tokens_revoked=%s %s",
                        user_id, revoked, self._log_context())

        logger.info("user updated user_id=%s fields=%s %s",
                    user.pk, sorted(fields), self._log_context())
        return user

    def get_user(self, user_id: int):
        """One user by id, for a caller that needs the row itself.

        Raises:
            UserNotFound: no user with that id.
        """
        user = get_user_model().objects.filter(pk=user_id).only(*USER_LIST_FIELDS).first()
        if user is None:
            raise UserNotFound()
        return user

    @staticmethod
    def _sync_roles(user, role_ids: list[int]) -> None:
        """Full desired-state replacement of ``user``'s role assignments.

        Raises:
            ValueError: a ``role_ids`` entry names no active role — translated to
                the client-safe ``InvalidRole`` by the caller.
            LastAdminRoleProtected: ``user`` is the platform's last active admin
                and the new set drops the Admin role. Without this, a plain
                ``role_ids: []`` (or any list omitting it) on the last admin would
                silently strip their only RBAC role — the same outcome
                ``UserRoleService.revoke()`` already refuses one role at a time,
                which this full-replace path bypassed entirely before this check
                existed.
        """

        deduped_ids = list(dict.fromkeys(role_ids))
        valid_roles = {r.id: r for r in Role.objects.filter(id__in=deduped_ids, is_active=True)}
        unknown = [rid for rid in deduped_ids if rid not in valid_roles]
        if unknown:
            raise ValueError(f"unknown or inactive role_ids: {unknown}")

        if is_last_active_admin(user):
            admin_role = Role.objects.filter(name__iexact=ADMIN_ROLE_NAME).first()
            if admin_role is not None and admin_role.pk not in deduped_ids:
                raise LastAdminRoleProtected()

        UserRole.objects.filter(user=user).delete()
        to_create = [UserRole(user=user, role=valid_roles[rid]) for rid in deduped_ids]
        if to_create:
            UserRole.objects.bulk_create(to_create)

    @staticmethod
    def _classify_conflict(username: str, email: str,
                           exclude_pk: int | None = None) -> DuplicateUser | None:
        """Which uniqueness rule the INSERT violated, or None if it was not one.

        Determined by querying, not by parsing the driver's error text: constraint
        names and message formats differ between Postgres and sqlite, and a string
        match that silently stopped matching would misreport every conflict. These
        queries run ONLY on the failure path, so the happy path pays nothing.

        ``email__iexact`` deliberately mirrors the ``LOWER(email)`` index from
        migration 0001 — if the two ever disagree, a real conflict would be reported
        as unattributable.

        ``exclude_pk`` is the row being updated: without it an email the user already
        owns would look like a conflict with itself.

        Returns None rather than guessing: the caller turns that into a 500, because
        an unexplained constraint failure is a server problem, not a duplicate.
        """
        others = get_user_model().objects.all()
        if exclude_pk is not None:
            others = others.exclude(pk=exclude_pk)
        if username and others.filter(username=username).exists():
            return UsernameTaken()
        if email and others.filter(email__iexact=email).exists():
            return EmailTaken()
        return None

    def _log_context(self) -> str:
        """``request_id``/``actor`` suffix for the audit line — who created whom.

        There is no queryable audit trail yet (tracked as M8 in
        ``AUTH_ISSUES_BACKLOG.md``); when one lands, this is the call site that
        should write to it.
        """
        actor = getattr(getattr(self._request, "user", None), "pk", None)
        return (f"request_id={getattr(self._request, 'request_id', '')} "
                f"created_by={actor if actor is not None else '-'}")
