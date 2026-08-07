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

from apps.core.messages import MESSAGES
from apps.core.token_revocation import revoke_all_refresh_tokens

from ..models import UserProfile
from .admin_guard import LastAdminProtected, is_last_active_admin
from .base import ConflictError, NotFoundError, paginate

logger = logging.getLogger(__name__)

CODE_USERNAME_TAKEN = "USERNAME_TAKEN"
CODE_EMAIL_TAKEN = "EMAIL_TAKEN"
CODE_USER_CONFLICT = "USER_CONFLICT"
CODE_USER_NOT_FOUND = "USER_NOT_FOUND"

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
                    first_name: str = "", last_name: str = ""):
        """Create one active, unprivileged user.

        Keyword-only by design: these are five same-typed strings, and a positional
        call that transposed ``username`` and ``email`` would be accepted silently.

        Uniqueness is enforced by the database, not by a preceding SELECT. The
        obvious implementation — "does this username exist? no? then insert" — is a
        check-then-insert race: two concurrent requests both find nothing and both
        proceed. Letting the INSERT fail and translating the error is the only
        version that cannot create a duplicate, and it costs one query fewer on the
        happy path.

        Args:
            username: Validated, unique-by-constraint. Stored as submitted.
            email: Validated address. Unique case-insensitively among non-blank
                values, via the index added in migration 0001.
            password: Plaintext, already policy-checked. Hashed by ``create_user``
                and never logged.
            first_name: Optional.
            last_name: Optional.

        Returns:
            The saved ``User``.

        Raises:
            UsernameTaken / EmailTaken: the corresponding value is in use.
            DuplicateUser: a uniqueness violation we could not attribute to a
                specific field (never guessed at).
        """
        user_model = get_user_model()
        try:
            # One statement, but wrapped so that any future work added to user
            # creation (role assignment is the next phase) joins the same unit —
            # a user must never be left half-created.
            with transaction.atomic():
                user = user_model.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                    first_name=first_name,
                    last_name=last_name,
                )
                UserProfile.objects.create(user=user)
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
        """Apply profile changes to one user — including, now, ``is_active``.

        Only the fields present in ``fields`` are written (``update_fields``), so a
        concurrent change to a column this request did not touch is not clobbered.

        The row is locked for the duration: read-modify-write on an unlocked row loses
        one of two concurrent updates. Cheap here — one row, one short transaction.

        Caveat worth knowing: ``select_for_update`` is a **Postgres-only** guarantee.
        SQLite reports ``has_select_for_update = False`` and Django silently ignores
        it there, so the local test suite exercises this path but does NOT prove the
        lock. Production runs Postgres, where it holds.

        ``is_active`` is one field among several here, not a separate endpoint —
        deactivating a user is a profile edit like any other, and giving it its own
        route would mean guarding "last admin" and "revoke tokens" in two places
        instead of one. Both still apply: turning ``is_active`` off is refused for
        the platform's last active admin (checked BEFORE anything is written, so a
        refused request never partially applies), and blacklists every live refresh
        token on success — same primitive ``AuthService.change_password`` uses, so a
        deactivated account cannot mint a fresh access token even though the ones
        already issued keep working until they expire (``JWTAuthentication.get_user``
        refuses those directly, on every request).

        Args:
            user_id: Target user.
            fields: Any of ``email``, ``first_name``, ``last_name``, ``is_active``.

        Returns:
            The saved ``User``.

        Raises:
            UserNotFound: no user with that id.
            EmailTaken: the new email belongs to someone else.
            LastAdminProtected: ``is_active=False`` and this is the platform's only
                active admin.
        """
        if not fields:  # the serializer rejects this; belt-and-braces for direct calls
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
                for name, value in fields.items():
                    setattr(user, name, value)
                user.save(update_fields=list(fields))

                if activity_changed:
                    # get_or_create: a user created before migration 0008 has no
                    # profile row yet — backfilled here rather than by a data
                    # migration that would have to touch every historical user.
                    profile, _ = UserProfile.objects.get_or_create(user=user)
                    profile.deleted_at = timezone.now() if deactivating else None
                    profile.save(update_fields=["deleted_at", "updated_at"])
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
