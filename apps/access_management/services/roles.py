"""Role administration business logic.

Views stay thin: validate a body, call in here, render the outcome. Everything that
decides whether a role can be created or changed lives in this module.

Mirrors ``services/users.py`` deliberately — same typed-error contract, same
"let the database arbitrate uniqueness" stance, same transaction discipline. Two
domains that behave the same way are easier to reason about than two that each
invented their own approach.

Deliberately absent: anything about permissions or assignment. A role is a named
record here and nothing more, until those phases land.
"""
from __future__ import annotations

import logging

from django.db import IntegrityError, transaction
from django.db.models import Q
from django.utils import timezone

from apps.core.messages import MESSAGES

from ..models import Role
from .base import ConflictError, NotFoundError, paginate

logger = logging.getLogger(__name__)

CODE_ROLE_NAME_TAKEN = "ROLE_NAME_TAKEN"
CODE_ROLE_NOT_FOUND = "ROLE_NOT_FOUND"

#: Exactly the columns ``views/roles.py::public_fields`` renders. Passed to
#: ``.only()`` so the set fetched and the set projected cannot drift: adding a column
#: to the response without adding it here would cost a deferred-field query per row.
ROLE_LIST_FIELDS = ("id", "name", "description", "is_active", "created_at", "updated_at",
                    "deleted_at")


class RoleNotFound(NotFoundError):
    """No role with that id. Inherits its 404 from ``NotFoundError``."""

    code = CODE_ROLE_NOT_FOUND
    message = MESSAGES["role"]["not_found"]


class RoleNameTaken(ConflictError):
    """The name is already used by another role, compared case-insensitively.

    Inherits its 409 from ``ConflictError``.
    """

    code = CODE_ROLE_NAME_TAKEN
    message = MESSAGES["role"]["name_taken"]


class RoleService:
    """Role administration for one request.

    ``request`` is optional so the service is usable (and testable) outside HTTP; it
    is read only to tag log lines with the ambient request id and the acting admin.
    """

    def __init__(self, request=None):
        self._request = request

    def create_role(self, *, name: str, description: str = "") -> Role:
        """Create one active role.

        Keyword-only by design: two same-typed strings, and a positional call that
        transposed them would be accepted silently.

        Uniqueness is enforced by the database, not by a preceding SELECT. The obvious
        implementation — "does this name exist? no? then insert" — is a
        check-then-insert race: two concurrent requests both find nothing and both
        proceed. Letting the INSERT fail and translating the error is the only version
        that cannot create a duplicate, and it costs one query fewer on the happy path.

        Raises:
            RoleNameTaken: the name is in use (case-insensitively).
        """
        try:
            # One statement today, but wrapped so that work added later — seeding
            # default permissions onto a new role, for instance — joins the same unit
            # and a role can never be left half-created.
            with transaction.atomic():
                role = Role.objects.create(name=name, description=description)
        except IntegrityError as exc:
            conflict = self._classify_conflict(name)
            if conflict is None:
                logger.exception("role creation failed on an unattributable integrity "
                                 "error name=%s %s", name, self._log_context())
                raise
            raise conflict from exc

        logger.info("role created role_id=%s name=%s %s",
                    role.pk, role.name, self._log_context())
        return role

    def list_roles(self, *, page: int, page_size: int, search: str = "",
                   is_active=None, ordering: str = "name") -> tuple[list, int]:
        """One page of roles, plus the total matching count.

        Owns only what "search" and "active" MEAN for a role; the paging mechanics are
        shared with every other list endpoint in the app (``base.paginate``).

        Args:
            search: Case-insensitive substring matched against name OR description.
            is_active: Tri-state — None means "no filter", not "False".
        """
        queryset = Role.objects.all()
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | Q(description__icontains=search))
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)

        return paginate(queryset, page=page, page_size=page_size, ordering=ordering,
                        only_fields=ROLE_LIST_FIELDS)

    def list_active_roles(self) -> list[Role]:
        """Every active role — id and name only. For a picker/dropdown, not the
        admin table.

        Deliberately UNPAGINATED, unlike ``list_roles``: a dropdown needs every
        option in one response, and paging it would just move the "fetch every
        page" work onto every frontend that renders one. Safe specifically because
        roles are administrator-authored (see ``models/roles.py`` — "tens to
        hundreds", never per-row user data), which is the same reasoning
        ``list_roles`` itself gives for skipping an index on ``is_active``: a table
        this small costs nothing to scan in full. This would NOT be a safe pattern
        for ``users`` or ``catalog`` — both are populated by something other than an
        administrator's own typing and have no such bound.
        """
        return list(Role.objects.filter(is_active=True).order_by("name")
                    .only("id", "name"))

    def get_role(self, role_id: int) -> Role:
        """One role by id.

        Raises:
            RoleNotFound: no role with that id.
        """
        role = Role.objects.filter(pk=role_id).only(*ROLE_LIST_FIELDS).first()
        if role is None:
            raise RoleNotFound()
        return role

    def update_role(self, role_id: int, **fields) -> Role:
        """Apply changes to one role.

        Only the fields present in ``fields`` are written (``update_fields``), so a
        concurrent change to a column this request did not touch is not clobbered.

        The row is locked for the duration: read-modify-write on an unlocked row loses
        one of two concurrent updates. Cheap here — one row, one short transaction.

        Caveat worth knowing: ``select_for_update`` is a **Postgres-only** guarantee.
        SQLite reports ``has_select_for_update = False`` and Django silently ignores
        it there, so the local test suite exercises this path but does NOT prove the
        lock. Production runs Postgres, where it holds.

        Setting ``is_active=False`` is how a role is retired; there is no hard
        delete. ``deleted_at`` is stamped in the same write when that happens
        (cleared if ``is_active`` flips back) — a timestamp on the retirement
        decision, not a second one; ``is_active`` alone still decides whether the
        role grants anything.

        Args:
            role_id: Target role.
            fields: Any of ``name``, ``description``, ``is_active``.

        Raises:
            RoleNotFound: no role with that id.
            RoleNameTaken: the new name belongs to a different role.
        """
        if not fields:  # the serializer rejects this; belt-and-braces for direct calls
            raise ValueError("update_role requires at least one field")

        try:
            with transaction.atomic():
                role = Role.objects.select_for_update().filter(pk=role_id).first()
                if role is None:
                    raise RoleNotFound()
                touched = list(fields)
                if "is_active" in fields and fields["is_active"] != role.is_active:
                    role.deleted_at = None if fields["is_active"] else timezone.now()
                    touched.append("deleted_at")
                for name, value in fields.items():
                    setattr(role, name, value)
                # updated_at is auto_now, so it must be named explicitly or the
                # timestamp silently stops tracking edits.
                role.save(update_fields=[*touched, "updated_at"])
        except IntegrityError as exc:
            conflict = self._classify_conflict(fields.get("name", ""), exclude_pk=role_id)
            if conflict is None:
                logger.exception("role update failed on an unattributable integrity "
                                 "error role_id=%s %s", role_id, self._log_context())
                raise
            raise conflict from exc

        logger.info("role updated role_id=%s fields=%s %s",
                    role.pk, sorted(fields), self._log_context())
        return role

    @staticmethod
    def _classify_conflict(name: str,
                           exclude_pk: int | None = None) -> ConflictError | None:
        """Which uniqueness rule the write violated, or None if it was not one.

        A pure classifier — it queries and returns, it never raises. The caller owns
        the control flow, matching ``UserService._classify_conflict`` exactly; a
        function named "classify" that secretly re-raises is a hidden side effect.

        Determined by querying, not by parsing the driver's error text: constraint
        names and message formats differ between Postgres and sqlite, and a string
        match that silently stopped matching would misreport every conflict. The query
        runs ONLY on the failure path, so the happy path pays nothing.

        ``exclude_pk`` is the row being updated: without it, a role keeping its own
        name would look like a conflict with itself.

        Returns None when the conflict cannot be attributed: an unexplained constraint
        failure is a server problem, and reporting it as "name already taken" would
        send an administrator chasing a role that is not there.
        """
        others = Role.objects.all()
        if exclude_pk is not None:
            others = others.exclude(pk=exclude_pk)
        if name and others.filter(name__iexact=name).exists():
            return RoleNameTaken()
        return None

    def _log_context(self) -> str:
        """``request_id``/``actor`` suffix for the audit line — who changed what.

        There is no queryable audit trail yet (tracked as M8 in
        ``AUTH_ISSUES_BACKLOG.md``); when one lands, this is a call site that should
        write to it.
        """
        actor = getattr(getattr(self._request, "user", None), "pk", None)
        return (f"request_id={getattr(self._request, 'request_id', '')} "
                f"actor={actor if actor is not None else '-'}")
