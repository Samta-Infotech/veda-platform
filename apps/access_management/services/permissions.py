"""Permission catalogue access — read-only by design.

There is no ``create_permission`` or ``update_permission``, and that absence is the
feature: only code can enforce a permission, so the catalogue is seeded by migration
(``0004_seed_permissions``) and this service only reads it. A permission an
administrator invented at runtime would be a row no gate ever checks.

Mirrors ``services/roles.py`` in shape — same typed errors, same paging primitive —
minus every write path.
"""
from __future__ import annotations

from django.db.models import Q

from apps.core.messages import MESSAGES

from ..codes import PermissionCode
from ..models import Permission
from .base import NotFoundError, paginate

CODE_PERMISSION_NOT_FOUND = "PERMISSION_NOT_FOUND"

#: Excluded from the dropdown/picker ONLY (``permissions/list`` still shows every
#: permission — this is a UI-picker concern, not a catalogue-visibility one).
#: ``data.read`` is resource-scoped: an admin never picks it standalone, they grant
#: it implicitly by ALLOW/DENY-ing a resource in the catalog tree (see
#: ``roles/permissions/grant``). Showing it in a flat picker meant for
#: non-resource-scoped permissions (``user.manage``, ``role.manage``, ...) would
#: invite granting it with no resource_path, which does nothing (a blank-path
#: ``data.read`` grant never covers any resource — see ``resolver.py``'s module
#: docstring).
_HIDDEN_FROM_DROPDOWN = frozenset({PermissionCode.DATA_READ})

#: Exactly the columns ``views/permissions.py::public_fields`` renders, passed to
#: ``.only()`` so the set fetched and the set projected cannot drift.
PERMISSION_LIST_FIELDS = (
    "id", "code", "name", "description", "is_active", "created_at", "updated_at",
)


class PermissionNotFound(NotFoundError):
    """No permission with that id. Inherits its 404 from ``NotFoundError``."""

    code = CODE_PERMISSION_NOT_FOUND
    message = MESSAGES["permission"]["not_found"]


class PermissionService:
    """Read access to the permission catalogue for one request.

    ``request`` is accepted for symmetry with the other services (and so a future
    audit hook has the actor); nothing here writes, so it is currently unused beyond
    that.
    """

    def __init__(self, request=None):
        self._request = request

    def list_permissions(self, *, page: int, page_size: int, search: str = "",
                         is_active=None, ordering: str = "code") -> tuple[list, int]:
        """One page of permissions, plus the total matching count.

        Owns only what "search" and "active" MEAN for a permission; the paging
        mechanics are shared with every other list endpoint (``base.paginate``).

        Args:
            search: Case-insensitive substring matched against code OR name.
                Description is excluded deliberately — it is prose, and matching it
                would make a search for "read" return most of the catalogue.
            is_active: Tri-state — None means "no filter", not "False".
        """
        queryset = Permission.objects.all()
        if search:
            queryset = queryset.filter(
                Q(code__icontains=search) | Q(name__icontains=search))
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)

        return paginate(queryset, page=page, page_size=page_size, ordering=ordering,
                        only_fields=PERMISSION_LIST_FIELDS)

    def get_permission(self, permission_id: int) -> Permission:
        """One permission by id.

        Raises:
            PermissionNotFound: no permission with that id.
        """
        permission = (Permission.objects.filter(pk=permission_id)
                      .only(*PERMISSION_LIST_FIELDS).first())
        if permission is None:
            raise PermissionNotFound()
        return permission

    def list_active_permissions(self) -> list[Permission]:
        """Every active, non-resource-scoped permission — for a picker/dropdown in
        the admin UI. See ``_HIDDEN_FROM_DROPDOWN`` for why ``data.read`` is
        excluded here specifically (and only here — ``list_permissions`` still
        shows it)."""
        return list(Permission.objects.filter(is_active=True)
                    .exclude(code__in=_HIDDEN_FROM_DROPDOWN)
                    .order_by("code")
                    .only("id", "code", "name", "description"))

