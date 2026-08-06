"""Shared service-layer contract for every access-management domain.

Two things live here, and only because more than one domain genuinely needs them:
the error base class, and the paging primitive. Domain-specific logic stays in
``services/users.py`` / ``services/roles.py`` — a ``base`` module that accumulates
one domain's helpers is how shared modules rot.
"""
from __future__ import annotations


class AccessManagementError(Exception):
    """An expected failure with a safe client rendering.

    Subclasses set ``code`` (stable, machine-readable) and ``message`` (safe,
    user-facing copy). The HTTP status is chosen by the view layer
    (``views/base.py::error_status``), so services stay free of HTTP concerns.

    The view renders these two class attributes and never ``str(exc)``, so a
    subclass raised with a debugging detail — ``raise UserNotFound(f"id={pk}")`` —
    cannot leak that detail to a client.
    """

    code = "ACCESS_MANAGEMENT_ERROR"
    message = "The request could not be completed."


class NotFoundError(AccessManagementError):
    """The addressed record does not exist. Rendered as 404.

    Domains subclass this (``UserNotFound``, ``RoleNotFound``) rather than being
    listed in a status registry. That is what keeps the view layer open for extension
    and closed for modification: a new domain's "not found" gets the right status by
    inheriting, with nothing central to remember to update — and forgetting is not a
    theoretical risk, it is exactly how the first cut of the role endpoints ended up
    answering 400 for every conflict.
    """

    code = "NOT_FOUND"
    message = "The requested record does not exist."


class ConflictError(AccessManagementError):
    """The change collides with an existing record. Rendered as 409.

    Same reasoning as ``NotFoundError``: inheritance, not registration.
    """

    code = "CONFLICT"
    message = "That change conflicts with an existing record."


def paginate(queryset, *, page: int, page_size: int, ordering: str,
             only_fields: tuple[str, ...] | None = None) -> tuple[list, int]:
    """One page of ``queryset``, plus the total matching count.

    Args:
        queryset: Already filtered by the caller — this function does not know or
            care what "search" means for a given domain.
        page: 1-based, already validated. A page past the end yields an empty list,
            not an error.
        page_size: Already capped by the serializer; an uncapped value would let one
            caller pull the whole table in a single response.
        ordering: Already allowlisted by the serializer, so it is safe to hand to
            ``order_by``. Passing user input straight through here would let a caller
            traverse relations or trigger a 500.
        only_fields: Restrict the columns fetched, so a list page never drags secrets
            or wide columns out of the database in the first place.

    Returns:
        ``(rows, total)``. Exactly two queries: one COUNT, one page fetch.

    The secondary sort on ``pk`` is not cosmetic: without it, rows that tie on the
    sort key can reappear on the next page or be skipped entirely, so paging through
    a list would silently miss records.

    On the COUNT: it is the expensive half on a large table, and it is kept because a
    client cannot render a pager without a total. If a table here ever reaches
    millions of rows, switch that domain to keyset pagination rather than dropping
    the count silently.
    """
    total = queryset.count()
    rows = queryset.order_by(ordering, "pk")
    if only_fields:
        rows = rows.only(*only_fields)
    offset = (page - 1) * page_size
    return list(rows[offset:offset + page_size]), total
