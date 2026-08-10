"""Shared view-layer plumbing for every access-management domain.

Two things live here because every domain in this app needs the same answer to them:

  * **who may call** — staff-only, via the ``IsAdminUser`` the platform already uses
    (``apps/query/views.py``). This is also the single place a real permission check
    will replace it once the RBAC model exists, instead of a check per endpoint.
  * **how a typed service error becomes an HTTP status** — the mapping that keeps the
    service layer free of HTTP concerns.

Deliberately not a framework. ``AdminView`` holds the access rule and the one
validate-or-400 branch every endpoint repeats; everything domain-specific stays in
the domain's own module.
"""
from __future__ import annotations

import logging

from rest_framework import status
from rest_framework.permissions import IsAdminUser
from rest_framework.views import APIView

from apps.core import api

from ..gate import RequiresPermission
from ..services import AccessManagementError, ConflictError, NotFoundError

logger = logging.getLogger(__name__)

# Semantic error class -> HTTP status. Deliberately only TWO entries: every domain's
# errors subclass ``ConflictError`` or ``NotFoundError``, and inherit the right status
# through the MRO walk in ``error_status``. Listing concrete classes here instead
# would mean every new domain has to remember to register — and the first cut of the
# role endpoints proved how that fails, silently answering 400 for every conflict.
ERROR_STATUS = {
    ConflictError: status.HTTP_409_CONFLICT,
    NotFoundError: status.HTTP_404_NOT_FOUND,
}
FALLBACK_ERROR_STATUS = status.HTTP_400_BAD_REQUEST


def error_status(exc: AccessManagementError) -> int:
    """Status for a typed failure, honouring subclasses.

    Walks the MRO so a concrete error inherits its status from ``ConflictError`` or
    ``NotFoundError`` — an exact type lookup would miss every subclass and quietly
    fall back to 400, which is precisely how the role endpoints first behaved.

    Anything not descending from a mapped base falls back to 400: a service error the
    view layer has no opinion about is a client-side problem by default, and a wrong
    4xx is safer than a wrong 2xx.
    """
    for klass in type(exc).__mro__:
        if klass in ERROR_STATUS:
            return ERROR_STATUS[klass]
    return FALLBACK_ERROR_STATUS


def pagination_payload(page: int, page_size: int, total: int) -> dict:
    """The ``pagination`` block every ``<resource>/list`` response carries.

    One definition so users and roles cannot answer the same question differently —
    a client that learned to read one pager reads them all. snake_case, matching
    every other key this platform's API returns — not the frontend's camelCase
    sample, which does not match this backend's naming standard.
    """
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        # Ceiling division without importing math; 0 results -> 0 pages.
        "total_pages": (total + page_size - 1) // page_size,
        "has_next": page * page_size < total,
        "has_previous": page > 1,
    }


def log_context(request) -> str:
    """``request_id``/``actor`` suffix for audit lines — who acted, on which request."""
    actor = getattr(getattr(request, "user", None), "pk", None)
    return (f"request_id={getattr(request, 'request_id', '')} "
            f"actor={actor if actor is not None else '-'}")


class AdminView(APIView):
    """Base for the staff-only endpoints in this app.

    An anonymous caller gets 401 and an authenticated non-staff caller gets 403 —
    both decided by DRF from the authenticators configured in settings, not by
    anything written here.
    """

    #: ``IsAdminUser`` is kept ALONGSIDE the RBAC gate, not replaced by it. DRF
    #: requires every class to pass, so enforcement is strictly narrower than staff
    #: alone and there is no configuration in which adding the gate grants access that
    #: was previously refused. The old check is removed only once enforcement has run
    #: in `shadow` long enough to trust it.
    permission_classes = [IsAdminUser, RequiresPermission]

    #: The permission this endpoint requires, consulted by ``RequiresPermission``
    #: (a no-op while ``VEDA_RBAC_MODE=off``). Subclasses that leave it unset are
    #: DENIED under enforcement — fail closed, loudly.
    required_permission = None

    #: Set by each subclass.
    serializer_class = None
    #: Noun phrase used in log lines, e.g. "user creation", "role list". Carries the
    #: domain because the log format is domain-neutral.
    action = "request"

    def validate(self, request):
        """``(validated_data, None)`` on success, ``(None, response)`` on failure.

        Read-only subclasses implement ``get`` (query params — no body on a GET
        request, so parameters travel there instead); mutating subclasses
        implement ``post`` (body) — this method serves both, reading whichever
        the request actually is, so the validation logic isn't duplicated per verb.
        """
        params = request.query_params if request.method == "GET" else request.data
        serializer = self.serializer_class(data=params)
        if not serializer.is_valid():
            # serializer.errors is safe to echo (it describes the caller's own
            # submission) but must never be logged wholesale — on a password-policy
            # failure it carries the submitted password. Field NAMES only.
            logger.warning("%s rejected: invalid payload fields=%s %s",
                           self.action, sorted(serializer.errors), log_context(request))
            return None, api.invalid_payload(serializer.errors)
        return serializer.validated_data, None

    def failure(self, request, exc: AccessManagementError):
        """Render a typed service error. Only the curated ``message``/``code`` reach
        the client — never ``str(exc)`` or a traceback."""
        logger.warning("%s rejected: %s %s",
                       self.action, exc.code, log_context(request))
        return api.error(exc.message, error_status(exc), code=exc.code)
