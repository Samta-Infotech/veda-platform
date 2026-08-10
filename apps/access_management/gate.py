"""Gate 2 — API authorization.

The first component that can **refuse** a request. Everything before it built and
answered the RBAC graph; this acts on the answer.

THREE MODES, ONE SETTING — ``VEDA_RBAC_MODE``
    ``off``      (default) the gate is a no-op. Behaviour is byte-identical to before
                 this module existed.
    ``shadow``   the gate decides and **logs** what it *would* have refused, then
                 allows anyway. This is how you find out what enforcement will break
                 *before* it breaks it, using real traffic.
    ``enforce``  the decision is honoured.

    Shadow is not a nicety. Flipping straight from ``off`` to ``enforce`` on a system
    where no grant has ever been exercised would deny every request from every user
    who has not been provisioned yet — which is everyone. Shadow turns that from an
    outage into a log query.

STRICTLY TIGHTER, NEVER LOOSER
    This class is added *alongside* the existing ``IsAdminUser``, never instead of it.
    DRF requires every permission class to pass, so:

      * ``off``     -> unchanged (the gate abstains)
      * ``enforce`` -> staff AND permission. Strictly narrower than staff alone.

    There is no configuration in which adding this gate grants access that was
    previously refused. That is the backward-compatibility guarantee, and it is why
    the old check is deliberately not removed yet.

FAIL CLOSED
    A view that opts into this gate without declaring what it needs is **denied**, not
    allowed. A misconfiguration must be a loud, visible failure rather than a silent
    hole — that is the whole point of the exercise.
"""
from __future__ import annotations

import logging

from django.conf import settings
from rest_framework.permissions import BasePermission

from .services import PermissionResolver

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ENFORCE = "enforce"
VALID_MODES = (MODE_OFF, MODE_SHADOW, MODE_ENFORCE)


def rbac_mode() -> str:
    """The configured mode, defaulting to ``off``.

    An unrecognised value falls back to ``off`` with an error logged, rather than
    guessing. Getting this wrong in the *other* direction — treating a typo as
    ``enforce`` — would take an entire deployment offline.
    """
    mode = getattr(settings, "VEDA_RBAC_MODE", MODE_OFF)
    if mode not in VALID_MODES:
        logger.error("VEDA_RBAC_MODE=%r is not one of %s; treating as %r",
                     mode, VALID_MODES, MODE_OFF)
        return MODE_OFF
    return mode


class RequiresPermission(BasePermission):
    """Consults the resolver for the permission a view declares.

    Views opt in by setting ``required_permission``, and optionally overriding
    ``get_required_resource(request)`` when the permission is resource-scoped::

        class SomeView(AdminView):
            required_permission = "user.manage"

    The resolver is called at most **once per request** and cached on the request
    object: several permission classes, or a later Gate 1 check, must not each pay for
    their own traversal.
    """

    #: Attribute name a view sets to declare what it needs.
    VIEW_ATTRIBUTE = "required_permission"

    def has_permission(self, request, view) -> bool:
        mode = rbac_mode()
        if mode == MODE_OFF:
            return True

        code = getattr(view, self.VIEW_ATTRIBUTE, None)
        if not code:
            # Fail closed: a view that opted in but declared nothing is a bug, and a
            # bug in an authorization gate must be loud, not permissive.
            logger.error("gate: %s uses RequiresPermission but declares no %s",
                         view.__class__.__name__, self.VIEW_ATTRIBUTE)
            return mode != MODE_ENFORCE and self._shadow(request, view, "", "", False)

        resource = self._resource_for(view, request)
        allowed = self._effective(request).allows(code, resource)

        if mode == MODE_SHADOW:
            return self._shadow(request, view, code, resource, allowed)

        if not allowed:
            logger.warning("gate: DENIED %s user_id=%s permission=%s resource=%s %s",
                           view.__class__.__name__, self._user_id(request),
                           code, resource or "(global)", self._context(request))
        return allowed

    # -- internals ----------------------------------------------------------

    def _shadow(self, request, view, code, resource, allowed) -> bool:
        """Record the decision without acting on it. Always returns True.

        Logged at WARNING only when it *would* have refused, so the shadow signal is
        the exception list rather than a copy of the access log — you can grep for it
        and get exactly the work left to do before flipping to ``enforce``.
        """
        if not allowed:
            logger.warning(
                "gate[shadow]: WOULD DENY %s user_id=%s permission=%s resource=%s %s",
                view.__class__.__name__, self._user_id(request),
                code or "(undeclared)", resource or "(global)", self._context(request))
        return True

    @staticmethod
    def _resource_for(view, request) -> str:
        """The resource this request targets, or "" when the permission is global.

        A view overrides ``get_required_resource`` when the answer depends on the
        request body — which is how a data-scoped endpoint will eventually name the
        source or table it is about.
        """
        getter = getattr(view, "get_required_resource", None)
        return getter(request) if callable(getter) else ""

    @staticmethod
    def _effective(request):
        """Resolve once per request, then reuse.

        Cached on the request object rather than in a module global: a global would
        leak one user's permissions into another's request under any concurrency.
        """
        cached = getattr(request, "_veda_effective_permissions", None)
        if cached is None:
            cached = PermissionResolver(request).resolve(getattr(request, "user", None))
            request._veda_effective_permissions = cached
        return cached

    @staticmethod
    def _user_id(request):
        return getattr(getattr(request, "user", None), "pk", None)

    @staticmethod
    def _context(request) -> str:
        return f"request_id={getattr(request, 'request_id', '')} path={request.path}"
