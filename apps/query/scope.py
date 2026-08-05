"""apps.query.scope — server-side resolution of a request's query SCOPE (§6.2, P5).

The scope is the ordered list of source ids a query is allowed to run against,
primary first. It is ALWAYS resolved server-side: a client-supplied
``source_id``/``source_ids`` is treated as a *request*, intersected with the
tenant's ready-source registry, never trusted verbatim.

Extracted from ``QueryView`` (2026-08) because both HTTP entry points need it —
``/api/v1/query`` (apps.query.views.QueryView) and the chat turn endpoint
(apps.chat.views.ConversationQueryView). The chat view previously imported the
view class and called its private ``_resolve_scope`` staticmethod across an app
boundary; both now depend on this one public function instead. Behaviour is
unchanged — ``QueryView._resolve_scope`` remains as a thin delegating alias.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Last-resort source id when the ready-source registry is unreadable or empty, so
# the fail-closed context seam (§4.1) always receives a source. Dev convenience:
# in a real deployment at least one Source row is ready.
ENV_DEFAULT_SOURCE_ID = "VEDA_DEFAULT_SOURCE_ID"
_FALLBACK_DEFAULT_SOURCE_ID = "1"


def resolve_query_scope(data, tenant) -> list[int]:
    """The validated query scope (list of source ids, primary first) — §6.2, P5.

    Precedence: an explicit request ``source_ids``/``source_id`` is intersected with
    the tenant's READY sources (ownership check — never trust the body verbatim);
    absent any request pin, the default scope is ALL ready sources of the tenant.
    Falls back to ``VEDA_DEFAULT_SOURCE_ID`` when the registry can't be read /
    nothing is ready yet, so the fail-closed context seam (§4.1) always gets a source.

    Args:
        data: The raw request body (any mapping supporting ``.get``). Untrusted.
        tenant: The server-resolved tenant. Accepted for call-site symmetry and
            future per-tenant registry filtering; the ready registry is currently
            global, so it does not narrow the result today.

    Returns:
        A non-empty, de-duplicated list of source ids, primary first.

    Raises:
        ValueError: if ``VEDA_DEFAULT_SOURCE_ID`` is set to a non-integer. This is
            a deployment misconfiguration and fails loudly on purpose rather than
            silently querying an unintended source.
    """
    default_source_id = int(os.environ.get(ENV_DEFAULT_SOURCE_ID, _FALLBACK_DEFAULT_SOURCE_ID))
    ready_source_ids = _ready_source_ids()
    ready_set = set(ready_source_ids)

    requested_ids = _requested_source_ids(data)
    if requested_ids is not None:
        # Ownership: keep only ids the tenant actually owns (ready registry). If the
        # registry is unreadable, trust the explicit pin rather than fail the request.
        scope = [i for i in requested_ids if i in ready_set] if ready_set else requested_ids
        if scope:
            return list(dict.fromkeys(scope))
        logger.info("query scope: requested sources %s are not ready; falling back to "
                    "the tenant default scope (tenant=%s)", requested_ids, tenant)

    # No valid request pin → default to all ready sources (plan default), else the
    # dev fallback so inference always receives a context.
    return ready_source_ids or [default_source_id]


def _ready_source_ids() -> list[int]:
    """Ready sources, ascending. An unreadable registry (no migrations yet, DB
    down) degrades to "unknown ownership" rather than failing the request — the
    caller then trusts an explicit pin, exactly as before."""
    try:
        from apps.sources.models import Source

        return list(Source.objects.filter(ready=True).order_by("id").values_list("id", flat=True))
    except Exception:  # noqa: BLE001 — registry unreadable must not fail the query
        logger.warning("query scope: ready-source registry unreadable; "
                       "falling back to the request pin / default source", exc_info=True)
        return []


def _requested_source_ids(data) -> list[int] | None:
    """The client's requested scope, coerced to ints — or None when the request
    pinned nothing. A malformed pin yields an empty list, which callers treat as
    "no usable pin" (falling through to the default scope) rather than an error."""
    requested = data.get("source_ids")
    if requested is None and data.get("source_id") is not None:
        requested = [data.get("source_id")]
    if requested is None:
        return None
    try:
        return [int(s) for s in requested]
    except (TypeError, ValueError):
        logger.warning("query scope: ignoring malformed source pin %r", requested)
        return []
