"""apps.core.api — the platform's HTTP response envelope, in one place.

Every DRF endpoint in this project answers in the same shape::

    {"status_code": 200, "message": "...", "data": {...}}          # success
    {"status_code": 400, "message": "Invalid request data.",         # bad body
     "errors": {"field": ["..."]}}
    {"status_code": 401, "message": "...", "code": "..."}            # failure

That shape used to be re-typed as a literal at every call site — six times in
``apps/chat/views.py``, twice more in ``apps/authentication/views.py``, and it was
about to be typed a ninth time by ``apps/access_management``. Each copy was a
chance for one endpoint to drift (a missing ``status_code``, a ``null`` where a key
should have been absent) and for a frontend to special-case it. These three
functions are that shape's single definition; ``apps.core`` is where they live
because it is the app the whole platform already depends on for shared framework
pieces (base models, request-id middleware, the settings bridge).

**Absent, not null.** Optional parts of the envelope are *omitted* when not
supplied, never emitted as ``null`` — a client that checks ``"data" in body`` must
keep working, and ``logout`` deliberately answers with no ``data`` at all.

These helpers only shape the payload; they make no decision about *what* to
return. Choosing the status and the copy stays with the view, which is the layer
that knows the HTTP semantics.
"""
from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response

# Copy for a request whose body failed serializer validation. Shared so the same
# words reach every endpoint — clients do (regrettably but observably) match on it.
MSG_INVALID_PAYLOAD = "Invalid request data."


def success(message: str, data=None, status_code: int = status.HTTP_200_OK) -> Response:
    """A success envelope.

    Args:
        message: Human-facing summary, e.g. ``"User created successfully."``.
        data: Payload body. Omitted from the response entirely when ``None`` —
            pass ``{}`` if an empty object is genuinely wanted.
        status_code: HTTP status, also mirrored into the body.
    """
    body = {"status_code": status_code, "message": message}
    if data is not None:
        body["data"] = data
    return Response(body, status=status_code)


def error(message: str, status_code: int, code: str | None = None, data=None) -> Response:
    """A failure envelope.

    Args:
        message: Safe, user-facing copy. Never a raw exception, never a traceback.
        status_code: HTTP status, also mirrored into the body.
        code: Stable machine-readable error code (e.g. ``"INVALID_CREDENTIALS"``)
            so clients branch on it instead of on prose. Omitted when not given.
        data: Extra context a client needs to act on the failure. Omitted when not
            given.
    """
    body = {"status_code": status_code, "message": message}
    if code is not None:
        body["code"] = code
    if data is not None:
        body["data"] = data
    return Response(body, status=status_code)


def iso_z(value) -> str | None:
    """A datetime as ``YYYY-MM-DDTHH:MM:SSZ``, or None.

    One timestamp format for the whole API. DRF's own renderer would emit
    microseconds and a ``+00:00`` offset, which does not match the ``Z`` form the
    chat endpoints have always returned — so every endpoint routes through here
    instead of each picking its own. ``apps/chat/views.py::_iso_z`` delegates to this.
    """
    return value.strftime("%Y-%m-%dT%H:%M:%SZ") if value else None


def human_date(value) -> str:
    """A datetime as ``"Jul 16, 2026"`` — a human-facing display date, or ``""``.

    NOT a replacement for ``iso_z()``: that format is the machine-facing contract
    every existing endpoint's timestamps already use (sortable, timezone-explicit,
    parseable without a locale). This one is for a field a UI renders directly in a
    table — a display string, never meant to be parsed back. Both live here so a
    third format never gets invented ad hoc at a call site; a view picks the one
    that matches what it's building.

    Returns ``""`` rather than ``None`` for a missing value — an empty table cell
    reads better than the literal text "None" in a UI that renders this directly
    without a null-check.
    """
    return value.strftime("%b %d, %Y") if value else ""


def invalid_payload(errors) -> Response:
    """The 400 for a malformed body, carrying DRF's own field-error mapping.

    Field errors are safe to echo: they describe the caller's own submission
    ("this field is required"). They must never be used to report on *state* the
    caller cannot see — whether a username is taken, whether an account exists —
    which is why uniqueness and credential failures are reported through
    ``error()`` with a deliberately generic message instead.
    """
    return Response(
        {"status_code": status.HTTP_400_BAD_REQUEST, "message": MSG_INVALID_PAYLOAD,
         "errors": errors},
        status=status.HTTP_400_BAD_REQUEST,
    )
