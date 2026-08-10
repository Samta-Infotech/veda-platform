"""Blacklist every outstanding refresh token for a user — the one primitive that
ends every session an account holds.

Lives in ``apps.core`` rather than ``apps.authentication`` on purpose: it is needed
by ``apps.access_management`` too (deactivating a user must not leave their refresh
tokens usable), and having either app import the other's service module to reach one
function would tangle two bounded contexts that otherwise have no reason to know
about each other. Both import this instead — a shared leaf, not a cross-app edge.

Access tokens need no equivalent call here: they are short-lived (minutes) and, for a
DEACTIVATED user specifically, ``JWTAuthentication.get_user`` already refuses them
because it checks ``user.is_active`` on every request — that check belongs to
``rest_framework_simplejwt``, not to this project.
"""
from __future__ import annotations

from django.utils import timezone
from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken, OutstandingToken


def revoke_all_refresh_tokens(user_id) -> int:
    """Blacklist every live (non-expired) outstanding refresh token of one account.

    Already-expired tokens are skipped: they are refused by the ``exp`` check
    anyway, so blacklisting them buys no security and would make this scale with an
    account's entire history rather than its live sessions.

    ``ignore_conflicts`` makes this safe to call concurrently, and idempotent over
    tokens already blacklisted by an earlier call. Returns the number of candidate
    rows considered — not all of them are necessarily new insertions.
    """
    if not user_id:
        return 0
    outstanding = list(OutstandingToken.objects.filter(
        user_id=user_id, expires_at__gt=timezone.now()))
    BlacklistedToken.objects.bulk_create(
        [BlacklistedToken(token=row) for row in outstanding], ignore_conflicts=True)
    return len(outstanding)
