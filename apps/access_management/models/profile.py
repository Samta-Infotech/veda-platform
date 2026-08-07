"""The one piece of user metadata that does not belong on ``auth_user`` itself.

WHY NOT A COLUMN ON ``auth_user``
    That table is Django's own — migration 0001's raw-SQL index already exists only
    because a real DB constraint had to live somewhere and Django gives no other
    way to add one to a model it owns. A business-data column is different: adding
    one directly risks colliding with a future Django release that adds a
    same-named field of its own, and there is a clean alternative that doesn't run
    that risk — a related table, exactly the pattern ``django.contrib.auth`` itself
    expects extensions to use.

WHY ``deleted_at`` IS NOT A SECOND LIFECYCLE FLAG
    ``is_active`` stays the single source of truth for "can this account do
    anything" — the login guard, the last-admin protection, ``JWTAuthentication``
    all key off it, and duplicating that decision into a second flag is how the two
    would eventually disagree. ``deleted_at`` answers a narrower, purely
    informational question: *when* did the account become inactive. It is written
    exactly when ``UserService.update_user`` flips ``is_active`` to False, and
    cleared when it flips back — never set directly.

NULLABLE PROFILE ROW
    Not every ``User`` has one yet — this table is added after users already
    exist. ``get_or_create`` at the two points that need it (``update_user``, the
    list projection) backfills lazily rather than requiring a data migration to
    touch every historical row.
"""
from __future__ import annotations

from django.conf import settings
from django.db import models

from apps.core.models import TimeStampedModel


class UserProfile(TimeStampedModel):
    """Metadata about a user that ``auth_user`` has no column for.

    ``updated_at`` comes from ``TimeStampedModel`` and reflects the last change to
    THIS row (right now: only ``deleted_at`` toggling) — not a general "last edited
    anything about this user" signal, since profile fields today and the auth_user
    columns are written independently.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")

    #: Set when ``is_active`` flips to False, cleared when it flips back. Never
    #: written directly — see the module docstring for why this is a timestamp on
    #: a decision, not a second decision.
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "User profile"
        verbose_name_plural = "User profiles"

    def __str__(self) -> str:
        return f"profile for user_id={self.user_id}"
