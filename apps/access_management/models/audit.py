"""Authorization decision audit (traceability Part 21).

WHY THIS EXISTS
    Until now an authorization denial left exactly one artifact: a
    ``logger.warning`` line in ``apps/access_management/gate.py``. That is not
    queryable, not retained past log rotation, and cannot answer the three
    questions an operator actually asks:

        How many queries were denied?
        Which users experienced access denials?
        Was a denial caused by a MISSING permission, or by an explicit DENY policy?

    The third one matters most and is the one a log line loses entirely — "not
    allowed" and "explicitly forbidden" need completely different remediation
    ("grant them the role" vs "the deny is intentional, the request is wrong").
    ``EffectivePermissions`` already distinguishes them (``denies()`` is
    deliberately separate from ``not allows()``); this records that distinction.

WHAT IS DELIBERATELY *NOT* STORED
    **The resource path.** ``db:crm_postgres:employee:salary`` names the schema,
    table and column of something the caller was just told they may not see —
    writing it into a durable table, on every denial, would build exactly the
    catalogue of restricted names the denial exists to protect. Only the coarse
    ``resource_kind`` (``db``/``nosql``/``files``/``lake``) is kept, which is
    enough to slice denials by source family and answers all three questions
    above. The full path stays in the request-scoped debug log where it is
    already available to an operator who genuinely needs it.

    Also not stored: any credential, any grant payload, any query text (that
    lives in ``QueryLog``, joined by ``request_id``).

DENIALS ONLY
    Allows are not recorded. A row per permitted request would be enormous and
    adds nothing — the denominator for "how many were denied" comes from
    ``QueryLog``, joined on ``request_id``. Shadow-mode "would have denied"
    decisions ARE recorded (with ``mode="shadow"``), because the entire point of
    shadow mode is to enumerate the work remaining before enforcing.

APPEND-ONLY
    Nothing in the codebase updates or deletes these rows; the admin registers it
    read-only. Retention is an ops decision, not a code one.
"""
from __future__ import annotations

from django.conf import settings
from django.db import models


class Decision(models.TextChoices):
    ALLOW = "allow", "Allow"
    DENY = "deny", "Deny"


class DenialReason(models.TextChoices):
    """WHY a decision came out the way it did — the actionable half of the record."""

    #: An explicit DENY grant matched. Intentional policy; the request is wrong.
    EXPLICIT_DENY = "explicit_deny", "Explicit deny grant"
    #: No grant reached the resource at all. Usually "this role needs the permission".
    NO_GRANT = "no_grant", "No matching grant"
    #: The view opted into the gate but declared no permission — a code bug, and the
    #: gate fails closed on it. Recorded so it is visible rather than silently denying.
    UNDECLARED = "undeclared", "View declared no permission"
    #: Resolution itself failed; the gate failed closed rather than guessing.
    RESOLUTION_ERROR = "resolution_error", "Permission resolution failed"


class AuthorizationDecision(models.Model):
    """One authorization decision worth keeping. See the module docstring."""

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    #: WHO. Nullable + SET_NULL for the same reasons as ``QueryLog.user``: an
    #: unauthenticated denial has no user, and deleting a user must never delete
    #: the audit history of what they were denied.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="authorization_decisions",
    )

    #: The permission code checked, e.g. "data.read" (apps.access_management.codes).
    action = models.CharField(max_length=64, db_index=True)

    #: COARSE resource family only — "db" | "nosql" | "files" | "lake", or "" for a
    #: permission that is not resource-scoped. Never the resource path (see docstring).
    resource_kind = models.CharField(max_length=16, blank=True)

    decision = models.CharField(max_length=8, choices=Decision.choices)
    reason_code = models.CharField(max_length=32, choices=DenialReason.choices, blank=True)

    #: "enforce" | "shadow" — a shadow row is a WOULD-have-denied, not a real denial,
    #: and mixing the two would make a shadow rollout look like an outage.
    mode = models.CharField(max_length=8, blank=True)

    #: Joins to QueryLog.request_id and to the engine trace_id (one id end to end).
    request_id = models.CharField(max_length=64, blank=True, db_index=True)

    #: Which gate refused, for locating the check in code. A class name, not a path.
    view_name = models.CharField(max_length=128, blank=True)

    class Meta:
        indexes = [
            # "which users experienced denials, most recent first"
            models.Index(fields=["user", "created_at"], name="authzdec_user_created_idx"),
            # "how many denials, by reason, over a window"
            models.Index(fields=["decision", "reason_code"], name="authzdec_dec_reason_idx"),
        ]
        ordering = ("-created_at",)
        verbose_name = "Authorization decision"
        verbose_name_plural = "Authorization decisions"

    def __str__(self) -> str:
        target = self.resource_kind or "(global)"
        return f"authzdecision#{self.pk} {self.decision} {self.action} on {target}"
