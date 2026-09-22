"""apps.query models — QueryLog (audit, L9) (migration_plan.md §5, §6.6).

Append-only mirror of ``query/audit_logger.py``: query text, tenant, route taken,
sub-results, terminal status, latency, parameterized SQL executed, refusal reason.
The parameterized SQL is stored as text with bind placeholders — never with
interpolated values (hard security constraint: parameterized-only).
"""
from __future__ import annotations

from django.conf import settings
from django.db import models


class TerminalStatus(models.TextChoices):
    # Frozen terminal statuses (migration_plan.md §2, §19 item 7).
    ANSWERED = "answered", "Answered"
    NO_TABLE = "no_table", "No table"
    CLARIFY = "clarify", "Clarify"
    REFUSE = "refuse", "Refuse"
    UNGROUNDED = "ungrounded", "Ungrounded"
    QUALIFIER_DROPPED = "qualifier_dropped", "Qualifier dropped"
    IR_MISMATCH = "ir_mismatch", "IR mismatch"
    INVALID = "invalid", "Invalid"
    EXEC_ERROR = "exec_error", "Exec error"


class QueryLog(models.Model):
    source = models.ForeignKey(
        "sources.Source", on_delete=models.SET_NULL, related_name="query_logs",
        null=True, blank=True,  # nullable so audit works before a Source is registered (dev)
    )
    tenant = models.CharField(max_length=128, db_index=True)
    query_text = models.TextField()
    route = models.CharField(max_length=16, blank=True)  # sql/rag/hybrid/nosql
    status = models.CharField(max_length=32, choices=TerminalStatus.choices)
    sub_results = models.JSONField(default=list, blank=True)
    executed_sql = models.TextField(blank=True)  # parameterized text only
    refusal_reason = models.TextField(blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    prompt_tokens = models.PositiveIntegerField(null=True, blank=True)
    completion_tokens = models.PositiveIntegerField(null=True, blank=True)
    total_tokens = models.PositiveIntegerField(null=True, blank=True)
    cache_hit = models.BooleanField(default=False)  # verified-query cache served this (§6.6)
    # The ONE correlation id: apps.core.middleware mints/honours X-Request-Id, the api
    # tier forwards it, and veda_core/veda/explain.py adopts it verbatim as the engine
    # trace_id. So this column joins an audit row to its full engine trace, and to the
    # `support.trace_id` the user was shown — for BOTH front doors (traceability Part 19).
    # No separate trace_id column: a second identifier for the same thing would drift.
    request_id = models.CharField(max_length=64, blank=True, db_index=True)

    # ── governance (traceability Parts 19/20) ────────────────────────────────
    # WHO ran it. Nullable + SET_NULL, deliberately:
    #   * nullable so the migration is safe on existing rows — the user of a
    #     historical query cannot be reconstructed, and inventing one would be
    #     worse than recording "unknown";
    #   * SET_NULL (never CASCADE) so deleting a user cannot delete audit history.
    # `tenant` above stays as-is: it is populated from `user.username` for an
    # authenticated caller, which made it an accidental identity proxy. This is the
    # real relationship; tenant goes back to meaning only tenancy.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="query_logs",
    )
    # Which sources ACTUALLY participated (ids of the sources that executed), as
    # opposed to `source` above which is only the primary. Answers "which sources
    # participated?" for a multi-source or federated answer.
    participating_sources = models.JSONField(default=list, blank=True)
    # Was the answer incomplete (a source did not respond, or RBAC withheld data)?
    partial = models.BooleanField(default=False)
    # Warning CODES only (veda/warnings.py's stable enum) — never the messages, which
    # are display copy and would bloat every row for no analytical gain.
    warning_codes = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "status"]),
            models.Index(fields=["created_at"]),
            # "which users hit refusals / access denials", the governance question
            # this table now has to answer (traceability Part 19).
            models.Index(fields=["user", "status"], name="querylog_user_status_idx"),
        ]

    def __str__(self) -> str:
        return f"querylog#{self.pk} [{self.status}]"
