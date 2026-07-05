"""apps.query models — QueryLog (audit, L9) (migration_plan.md §5, §6.6).

Append-only mirror of ``query/audit_logger.py``: query text, tenant, route taken,
sub-results, terminal status, latency, parameterized SQL executed, refusal reason.
The parameterized SQL is stored as text with bind placeholders — never with
interpolated values (hard security constraint: parameterized-only).
"""
from __future__ import annotations

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
    cache_hit = models.BooleanField(default=False)  # verified-query cache served this (§6.6)
    request_id = models.CharField(max_length=64, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "status"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self) -> str:
        return f"querylog#{self.pk} [{self.status}]"
