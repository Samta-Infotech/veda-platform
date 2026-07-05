"""apps.evaluation models — EvalRun / EvalCaseResult (migration_plan.md §5).

Wraps the ``evaluation/`` harnesses as tracked runs and stores the HTML report
artifact (replacing the loose ``poc_report.html``). Runs via Celery, viewable in
admin. The parity suite (Phase 7) diffs these against the Phase-0 golden baseline.
"""
from __future__ import annotations

from django.db import models


class EvalRun(models.Model):
    source = models.ForeignKey(
        "sources.Source", on_delete=models.CASCADE, related_name="eval_runs"
    )
    tenant = models.CharField(max_length=128, db_index=True)
    label = models.CharField(max_length=200, blank=True)
    recall_at_k = models.FloatField(null=True, blank=True)
    hit_rate = models.FloatField(null=True, blank=True)
    sql_success_rate = models.FloatField(null=True, blank=True)
    report_html = models.TextField(blank=True)  # stored artifact
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"evalrun#{self.pk} {self.label}"


class EvalCaseResult(models.Model):
    run = models.ForeignKey(EvalRun, on_delete=models.CASCADE, related_name="cases")
    query_id = models.CharField(max_length=32)  # e.g. D01, S03, M02, T01, A05
    query_type = models.CharField(max_length=32)
    difficulty = models.CharField(max_length=16)
    recall = models.FloatField(null=True, blank=True)
    hit = models.BooleanField(default=False)
    status = models.CharField(max_length=32, blank=True)
    details = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [models.Index(fields=["run", "query_id"])]
