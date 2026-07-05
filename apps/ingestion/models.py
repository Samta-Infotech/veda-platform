"""apps.ingestion models — IngestionJob / IngestionStage (migration_plan.md §5, §7).

Tracks a full L0 run per source and each stage's status, checkpoint, row counts,
errors, and timing — making ingestion resumable and observable in admin. Resume
restarts from the last incomplete stage; stage 6 (embeddings) additionally
records a batch checkpoint for mid-stage resume (§4.2a).
"""
from __future__ import annotations

from django.db import models


class JobStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    RUNNING = "running", "Running"
    SUCCESS = "success", "Success"
    FAILED = "failed", "Failed"


class IngestionJob(models.Model):
    source = models.ForeignKey(
        "sources.Source", on_delete=models.CASCADE, related_name="ingestion_jobs"
    )
    tenant = models.CharField(max_length=128, db_index=True)
    status = models.CharField(
        max_length=16, choices=JobStatus.choices, default=JobStatus.PENDING
    )
    encoder_mode = models.CharField(max_length=32, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"job#{self.pk} {self.source_id}/{self.tenant} [{self.status}]"


class IngestionStage(models.Model):
    """One row per L0 stage of a job (the ten stages of §7)."""

    job = models.ForeignKey(
        IngestionJob, on_delete=models.CASCADE, related_name="stages"
    )
    order = models.PositiveSmallIntegerField()
    name = models.CharField(max_length=64)
    status = models.CharField(
        max_length=16, choices=JobStatus.choices, default=JobStatus.PENDING
    )
    row_count = models.BigIntegerField(default=0)
    # Mid-stage resume for the batched embedding stage (§4.2a).
    batch_checkpoint = models.JSONField(default=dict, blank=True)
    error_traceback = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["job", "order"], name="uq_ingestionstage_order"
            )
        ]
        ordering = ["order"]

    def __str__(self) -> str:
        return f"{self.job_id}:{self.order}:{self.name} [{self.status}]"
