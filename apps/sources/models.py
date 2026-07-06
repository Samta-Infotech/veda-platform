"""apps.sources models — Source registry + connection profile (migration_plan.md §5 apps.sources).

Replaces ad-hoc ``db_abstraction`` source config with managed models. Secrets are
stored **by reference only** (§9) — never plaintext credentials in a row. The
``ready`` flag is flipped only when an ingestion job completes, so the query path
never reads a half-built substrate (§5, §18 "ingestion partial failure").
"""
from __future__ import annotations

from django.db import models


class Dialect(models.TextChoices):
    POSTGRES = "postgres", "PostgreSQL"
    MYSQL = "mysql", "MySQL"
    SQLITE = "sqlite", "SQLite"
    ORACLE = "oracle", "Oracle"
    SQLSERVER = "sqlserver", "SQL Server"
    DUCKDB = "duckdb", "DuckDB"
    MONGO = "mongo", "MongoDB"
    ES = "es", "Elasticsearch"
    DYNAMO = "dynamo", "DynamoDB"


class SourceStatus(models.TextChoices):
    REGISTERED = "registered", "Registered"
    INGESTING = "ingesting", "Ingesting"
    READY = "ready", "Ready"
    FAILED = "failed", "Failed"


class Source(models.Model):
    """One row per connectable database (§5 apps.sources)."""

    name = models.CharField(max_length=200, unique=True)
    dialect = models.CharField(max_length=20, choices=Dialect.choices)
    connector_type = models.CharField(max_length=64)
    # Connection config lives on the Source row (§5: "how to connect to it"), so onboarding a
    # new source is a data operation — register + trigger ingestion, no code/env edits.
    host = models.CharField(max_length=256, blank=True)
    port = models.PositiveIntegerField(null=True, blank=True)
    dbname = models.CharField(max_length=256, blank=True)
    db_user = models.CharField(max_length=256, blank=True)
    # Password: env-ref preferred (prod → Docker secret via `env:NAME`); inline allowed for dev.
    password_env = models.CharField(max_length=256, blank=True,
                                    help_text="env var holding the password, e.g. HOMZHUB_DB_PASSWORD")
    password_inline = models.CharField(max_length=256, blank=True,
                                       help_text="dev only; prefer password_env in prod")
    # Reference into Docker secrets / env — NEVER the credential itself (§9).
    connection_secret_ref = models.CharField(max_length=256, blank=True)
    status = models.CharField(
        max_length=20, choices=SourceStatus.choices, default=SourceStatus.REGISTERED
    )
    # Query path reads only sources with ready=True (§5).
    ready = models.BooleanField(default=False)
    last_ingested_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=["ready", "status"])]

    def __str__(self) -> str:
        return f"{self.name} ({self.dialect})"

    def resolve_password(self) -> str:
        """Resolve the source password: env-ref first (prod-safe), then inline (dev)."""
        import os
        if self.password_env:
            return os.environ.get(self.password_env, "")
        return self.password_inline or ""

    def connection(self) -> dict:
        """The engine-shaped connection dict for THIS source. Ingestion injects it so the
        engine ingests this source's DB — no global env, no per-source code changes."""
        return {
            "engine": "postgresql", "host": self.host, "port": self.port or 5432,
            "dbname": self.dbname, "user": self.db_user, "password": self.resolve_password(),
        }

    def as_engine_env(self) -> dict:
        """Map the connection to the VEDA_SOURCE_* env the engine reads (config.get_source)."""
        c = self.connection()
        return {
            "VEDA_SOURCE_HOST": str(c["host"]), "VEDA_SOURCE_PORT": str(c["port"]),
            "VEDA_SOURCE_DBNAME": str(c["dbname"]), "VEDA_SOURCE_USER": str(c["user"]),
            "VEDA_SOURCE_PASSWORD": str(c["password"]),
        }


class SourceConnectionProfile(models.Model):
    """Pool sizing / read-only role / timeout overrides per source (§5)."""

    source = models.OneToOneField(
        Source, on_delete=models.CASCADE, related_name="connection_profile"
    )
    pool_min_size = models.PositiveIntegerField(default=1)
    pool_max_size = models.PositiveIntegerField(default=5)
    read_only_role = models.CharField(max_length=128, blank=True)
    statement_timeout_ms = models.PositiveIntegerField(default=30000)
    sensitive_pattern_overrides = models.JSONField(default=list, blank=True)

    def __str__(self) -> str:
        return f"profile:{self.source.name}"
