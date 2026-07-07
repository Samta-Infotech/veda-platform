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
    # Non-DB source kinds (review Finding 5): now representable from a Source row,
    # so document/datalake onboarding is the same pure data operation via the API.
    FILESYSTEM = "filesystem", "Filesystem documents"
    S3_DOCS = "s3_docs", "S3 documents"
    DELTA = "delta", "Delta Lake"
    PARQUET = "parquet", "Parquet"
    CSV_LAKE = "csv_lake", "CSV data lake"
    ICEBERG = "iceberg", "Iceberg"


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
    # Client-specific table exclusions (§3.1) — moved out of config.VEDA_SOURCES so no
    # client table names live in code. Framework-noise defaults (django_*/celery_*) are
    # applied by the engine scanner on top of this list.
    exclude_tables = models.JSONField(default=list, blank=True)
    # Restrict scanning to a single schema (None/"" = all public schemas).
    schema_filter = models.CharField(max_length=128, blank=True)
    # Document / datalake sources (review Finding 5): the connector-facing fields
    # (path, formats, recursion, size cap) live on the row, same as DB connections.
    source_path = models.CharField(
        max_length=512, blank=True,
        help_text="Root path/URI for document or datalake sources "
                  "(e.g. /data/contracts, s3://bucket/prefix)")
    doc_formats = models.JSONField(
        default=list, blank=True,
        help_text='Document formats to ingest, e.g. ["pdf", "docx", "md"]; '
                  "empty = connector defaults")
    doc_recursive = models.BooleanField(default=True)
    doc_max_file_mb = models.PositiveIntegerField(null=True, blank=True)
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
        """Map the connection to the VEDA_SOURCE_* env the engine reads (config.get_source).

        Also carries the client exclude_tables (JSON) + schema filter so the engine
        scanner honours THIS source's exclusions without any config-file registry (§3.1)."""
        import json
        c = self.connection()
        env = {
            "VEDA_SOURCE_ID": str(self.pk),
            "VEDA_SOURCE_HOST": str(c["host"]), "VEDA_SOURCE_PORT": str(c["port"]),
            "VEDA_SOURCE_DBNAME": str(c["dbname"]), "VEDA_SOURCE_USER": str(c["user"]),
            "VEDA_SOURCE_PASSWORD": str(c["password"]),
            "VEDA_EXCLUDE_TABLES": json.dumps(list(self.exclude_tables or [])),
        }
        if self.schema_filter:
            env["VEDA_SOURCE_SCHEMA"] = str(self.schema_filter)
        return env

    # Dialect → (engine source-type vocabulary, connector engine name) for the engine's
    # connector registry (connectors/base.register_connector "<type>:<engine>" keys).
    # All four source kinds are representable from a Source row (review Finding 5):
    # relational/nosql via the connection fields, document/datalake via source_path.
    _DIALECT_TO_ENGINE = {
        "postgres":   ("relational", "postgresql"),
        "mysql":      ("relational", "mysql"),
        "sqlite":     ("relational", "sqlite"),
        "oracle":     ("relational", "generic"),
        "sqlserver":  ("relational", "generic"),
        "duckdb":     ("relational", "generic"),
        "mongo":      ("nosql", "mongodb"),
        "es":         ("nosql", "elasticsearch"),
        "dynamo":     ("nosql", "dynamodb"),
        "filesystem": ("document", "filesystem"),
        "s3_docs":    ("document", "s3"),
        "delta":      ("datalake", "delta"),
        "parquet":    ("datalake", "parquet"),
        "csv_lake":   ("datalake", "csv"),
        "iceberg":    ("datalake", "iceberg"),
    }

    def source_kind(self) -> str:
        """The engine source-type ('relational' | 'nosql' | 'document' | 'datalake')
        this source routes to (apps.ingestion.tasks / ingestion.dispatcher)."""
        return self._DIALECT_TO_ENGINE.get(self.dialect, ("relational", "generic"))[0]

    def as_source_config(self) -> dict:
        """Engine source_config for THIS source, consumed by
        source_dispatcher.dispatch_ingestion / connectors.build_connector."""
        stype, engine = self._DIALECT_TO_ENGINE.get(self.dialect, ("relational", "generic"))
        cfg = {
            "id": str(self.pk),
            "type": stype,
            "engine": engine,
            "enabled": True,
            # documents are chunk-retrieval (RAG) sources; the rest generate queries
            "role": "searchable" if stype == "document" else "queryable",
        }
        if stype in ("document", "datalake"):
            cfg["path"] = self.source_path or ""
            if stype == "document":
                if self.doc_formats:
                    cfg["formats"] = list(self.doc_formats)
                cfg["recursive"] = bool(self.doc_recursive)
                if self.doc_max_file_mb:
                    cfg["max_file_mb"] = int(self.doc_max_file_mb)
        else:
            c = self.connection()
            cfg.update({
                "host": c["host"], "port": c["port"], "dbname": c["dbname"],
                "user": c["user"], "password": c["password"],
                "schema": self.schema_filter or None,
                "exclude_tables": list(self.exclude_tables or []),
            })
        return cfg


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
