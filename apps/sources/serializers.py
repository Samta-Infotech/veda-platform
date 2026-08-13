"""Serializers and helpers for the Data Sources list API.

Returns exactly what is in the database (``Source`` table). When the table is
empty — common in dev/demo where sources are still configured via env
(``veda_core/config.py``) — the API falls back to building a single item from
the injected source config so the frontend always has something to render.
"""
from __future__ import annotations

from rest_framework import serializers
from apps.core import api
from apps.sources.models import Source, SourceStatus


def dialect_to_source_type(dialect: str) -> str:
    """Map a Source.dialect value to a UI-friendly source_type label."""
    d = (dialect or "").lower()
    if d in ("postgres", "mysql", "sqlite", "oracle", "sqlserver", "duckdb"):
        return "DATABASE"
    elif d in ("mongo", "es", "dynamo"):
        return "DATABASE"
    elif d in ("delta", "parquet", "csv_lake", "iceberg"):
        return "DATALAKE"
    elif d in ("filesystem", "s3_docs"):
        return "FILE_SYSTEM"
    return "DATABASE"


# api_contract.md §5.2/§5.6: metadata.db_type is an uppercase engine label
# ("POSTGRESQL", "MYSQL", ...), distinct from source_type (DATABASE/DATALAKE/
# FILE_SYSTEM) — one Source.dialect value maps to exactly one db_type.
_DIALECT_TO_DB_TYPE = {
    "postgres": "POSTGRESQL", "mysql": "MYSQL", "sqlite": "SQLITE",
    "oracle": "ORACLE", "sqlserver": "SQLSERVER", "duckdb": "DUCKDB",
    "mongo": "MONGODB", "es": "ELASTICSEARCH", "dynamo": "DYNAMODB",
    "filesystem": "FILESYSTEM", "s3_docs": "S3", "delta": "DELTA",
    "parquet": "PARQUET", "csv_lake": "CSV_LAKE", "iceberg": "ICEBERG",
}


def dialect_to_db_type(dialect: str) -> str:
    return _DIALECT_TO_DB_TYPE.get((dialect or "").lower(), (dialect or "").upper())


# api_contract.md §5.2 documents exactly two statuses (CONNECTED/ERROR) — a
# source only ever exists in this table once §5.6's pre-create connection
# validation has passed, so REGISTERED/INGESTING (this app's own in-between
# states, before `ready` flips) are surfaced as ERROR with a status_message
# that says why, rather than inventing a third status the contract disallows.
def _status_and_message(source: Source) -> tuple[str, str | None]:
    if source.ready:
        return "CONNECTED", None
    if source.status == SourceStatus.FAILED:
        return "ERROR", "The connection could not be established."
    if source.status == SourceStatus.INGESTING:
        return "ERROR", "Ingestion is in progress; this source is not yet queryable."
    return "ERROR", "This source is registered but has not completed its initial connection yet."


class DataSourceListSerializer(serializers.Serializer):
    # No status filter: this list only ever returns connected sources (view
    # filters to ready=True), and no page/page_size: it's a small, admin-curated
    # set, never large enough to need paging (see views.py's module docstring).
    source_type = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    search = serializers.CharField(required=False, allow_blank=True, allow_null=True, default="")


# Only surface the fields that actually mean something for the source_type —
# a DATALAKE/FILE_SYSTEM source has no host/port/username, and showing those
# as empty strings buries the field the caller actually needs (source_path)
# in noise. Keyed by source_type so a new dialect just needs a new entry in
# dialect_to_source_type(), not a new metadata branch here.
def _metadata_for(source: Source, source_type: str) -> dict:
    if source_type == "DATABASE":
        return {
            "db_type": dialect_to_db_type(source.dialect),
            "host": source.host,
            "port": source.port,
            "database": source.dbname,
            "username": source.db_user,
            "schema_filter": source.schema_filter,
        }
    if source_type == "DATALAKE":
        return {
            "db_type": dialect_to_db_type(source.dialect),
            "source_path": source.source_path,
        }
    # FILE_SYSTEM
    return {
        "path": source.source_path,
        "doc_formats": source.doc_formats,
        "doc_recursive": source.doc_recursive,
        "doc_max_file_mb": source.doc_max_file_mb,
    }


def serialize_source(source: Source) -> dict:
    """Serialize a Django ``Source`` model instance per api_contract.md §5.2."""
    source_type = dialect_to_source_type(source.dialect)
    status_str, status_message = _status_and_message(source)

    return {
        "source_id": source.pk,
        "source_type": source_type,
        "name": source.name,
        "status": status_str,
        "is_connected": source.ready,
        "metadata": _metadata_for(source, source_type),
        "last_checked_at": api.iso_z(source.last_ingested_at),
        "status_message": status_message,
        "created_at": api.iso_z(source.created_at),
        "updated_at": api.iso_z(source.updated_at),
    }


def _serialize_config_source(cfg: dict) -> dict:
    """Serialize a source dict from ``veda_core/config.py`` (env-injected).

    This is the fallback when the Django ``Source`` table is empty — the engine
    was onboarded via env vars, not through the admin API yet.
    """
    engine = (cfg.get("engine") or "").lower()
    src_type = cfg.get("type", "relational")

    if src_type == "relational":
        source_type = "DATABASE"
    elif src_type in ("datalake", "delta", "parquet"):
        source_type = "DATALAKE"
    elif src_type in ("document", "filesystem"):
        source_type = "FILE_SYSTEM"
    else:
        source_type = "DATABASE"

    if source_type == "DATABASE":
        metadata = {
            "db_type": dialect_to_db_type(engine),
            "host": cfg.get("host", ""),
            "port": cfg.get("port"),
            "database": cfg.get("dbname", ""),
            "username": cfg.get("user", ""),
            "schema_filter": cfg.get("schema", "") or "",
        }
    elif source_type == "DATALAKE":
        metadata = {
            "db_type": dialect_to_db_type(engine),
            "source_path": cfg.get("source_path", ""),
        }
    else:  # FILE_SYSTEM
        metadata = {"path": cfg.get("source_path", "")}

    return {
        "source_id": cfg.get("id", "config_source"),
        "source_type": source_type,
        "name": cfg.get("dbname") or cfg.get("id", "primary_db"),
        "status": "CONNECTED",
        "is_connected": True,
        "metadata": metadata,
        "last_checked_at": None,
        "status_message": None,
        "created_at": None,
        "updated_at": None,
    }


# User's call: group by category (database/datalake/file_system) instead of a
# flat items array. "database" and "file_system" hold exactly ONE source each
# (the first CONNECTED one of that type) — V1 assumes single-source-per-type
# for those. "datalake" is the one category confirmed to already have more
# than one connected source live (invoices_csv + catalog_parquet), so it's a
# LIST, not a single object, to avoid silently dropping the second one.
_SINGLE_TYPE_TO_KEY = {"DATABASE": "database", "FILE_SYSTEM": "file_system"}
_LIST_TYPE_TO_KEY = {"DATALAKE": "datalake"}


def _shape(item: dict) -> dict:
    metadata = {"connection_id": item["source_id"], **item["metadata"]}
    return {
        "status": "Connected" if item["is_connected"] else "Not Connected",
        "is_connected": item["is_connected"],
        "metadata": metadata,
    }


def group_by_type(items: list[dict]) -> dict:
    grouped: dict = {}

    for source_type, key in _SINGLE_TYPE_TO_KEY.items():
        first = next((i for i in items if i["source_type"] == source_type), None)
        grouped[key] = (_shape(first) if first is not None
                        else {"status": "Not Connected", "is_connected": False, "metadata": {}})

    for source_type, key in _LIST_TYPE_TO_KEY.items():
        grouped[key] = [_shape(i) for i in items if i["source_type"] == source_type]

    return grouped


def get_config_sources() -> list[dict]:
    """Try to load configured sources from ``veda_core/config.py`` env fallback."""
    try:
        from veda_core.config import get_enabled_sources
        sources = get_enabled_sources()
        return [_serialize_config_source(s) for s in sources]
    except Exception:
        return []
