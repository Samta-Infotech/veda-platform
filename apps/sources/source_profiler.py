"""apps.sources.source_profiler — post-ingestion source profiling (multi-source routing, Phase 1.3/1.4).

Generates a grounded, business-facing ``description`` for a just-ingested source so the
query-time routing coordinator has something to reason over besides raw connection details.

Design decisions (see ``docs/multisource_routing/MEMORY.md``):
- **Django-side, no boundary violation.** After the ingestion warm stage, observed schema is
  already synced into Django substrate models (``apps.substrate.SchemaTable``/``SchemaColumn``),
  so the profiler reads those — it never imports ``veda_core`` or re-queries the engine store.
- **Deterministic / zero-hallucination.** The description is composed from real observed table and
  column names + their semantic types. No SLM call in v1 — nothing is invented. (An SLM-enriched
  variant can be added later as an opt-in.)
- **Manual wins.** A human-entered ``description`` (``description_generated=False``) is never
  overwritten. The profiler only fills a blank description or refreshes one it previously generated.
- **Domain/canonical are NOT auto-inferred.** ``domain_tags`` and ``is_canonical`` are business
  judgments left to a human; the profiler never sets them.

Flag-gated default-OFF (``SOURCE_PROFILER_ENABLED``) so prod stays byte-identical until opted in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.conf import settings

from apps.sources.models import Source
from apps.sources.serializers import dialect_to_source_type

logger = logging.getLogger(__name__)


# Human-readable capability phrases per observed semantic type — what the source is good FOR.
# Grounded: only emitted when a column of that type was actually observed.
_SEMANTIC_CAPABILITY = {
    "MONETARY": "financial and numeric aggregation",
    "TEMPORAL": "time-based filtering and trend analysis",
    "CATEGORICAL": "filtering and grouping",
    "IDENTIFIER": "record lookups and joins",
    "FLAG": "boolean/status filtering",
    "TEXT": "text lookups",
}


@dataclass
class ProfileResult:
    source_id: object
    updated: bool = False
    reason: str = ""
    description: str = ""
    table_count: int = 0
    capabilities: list = field(default_factory=list)


def _is_enabled() -> bool:
    return bool(getattr(settings, "SOURCE_PROFILER_ENABLED", False))


def _describe_tabular(source: Source, tenant: str) -> tuple[str, int, list]:
    """Compose a grounded description from observed tables/columns in Django substrate.

    Returns (description, table_count, capabilities). Empty description when no schema was
    observed (e.g. a document source, or substrate not yet synced)."""
    from apps.substrate.models import SchemaTable, SchemaColumn

    tables = list(
        SchemaTable.objects.filter(source_id=source.pk, tenant=tenant)
        .order_by("-row_count", "name")
        .values_list("name", flat=True)[:200]
    )
    if not tables:
        return "", 0, []

    sem_types = list(
        SchemaColumn.objects.filter(source_id=source.pk, tenant=tenant, excluded=False)
        .exclude(semantic_type="")
        .values_list("semantic_type", flat=True)
        .distinct()
    )
    caps = [_SEMANTIC_CAPABILITY[t] for t in _SEMANTIC_CAPABILITY if t in sem_types]

    sample = ", ".join(tables[:8])
    more = f" and {len(tables) - 8} more" if len(tables) > 8 else ""
    parts = [f"Structured data across {len(tables)} table(s): {sample}{more}."]
    if caps:
        parts.append("Suitable for " + ", ".join(caps) + ".")
    return " ".join(parts), len(tables), caps


def _describe_document(source: Source) -> str:
    """Grounded description for a document/filesystem source (no relational schema)."""
    where = source.source_path or "the configured location"
    fmts = ", ".join(source.doc_formats) if source.doc_formats else "supported document formats"
    return (f"Document source at {where} ({fmts}). Suitable for retrieval and grounded "
            f"question answering over its documents.")


def profile_source(source_id, tenant: str = "default") -> ProfileResult:
    """Generate + persist a grounded description for one just-ingested source.

    Respects manual-wins: only fills a blank description or refreshes a previously
    auto-generated one; never touches domain_tags/is_canonical. Idempotent.
    """
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist:
        return ProfileResult(source_id=source_id, reason="source_not_found")

    # Manual-wins: a human-entered description is authoritative — never overwrite it.
    if source.description and not source.description_generated:
        return ProfileResult(source_id=source_id, reason="manual_description_kept",
                             description=source.description)

    source_type = dialect_to_source_type(source.dialect)
    if source_type == "FILE_SYSTEM":
        description, table_count, caps = _describe_document(source), 0, []
    else:
        description, table_count, caps = _describe_tabular(source, tenant)
        if not description:
            # Tabular source with no observed schema yet (substrate not synced) — skip
            # rather than write a hollow description.
            return ProfileResult(source_id=source_id, reason="no_observed_schema")

    Source.objects.filter(pk=source_id).update(
        description=description, description_generated=True)
    logger.info("source profiled source_id=%s tables=%s caps=%s", source_id, table_count, caps)
    return ProfileResult(source_id=source_id, updated=True, reason="generated",
                        description=description, table_count=table_count, capabilities=caps)


def profile_source_if_enabled(source_id, tenant: str = "default") -> None:
    """Flag-gated, never-raises entry point for the ingestion hook. A profiling failure
    must never fail an ingestion job that already succeeded (mirrors _sync_catalog_if_enabled)."""
    if not _is_enabled():
        return
    try:
        profile_source(source_id, tenant=tenant)
    except Exception:
        logger.exception("source profiling failed source_id=%s — ingestion job still "
                         "succeeded; description left unset", source_id)
