"""query/source_capabilities.py — Source Capability model (Phase A1, foundation only).

Phase A1 of the Source Capability & Execution Adapter Foundation
(docs/architecture/VEDA_SOURCE_CAPABILITY_ADAPTER_AUDIT.md). Pure, additive, read-only-in-effect:
this module has NO call sites anywhere else in the codebase yet. It does not wire into routing,
planning, execution, fast_path, SQL generation, or the alignment layer. Importing it changes
nothing; it exists so later phases (adapter interface, relational wrap) have a typed capability
model to build on instead of the ad-hoc source_type strings scattered across the query tier
(CandidateSource.source_type, federated_route.py's "postgres"/"parquet", etc. — see the audit's
§3 duplication analysis for why those are NOT touched here).

`source_kind` values here are exactly apps/sources/models.py::Source.source_kind()'s four
strings ("relational" | "nosql" | "document" | "datalake"). This module never re-derives kind
from a dialect or connection string — callers pass the already-resolved kind string (e.g. from
CandidateSource.source_type or a source profile dict), which itself ultimately traces back to
Source.source_kind() on the Django side. Keeping ONE derivation path is deliberate: the audit
found three independent kind-keyed dispatch layers and at least three source-kind string
vocabularies already coexisting; this module must not become a fourth.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class SourceCapability(str, Enum):
    """What a source can DO, not what product/vendor it is. Each value below is backed by either
    an existing connectors/base.py `supports_*` flag, or a concrete, evidenced query-tier code path
    found during the audit — nothing here is speculative. See the audit's §4 for the evidence
    behind each value; deliberately excludes vector_search/entity_extraction/api_request/
    file_metadata_search, which the audit found no evidence for in this codebase today."""

    STRUCTURED_QUERY = "structured_query"        # backs connectors/base.py supports_query
    SCHEMA_DISCOVERY = "schema_discovery"        # backs connectors/base.py supports_schema
    DOCUMENT_RETRIEVAL = "document_retrieval"    # backs connectors/base.py supports_chunks
    AGGREGATION = "aggregation"                  # fast_path.py / generation.py SQL aggregation
    FILTERING = "filtering"                      # fast_path.py / value_filter.py WHERE-clause support
    JOINING = "joining"                          # join_planner.py (intra-source); DuckDB ATTACH for cross-source
    FEDERATION = "federation"                    # can participate in a federated_executor.py DuckDB plan


@dataclass(frozen=True)
class SourceCapabilities:
    """A source kind's declared capability set. Immutable — callers must not construct ad-hoc
    variants; use `capabilities_for()` so the mapping stays in one place."""

    source_kind: str
    capabilities: FrozenSet[SourceCapability]

    def has(self, capability: SourceCapability) -> bool:
        return capability in self.capabilities


# One profile per Source.source_kind() value. Evidence for each is in the audit doc's §4/§6/§8:
# - relational: full 10-stage ingestion pipeline, fast_path, semantic-layer SQL-expression metrics.
# - datalake: DuckDB-catalog execution (cross_source_composer.py), query-time value sampling
#   (datalake_values.py). JOINING is present but execution-path-limited (only the hardcoded
#   postgres/parquet semi-join today) — not a general multi-way join; recorded as present anyway
#   since the capability describes intent-level capability, not today's federation-code coverage.
# - document: FilesystemDocumentConnector + run_rag_layer. No STRUCTURED_QUERY/AGGREGATION/JOINING —
#   confirmed absent; the relational semantic layer cannot represent a document (audit §9).
# - nosql: connector exists (connectors/nosql.py) and is query-time-live (veda_hybrid.py::_run_nosql).
#   Schema-flexible structured query and filtering are supported; JOINING/FEDERATION are not
#   evidenced for nosql in this codebase and are left out rather than assumed.
_PROFILES: dict = {
    "relational": frozenset({
        SourceCapability.STRUCTURED_QUERY,
        SourceCapability.SCHEMA_DISCOVERY,
        SourceCapability.AGGREGATION,
        SourceCapability.FILTERING,
        SourceCapability.JOINING,
        SourceCapability.FEDERATION,
    }),
    "datalake": frozenset({
        SourceCapability.STRUCTURED_QUERY,
        SourceCapability.SCHEMA_DISCOVERY,
        SourceCapability.AGGREGATION,
        SourceCapability.FILTERING,
        SourceCapability.JOINING,
        SourceCapability.FEDERATION,
    }),
    "document": frozenset({
        SourceCapability.DOCUMENT_RETRIEVAL,
        SourceCapability.SCHEMA_DISCOVERY,
    }),
    "nosql": frozenset({
        SourceCapability.STRUCTURED_QUERY,
        SourceCapability.SCHEMA_DISCOVERY,
        SourceCapability.FILTERING,
    }),
}

_UNKNOWN_KIND_CAPABILITIES: FrozenSet[SourceCapability] = frozenset()


def capabilities_for(source_kind: str) -> SourceCapabilities:
    """Look up the capability set for a source kind. An unrecognized kind gets an EMPTY capability
    set (safe default — no capability is assumed, never a guess), not an exception; this mirrors
    Source.source_kind()'s own fallback-to-"relational" behavior being a DB-side concern, not this
    module's. Callers that need to distinguish "unknown kind" from "kind with no capabilities"
    should check `source_kind in KNOWN_SOURCE_KINDS` themselves."""
    return SourceCapabilities(
        source_kind=source_kind,
        capabilities=_PROFILES.get(source_kind, _UNKNOWN_KIND_CAPABILITIES),
    )


KNOWN_SOURCE_KINDS: FrozenSet[str] = frozenset(_PROFILES.keys())
