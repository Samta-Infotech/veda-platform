"""storage_adapters/assembler.py — SemanticModelAssembler (migration plan §8a).

The deterministic engine consumes ONE nested `sm` dict (pipeline.py:17, 346,
553, 563, 613, 631, 660, ...), not the normalized substrate rows. Today
`veda_hybrid._load_semantic_model()` loads that dict whole from
`data/veda_semantic_model.json`; in the platform the same information lives in
the Django-owned `Sm*` substrate models, so this Builder reconstructs the
byte-identical dict from those rows — and `persist()` is the inverse the writer
uses to store what ingestion produced.

Parity gate (§8a, Phase 3 exit 6): for the home schema, `assemble(source, tenant)`
must be deep-equal to the legacy `veda_semantic_model.json` (ordering normalized
before diffing). The assembled dict is cached in Redis + process memory keyed by
`substrate_version` (bumped on ingestion completion, broadcast via §8.4).
"""
from __future__ import annotations

from typing import Any, Tuple

# sm top-level keys, in the legacy file's order.
_SM_KEYS = ("version", "tables", "columns", "retrieval_documents", "domain_synonyms", "concept_graph")


def persist(sm: dict, *, source_id: int, tenant: str, version: str | None = None) -> None:
    """Persist a semantic-model dict into the normalized `Sm*` substrate.

    Idempotent per (source, tenant): replaces the scope's rows. This is what the
    ingestion writer calls with the semantic_layer_v2 output (§7); inverse of assemble.
    """
    from django.db import transaction

    from apps.substrate.models import (
        SmColumn, SmConcept, SmRetrievalDoc, SmSynonym, SmTable, SubstrateVersion,
    )

    scope = dict(source_id=source_id, tenant=tenant)
    with transaction.atomic():
        for model in (SmColumn, SmConcept, SmRetrievalDoc, SmSynonym, SmTable):
            model.objects.all_tenants().filter(**scope).delete()

        SmTable.objects.bulk_create(
            [SmTable(name=k, payload=v, **scope) for k, v in sm.get("tables", {}).items()]
        )
        SmColumn.objects.bulk_create(
            [SmColumn(key=k, payload=v, **scope) for k, v in sm.get("columns", {}).items()]
        )
        SmRetrievalDoc.objects.bulk_create(
            [SmRetrievalDoc(key=k, payload=v, **scope)
             for k, v in sm.get("retrieval_documents", {}).items()]
        )
        SmSynonym.objects.bulk_create(
            [SmSynonym(phrase=k, mappings=v, **scope)
             for k, v in sm.get("domain_synonyms", {}).items()]
        )
        SmConcept.objects.bulk_create(
            [SmConcept(name=k, payload=v, **scope)
             for k, v in sm.get("concept_graph", {}).items()]
        )
        SubstrateVersion.objects.update_or_create(
            source_id=source_id, tenant=tenant,
            defaults=dict(version=version or "1", sm_version=str(sm.get("version", "1.0"))),
        )


def sm_redis_key(source_id: int, tenant: str) -> str:
    return f"veda:sm:{source_id}:{tenant}"


def publish_sm(source_id: int, tenant: str, redis_url: str | None = None) -> int:
    """Assemble the `sm` from substrate and publish it to `redis-cache` so the
    inference tier (Django-free) can load it without touching the ORM (§8a
    "cached in Redis + inference process memory"). Returns the byte length written.
    """
    import json
    import os

    import redis as _redis

    sm, _cols = SemanticModelAssembler.assemble(source_id, tenant)
    payload = json.dumps(sm)
    url = redis_url or os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
    client = _redis.Redis.from_url(url)
    client.set(sm_redis_key(source_id, tenant), payload)
    return len(payload)


def rehydrate_channel(source_id: int, tenant: str, scope: str = "all") -> str:
    return f"veda:rehydrate:{source_id}:{tenant}:{scope}"


def publish_rehydrate(source_id: int, tenant: str, scope: str = "all") -> int:
    """Publish a rehydrate message on the redis-cache pub/sub channel (§8.4) so every
    inference replica reloads the named scope (FK/glossary/KG/verified-cache/sm) and
    bumps its in-memory substrate_version. Returns the number of subscribers reached."""
    import json
    import os

    import redis as _redis

    url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
    client = _redis.Redis.from_url(url)
    msg = json.dumps({"source_id": source_id, "tenant": tenant, "scope": scope})
    return client.publish(rehydrate_channel(source_id, tenant, scope), msg)


class SemanticModelAssembler:
    """Builder: normalized `Sm*` rows -> the `sm` read-model (§8a)."""

    @staticmethod
    def assemble(source_id: int, tenant: str) -> Tuple[dict, list]:
        """Return `(sm, all_cols)` for `(source_id, tenant)`, rebuilt from substrate."""
        from apps.substrate.models import (
            SmColumn, SmConcept, SmRetrievalDoc, SmSynonym, SmTable, SubstrateVersion,
        )

        scope = dict(source_id=source_id, tenant=tenant)
        sv = SubstrateVersion.objects.all_tenants().filter(**scope).first()

        sm: dict[str, Any] = {
            "version": sv.sm_version if sv else "1.0",
            "tables": {r.name: r.payload for r in SmTable.objects.all_tenants().filter(**scope)},
            "columns": {r.key: r.payload for r in SmColumn.objects.all_tenants().filter(**scope)},
            "retrieval_documents": {
                r.key: r.payload for r in SmRetrievalDoc.objects.all_tenants().filter(**scope)
            },
            "domain_synonyms": {
                r.phrase: r.mappings for r in SmSynonym.objects.all_tenants().filter(**scope)
            },
            "concept_graph": {
                r.name: r.payload for r in SmConcept.objects.all_tenants().filter(**scope)
            },
        }
        all_cols = list(sm["columns"].keys())
        return sm, all_cols
