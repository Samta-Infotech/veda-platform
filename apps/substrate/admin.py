"""apps.substrate admin — every substrate model listed, scoped by source+tenant (§2.3 exit 3).

Admin uses ``all_tenants()`` (the escape hatch) so staff can inspect across tenants;
the tenant-scoped default manager governs the request/query paths.
"""
from django.contrib import admin

from . import models

_TENANT_MODELS = [
    models.SchemaTable,
    models.SchemaColumn,
    models.FkEdge,
    models.TableMetadata,
    models.SemanticType,
    models.GlossaryEntry,
    models.Synonym,
    models.SyntheticPair,
    models.SemanticConcept,
    models.ColumnValueSample,
    models.ColumnProfile,
    models.GraphNode,
    models.GraphEdge,
    models.GraphArtifact,
    models.VerifiedQueryCache,
]

for _m in _TENANT_MODELS:
    admin.site.register(_m)
