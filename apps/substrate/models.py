"""apps.substrate models — everything ingestion produces (migration_plan.md §6).

1:1 with ARCHITECTURE.md §4 (ingestion) and §11 (stores). Nothing dropped,
nothing invented. Structured rows are Django models here; the six embedding
tables (§6.4) are pgvector tables created via a raw-SQL migration with HNSW
indexes — the ORM cannot express ``embedding <=> %s::vector`` — and are mirrored
below by ``managed = False`` models purely for admin visibility. ANN search runs
as raw SQL in ``storage_adapters.reader`` (§4 seam), never through the ORM.

Field names deliberately track the ingestion function outputs so
``storage_adapters.writer`` maps cleanly (§2.3 exit criterion 4).
"""
from __future__ import annotations

from django.db import models

from apps.core.models import TenantScopedModel


# ─────────────────────────────────────────────────────────────────────────────
# §6.1 Structural / schema substrate
# ─────────────────────────────────────────────────────────────────────────────
class SchemaTable(TenantScopedModel):
    """schema_scanner → ScanResult. `id` (UUID) matches ingestion's per-table UUID."""

    name = models.CharField(max_length=256)
    row_count = models.BigIntegerField(null=True, blank=True)
    display_column = models.CharField(max_length=256, blank=True)
    is_sensitive = models.BooleanField(default=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["source", "tenant", "name"], name="uq_schematable_natural"
            )
        ]
        indexes = [models.Index(fields=["source", "tenant", "name"])]

    def __str__(self) -> str:
        return self.name


class SemanticTypeChoice(models.TextChoices):
    MONETARY = "MONETARY", "Monetary"
    TEMPORAL = "TEMPORAL", "Temporal"
    CATEGORICAL = "CATEGORICAL", "Categorical"
    IDENTIFIER = "IDENTIFIER", "Identifier"
    FLAG = "FLAG", "Flag"
    TEXT = "TEXT", "Text"


class SchemaColumn(TenantScopedModel):
    """schema_scanner column output (sensitive columns excluded at ingestion)."""

    table = models.ForeignKey(
        SchemaTable, on_delete=models.CASCADE, related_name="columns"
    )
    name = models.CharField(max_length=256)
    data_type = models.CharField(max_length=128)
    is_pk = models.BooleanField(default=False)
    is_fk = models.BooleanField(default=False)
    semantic_type = models.CharField(
        max_length=20, choices=SemanticTypeChoice.choices, blank=True
    )
    confidence = models.FloatField(null=True, blank=True)
    review_flag = models.BooleanField(default=False)
    excluded = models.BooleanField(default=False)  # sensitive → excluded

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["table", "name"], name="uq_schemacolumn_natural"
            )
        ]
        indexes = [models.Index(fields=["source", "tenant", "table"])]

    def __str__(self) -> str:
        return f"{self.table.name}.{self.name}"


class JoinType(models.TextChoices):
    INNER = "INNER", "Inner"
    LEFT = "LEFT", "Left"
    RIGHT = "RIGHT", "Right"
    FULL = "FULL", "Full"


class FkEdge(TenantScopedModel):
    """vector_store.store_fk_adjacency + data_graph — the join engine's source of truth.

    Authoritative for join inference (compiler / join_planner / graph_guard).
    ``data_graph``'s undeclared FKs (overlap ≥ 0.70) are merged into the same
    table with ``is_declared=False`` (§6.1 callout). Loaded into a Redis hash +
    in-process adjacency map at inference startup.
    """

    from_table = models.ForeignKey(
        SchemaTable, on_delete=models.CASCADE, related_name="fk_out"
    )
    from_col = models.ForeignKey(
        SchemaColumn, on_delete=models.CASCADE, related_name="fk_out_cols"
    )
    to_table = models.ForeignKey(
        SchemaTable, on_delete=models.CASCADE, related_name="fk_in"
    )
    to_col = models.ForeignKey(
        SchemaColumn, on_delete=models.CASCADE, related_name="fk_in_cols"
    )
    join_type = models.CharField(
        max_length=8, choices=JoinType.choices, default=JoinType.INNER
    )
    is_declared = models.BooleanField(default=True)
    overlap_score = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=["source", "tenant", "from_table"])]


class TableMetadata(TenantScopedModel):
    """table_metadata store — display column + notes."""

    table = models.OneToOneField(
        SchemaTable, on_delete=models.CASCADE, related_name="metadata"
    )
    display_column = models.CharField(max_length=256, blank=True)
    notes = models.TextField(blank=True)


# ─────────────────────────────────────────────────────────────────────────────
# §6.2 Semantic / language substrate
# ─────────────────────────────────────────────────────────────────────────────
class SemanticType(TenantScopedModel):
    """semantic_type_inference (3-layer)."""

    column = models.ForeignKey(
        SchemaColumn, on_delete=models.CASCADE, related_name="semantic_types"
    )
    type = models.CharField(max_length=20, choices=SemanticTypeChoice.choices)
    confidence = models.FloatField(null=True, blank=True)
    layer_hit = models.CharField(max_length=64, blank=True)
    review_flag = models.BooleanField(default=False)


class Provenance(models.TextChoices):
    LLM = "LLM", "LLM"
    RULE = "RULE", "Rule"
    DECLARED = "DECLARED", "Declared"


class GlossaryEntry(TenantScopedModel):
    """domain_glossary / glossary_builder."""

    term = models.CharField(max_length=256)
    canonical = models.CharField(max_length=256, blank=True)
    definition = models.TextField(blank=True)
    provenance = models.CharField(
        max_length=16, choices=Provenance.choices, default=Provenance.LLM
    )
    scope = models.CharField(max_length=128, blank=True)

    class Meta:
        indexes = [models.Index(fields=["source", "tenant", "term"])]


class Synonym(TenantScopedModel):
    """glossary_builder synonyms."""

    term = models.CharField(max_length=256)
    synonym = models.CharField(max_length=256)
    weight = models.FloatField(default=1.0)

    class Meta:
        indexes = [models.Index(fields=["source", "tenant", "term"])]


class SyntheticPair(TenantScopedModel):
    """synthetic_query_gen — NL/IR pairs for optional fine-tune."""

    nl_text = models.TextField()
    ir_json = models.JSONField()
    target_column = models.CharField(max_length=256, blank=True)
    used_for_finetune = models.BooleanField(default=False)


class SemanticConcept(TenantScopedModel):
    """semantic/registry (compiled) — concept/dimension/metric share one shape."""

    KIND = models.TextChoices("Kind", "CONCEPT DIMENSION METRIC")
    kind = models.CharField(max_length=16, choices=KIND.choices)
    name = models.CharField(max_length=256)
    definition = models.TextField(blank=True)
    mapping_json = models.JSONField(default=dict, blank=True)
    manifest_version = models.CharField(max_length=64, blank=True)

    class Meta:
        indexes = [models.Index(fields=["source", "tenant", "kind", "name"])]


# ─────────────────────────────────────────────────────────────────────────────
# §6.3 Value-grounding substrate
# ─────────────────────────────────────────────────────────────────────────────
class ColumnValueSample(TenantScopedModel):
    """value_sampler — used by value grounding (Gate L6a) + arbitration.

    Also mirrored to a Redis SET (vg:{source}:{tenant}:{column_uuid}) so
    validation.value_grounding is O(1) set-membership, not a table scan (§6.3).
    """

    column = models.ForeignKey(
        SchemaColumn, on_delete=models.CASCADE, related_name="value_samples"
    )
    value = models.TextField()
    freq = models.BigIntegerField(default=0)
    sampled_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["column"])]


class ColumnProfile(TenantScopedModel):
    """data_profiler — distinct/null/min/max + top-N."""

    column = models.OneToOneField(
        SchemaColumn, on_delete=models.CASCADE, related_name="profile"
    )
    distinct_count = models.BigIntegerField(null=True, blank=True)
    top_n_json = models.JSONField(default=list, blank=True)
    null_ratio = models.FloatField(null=True, blank=True)
    min_value = models.TextField(blank=True)
    max_value = models.TextField(blank=True)


# ─────────────────────────────────────────────────────────────────────────────
# §6.4 Vector / embedding substrate — pgvector tables (managed=False mirrors)
#
# The real tables + HNSW cosine indexes are created by a RunSQL migration with
# the tuned m / ef_construction from §7.1a. These mirror models exist ONLY for
# admin visibility; ANN search is raw SQL in storage_adapters.reader.
# ─────────────────────────────────────────────────────────────────────────────
class ChunkEmbedding(models.Model):
    chunk_id = models.UUIDField()
    source_id = models.IntegerField()
    tenant = models.TextField()

    class Meta:
        managed = False
        db_table = "chunk_embeddings"  # 1024-dim doc chunks (RAG substrate)


# ─────────────────────────────────────────────────────────────────────────────
# §6.5 Graph substrate (unified knowledge graph)
# ─────────────────────────────────────────────────────────────────────────────
class GraphNode(TenantScopedModel):
    """unified_graph_builder / relationship_graph — schema+chunk nodes."""

    node_key = models.CharField(max_length=512)
    node_type = models.CharField(max_length=64)
    payload = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [models.Index(fields=["source", "tenant", "node_type"])]


class GraphEdge(TenantScopedModel):
    """Typed edges for query-time expansion (suggest_expansions)."""

    from_node = models.ForeignKey(
        GraphNode, on_delete=models.CASCADE, related_name="edges_out"
    )
    to_node = models.ForeignKey(
        GraphNode, on_delete=models.CASCADE, related_name="edges_in"
    )
    edge_type = models.CharField(max_length=64)
    weight = models.FloatField(default=1.0)


class GraphNodeEmbedding(models.Model):
    """graph_embedder node embeddings — pgvector, managed=False mirror."""

    node_id = models.UUIDField()
    source_id = models.IntegerField()
    tenant = models.TextField()

    class Meta:
        managed = False
        db_table = "graph_node_embeddings"


class GraphArtifact(TenantScopedModel):
    """Registers a persisted graph-artifact file path + version (§6.5).

    Currently the relationship graph (``kind="relationship_graph"``, read by the
    query path's join_planner). The former Kùzu graph-DB backend was removed.
    """

    path = models.CharField(max_length=1024)
    version = models.CharField(max_length=64)
    kind = models.CharField(max_length=32, default="relationship_graph")


# ─────────────────────────────────────────────────────────────────────────────
# §6.6 Verified-query cache (query-time WRITE — inference's one documented write)
# ─────────────────────────────────────────────────────────────────────────────
class VerifiedQueryCache(TenantScopedModel):
    """veda/cache.py (file-based cosine ≥ 0.85) → Postgres + pgvector lookup.

    Written on the query hot path via INSERT … ON CONFLICT (query_hash) DO NOTHING
    (idempotent under N replicas), then fanned out to peers via pub/sub (§8.4).
    Skip rules preserved verbatim in the writer: existence / fast-path / temporal
    answers are NEVER cached (§6.6 callout). ``query_embedding`` is a pgvector
    column added by the RunSQL migration (not expressible in the ORM).
    """

    query_hash = models.CharField(max_length=64)
    query_text = models.TextField()
    verified_sql = models.TextField()
    columns_json = models.JSONField(default=list, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["source", "tenant", "query_hash"], name="uq_verifiedcache_hash"
            )
        ]
        indexes = [models.Index(fields=["source", "tenant", "query_hash"])]


# ─────────────────────────────────────────────────────────────────────────────
# §8a Semantic-model substrate — the normalized rows the SemanticModelAssembler
# reconstructs the `sm` read-model from. The engine consumes one nested `sm` dict
# (pipeline.py); here it is stored per-entity so Django owns it, and the assembler
# rebuilds the byte-identical dict (deep-equality parity gate, §8a).
#
# The rich per-entity attributes are LLM/rule-derived free-form fields, so they
# live in a JSON `payload` rather than 15 narrow columns — the normalization is
# per column / retrieval-doc / synonym / concept / table, which is what makes
# Django the owner and the assembler a faithful Builder.
# ─────────────────────────────────────────────────────────────────────────────
class SubstrateVersion(TenantScopedModel):
    """Monotonic version bumped when an ingestion job completes; invalidates the
    assembled `sm` cache and drives the rehydrate fan-out (§8.4, §8a)."""

    version = models.CharField(max_length=64)
    sm_version = models.CharField(max_length=16, default="1.0")  # sm["version"]
    # P7/Q-10: per-source HNSW ef_search tuned at L5 by source size, served at query
    # time via VEDA_HNSW_EF_SEARCH_<source_id> (reader._resolve_ef_search) instead of
    # one global value. 40 == the shipped default (§7.1a-tuned).
    hnsw_ef_search = models.PositiveIntegerField(default=40)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["source", "tenant"], name="uq_substrateversion_scope"
            )
        ]


class SmTable(TenantScopedModel):
    """sm["tables"][name] — one row per table."""

    name = models.CharField(max_length=256)
    payload = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["source", "tenant", "name"], name="uq_smtable")
        ]


class SmColumn(TenantScopedModel):
    """sm["columns"]["table.column"] — one row per column (rich payload)."""

    key = models.CharField(max_length=512)  # "table.column"
    payload = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["source", "tenant", "key"], name="uq_smcolumn")
        ]
        indexes = [models.Index(fields=["source", "tenant", "key"])]


class SmRetrievalDoc(TenantScopedModel):
    """sm["retrieval_documents"]["table.column"] — one row per retrieval doc."""

    key = models.CharField(max_length=512)
    payload = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["source", "tenant", "key"], name="uq_smretrdoc")
        ]


class SmSynonym(TenantScopedModel):
    """sm["domain_synonyms"][phrase] — phrase → mapping (list or value)."""

    phrase = models.CharField(max_length=512)
    mappings = models.JSONField(default=list)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["source", "tenant", "phrase"], name="uq_smsynonym")
        ]
        indexes = [models.Index(fields=["source", "tenant", "phrase"])]


class SmConcept(TenantScopedModel):
    """sm["concept_graph"][name] — one row per concept."""

    name = models.CharField(max_length=256)
    payload = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["source", "tenant", "name"], name="uq_smconcept")
        ]
