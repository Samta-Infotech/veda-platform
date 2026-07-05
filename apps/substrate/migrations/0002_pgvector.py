"""pgvector tables + HNSW indexes (migration_plan.md §2.4, §6.4, §7.1a).

The ORM cannot express ``vector(N)`` columns, HNSW indexes, or ``<=> ::vector``
search, so the six embedding tables (plus graph-node embeddings and the verified
cache's query embedding) are created here with raw SQL. The managed=False mirror
models in ``models.py`` give admin visibility over these same tables.

HNSW build params (m, ef_construction) come from Django settings (bridged from
config.py, §9) so §7.1a can retune them reversibly — the shipping index IS the
gated index. Dims are pinned per §6.4 / config.py:
  column_embeddings, column_embeddings_lt, relgt_structural = 256
  column_embeddings_hybrid                                  = 640
  column_embeddings_bge, chunk_embeddings, graph_node_embeddings,
  verifiedquerycache.query_embedding                        = 1024
"""
from django.conf import settings
from django.db import migrations

_M = settings.VEDA["HNSW_M"]
_EFC = settings.VEDA["HNSW_EF_CONSTRUCTION"]

# (table, id_column, dim) — key column matches the managed=False mirror models.
_EMBED_TABLES = [
    ("column_embeddings", "column_uuid", 256),
    ("column_embeddings_lt", "column_uuid", 256),
    ("column_embeddings_hybrid", "column_uuid", 640),
    ("column_embeddings_bge", "column_uuid", 1024),
    ("relgt_structural", "column_uuid", 256),
    ("chunk_embeddings", "chunk_id", 1024),
    ("graph_node_embeddings", "node_id", 1024),
]


def _create(table: str, id_col: str, dim: int) -> str:
    idx = f"{table}_hnsw"
    return (
        f'CREATE TABLE IF NOT EXISTS "{table}" ('
        f'  {id_col} uuid NOT NULL,'
        f'  source_id integer NOT NULL,'
        f'  tenant text NOT NULL,'
        f'  embedding vector({dim}) NOT NULL,'
        f'  PRIMARY KEY ({id_col}, source_id, tenant)'
        f');'
        f' CREATE INDEX IF NOT EXISTS "{idx}" ON "{table}"'
        f'  USING hnsw (embedding vector_cosine_ops)'
        f'  WITH (m = {_M}, ef_construction = {_EFC});'
    )


def _drop(table: str) -> str:
    return f'DROP TABLE IF EXISTS "{table}" CASCADE;'


_forward = "\n".join(_create(t, c, d) for t, c, d in _EMBED_TABLES)
_reverse = "\n".join(_drop(t) for t, _, _ in _EMBED_TABLES)

# Verified-query cache: add the pgvector query_embedding column + HNSW index
# (cosine ≥ 0.85 replay lookup, §6.6). Table is the ORM-created substrate table.
_VQC = "substrate_verifiedquerycache"
_forward_vqc = (
    f'ALTER TABLE "{_VQC}" ADD COLUMN IF NOT EXISTS query_embedding vector(1024);'
    f' CREATE INDEX IF NOT EXISTS "{_VQC}_qemb_hnsw" ON "{_VQC}"'
    f'  USING hnsw (query_embedding vector_cosine_ops)'
    f'  WITH (m = {_M}, ef_construction = {_EFC});'
)
_reverse_vqc = (
    f'DROP INDEX IF EXISTS "{_VQC}_qemb_hnsw";'
    f' ALTER TABLE "{_VQC}" DROP COLUMN IF EXISTS query_embedding;'
)


class Migration(migrations.Migration):
    dependencies = [("substrate", "0001_initial")]

    operations = [
        migrations.RunSQL(_forward, reverse_sql=_reverse),
        migrations.RunSQL(_forward_vqc, reverse_sql=_reverse_vqc),
    ]
