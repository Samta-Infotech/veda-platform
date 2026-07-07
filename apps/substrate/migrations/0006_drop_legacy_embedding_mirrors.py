"""WP3 — drop the legacy relgt/light-text/hybrid embedding mirrors.

One embedding space (BGE-M3) means one column store (column_embeddings_bge). The
managed=False mirror models (ColumnEmbedding / ColumnEmbeddingLT / ColumnEmbeddingHybrid)
and their physical tables are removed. A clean re-ingest (WP9) recreates only the BGE
store, so DROP … IF EXISTS is safe.
"""
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('substrate', '0005_alter_graphartifact_kind'),
    ]

    operations = [
        migrations.DeleteModel(name='ColumnEmbedding'),
        migrations.DeleteModel(name='ColumnEmbeddingLT'),
        migrations.DeleteModel(name='ColumnEmbeddingHybrid'),
        migrations.RunSQL(
            sql=[
                "DROP TABLE IF EXISTS column_embeddings;",
                "DROP TABLE IF EXISTS column_embeddings_lt;",
                "DROP TABLE IF EXISTS column_embeddings_hybrid;",
            ],
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
