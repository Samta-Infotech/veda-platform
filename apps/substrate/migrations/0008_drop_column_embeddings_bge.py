"""WP3 — drop the unused column_embeddings_bge mirror.

storage_adapters.writer.store_column_embeddings() (the intended sync from the
engine's column_embeddings_v2 into this Django mirror) was never implemented,
so column_embeddings_bge stayed permanently empty. storage_adapters.reader.
ann_search() now reads column_embeddings_v2 directly instead of this mirror,
so the table has no reader or writer left.
"""
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('substrate', '0007_drop_relgt_structural'),
    ]

    operations = [
        migrations.DeleteModel(name='ColumnEmbeddingBGE'),
        migrations.RunSQL(
            sql="DROP TABLE IF EXISTS column_embeddings_bge;",
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
