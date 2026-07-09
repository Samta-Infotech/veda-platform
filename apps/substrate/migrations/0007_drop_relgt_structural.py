"""WP3 — drop the last legacy RELGT structural encoder mirror.

Migration 0006 dropped the column_embeddings / _lt / _hybrid mirrors but missed
relgt_structural (256-dim RELGT structural encoder). Nothing writes to it — one
embedding space (BGE-M3) means one column store (column_embeddings_bge).
"""
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('substrate', '0006_drop_legacy_embedding_mirrors'),
    ]

    operations = [
        migrations.DeleteModel(name='RelgtStructural'),
        migrations.RunSQL(
            sql="DROP TABLE IF EXISTS relgt_structural;",
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
