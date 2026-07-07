"""P7/Q-10 — per-source HNSW ef_search stored on the SubstrateVersion."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('substrate', '0003_smconcept_smretrievaldoc_smsynonym_smtable_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='substrateversion',
            name='hnsw_ef_search',
            field=models.PositiveIntegerField(default=40),
        ),
    ]
