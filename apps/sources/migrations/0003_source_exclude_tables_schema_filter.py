"""P3 (§3.1) — move client-specific table exclusions + schema filter onto the
Source row so no client table names live in config.py."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sources', '0002_source_db_user_source_dbname_source_host_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='source',
            name='exclude_tables',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name='source',
            name='schema_filter',
            field=models.CharField(blank=True, max_length=128),
        ),
    ]
