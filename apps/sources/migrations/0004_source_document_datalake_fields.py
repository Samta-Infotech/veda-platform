"""Review Finding 5 — make document/datalake sources representable from a Source
row (path/formats/recursion/size cap), so all four source kinds onboard through
the same API data operation. Also widens `dialect` choices (no schema change —
choices are app-level)."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sources', '0003_source_exclude_tables_schema_filter'),
    ]

    operations = [
        migrations.AddField(
            model_name='source',
            name='source_path',
            field=models.CharField(
                blank=True, max_length=512,
                help_text='Root path/URI for document or datalake sources '
                          '(e.g. /data/contracts, s3://bucket/prefix)'),
        ),
        migrations.AddField(
            model_name='source',
            name='doc_formats',
            field=models.JSONField(
                blank=True, default=list,
                help_text='Document formats to ingest, e.g. ["pdf", "docx", "md"]; '
                          'empty = connector defaults'),
        ),
        migrations.AddField(
            model_name='source',
            name='doc_recursive',
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name='source',
            name='doc_max_file_mb',
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='source',
            name='dialect',
            field=models.CharField(choices=[
                ('postgres', 'PostgreSQL'), ('mysql', 'MySQL'), ('sqlite', 'SQLite'),
                ('oracle', 'Oracle'), ('sqlserver', 'SQL Server'), ('duckdb', 'DuckDB'),
                ('mongo', 'MongoDB'), ('es', 'Elasticsearch'), ('dynamo', 'DynamoDB'),
                ('filesystem', 'Filesystem documents'), ('s3_docs', 'S3 documents'),
                ('delta', 'Delta Lake'), ('parquet', 'Parquet'),
                ('csv_lake', 'CSV data lake'), ('iceberg', 'Iceberg'),
            ], max_length=20),
        ),
    ]
