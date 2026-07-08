"""Vertical-aware glossary (G0) — add Source.industry_vertical so an admin can
explicitly select BFSI / Real Estate / Healthcare / Retail / Generic at
registration time, driving domain-aware glossary generation during L3
enrichment instead of a hardcoded BFSI/AML framing for every source."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sources', '0004_source_document_datalake_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='source',
            name='industry_vertical',
            field=models.CharField(
                choices=[
                    ('bfsi', 'BFSI / Banking & Financial Services'),
                    ('real_estate', 'Real Estate'),
                    ('healthcare', 'Healthcare'),
                    ('retail', 'Retail'),
                    ('generic', 'Generic / Other'),
                ],
                default='generic', max_length=32,
                help_text='Drives which domain glossary and LLM domain-framing is used '
                          'during L3 enrichment. Set once at registration; does not '
                          'auto-detect.'),
        ),
    ]
