from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0003_querylog_cache_hit'),
    ]

    operations = [
        migrations.AddField(
            model_name='querylog',
            name='prompt_tokens',
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='querylog',
            name='completion_tokens',
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='querylog',
            name='total_tokens',
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
    ]
