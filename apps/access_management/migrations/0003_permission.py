"""Create the ``Permission`` table — the vocabulary of authorizable actions.

Additive: a new table plus its case-insensitive unique constraint on ``code``.
Nothing existing is altered. The rows themselves are seeded by 0004, kept separate so
a schema change and a data change are never entangled in one migration.
"""

import django.db.models.functions.text
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('access_management', '0002_role'),
    ]

    operations = [
        migrations.CreateModel(
            name='Permission',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('code', models.CharField(max_length=100)),
                ('name', models.CharField(max_length=150)),
                ('description', models.TextField(blank=True)),
                ('is_active', models.BooleanField(default=True)),
            ],
            options={
                'verbose_name': 'Permission',
                'verbose_name_plural': 'Permissions',
                'ordering': ('code',),
            },
        ),
        migrations.AddConstraint(
            model_name='permission',
            constraint=models.UniqueConstraint(django.db.models.functions.text.Lower('code'), name='access_management_permission_code_ci_uniq'),
        ),
    ]
