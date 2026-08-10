"""Create the ``Role`` table — the first model this app owns.

Additive: a new table plus its constraint. Nothing existing is altered, so the
migration is safe to apply to a live database and trivially reversible.

The unique constraint is on ``LOWER(name)``, so "Admin" and "admin" cannot coexist.
Expressed as a Django ``UniqueConstraint`` rather than raw SQL because we own this
model — migration 0001 needed raw SQL for the same rule only because it targets
``auth_user``, which belongs to Django.
"""

import django.db.models.functions.text
from django.db import migrations, models


class Migration(migrations.Migration):


    dependencies = [
        ('access_management', '0001_user_email_unique_index'),
    ]

    operations = [
        migrations.CreateModel(
            name='Role',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=150)),
                ('description', models.TextField(blank=True)),
                ('is_active', models.BooleanField(default=True)),
            ],
            options={
                'verbose_name': 'Role',
                'verbose_name_plural': 'Roles',
                'ordering': ('name',),
            },
        ),
        migrations.AddConstraint(
            model_name='role',
            constraint=models.UniqueConstraint(django.db.models.functions.text.Lower('name'), name='access_management_role_name_ci_uniq'),
        ),
    ]
