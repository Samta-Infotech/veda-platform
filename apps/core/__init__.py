"""apps.core — tenancy base models, request-id middleware, health/metrics views.

``default_app_config`` was removed here deliberately: Django deprecated it in 3.2
and DELETED the machinery in 4.1, so on this project's Django 5.0 it was dead
code with no effect. ``apps.core.apps.CoreConfig`` is still picked up
automatically (Django's default AppConfig discovery finds the single AppConfig
subclass in ``apps/core/apps.py``).
"""
