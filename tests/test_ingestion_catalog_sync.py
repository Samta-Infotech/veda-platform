"""Coverage for the ingestion→catalog auto-sync hook (``VEDA_AUTO_SYNC_CATALOG``).

Scoped tightly to ``apps.ingestion.tasks._sync_catalog_if_enabled`` — the one new
function this change adds — rather than the whole ingestion task, which needs a
running engine subprocess and is out of scope here. The three things that matter:

  * flag off -> no Django access_management import, no DB touch (cost-free no-op)
  * flag on, success -> CatalogDiscoveryService.sync_source is actually called
  * flag on, failure -> swallowed and logged, NEVER raised (the ingestion job that
    just succeeded must not be turned into a failure by this)

Run from repo root: ``pytest tests/test_ingestion_catalog_sync.py``
"""
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_django():
    import config  # noqa: F401

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    import django

    django.setup()


_setup_django()

from django.test import override_settings  # noqa: E402

from apps.ingestion.tasks import _sync_catalog_if_enabled  # noqa: E402


def test_flag_off_never_touches_the_catalog_service():
    with override_settings(VEDA_AUTO_SYNC_CATALOG=False):
        with mock.patch("apps.access_management.services.CatalogDiscoveryService") as svc:
            _sync_catalog_if_enabled(source_id=1)

    svc.assert_not_called()


def test_flag_defaults_to_off():
    """The setting genuinely defaults OFF, as shipped in config/settings/base.py —
    not relying on every deployment remembering to set the env var."""
    from django.conf import settings

    assert settings.VEDA_AUTO_SYNC_CATALOG is False


@override_settings(VEDA_AUTO_SYNC_CATALOG=True)
def test_flag_on_calls_sync_source_for_the_ingested_source():
    fake_source = object()
    with mock.patch("apps.sources.models.Source.objects") as manager:
        manager.get.return_value = fake_source
        with mock.patch(
            "apps.access_management.services.CatalogDiscoveryService"
        ) as service_cls:
            service_cls.return_value.sync_source.return_value.as_dict.return_value = {}
            _sync_catalog_if_enabled(source_id=42)

    manager.get.assert_called_once_with(pk=42)
    service_cls.return_value.sync_source.assert_called_once_with(fake_source)


@override_settings(VEDA_AUTO_SYNC_CATALOG=True)
def test_a_sync_failure_is_swallowed_not_raised():
    with mock.patch("apps.sources.models.Source.objects") as manager:
        manager.get.side_effect = RuntimeError("boom")

        _sync_catalog_if_enabled(source_id=42)  # must not raise
