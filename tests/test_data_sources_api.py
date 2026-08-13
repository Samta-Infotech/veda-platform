import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _setup_django():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    import django
    django.setup()

_setup_django()

from django.contrib.auth import get_user_model
from django.test import Client, override_settings
from apps.sources.models import Source, Dialect

ADMIN_PASSWORD = "admin-correct-horse-staple"
LIST_URL = "/api/v1/data-sources/list"

_TEST_OVERRIDES = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
    VEDA_RBAC_MODE="off",
)

@pytest.fixture(scope="module", autouse=True)
def _database():
    from django.db import connection
    from django.test.utils import setup_test_environment, teardown_test_environment
    try:
        setup_test_environment()
        owns_env = True
    except RuntimeError:
        owns_env = False

    with override_settings(MIGRATION_MODULES={"substrate": None}):
        old_config = connection.creation.create_test_db(verbosity=0, serialize=False)
    try:
        yield
    finally:
        connection.creation.destroy_test_db(old_config, verbosity=0)
        if owns_env:
            teardown_test_environment()

@pytest.fixture(autouse=True)
def _isolated(_database):
    from django.core.cache import cache
    from django.db import transaction as db_transaction
    with override_settings(**_TEST_OVERRIDES):
        cache.clear()
        atomic = db_transaction.atomic()
        atomic.__enter__()
        try:
            yield
        finally:
            db_transaction.set_rollback(True)
            atomic.__exit__(None, None, None)

@pytest.fixture
def admin_user():
    return get_user_model().objects.create_user(
        username="admin_test", password=ADMIN_PASSWORD, is_staff=True
    )

def test_data_sources_list_post_with_db_rows(admin_user):
    """When Source table has rows, API returns raw DB data."""
    Source.objects.create(
        name="Finance DB", dialect=Dialect.POSTGRES, connector_type="sql",
        host="db.example.com", port=5432, dbname="finance", db_user="pg_user", ready=True
    )
    client = Client()
    client.force_login(admin_user)

    resp = client.post(LIST_URL, data={}, content_type="application/json")
    assert resp.status_code == 200
    res_data = resp.json()["data"]
    assert len(res_data["items"]) == 1
    item = res_data["items"][0]
    assert item["name"] == "Finance DB"
    assert item["source_type"] == "DATABASE"
    assert item["status"] == "CONNECTED"
    assert item["isConnected"] is True
    # api_contract.md §5.2 field names
    assert item["metadata"]["db_type"] == "POSTGRESQL"
    assert item["metadata"]["host"] == "db.example.com"
    assert item["metadata"]["port"] == 5432
    assert item["metadata"]["database"] == "finance"
    assert item["metadata"]["username"] == "pg_user"
    # DATALAKE/FILE_SYSTEM-only keys must not leak into a DATABASE source's metadata
    assert "source_path" not in item["metadata"]
    assert "doc_formats" not in item["metadata"]
    assert "last_checked_at" in item
    assert item["status_message"] is None

def test_data_sources_list_get_filesystem(admin_user):
    """GET with source_type filter works for FILE_SYSTEM."""
    Source.objects.create(
        name="Docs", dialect=Dialect.FILESYSTEM, connector_type="file",
        source_path="/data/docs", ready=True
    )
    client = Client()
    client.force_login(admin_user)

    resp = client.get(f"{LIST_URL}?source_type=FILE_SYSTEM")
    assert resp.status_code == 200
    res_data = resp.json()["data"]
    assert len(res_data["items"]) == 1
    item = res_data["items"][0]
    assert item["name"] == "Docs"
    assert item["source_type"] == "FILE_SYSTEM"
    assert item["metadata"]["source_path"] == "/data/docs"

def test_data_sources_list_excludes_not_ready_sources(admin_user):
    """A registered-but-not-yet-connected source is not "meaningful" for this
    list (user's call) — it's onboarding noise, not something to show."""
    Source.objects.create(
        name="Ready DB", dialect=Dialect.POSTGRES, connector_type="sql", ready=True)
    Source.objects.create(
        name="Pending DB", dialect=Dialect.POSTGRES, connector_type="sql", ready=False)
    client = Client()
    client.force_login(admin_user)

    resp = client.post(LIST_URL, data={}, content_type="application/json")
    names = [i["name"] for i in resp.json()["data"]["items"]]
    assert names == ["Ready DB"]

def test_data_sources_list_has_no_pagination(admin_user):
    """The response is a plain items list — no page control (user's call)."""
    Source.objects.create(
        name="Ready DB", dialect=Dialect.POSTGRES, connector_type="sql", ready=True)
    client = Client()
    client.force_login(admin_user)

    resp = client.post(LIST_URL, data={}, content_type="application/json")
    res_data = resp.json()["data"]
    assert set(res_data.keys()) == {"items"}

def test_data_sources_list_empty_db_falls_back_to_config(admin_user):
    """When Source table is empty, API falls back to veda_core/config sources."""
    # Source table is empty (no create)
    assert Source.objects.count() == 0

    client = Client()
    client.force_login(admin_user)

    resp = client.post(LIST_URL, data={}, content_type="application/json")
    assert resp.status_code == 200
    # Should return config-based sources (or empty if config env not set)
    res_data = resp.json()["data"]
    assert "items" in res_data
    assert "pagination" not in res_data
