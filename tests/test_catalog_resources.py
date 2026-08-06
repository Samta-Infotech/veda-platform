"""Coverage for apps/access_management — resource paths and the catalog projection.

  apps/access_management/resource_path.py   (pure, no database)
  POST /api/v1/catalog/list
  POST /api/v1/catalog/detail
  manage.py sync_catalog

Implements ADR-0001. Two areas carry most of the weight:

  * **Segment-boundary prefix matching.** ``db:crm`` must not cover ``db:crm_postgres``.
    A string ``startswith`` would grant every source whose name merely begins with
    another's — the prefix-authorization bug, tested explicitly.
  * **Reconciliation instead of referential integrity.** ``CatalogResource`` has no FK
    to the substrate because re-ingestion deletes and recreates it. The regression test
    for that is ``test_reingestion_does_not_destroy_catalog_rows`` — if it ever fails,
    grants are being silently revoked.

Run from repo root: ``pytest tests/test_catalog_resources.py``
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_django():
    import config  # noqa: F401

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    import django

    django.setup()


_setup_django()

from django.contrib.auth import get_user_model  # noqa: E402
from django.db import IntegrityError, transaction  # noqa: E402
from django.test import Client, override_settings  # noqa: E402

from apps.access_management import resource_path as rp  # noqa: E402
from apps.access_management.models import CatalogResource  # noqa: E402
from apps.access_management.services import (  # noqa: E402
    CODE_RESOURCE_NOT_FOUND,
    CatalogDiscoveryService,
    CatalogService,
    ResourceNotFound,
)

LIST_URL = "/api/v1/catalog/list"
DETAIL_URL = "/api/v1/catalog/detail"

ADMIN_PASSWORD = "admin-correct-horse-staple"

PUBLIC_FIELDS = {"path", "kind", "parent_path", "source_id", "substrate_id",
                 "is_active", "created_at", "updated_at"}

_TEST_OVERRIDES = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
)


# ===========================================================================
# resource_path — pure unit tests, no database
# ===========================================================================


def test_every_source_dialect_has_a_mapped_kind():
    """The drift guard for the decoupling in ``resource_path``.

    The module is deliberately Django-free, so it keys its map on plain strings rather
    than importing ``Source.Dialect``. This is the test that stops the two silently
    diverging: adding a dialect without a kind makes that source unaddressable, so
    nothing under it could ever be granted.
    """
    from apps.sources.models import Dialect

    assert set(Dialect.values) == set(rp.KIND_BY_DIALECT), (
        "every Source.dialect needs a kind in resource_path.KIND_BY_DIALECT "
        "(ADR-0001 §3.2)")


def test_kind_for_dialect_maps_the_known_families():
    assert rp.kind_for_dialect("postgres") == rp.KIND_DB
    assert rp.kind_for_dialect("mongo") == rp.KIND_NOSQL
    assert rp.kind_for_dialect("s3_docs") == rp.KIND_FILES
    assert rp.kind_for_dialect("iceberg") == rp.KIND_LAKE


def test_kind_for_dialect_fails_closed_on_an_unknown_dialect():
    """Never guesses. An unmappable source is unaddressable, which means nothing can
    be granted on it — better than granting into the wrong namespace."""
    with pytest.raises(rp.UnknownDialect):
        rp.kind_for_dialect("quantumdb")


def test_build_canonicalises():
    assert rp.build("DB", "  CRM_Postgres ", "Employee") == "db:crm_postgres:employee"


def test_validate_returns_the_canonical_form():
    assert rp.validate("DB:CRM:Employee") == "db:crm:employee"


@pytest.mark.parametrize("bad,reason", [
    ("db", "a kind alone is not a resource"),
    ("", "empty"),
    ("db::crm", "blank segment"),
    (":db:crm", "leading separator"),
    ("db:crm:emp loyee", "whitespace in a segment"),
    ("db:crm:employee/salary", "character outside the charset"),
    ("nope:crm", "unknown kind"),
    ("db:" + ":".join(str(i) for i in range(20)), "too many segments"),
])
def test_validate_rejects_unexpressible_paths(bad, reason):
    with pytest.raises(rp.InvalidResourcePath):
        rp.validate(bad)


def test_validate_rejects_non_strings():
    for value in (None, 42, ["db", "crm"]):
        with pytest.raises(rp.InvalidResourcePath):
            rp.validate(value)


def test_validate_rejects_an_over_long_path():
    with pytest.raises(rp.InvalidResourcePath):
        rp.build("db", "s", "x" * rp.MAX_LENGTH)


def test_parent_walks_up_and_stops_at_the_source():
    assert rp.parent("db:crm:employee:salary") == "db:crm:employee"
    assert rp.parent("db:crm:employee") == "db:crm"
    assert rp.parent("db:crm") is None      # a source is the root of its own tree


def test_prefixes_are_broadest_first_and_include_self():
    assert rp.prefixes("db:crm:employee:salary") == [
        "db:crm", "db:crm:employee", "db:crm:employee:salary"]


def test_prefixes_never_include_a_bare_kind():
    """A grant on ``db`` would silently cover every database source on the platform."""
    assert "db" not in rp.prefixes("db:crm:employee")
    assert rp.prefixes("db:crm") == ["db:crm"]


@pytest.mark.parametrize("ancestor,descendant,expected", [
    ("db:crm", "db:crm", True),                    # prefix-or-equal
    ("db:crm", "db:crm:employee", True),
    ("db:crm", "db:crm:employee:salary", True),
    ("db:crm:employee", "db:crm:invoice", False),
    ("db:crm:employee", "db:crm", False),          # not upward
    ("db:crm", "db:crm_postgres", False),          # THE prefix-authorization bug
    ("db:crm", "db:crmx:employee", False),
    ("db:crm", "files:crm:doc.pdf", False),        # different kind
])
def test_is_prefix_of_matches_on_segment_boundaries(ancestor, descendant, expected):
    """The single most security-relevant function in the module.

    A naive ``descendant.startswith(ancestor)`` returns True for
    ``("db:crm", "db:crm_postgres")`` and grants an unrelated source.
    """
    assert rp.is_prefix_of(ancestor, descendant) is expected


# ===========================================================================
# Database-backed
# ===========================================================================


@pytest.fixture(scope="module", autouse=True)
def _database():
    from django.db import connection
    from django.test.utils import setup_test_environment, teardown_test_environment

    try:
        setup_test_environment()
        owns_environment = True
    except RuntimeError:
        owns_environment = False

    with override_settings(MIGRATION_MODULES={"substrate": None}):
        old_config = connection.creation.create_test_db(verbosity=0, serialize=False)
    try:
        yield
    finally:
        connection.creation.destroy_test_db(old_config, verbosity=0)
        if owns_environment:
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
def source():
    from apps.sources.models import Source

    return Source.objects.create(
        name="crm_postgres", dialect="postgres", connector_type="relational")


def _add_table(source, name, tenant="default"):
    from apps.substrate.models import SchemaTable

    return SchemaTable.objects.all_tenants().create(
        source=source, tenant=tenant, name=name)


def _add_column(source, table, name, tenant="default"):
    from apps.substrate.models import SchemaColumn

    return SchemaColumn.objects.all_tenants().create(
        source=source, tenant=tenant, table=table, name=name, data_type="text")


@pytest.fixture
def populated_source(source):
    """A source with two tables, one of which has two columns."""
    employee = _add_table(source, "employee")
    _add_column(source, employee, "salary")
    _add_column(source, employee, "hired_on")
    _add_table(source, "invoice")
    return source


@pytest.fixture
def admin_client():
    user = get_user_model().objects.create_user(
        username="root", password=ADMIN_PASSWORD, email="root@example.com",
        is_staff=True)
    client = Client()
    client.force_login(user)
    return client


def _list(client, **body):
    return client.post(LIST_URL, body, content_type="application/json")


def _detail(client, **body):
    return client.post(DETAIL_URL, body, content_type="application/json")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_path_is_unique(source):
    CatalogResource.objects.create(path="db:crm_postgres", kind="db", source=source)

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            CatalogResource.objects.create(
                path="db:crm_postgres", kind="db", source=source)


def test_there_is_no_foreign_key_to_the_substrate():
    """Pins the central ADR decision (§3.7 as amended).

    A FK here would be deleted by re-ingestion (CASCADE) or would block it (PROTECT).
    ``substrate_id`` must stay a plain UUID column. If someone "improves" this into a
    ForeignKey, grants start disappearing on re-ingestion.
    """
    from django.db.models import ForeignKey, UUIDField

    field = CatalogResource._meta.get_field("substrate_id")
    assert isinstance(field, UUIDField)
    assert not isinstance(field, ForeignKey)

    related = [f.name for f in CatalogResource._meta.fields
               if isinstance(f, ForeignKey)]
    assert related == ["source"], "the only FK may be to sources.Source"


def test_the_source_foreign_key_protects():
    """Deleting a source that still has catalog rows — and therefore possibly grants —
    must be blocked, not cascade."""
    from django.db.models import PROTECT

    assert CatalogResource._meta.get_field("source").remote_field.on_delete is PROTECT


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovery_projects_source_tables_and_columns(populated_source):
    report = CatalogDiscoveryService().sync_source(populated_source)

    paths = set(CatalogResource.objects.values_list("path", flat=True))
    assert paths == {
        "db:crm_postgres",
        "db:crm_postgres:employee",
        "db:crm_postgres:employee:salary",
        "db:crm_postgres:employee:hired_on",
        "db:crm_postgres:invoice",
    }
    assert report.created == 5
    assert report.skipped == []


def test_discovery_records_the_hierarchy(populated_source):
    CatalogDiscoveryService().sync_source(populated_source)

    by_path = {r.path: r for r in CatalogResource.objects.all()}
    assert by_path["db:crm_postgres"].parent_path == ""
    assert by_path["db:crm_postgres:employee"].parent_path == "db:crm_postgres"
    assert by_path["db:crm_postgres:employee:salary"].parent_path == \
        "db:crm_postgres:employee"


def test_discovery_links_substrate_ids_without_a_foreign_key(populated_source):
    from apps.substrate.models import SchemaTable

    CatalogDiscoveryService().sync_source(populated_source)

    table = SchemaTable.objects.all_tenants().get(name="employee")
    resource = CatalogResource.objects.get(path="db:crm_postgres:employee")
    assert resource.substrate_id == table.id


def test_discovery_is_idempotent(populated_source):
    service = CatalogDiscoveryService()
    service.sync_source(populated_source)

    second = service.sync_source(populated_source)

    assert second.created == 0
    assert second.deactivated == 0
    assert second.unchanged == 5
    assert CatalogResource.objects.count() == 5


def test_discovery_deactivates_vanished_resources_without_deleting(populated_source):
    """Deleting would silently drop every grant referencing the path."""
    from apps.substrate.models import SchemaTable

    service = CatalogDiscoveryService()
    service.sync_source(populated_source)
    SchemaTable.objects.all_tenants().filter(name="invoice").delete()

    report = service.sync_source(populated_source)

    assert report.deactivated == 1
    resource = CatalogResource.objects.get(path="db:crm_postgres:invoice")
    assert resource.is_active is False      # deactivated...
    assert CatalogResource.objects.filter(path="db:crm_postgres:invoice").exists()


def test_discovery_reactivates_a_returning_resource(populated_source):
    from apps.substrate.models import SchemaTable

    service = CatalogDiscoveryService()
    service.sync_source(populated_source)
    SchemaTable.objects.all_tenants().filter(name="invoice").delete()
    service.sync_source(populated_source)

    _add_table(populated_source, "invoice")
    report = service.sync_source(populated_source)

    assert report.reactivated >= 1
    assert CatalogResource.objects.get(path="db:crm_postgres:invoice").is_active is True


def test_reingestion_does_not_destroy_catalog_rows(populated_source):
    """THE regression test for the amended ADR §3.7.

    ``storage_adapters/writer.py:137-139`` deletes every SchemaTable/SchemaColumn row
    for a source on each re-ingestion. With the originally-specified CASCADE foreign
    key, this would have deleted every CatalogResource row — and every grant pointing
    at it. Simulated here exactly as the writer does it.
    """
    from apps.substrate.models import SchemaColumn, SchemaTable

    service = CatalogDiscoveryService()
    service.sync_source(populated_source)
    assert CatalogResource.objects.count() == 5

    # Exactly what writer.py does at the start of every re-sync.
    SchemaColumn.objects.all_tenants().filter(source_id=populated_source.pk).delete()
    SchemaTable.objects.all_tenants().filter(source_id=populated_source.pk).delete()

    # The catalog — and therefore any grants — must survive untouched.
    assert CatalogResource.objects.count() == 5

    # Re-ingestion recreates them; discovery then reactivates rather than re-creating.
    employee = _add_table(populated_source, "employee")
    _add_column(populated_source, employee, "salary")
    _add_column(populated_source, employee, "hired_on")
    _add_table(populated_source, "invoice")
    report = service.sync_source(populated_source)

    assert report.created == 0, "rows must be reused, not recreated"
    assert CatalogResource.objects.filter(is_active=True).count() == 5


def test_unaddressable_names_are_reported_not_silently_dropped(source):
    """A table whose name cannot appear in a path is one nobody can grant access to —
    the operator has to know."""
    _add_table(source, "weird:name")
    _add_table(source, "fine_name")

    report = CatalogDiscoveryService().sync_source(source)

    assert any("weird:name" in entry for entry in report.skipped)
    assert CatalogResource.objects.filter(path="db:crm_postgres:fine_name").exists()
    assert not CatalogResource.objects.filter(path__contains="weird").exists()


def test_discovery_fails_closed_on_an_unmapped_dialect():
    from apps.sources.models import Source

    exotic = Source.objects.create(
        name="mystery", dialect="quantumdb", connector_type="x")

    with pytest.raises(rp.UnknownDialect):
        CatalogDiscoveryService().sync_source(exotic)
    assert not CatalogResource.objects.filter(source=exotic).exists()


def test_sync_all_skips_an_unmappable_source_and_continues(populated_source):
    """One bad source must not stop the rest of the catalog from being correct."""
    from apps.sources.models import Source

    Source.objects.create(name="mystery", dialect="quantumdb", connector_type="x")

    reports = CatalogDiscoveryService().sync_all()

    assert populated_source.pk in reports
    assert CatalogResource.objects.filter(source=populated_source).count() == 5


def test_two_tenants_sharing_a_source_collapse_to_one_path(source):
    """Paths carry no tenant (ADR §7 option 1), so the same table in two tenants is
    one resource. Documented behaviour, pinned so a change is deliberate."""
    _add_table(source, "employee", tenant="default")
    _add_table(source, "employee", tenant="other")

    CatalogDiscoveryService().sync_source(source)

    assert CatalogResource.objects.filter(path="db:crm_postgres:employee").count() == 1


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def test_list_returns_the_projection(admin_client, populated_source):
    CatalogDiscoveryService().sync_source(populated_source)

    response = _list(admin_client, page_size=100)

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["pagination"]["total"] == 5
    assert set(body["resources"][0]) == PUBLIC_FIELDS


def test_list_navigates_the_tree_one_level_at_a_time(admin_client, populated_source):
    """``parent_path`` is the lazy-tree access pattern."""
    CatalogDiscoveryService().sync_source(populated_source)

    children = _list(admin_client, parent_path="db:crm_postgres").json()["data"]
    grandchildren = _list(
        admin_client, parent_path="db:crm_postgres:employee").json()["data"]

    assert {r["path"] for r in children["resources"]} == {
        "db:crm_postgres:employee", "db:crm_postgres:invoice"}
    assert {r["path"] for r in grandchildren["resources"]} == {
        "db:crm_postgres:employee:salary", "db:crm_postgres:employee:hired_on"}


def test_list_parent_path_is_tri_state(admin_client, populated_source):
    """Omitted = every resource · "" = the source-level roots · a path = its children.

    Without the "" case a client cannot ask for the tree's top level, and with "" as
    the default a plain list would silently return only roots.
    """
    CatalogDiscoveryService().sync_source(populated_source)

    everything = _list(admin_client, page_size=100).json()["data"]
    roots = _list(admin_client, parent_path="").json()["data"]
    children = _list(admin_client, parent_path="db:crm_postgres").json()["data"]

    assert everything["pagination"]["total"] == 5
    assert roots["pagination"]["total"] == 1
    assert roots["resources"][0]["path"] == "db:crm_postgres"
    assert children["pagination"]["total"] == 2


def test_reactivation_refreshes_updated_at(populated_source):
    """``bulk_update`` does NOT call ``pre_save``, so ``auto_now`` never fires — the
    reactivation path has to stamp the timestamp itself, or a reactivated row silently
    claims it never changed."""
    from apps.substrate.models import SchemaTable

    service = CatalogDiscoveryService()
    service.sync_source(populated_source)
    SchemaTable.objects.all_tenants().filter(name="invoice").delete()
    service.sync_source(populated_source)
    before = CatalogResource.objects.get(path="db:crm_postgres:invoice").updated_at

    _add_table(populated_source, "invoice")
    service.sync_source(populated_source)

    after = CatalogResource.objects.get(path="db:crm_postgres:invoice").updated_at
    assert after > before


def test_list_canonicalises_parent_path(admin_client, populated_source):
    """A caller navigating with a differently-cased path must still find the children."""
    CatalogDiscoveryService().sync_source(populated_source)

    body = _list(admin_client, parent_path="DB:CRM_Postgres").json()["data"]

    assert body["pagination"]["total"] == 2


def test_list_filters_by_kind_and_source(admin_client, populated_source):
    CatalogDiscoveryService().sync_source(populated_source)

    by_kind = _list(admin_client, kind="db", page_size=100).json()["data"]
    by_source = _list(admin_client, source_id=populated_source.pk,
                      page_size=100).json()["data"]

    assert by_kind["pagination"]["total"] == 5
    assert by_source["pagination"]["total"] == 5


def test_list_rejects_an_unknown_kind(admin_client):
    """An empty page and a typo look identical to a client otherwise."""
    response = _list(admin_client, kind="nope")

    assert response.status_code == 400
    assert "kind" in response.json()["errors"]


def test_list_rejects_a_malformed_parent_path(admin_client):
    response = _list(admin_client, parent_path="db::bad")

    assert response.status_code == 400
    assert "parent_path" in response.json()["errors"]


def test_list_search_matches_the_path_substring(admin_client, populated_source):
    CatalogDiscoveryService().sync_source(populated_source)

    body = _list(admin_client, search="employee", page_size=100).json()["data"]

    assert body["pagination"]["total"] == 3      # the table and its two columns


def test_list_filters_by_is_active(admin_client, populated_source):
    from apps.substrate.models import SchemaTable

    service = CatalogDiscoveryService()
    service.sync_source(populated_source)
    SchemaTable.objects.all_tenants().filter(name="invoice").delete()
    service.sync_source(populated_source)

    inactive = _list(admin_client, is_active=False).json()["data"]

    assert inactive["pagination"]["total"] == 1
    assert inactive["resources"][0]["path"] == "db:crm_postgres:invoice"


def test_list_costs_two_queries(populated_source):
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    CatalogDiscoveryService().sync_source(populated_source)

    with CaptureQueriesContext(connection) as ctx:
        CatalogService().list_resources(page=1, page_size=100)

    assert len(ctx.captured_queries) == 2


def test_detail_returns_one_resource(admin_client, populated_source):
    CatalogDiscoveryService().sync_source(populated_source)

    response = _detail(admin_client, path="db:crm_postgres:employee")

    assert response.status_code == 200
    assert response.json()["data"]["path"] == "db:crm_postgres:employee"
    assert set(response.json()["data"]) == PUBLIC_FIELDS


def test_detail_canonicalises_the_path(admin_client, populated_source):
    """Otherwise one resource is addressable under two strings, only one of which
    matches its grants."""
    CatalogDiscoveryService().sync_source(populated_source)

    response = _detail(admin_client, path="DB:CRM_Postgres:Employee")

    assert response.status_code == 200
    assert response.json()["data"]["path"] == "db:crm_postgres:employee"


def test_detail_of_an_unknown_path_is_404(admin_client):
    response = _detail(admin_client, path="db:nowhere:nothing")

    assert response.status_code == 404
    assert response.json()["code"] == CODE_RESOURCE_NOT_FOUND


def test_detail_of_an_unexpressible_path_is_404_not_400(admin_client):
    """An unaddressable string names nothing. Reported as not-found rather than as a
    validation error, so it cannot be used to probe which paths are well-formed."""
    assert _detail(admin_client, path="db::bad").status_code == 404


def test_detail_requires_a_path(admin_client):
    assert _detail(admin_client).status_code == 400


def test_service_raises_a_typed_not_found():
    with pytest.raises(ResourceNotFound):
        CatalogService().get_resource("db:nowhere:nothing")


# ---------------------------------------------------------------------------
# Read-only, access control, wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "/api/v1/catalog/create", "/api/v1/catalog/update", "/api/v1/catalog/delete",
    "/api/v1/catalog/sync",
])
def test_there_are_no_write_endpoints(path):
    """The catalog is a projection. An operator-authored row would name a resource
    nothing upstream corresponds to; re-sync is an ops action, not an API one."""
    from django.urls import Resolver404, resolve

    with pytest.raises(Resolver404):
        resolve(path)


@pytest.mark.parametrize("url,body", [(LIST_URL, {}),
                                     (DETAIL_URL, {"path": "db:crm_postgres"})])
def test_anonymous_and_non_staff_are_rejected(url, body):
    anonymous = Client().post(url, body, content_type="application/json")
    plain = Client()
    plain.force_login(get_user_model().objects.create_user(
        username="nobody", password=ADMIN_PASSWORD))

    assert anonymous.status_code == 401
    assert plain.post(url, body, content_type="application/json").status_code == 403


@pytest.mark.parametrize("url", [LIST_URL, DETAIL_URL])
def test_endpoints_are_post_only(admin_client, url):
    assert admin_client.get(url).status_code == 405
    assert admin_client.put(url).status_code == 405
    assert admin_client.delete(url).status_code == 405


def test_routes_are_wired():
    from django.urls import resolve

    from apps.access_management.views import CatalogDetailView, CatalogListView

    assert resolve(LIST_URL).func.view_class is CatalogListView
    assert resolve(DETAIL_URL).func.view_class is CatalogDetailView


def test_sync_catalog_command_runs(populated_source):
    from io import StringIO

    from django.core.management import call_command

    out = StringIO()
    call_command("sync_catalog", "--source-id", str(populated_source.pk), stdout=out)

    assert "created=5" in out.getvalue()
    assert CatalogResource.objects.count() == 5


def test_sync_catalog_command_reports_unaddressable_resources(source):
    from io import StringIO

    from django.core.management import call_command

    _add_table(source, "weird:name")
    out, err = StringIO(), StringIO()

    call_command("sync_catalog", stdout=out, stderr=err)

    assert "unaddressable" in err.getvalue()
