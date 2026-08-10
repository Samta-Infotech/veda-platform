"""Coverage for the table/column allow-payload computed by Gate 1 Task 14.

  apps/access_management/services/data_scope.py :: compute_data_scope

Builds a real catalog via ``CatalogDiscoveryService`` (the same reconciliation path
production ingestion uses) from real ``SchemaTable``/``SchemaColumn`` rows, then
grants real permissions and asserts on the resulting payload shape — never asserts
on a hand-built ``CatalogResource`` fixture, so a change to how paths are built
would be caught here too.

Run from repo root: ``pytest tests/test_data_scope.py``
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
from django.test import override_settings  # noqa: E402

from apps.access_management.gate import MODE_ENFORCE, MODE_OFF  # noqa: E402
from apps.access_management.models import Effect, Permission, Role, RolePermission, UserRole  # noqa: E402
from apps.access_management.services.catalog import CatalogDiscoveryService  # noqa: E402
from apps.access_management.services.data_scope import compute_data_scope  # noqa: E402
from apps.sources.models import Dialect, Source  # noqa: E402
from apps.substrate.models import SchemaColumn, SchemaTable  # noqa: E402

ADMIN_PASSWORD = "admin-correct-horse-staple"
TENANT = "default"

_TEST_OVERRIDES = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
    VEDA_RBAC_MODE=MODE_ENFORCE,
)


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
def member():
    return get_user_model().objects.create_user(username="alice", password=ADMIN_PASSWORD)


@pytest.fixture
def staff_member():
    return get_user_model().objects.create_user(
        username="root", password=ADMIN_PASSWORD, is_staff=True)


@pytest.fixture
def read_perm():
    return Permission.objects.get(code="data.read")


def _grant(role, permission, path="", effect=Effect.ALLOW):
    return RolePermission.objects.create(
        role=role, permission=permission, resource_path=path, effect=effect)


def _role_for(user, name="Analyst"):
    role = Role.objects.create(name=name, is_active=True)
    UserRole.objects.create(user=user, role=role)
    return role


def _crm_source(with_catalog=True):
    """A CRM source with two tables (employee: id/name/salary; department:
    id/name), catalog-discovered so real CatalogResource rows exist."""
    source = Source.objects.create(
        name="crm", dialect=Dialect.POSTGRES, connector_type="sql", ready=True)
    employee = SchemaTable.objects.create(source=source, tenant=TENANT, name="Employee")
    for col in ("id", "name", "salary"):
        SchemaColumn.objects.create(
            source=source, tenant=TENANT, table=employee, name=col, data_type="text")
    department = SchemaTable.objects.create(source=source, tenant=TENANT, name="Department")
    for col in ("id", "name"):
        SchemaColumn.objects.create(
            source=source, tenant=TENANT, table=department, name=col, data_type="text")
    if with_catalog:
        CatalogDiscoveryService().sync_source(source)
    return source


def _path(*parts):
    from apps.access_management import resource_path as rp
    return rp.build("db", "crm", *parts)


# ---------------------------------------------------------------------------
# No restriction: off / staff
# ---------------------------------------------------------------------------

@override_settings(VEDA_RBAC_MODE=MODE_OFF)
def test_rbac_off_returns_none(member, read_perm):
    source = _crm_source()
    role = _role_for(member)
    _grant(role, read_perm, _path())
    assert compute_data_scope(member, [source.id]) is None


def test_no_user_returns_none():
    assert compute_data_scope(None, [1]) is None


def test_staff_bypasses_regardless_of_grants(staff_member):
    source = _crm_source()
    assert compute_data_scope(staff_member, [source.id]) is None


# ---------------------------------------------------------------------------
# Fully open: source-level and table-level
# ---------------------------------------------------------------------------

def test_bare_source_allow_with_no_deeper_grant_is_fully_open(member, read_perm):
    source = _crm_source()
    role = _role_for(member)
    _grant(role, read_perm, _path())

    scope = compute_data_scope(member, [source.id])
    assert scope[source.id].open is True
    assert scope[source.id].tables == ()


def test_source_allow_plus_a_deeper_deny_is_not_fully_open(member, read_perm):
    """The deny carves out an exception, so the source can no longer be reported
    as blanket-open — it must be enumerated instead."""
    source = _crm_source()
    role = _role_for(member)
    _grant(role, read_perm, _path())
    _grant(role, read_perm, _path("employee", "salary"), effect=Effect.DENY)

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.open is False
    by_name = {t.name: t for t in scope.tables}
    assert set(by_name) == {"Employee", "Department"}
    assert by_name["Department"].columns is None  # unaffected table stays fully open
    assert set(by_name["Employee"].columns) == {"id", "name"}  # salary excluded


def test_source_allow_plus_a_redundant_deeper_allow_stays_fully_open(member, read_perm):
    """An extra ALLOW under an already-open source restricts nothing — only a DENY
    should force enumeration, keeping the payload minimal in the common case."""
    source = _crm_source()
    role = _role_for(member)
    _grant(role, read_perm, _path())
    _grant(role, read_perm, _path("employee"))  # redundant, not restrictive

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.open is True
    assert scope.tables == ()


# ---------------------------------------------------------------------------
# Restricted: table- and column-level enumeration
# ---------------------------------------------------------------------------

def test_table_level_grant_without_source_grant_enumerates_that_table_only(member, read_perm):
    source = _crm_source()
    role = _role_for(member)
    _grant(role, read_perm, _path("employee"))

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.open is False
    assert [t.name for t in scope.tables] == ["Employee"]
    assert scope.tables[0].columns is None  # whole table granted, no column carve-out


def test_ungranted_table_is_omitted_entirely(member, read_perm):
    source = _crm_source()
    role = _role_for(member)
    _grant(role, read_perm, _path("employee"))
    # No grant at all on "department" -> must not appear.

    scope = compute_data_scope(member, [source.id])[source.id]
    assert {t.name for t in scope.tables} == {"Employee"}


def test_column_level_grants_list_exactly_the_granted_columns(member, read_perm):
    source = _crm_source()
    role = _role_for(member)
    _grant(role, read_perm, _path("employee", "id"))
    _grant(role, read_perm, _path("employee", "name"))
    # "salary" ungranted.

    scope = compute_data_scope(member, [source.id])[source.id]
    assert len(scope.tables) == 1
    table = scope.tables[0]
    assert table.name == "Employee"
    assert set(table.columns) == {"id", "name"}


def test_a_table_with_zero_reachable_columns_is_omitted(member, read_perm):
    """A table-level ALLOW that is then denied at every one of its columns leaves
    nothing addressable — it must not appear as an empty, misleading entry."""
    source = _crm_source()
    role = _role_for(member)
    _grant(role, read_perm, _path("employee"))
    for col in ("id", "name", "salary"):
        _grant(role, read_perm, _path("employee", col), effect=Effect.DENY)

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.tables == ()


def test_no_grants_at_all_yields_a_restricted_empty_source(member, read_perm):
    source = _crm_source()
    _role_for(member)  # a role, but with zero grants

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.open is False
    assert scope.tables == ()


def test_multiple_sources_are_each_scoped_independently(member, read_perm):
    crm = _crm_source()
    billing = Source.objects.create(
        name="billing", dialect=Dialect.POSTGRES, connector_type="sql", ready=True)
    invoice = SchemaTable.objects.create(source=billing, tenant=TENANT, name="Invoice")
    SchemaColumn.objects.create(
        source=billing, tenant=TENANT, table=invoice, name="id", data_type="text")
    CatalogDiscoveryService().sync_source(billing)

    role = _role_for(member)
    _grant(role, read_perm, _path())  # fully open on crm
    from apps.access_management import resource_path as rp
    _grant(role, read_perm, rp.build("db", "billing"))  # fully open on billing too

    scope = compute_data_scope(member, [crm.id, billing.id])
    assert scope[crm.id].open is True
    assert scope[billing.id].open is True


def test_no_catalog_projection_fails_closed(member, read_perm):
    """Discovery never having run for a source (or a source with no schema at all)
    must deny, not silently permit everything under it."""
    source = _crm_source(with_catalog=False)
    role = _role_for(member)
    _grant(role, read_perm, _path())  # a grant exists, but nothing was ever projected

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.open is False
    assert scope.tables == ()
