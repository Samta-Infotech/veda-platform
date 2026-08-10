"""Coverage for Gate 2 — the first component that can refuse a request.

  apps/access_management/gate.py   (RequiresPermission, VEDA_RBAC_MODE)

This is the highest-risk change in the RBAC programme: it is the only one that can
*break* working traffic. The tests are weighted accordingly:

  * ``off`` must be byte-identical to no gate at all
  * adding the gate must never GRANT anything — strictly tighter, never looser
  * ``shadow`` must decide and log but never refuse
  * a misconfigured view must be denied, not allowed

Run from repo root: ``pytest tests/test_gate.py``
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

import logging  # noqa: E402

from django.contrib.auth import get_user_model  # noqa: E402
from django.test import Client, override_settings  # noqa: E402

from apps.access_management.gate import (  # noqa: E402
    MODE_ENFORCE,
    MODE_OFF,
    MODE_SHADOW,
    RequiresPermission,
    rbac_mode,
)
from apps.access_management.models import (  # noqa: E402
    Effect,
    Permission,
    Role,
    RolePermission,
    UserRole,
)

ROLES_LIST = "/api/v1/roles/list"
USERS_LIST = "/api/v1/users/list"
PERMS_LIST = "/api/v1/permissions/list"

PASSWORD = "gate-correct-horse-staple"

_BASE = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
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

    with override_settings(**_BASE):
        cache.clear()
        atomic = db_transaction.atomic()
        atomic.__enter__()
        try:
            yield
        finally:
            db_transaction.set_rollback(True)
            atomic.__exit__(None, None, None)


@pytest.fixture
def staff():
    """A staff user with NO RBAC grants — the account every deployment already has."""
    return get_user_model().objects.create_user(
        username="root", password=PASSWORD, is_staff=True)


@pytest.fixture
def staff_client(staff):
    client = Client()
    client.force_login(staff)
    return client


def _grant(user, code, path="", effect=Effect.ALLOW, role_name=None):
    role = Role.objects.create(name=role_name or f"role-for-{code}-{path or 'global'}")
    UserRole.objects.create(user=user, role=role)
    RolePermission.objects.create(
        role=role, permission=Permission.objects.get(code=code),
        resource_path=path, effect=effect)
    return role


# ---------------------------------------------------------------------------
# off — the default must change nothing
# ---------------------------------------------------------------------------


def test_mode_defaults_to_off():
    """Prod behaviour must not change until someone deliberately changes it."""
    assert rbac_mode() == MODE_OFF


@override_settings(VEDA_RBAC_MODE=MODE_OFF)
@pytest.mark.parametrize("url", [ROLES_LIST, USERS_LIST, PERMS_LIST])
def test_off_leaves_a_grantless_staff_user_working(staff_client, url):
    """THE backward-compatibility test. Every existing deployment has staff accounts
    with zero RBAC grants; with the gate off they must be entirely unaffected."""
    assert staff_client.get(url).status_code == 200


@override_settings(VEDA_RBAC_MODE=MODE_OFF)
def test_off_does_not_even_resolve(staff_client):
    """The gate abstains before touching the database — no cost when disabled.

    Asserted by mocking the resolver directly, not by scanning the SQL log for
    ``rolepermission`` — ``roles/list`` itself legitimately queries that table now
    (``connected_sources`` in its response), which is unrelated to the gate and
    would otherwise make this test fail for a reason it was never about.
    """
    from unittest import mock

    with mock.patch("apps.access_management.gate.PermissionResolver") as resolver_cls:
        staff_client.get(ROLES_LIST)

    resolver_cls.assert_not_called()


# ---------------------------------------------------------------------------
# enforce — strictly tighter, never looser
# ---------------------------------------------------------------------------


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_enforce_denies_staff_without_the_permission(staff_client):
    assert staff_client.get(ROLES_LIST).status_code == 403


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_enforce_allows_staff_with_the_permission(staff, staff_client):
    _grant(staff, "role.manage")

    assert staff_client.get(ROLES_LIST).status_code == 200


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_the_gate_never_grants_what_staff_alone_refused(staff):
    """The guarantee that makes this safe to add: `IsAdminUser` is kept ALONGSIDE the
    gate, so a non-staff user with every RBAC permission is still refused."""
    member = get_user_model().objects.create_user(username="alice", password=PASSWORD)
    _grant(member, "role.manage")
    _grant(member, "user.manage")
    client = Client()
    client.force_login(member)

    assert client.get(ROLES_LIST).status_code == 403


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_enforce_still_rejects_anonymous_callers(staff):
    assert Client().get(ROLES_LIST).status_code == 401


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_permissions_are_per_endpoint_family(staff, staff_client):
    """`user.manage` must not open the role endpoints. A gate that granted everything
    to anyone holding any permission would be worse than none."""
    _grant(staff, "user.manage")

    assert staff_client.get(USERS_LIST).status_code == 200
    assert staff_client.get(ROLES_LIST).status_code == 403


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_a_deny_grant_refuses_even_with_an_allow(staff, staff_client):
    """DENY precedence reaches all the way to the HTTP response."""
    _grant(staff, "role.manage", role_name="Permissive")
    _grant(staff, "role.manage", effect=Effect.DENY, role_name="Restricted")

    assert staff_client.get(ROLES_LIST).status_code == 403


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_retiring_the_role_revokes_access_immediately(staff, staff_client):
    role = _grant(staff, "role.manage")
    assert staff_client.get(ROLES_LIST).status_code == 200

    role.is_active = False
    role.save(update_fields=["is_active", "updated_at"])

    assert staff_client.get(ROLES_LIST).status_code == 403


# ---------------------------------------------------------------------------
# shadow — decide, log, allow
# ---------------------------------------------------------------------------


@override_settings(VEDA_RBAC_MODE=MODE_SHADOW)
def test_shadow_never_refuses(staff_client):
    """The whole point: find out what enforcement would break, without breaking it."""
    assert staff_client.get(ROLES_LIST).status_code == 200


@override_settings(VEDA_RBAC_MODE=MODE_SHADOW)
def test_shadow_logs_what_it_would_have_denied(staff_client, caplog):
    with caplog.at_level(logging.WARNING, logger="apps.access_management.gate"):
        staff_client.get(ROLES_LIST)

    assert "WOULD DENY" in caplog.text
    assert "role.manage" in caplog.text


@override_settings(VEDA_RBAC_MODE=MODE_SHADOW)
def test_shadow_is_silent_when_it_would_have_allowed(staff, staff_client, caplog):
    """The shadow signal must be the exception list, not a copy of the access log —
    otherwise nobody can grep it for the work left to do."""
    _grant(staff, "role.manage")

    with caplog.at_level(logging.WARNING, logger="apps.access_management.gate"):
        staff_client.get(ROLES_LIST)

    assert "WOULD DENY" not in caplog.text


# ---------------------------------------------------------------------------
# Fail closed / misconfiguration
# ---------------------------------------------------------------------------


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_a_view_that_declares_no_permission_is_denied(staff_client, caplog):
    """Fail closed: a view opting into the gate without saying what it needs is a bug,
    and a bug in an authorization gate must be loud, not permissive."""
    from unittest import mock

    from apps.access_management.views import RoleListView

    with mock.patch.object(RoleListView, "required_permission", None):
        with caplog.at_level(logging.ERROR, logger="apps.access_management.gate"):
            response = staff_client.get(ROLES_LIST)

    assert response.status_code == 403
    assert "declares no required_permission" in caplog.text


@override_settings(VEDA_RBAC_MODE="typo-not-a-mode")
def test_an_unrecognised_mode_falls_back_to_off(staff_client, caplog):
    """Deliberately NOT fail-closed. Treating a typo as `enforce` would take an entire
    deployment offline; treating it as `off` preserves the status quo and logs loudly.
    """
    with caplog.at_level(logging.ERROR, logger="apps.access_management.gate"):
        response = staff_client.get(ROLES_LIST)

    assert rbac_mode() == MODE_OFF
    assert response.status_code == 200
    assert "not one of" in caplog.text


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


@override_settings(VEDA_RBAC_MODE=MODE_ENFORCE)
def test_the_resolver_runs_at_most_once_per_request(staff, staff_client):
    """Several permission classes — and, later, Gate 1 — must not each pay for their
    own traversal."""
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    _grant(staff, "role.manage")

    with CaptureQueriesContext(connection) as ctx:
        staff_client.get(ROLES_LIST)

    resolutions = [q for q in ctx.captured_queries
                   if "rolepermission" in q["sql"].lower()
                   and "userrole" in q["sql"].lower()]
    assert len(resolutions) == 1


def test_the_per_request_cache_is_on_the_request_not_a_global():
    """A module-level cache would leak one user's permissions into another's request
    under any concurrency. Pinned because it is invisible until it is catastrophic."""
    import apps.access_management.gate as gate_module

    source = open(gate_module.__file__).read()
    assert "_veda_effective_permissions" in source
    assert "request._veda_effective_permissions" in source


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_every_routed_endpoint_declares_a_permission():
    """A view that inherits the gate but forgets to declare is DENIED under
    enforcement — so the whole routed surface has to be checked, not assumed.

    Derived from the URL conf rather than from exported names: what matters is what is
    reachable, and the abstract ``AdminView`` base is deliberately undeclared because
    it is not an endpoint.
    """
    from apps.access_management.urls import urlpatterns

    routed = [(str(p.pattern), p.callback.view_class) for p in urlpatterns]
    assert routed, "no routes discovered — the check would pass vacuously"

    missing = [path for path, cls in routed
               if not getattr(cls, "required_permission", None)]
    assert missing == [], f"routed endpoints with no required_permission: {missing}"


def test_the_abstract_base_declares_nothing_on_purpose():
    """``AdminView`` must NOT carry a default permission — a default is what lets a
    concrete view forget to declare one and still pass the check above."""
    from apps.access_management.views import AdminView

    assert AdminView.required_permission is None


def test_the_gate_is_installed_alongside_is_admin_user():
    """Never instead of. Removing IsAdminUser here would make the gate the only
    barrier, and a bug in it would open every admin endpoint."""
    from rest_framework.permissions import IsAdminUser

    from apps.access_management.views import RoleListView

    assert IsAdminUser in RoleListView.permission_classes
    assert RequiresPermission in RoleListView.permission_classes


@override_settings(VEDA_RBAC_MODE=MODE_OFF)
def test_the_query_endpoint_is_still_untouched():
    """Gate 2 covers the admin surface only. `/api/v1/query` — the endpoint that
    actually reaches customer data — is NOT yet gated; that is Gate 1's phase, and
    this test should only change when it deliberately does."""
    from django.urls import resolve

    from rest_framework.permissions import AllowAny

    assert AllowAny in resolve("/api/v1/query").func.view_class.permission_classes
