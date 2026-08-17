"""Coverage for the Django-side of this session's filesystem/document (RAG)
pipeline fixes:

  1. apps.access_management.services.catalog::CatalogDiscoveryService
     — "files"-kind sources get per-document CatalogResource children
       (previously: only the source-level root, ever — no document was ever
       individually addressable/grantable).

  2. apps.access_management.services.data_scope::compute_data_scope
     — "files"-kind sources enumerate their allowed DOCUMENTS in the wire
       payload (previously: always empty/broken — _substrate_names only knows
       SchemaTable/SchemaColumn, which documents have neither of).

  3. chatbot/nodes.py::_extract_engine_result
     — a RAG/hybrid engine result (no pipeline-level "status" key of its own)
       is correctly recognized as "answered" instead of silently discarded as
       a generic "Could you clarify?" (the SQL/Tier-1/Tier-2 pipeline's
       _done()-minted "status" is the only kind this used to recognize).
       chatbot.nodes imports standalone (no django.setup() needed — same as
       tests/test_chatbot_classify.py), so it's covered here too rather than
       in a third file.

Deliberately does NOT put veda_core/ on sys.path (unlike test_rbac_filter.py):
veda_core/config.py shadows Django's own "config" package the moment it's
importable via a bare `config` module name, which breaks django.setup() here.
See tests/test_filesystem_rbac_and_retrieval.py for the veda_core-side coverage.

Run from repo root: ``pytest tests/test_filesystem_catalog_and_scope.py``
"""
from __future__ import annotations

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

from django.contrib.auth import get_user_model  # noqa: E402
from django.test import override_settings  # noqa: E402

from apps.access_management.gate import MODE_ENFORCE  # noqa: E402
from apps.access_management.models import Effect, Permission, Role, RolePermission, UserRole  # noqa: E402
from apps.access_management.services.catalog import CatalogDiscoveryService  # noqa: E402
from apps.access_management.services.data_scope import compute_data_scope  # noqa: E402
from apps.sources.models import Dialect, Source  # noqa: E402

import chatbot.nodes as chatbot_nodes  # noqa: E402

TENANT = "default"

_TEST_OVERRIDES = dict(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
    VEDA_RBAC_MODE=MODE_ENFORCE,
)


# ---------------------------------------------------------------------------
# Django DB fixtures (same shape as tests/test_data_scope.py)
# ---------------------------------------------------------------------------

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
    return get_user_model().objects.create_user(username="alice", password="x")


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


def _docs_source():
    """A filesystem source with no real doc_chunks connection — callers mock
    CatalogDiscoveryService._document_rows instead of touching veda_engine."""
    return Source.objects.create(
        name="contracts", dialect=Dialect.FILESYSTEM, connector_type="filesystem",
        ready=True)


def _path(*parts):
    from apps.access_management import resource_path as rp
    return rp.build("files", "contracts", *parts)


def _mock_doc_rows(*rows):
    """rows: (doc_id, doc_name) tuples, as CatalogDiscoveryService._document_rows
    (a raw psycopg2 read against veda_engine's doc_chunks) would return."""
    return mock.patch.object(
        CatalogDiscoveryService, "_document_rows", staticmethod(lambda source_id: list(rows)))


# ---------------------------------------------------------------------------
# 1. CatalogDiscoveryService — per-document catalog resources for "files" kind
# ---------------------------------------------------------------------------

def test_discovery_creates_one_child_resource_per_document():
    source = _docs_source()
    with _mock_doc_rows(("doc-1", "msa.pdf"), ("doc-2", "notes.md")):
        report = CatalogDiscoveryService().sync_source(source)

    assert report.created == 3  # root + 2 documents
    from apps.access_management.models import CatalogResource
    paths = set(CatalogResource.objects.filter(source=source).values_list("path", flat=True))
    assert paths == {"files:contracts", "files:contracts:msa.pdf", "files:contracts:notes.md"}


def test_discovery_document_child_carries_the_engine_doc_id_as_substrate_id():
    source = _docs_source()
    with _mock_doc_rows(("doc-1", "msa.pdf")):
        CatalogDiscoveryService().sync_source(source)

    from apps.access_management.models import CatalogResource
    row = CatalogResource.objects.get(source=source, path="files:contracts:msa.pdf")
    assert str(row.substrate_id) == "doc-1"
    assert row.parent_path == "files:contracts"


def test_discovery_with_zero_documents_creates_only_the_root():
    source = _docs_source()
    with _mock_doc_rows():
        report = CatalogDiscoveryService().sync_source(source)

    assert report.created == 1
    from apps.access_management.models import CatalogResource
    assert list(CatalogResource.objects.filter(source=source).values_list("path", flat=True)) \
        == ["files:contracts"]


def test_discovery_rerun_deactivates_a_document_removed_upstream():
    source = _docs_source()
    with _mock_doc_rows(("doc-1", "msa.pdf"), ("doc-2", "notes.md")):
        CatalogDiscoveryService().sync_source(source)

    with _mock_doc_rows(("doc-1", "msa.pdf")):  # notes.md gone upstream
        report = CatalogDiscoveryService().sync_source(source)

    assert report.deactivated == 1
    from apps.access_management.models import CatalogResource
    notes = CatalogResource.objects.get(source=source, path="files:contracts:notes.md")
    assert notes.is_active is False
    msa = CatalogResource.objects.get(source=source, path="files:contracts:msa.pdf")
    assert msa.is_active is True


def test_db_kind_source_is_completely_unaffected_by_the_files_branch():
    """The files-kind branch must never fire for a relational source — regression
    guard for the exact bug this session found in vendor/table_metadata drift."""
    source = Source.objects.create(
        name="crm", dialect=Dialect.POSTGRES, connector_type="sql", ready=True)
    from apps.substrate.models import SchemaColumn, SchemaTable
    table = SchemaTable.objects.create(source=source, tenant=TENANT, name="employee")
    SchemaColumn.objects.create(source=source, tenant=TENANT, table=table, name="id",
                                data_type="text")

    with mock.patch.object(CatalogDiscoveryService, "_document_rows") as doc_rows:
        CatalogDiscoveryService().sync_source(source)
    doc_rows.assert_not_called()


# ---------------------------------------------------------------------------
# 2. compute_data_scope — per-document allow-list for "files" kind
# ---------------------------------------------------------------------------

def test_data_scope_enumerates_only_the_allowed_document(member, read_perm):
    source = _docs_source()
    with _mock_doc_rows(("doc-1", "msa.pdf"), ("doc-2", "notes.md")):
        CatalogDiscoveryService().sync_source(source)
    role = _role_for(member)
    _grant(role, read_perm, _path("msa.pdf"))  # notes.md NOT granted

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.open is False
    assert [t.name for t in scope.tables] == ["msa.pdf"]
    assert scope.tables[0].columns is None  # a document has no column grain


def test_data_scope_no_document_grants_is_fully_restricted(member, read_perm):
    source = _docs_source()
    with _mock_doc_rows(("doc-1", "msa.pdf")):
        CatalogDiscoveryService().sync_source(source)
    _role_for(member)  # role exists, but grants nothing under this source

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.open is False
    assert scope.tables == ()


def test_data_scope_source_level_allow_is_fully_open_for_files_kind(member, read_perm):
    source = _docs_source()
    with _mock_doc_rows(("doc-1", "msa.pdf"), ("doc-2", "notes.md")):
        CatalogDiscoveryService().sync_source(source)
    role = _role_for(member)
    _grant(role, read_perm, _path())  # whole source, no narrower deny

    scope = compute_data_scope(member, [source.id])[source.id]
    assert scope.open is True
    assert scope.tables == ()


# ---------------------------------------------------------------------------
# 3. _extract_engine_result — RAG/hybrid results must not be discarded as
#    "Could you clarify?" for lacking a pipeline-level "status" key
# ---------------------------------------------------------------------------

def _payload(item0):
    return {"result": {"items": [item0], "trace_id": "t"}}


def test_sql_result_keeps_its_own_pipeline_status_unchanged():
    """Regression guard: the fix must not touch a route that DOES set status."""
    res0, status = chatbot_nodes._extract_engine_result(
        _payload({"status": "ok", "route": "deterministic",
                  "result": {"status": "answered", "rows": [[1]], "cols": ["n"]}}))
    assert status == "answered"
    assert res0["rows"] == [[1]]


def test_sql_exec_error_is_still_an_error():
    res0, status = chatbot_nodes._extract_engine_result(
        _payload({"status": "error", "route": "deterministic",
                  "result": {"status": "exec_error", "error": "boom"}}))
    assert status == "exec_error"


def test_rag_result_with_no_status_key_and_ok_subresult_is_answered():
    """The exact bug: a RAG answer has no res0['status'] at all — before the fix
    this always fell through to 'error' and got shown as a generic clarify."""
    res0, status = chatbot_nodes._extract_engine_result(
        _payload({"status": "ok", "route": "rag",
                  "result": {"answer": "the late fee is 2%", "citations": ["msa.pdf"],
                             "error": None}}))
    assert status == "answered"
    assert res0["answer"] == "the late fee is 2%"


def test_hybrid_result_with_no_status_key_and_ok_subresult_is_answered():
    res0, status = chatbot_nodes._extract_engine_result(
        _payload({"status": "ok", "route": "hybrid",
                  "result": {"answer": "100 rows; late fee is 2%", "error": None}}))
    assert status == "answered"


def test_rag_result_with_failed_subresult_is_an_error_not_a_silent_answer():
    """A genuinely failed RAG call (item0.status != 'ok') must NOT be upgraded to
    'answered' just because res0 also lacks a status key."""
    res0, status = chatbot_nodes._extract_engine_result(
        _payload({"status": "error", "route": "rag",
                  "result": {"answer": "", "error": "embedding failed"}}))
    assert status == "error"


def test_no_items_at_all_is_an_error():
    res0, status = chatbot_nodes._extract_engine_result({"result": {"items": []}})
    assert status == "error"
    assert res0 == {}
