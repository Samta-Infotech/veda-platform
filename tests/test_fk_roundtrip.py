"""FkEdge writer→reader round-trip (migration plan Phase 3 exit criterion 2).

Writes an FK edge through storage_adapters.writer and reads it back through
storage_adapters.reader, asserting the reader returns the identical legacy `FKEdge`
structure the engine's callers expect. Run in a Django-configured process.

Runnable two ways:
  - standalone script: ``python tests/test_fk_roundtrip.py`` (Django-configured process)
  - collected by pytest: ``pytest tests/test_fk_roundtrip.py``
django.setup() and the Django-app imports are deferred into the test body (they
crashed ``pytest tests/`` collection at import time). In a mixed ``pytest tests/``
run the veda_core-tier tests put veda_core/ on sys.path, whose config.py shadows
the Django ``config`` package — so django.setup() cannot configure here and the
test SKIPS rather than crashing collection. Run it in its own Django-configured
invocation to actually exercise the round-trip.
"""
import os
import uuid


def test_fk_roundtrip():
    import pytest
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    try:
        import django
        django.setup()
        from veda_core.context import RequestContext, set_context
        from storage_adapters import reader, writer
        from storage_adapters.reader import FKEdge  # numpy-free, field-identical to legacy
        from apps.substrate.models import FkEdge as M, SchemaColumn, SchemaTable
    except Exception as exc:  # settings unconfigurable (config shadow) / app not ready
        pytest.skip(f"Django app context unavailable in this process: {type(exc).__name__}: {exc}")

    set_context(RequestContext(source_id=1, tenant="fk_test"))

    ft, tt = str(uuid.uuid4()), str(uuid.uuid4())
    fc, tc = str(uuid.uuid4()), str(uuid.uuid4())
    edge = FKEdge(
        from_col_id=fc, from_col_name="user_id", from_table_id=ft, from_table_name="orders",
        to_col_id=tc, to_col_name="id", to_table_id=tt, to_table_name="user",
    )

    # clean any prior test rows for this tenant
    M.objects.all_tenants().filter(tenant="fk_test").delete()
    SchemaColumn.objects.all_tenants().filter(tenant="fk_test").delete()
    SchemaTable.objects.all_tenants().filter(tenant="fk_test").delete()

    n = writer.store_fk_adjacency([edge])
    assert n == 1, n

    got = reader.get_fk_adjacency([ft])
    assert len(got) == 1, f"expected 1 edge, got {len(got)}"
    g = got[0]
    for field in ("from_col_id", "from_col_name", "from_table_id", "from_table_name",
                  "to_col_id", "to_col_name", "to_table_id", "to_table_name"):
        assert getattr(g, field) == getattr(edge, field), (
            f"{field}: reader={getattr(g, field)!r} != written={getattr(edge, field)!r}"
        )
    assert type(g).__name__ == "FKEdge", type(g)

    # cleanup
    M.objects.all_tenants().filter(tenant="fk_test").delete()
    SchemaColumn.objects.all_tenants().filter(tenant="fk_test").delete()
    SchemaTable.objects.all_tenants().filter(tenant="fk_test").delete()
    print("FK ROUND-TRIP OK — writer→reader returns identical legacy FKEdge")


if __name__ == "__main__":
    test_fk_roundtrip()
