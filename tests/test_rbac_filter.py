"""Coverage for Gate 1 (User Story 3) Task 16 — engine-side RBAC filtering.

  veda_core/veda/rbac_filter.py :: filter_sm, filter_retrieval_results, narrow_allowed

Pure functions, no Django, no heavy ML deps — runs against ``veda_core`` on
``sys.path`` directly, matching how the inference tier imports it.

The core safety property under test: ``filter_sm``/``filter_retrieval_results``
NEVER get called on the object handed to ``get_engine()`` (see the module
docstring and ``veda_hybrid.py::_load_semantic_model``'s own comment) — that
invariant is enforced by review/design, not by a unit test here; this file only
covers the filtering logic itself.

``narrow_allowed`` additionally closes the real gap found by the 2026-08-08
production audit: candidate filtering alone does not stop a forbidden
table/column from reaching generated SQL (FastPath, verified-cache replay,
FK/entity expansion, and single/multi-table planning all consult the raw,
unfiltered semantic model independently). ``test_narrow_allowed_end_to_end_*``
below proves the closed loop against the REAL ``validate_and_parameterize`` —
not just that the allowlist shrinks, but that shrinking it actually causes the
downstream firewall to reject SQL referencing what was trimmed out.

Run from repo root: ``pytest tests/test_rbac_filter.py``
"""
import os
import sys
from dataclasses import dataclass, field

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

from veda.rbac_filter import (  # noqa: E402
    filter_nosql_collections,
    filter_retrieval_results,
    filter_sm,
    narrow_allowed,
)
from veda.validation import validate_and_parameterize  # noqa: E402


def _ctx(source_ids, allowed_resources):
    @dataclass(frozen=True)
    class _Ctx:
        source_ids: tuple
        allowed_resources: tuple = None
    return _Ctx(source_ids=tuple(source_ids), allowed_resources=allowed_resources)


@dataclass
class _Result:
    col_id: str
    column_name: str = ""
    table_name: str = ""
    final_score: float = 0.0


def _sm(tables=None, columns=None, docs=None):
    return {"tables": tables or {}, "columns": columns or {}, "retrieval_documents": docs or {}}


# ---------------------------------------------------------------------------
# filter_sm
# ---------------------------------------------------------------------------

def test_no_ctx_is_identity():
    sm = _sm(tables={"t": {}})
    assert filter_sm(sm, None) is sm


def test_ctx_with_no_allowed_resources_is_identity():
    sm = _sm(tables={"t": {}})
    ctx = _ctx([1], None)
    assert filter_sm(sm, ctx) is sm


def test_open_source_keeps_everything():
    sm = _sm(
        tables={"employee": {}, "department": {}},
        columns={"employee.id": {}, "employee.salary": {}, "department.id": {}},
    )
    ctx = _ctx([1], ((1, (True, ())),))
    out = filter_sm(sm, ctx)
    assert set(out["tables"]) == {"employee", "department"}
    assert set(out["columns"]) == {"employee.id", "employee.salary", "department.id"}


def test_restricted_source_keeps_only_listed_tables():
    sm = _sm(
        tables={"employee": {}, "department": {}},
        columns={"employee.id": {}, "department.id": {}},
    )
    ctx = _ctx([1], ((1, (False, (("employee", None),))),))  # employee: whole table open
    out = filter_sm(sm, ctx)
    assert set(out["tables"]) == {"employee"}
    assert set(out["columns"]) == {"employee.id"}


def test_restricted_table_keeps_only_listed_columns():
    sm = _sm(
        tables={"employee": {}},
        columns={"employee.id": {}, "employee.name": {}, "employee.salary": {}},
    )
    ctx = _ctx([1], ((1, (False, (("employee", ("id", "name")),))),))
    out = filter_sm(sm, ctx)
    assert set(out["tables"]) == {"employee"}
    assert set(out["columns"]) == {"employee.id", "employee.name"}


def test_source_absent_from_allowed_resources_is_denied():
    sm = _sm(tables={"employee": {}}, columns={"employee.id": {}})
    ctx = _ctx([1, 2], ((1, (True, ())),))  # source 2 never mentioned at all
    out = filter_sm(sm, ctx)
    assert out["tables"] == {"employee": sm["tables"]["employee"]}  # source 1's table, untouched
    # a second-source table would be denied — simulate via _source_id tag:
    sm2 = _sm(tables={"other": {"_source_id": 2}}, columns={"other.id": {"_source_id": 2}})
    out2 = filter_sm(sm2, ctx)
    assert out2["tables"] == {}
    assert out2["columns"] == {}


def test_multi_source_merged_qualified_table_name_is_resolved_correctly():
    """`_merge_scoped_sms` qualifies a colliding table name as `src{ID}.{table}` —
    the column key then becomes `src{ID}.{table}.{col}`, THREE dots deep. Splitting
    naively on the first dot would wrongly separate the qualifier from the table."""
    sm = _sm(
        tables={"src2.employee": {"_source_id": 2}},
        columns={"src2.employee.id": {"_source_id": 2},
                 "src2.employee.salary": {"_source_id": 2}},
    )
    ctx = _ctx([2], ((2, (False, (("employee", ("id",)),))),))
    out = filter_sm(sm, ctx)
    assert set(out["tables"]) == {"src2.employee"}
    assert set(out["columns"]) == {"src2.employee.id"}


def test_retrieval_documents_follow_their_column_or_table():
    sm = _sm(
        tables={"employee": {}, "department": {}},
        columns={"employee.id": {}, "employee.salary": {}},
        docs={"employee.salary": {"chunk": "x"}, "department": {"chunk": "y"}},
    )
    ctx = _ctx([1], ((1, (False, (("employee", ("id",)),))),))  # salary denied, employee id ok
    out = filter_sm(sm, ctx)
    # employee.salary denied -> its doc must drop; department table denied -> its doc must drop.
    assert out["retrieval_documents"] == {}


def test_never_mutates_the_input():
    sm = _sm(tables={"employee": {}}, columns={"employee.id": {}})
    original_tables = sm["tables"]
    ctx = _ctx([1], ((1, (False, (("employee", ("id",)),))),))
    filter_sm(sm, ctx)
    assert sm["tables"] is original_tables
    assert sm["tables"] == {"employee": {}}


# ---------------------------------------------------------------------------
# filter_retrieval_results
# ---------------------------------------------------------------------------

def test_results_no_ctx_is_identity():
    results = [_Result(col_id="employee.id")]
    assert filter_retrieval_results(results, _sm(), None) is results


def test_results_none_and_empty_pass_through():
    ctx = _ctx([1], ((1, (True, ())),))
    assert filter_retrieval_results(None, _sm(), ctx) is None
    assert filter_retrieval_results([], _sm(), ctx) == []


def test_results_open_source_keeps_all_candidates():
    sm = _sm(tables={"employee": {}, "department": {}})
    results = [_Result(col_id="employee.id"), _Result(col_id="department.name")]
    ctx = _ctx([1], ((1, (True, ())),))
    assert filter_retrieval_results(results, sm, ctx) == results


def test_results_restricted_source_drops_forbidden_candidates():
    sm = _sm(tables={"employee": {}, "department": {}})
    results = [_Result(col_id="employee.id"), _Result(col_id="employee.salary"),
              _Result(col_id="department.name")]
    ctx = _ctx([1], ((1, (False, (("employee", ("id",)),))),))
    kept = filter_retrieval_results(results, sm, ctx)
    assert [r.col_id for r in kept] == ["employee.id"]


def test_results_a_candidate_for_an_unlisted_table_is_dropped():
    sm = _sm(tables={"ghost": {}})
    results = [_Result(col_id="ghost.id")]
    ctx = _ctx([1], ((1, (False, ())),))  # nothing granted at all
    assert filter_retrieval_results(results, sm, ctx) == []


def test_results_multi_source_candidate_resolved_via_source_id_tag():
    sm = _sm(tables={"employee": {"_source_id": 1}, "invoice": {"_source_id": 2}})
    results = [_Result(col_id="employee.id"), _Result(col_id="invoice.amount")]
    ctx = _ctx([1, 2], ((1, (True, ())), (2, (False, ()))))  # source 1 open, source 2 denied
    kept = filter_retrieval_results(results, sm, ctx)
    assert [r.col_id for r in kept] == ["employee.id"]


# ---------------------------------------------------------------------------
# narrow_allowed — the centralized final gate before validate_and_parameterize
# ---------------------------------------------------------------------------

def test_narrow_no_ctx_is_identity():
    tables, cols = narrow_allowed({"employee"}, ["id", "salary"], _sm(), None)
    assert tables == {"employee"} and cols == ["id", "salary"]


def test_narrow_ctx_with_no_allowed_resources_is_identity():
    sm = _sm(tables={"employee": {}})
    ctx = _ctx([1], None)
    tables, cols = narrow_allowed({"employee"}, ["id"], sm, ctx)
    assert tables == {"employee"} and cols == ["id"]


def test_narrow_open_source_keeps_everything_proposed():
    sm = _sm(tables={"employee": {}, "department": {}},
            columns={"employee.id": {}, "department.name": {}})
    ctx = _ctx([1], ((1, (True, ())),))
    tables, cols = narrow_allowed({"employee", "department"}, ["id", "name"], sm, ctx)
    assert tables == {"employee", "department"}
    assert sorted(cols) == ["id", "name"]


def test_narrow_drops_a_table_not_in_the_data_scope():
    sm = _sm(tables={"employee": {}, "department": {}})
    ctx = _ctx([1], ((1, (False, (("employee", None),))),))  # only employee granted
    tables, cols = narrow_allowed({"employee", "department"}, ["id"], sm, ctx)
    assert tables == {"employee"}


def test_narrow_drops_forbidden_columns_of_a_partially_open_table():
    sm = _sm(tables={"employee": {}},
            columns={"employee.id": {}, "employee.salary": {}})
    ctx = _ctx([1], ((1, (False, (("employee", ("id",)),))),))  # salary denied
    tables, cols = narrow_allowed({"employee"}, ["id", "salary"], sm, ctx)
    assert tables == {"employee"}
    assert cols == ["id"]


def test_narrow_zero_permitted_tables_yields_zero_columns():
    sm = _sm(tables={"ghost": {}})
    ctx = _ctx([1], ((1, (False, ())),))  # nothing granted
    tables, cols = narrow_allowed({"ghost"}, ["id"], sm, ctx)
    assert tables == set()
    assert cols == []


def test_narrow_a_table_absent_from_sm_is_never_assumed_safe():
    """A caller proposing a table that doesn't even exist in `sm` (a CTE alias,
    or a name from a stale/mismatched source) must not be waved through just
    because it isn't explicitly denied."""
    sm = _sm(tables={})
    ctx = _ctx([1], ((1, (True, ())),))  # source-wide open — but the table is unknown
    tables, cols = narrow_allowed({"mystery"}, ["id"], sm, ctx)
    assert tables == set()


def test_narrow_never_mutates_its_inputs():
    sm = _sm(tables={"employee": {}}, columns={"employee.id": {}, "employee.salary": {}})
    original_tables = {"employee"}
    ctx = _ctx([1], ((1, (False, (("employee", ("id",)),))),))
    narrow_allowed(original_tables, ["id", "salary"], sm, ctx)
    assert original_tables == {"employee"}  # the caller's own set is untouched


# --- end-to-end: proves the closed loop against the REAL SQL firewall -------

def test_narrow_allowed_end_to_end_forbidden_column_is_rejected_by_validate_and_parameterize():
    """The actual security property: SQL that references a column RBAC narrowed
    away is REJECTED — not silently rewritten, not silently permitted — by the
    same firewall that already catches LLM hallucination. This is what closes
    the 2026-08-08 audit's finding that candidate-list filtering alone doesn't
    stop a forbidden column from reaching generated SQL."""
    sm = _sm(tables={"employee": {}},
            columns={"employee.id": {}, "employee.salary": {}})
    ctx = _ctx([1], ((1, (False, (("employee", ("id",)),))),))  # salary denied

    # Some upstream path (FastPath / cache replay / FK expansion / Tier-2 — it
    # doesn't matter which) proposed BOTH columns, unaware of RBAC.
    proposed_tables, proposed_columns = {"employee"}, ["id", "salary"]
    tables, cols = narrow_allowed(proposed_tables, proposed_columns, sm, ctx)

    sql = "SELECT employee.id, employee.salary FROM employee"
    _, _, err = validate_and_parameterize(sql, tables, cols)
    assert err is not None
    assert "salary" in err.lower()


def test_narrow_allowed_end_to_end_permitted_query_still_succeeds():
    """The negative control: narrowing must not break a query that only touches
    what's actually permitted."""
    sm = _sm(tables={"employee": {}},
            columns={"employee.id": {}, "employee.salary": {}})
    ctx = _ctx([1], ((1, (False, (("employee", ("id",)),))),))

    tables, cols = narrow_allowed({"employee"}, ["id", "salary"], sm, ctx)

    sql = "SELECT employee.id FROM employee"
    param_sql, params, err = validate_and_parameterize(sql, tables, cols)
    assert err is None
    assert param_sql is not None


def test_narrow_allowed_end_to_end_forbidden_table_from_fk_expansion_is_rejected():
    """Simulates the audit's FK-neighbour-expansion finding directly: a second
    table gets added to the allowlist by some join-planner that never consulted
    RBAC (exactly how pipeline.py's `allowed_tables = {primary, _fk["target"]}`
    reads). narrow_allowed must strip it before the firewall sees it."""
    sm = _sm(tables={"employee": {}, "salary_history": {}},
            columns={"employee.id": {}, "salary_history.amount": {}})
    ctx = _ctx([1], ((1, (False, (("employee", None),))),))  # only employee granted

    # A join planner proposed a second table it discovered via an FK graph,
    # oblivious to RBAC.
    proposed_tables = {"employee", "salary_history"}
    proposed_columns = ["id", "amount"]
    tables, cols = narrow_allowed(proposed_tables, proposed_columns, sm, ctx)

    sql = "SELECT employee.id, salary_history.amount FROM employee JOIN salary_history"
    _, _, err = validate_and_parameterize(sql, tables, cols)
    assert err is not None
    assert "salary_history" in err.lower()


# ---------------------------------------------------------------------------
# filter_nosql_collections — the NoSQL mirror of filter_sm (2026-08-08 audit:
# _run_nosql had no RBAC filtering at all, unlike the relational path)
# ---------------------------------------------------------------------------

@dataclass
class _Collection:
    collection_id: str
    collection_name: str
    source_id: str
    engine: str
    inferred_fields: list
    doc_count: int
    metadata: dict = field(default_factory=dict)


def _coll(name, fields):
    return _Collection(collection_id=name, collection_name=name, source_id="1",
                       engine="mongodb",
                       inferred_fields=[{"name": f} for f in fields], doc_count=0)


def test_nosql_no_ctx_is_identity():
    colls = [_coll("employees", ["id", "salary"])]
    assert filter_nosql_collections(colls, 1, None) is colls


def test_nosql_open_source_keeps_everything():
    colls = [_coll("employees", ["id", "salary"]), _coll("departments", ["id"])]
    ctx = _ctx([1], ((1, (True, ())),))
    assert filter_nosql_collections(colls, 1, ctx) is colls


def test_nosql_restricted_source_drops_an_unlisted_collection():
    colls = [_coll("employees", ["id"]), _coll("departments", ["id"])]
    ctx = _ctx([1], ((1, (False, (("employees", None),))),))  # employees: whole collection open
    kept = filter_nosql_collections(colls, 1, ctx)
    assert [c.collection_name for c in kept] == ["employees"]


def test_nosql_partial_collection_keeps_only_permitted_fields():
    colls = [_coll("employees", ["id", "salary"])]
    ctx = _ctx([1], ((1, (False, (("employees", ("id",)),))),))  # salary denied
    kept = filter_nosql_collections(colls, 1, ctx)
    assert len(kept) == 1
    assert [f["name"] for f in kept[0].inferred_fields] == ["id"]


def test_nosql_collection_with_zero_reachable_fields_is_dropped():
    colls = [_coll("employees", ["id", "salary"])]
    ctx = _ctx([1], ((1, (False, (("employees", ()),))),))  # granted, but no fields at all
    assert filter_nosql_collections(colls, 1, ctx) == []


def test_nosql_a_second_source_id_is_denied_independently():
    colls = [_coll("employees", ["id"])]
    ctx = _ctx([1, 2], ((1, (True, ())),))  # only source 1 is open; source 2 unmentioned
    assert filter_nosql_collections(colls, 2, ctx) == []


def test_nosql_never_mutates_a_kept_collection_in_place():
    coll = _coll("employees", ["id", "salary"])
    original_fields = coll.inferred_fields
    ctx = _ctx([1], ((1, (False, (("employees", ("id",)),))),))
    kept = filter_nosql_collections([coll], 1, ctx)
    assert coll.inferred_fields is original_fields  # the input object is untouched
    assert kept[0] is not coll  # a new instance was returned instead
