"""Two more follow-up fixes from the implementation audit:

1. veda/planning.py::build_existence_sql() — the row-returning ("...list")
   mode used to build `SELECT a.* FROM "<anchor>" a WHERE ...` unconditionally.
   It now accepts optional sm/results/query/all_cols and, when given, projects
   veda/routing.py::recommended_projection() instead — omitting them keeps the
   historical a.* behavior byte-for-byte (regression guard below).

2. veda/generation.py::generate_join_sql() — the multi-table LLM prompt had NO
   Recommended Projection section at all (a completely separate function from
   the single-table generate_sql(), which already had one). It now accepts an
   optional `results` param and renders a per-alias Recommended Projection
   block when given.

Both are pure-function tests — no DB, no LLM (call_slm is monkeypatched in
the join test), no pipeline.run_query() needed since neither function is a
closure.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

ANCHOR = "worklists_incident"
CHILD = "worklists_comment"


class _FakeResult:
    def __init__(self, col_id, score):
        self.col_id = col_id
        self.column_name = col_id.split(".", 1)[1]
        self.final_score = score


_EDGES = [{"source_table": ANCHOR, "target_table": CHILD,
          "source_column": "id", "target_column": "incident_id"}]
_ALL_COLS = [f"{ANCHOR}.{c}" for c in
            ("id", "title", "status", "created_by_id", "deleted_at")]
_SM = {"columns": {
    f"{ANCHOR}.title": {"importance_class": "HIGH"},
    f"{ANCHOR}.status": {"importance_class": "HIGH"},
    f"{ANCHOR}.created_by_id": {"importance_class": "LOW"},
    f"{ANCHOR}.deleted_at": {"importance_class": "LOW"},
}}


def test_build_existence_sql_without_kwargs_keeps_select_star():
    """Regression guard: omitting the new optional params must reproduce
    today's exact SQL for any existing/other caller."""
    from veda.planning import build_existence_sql
    sql, tables = build_existence_sql(ANCHOR, _EDGES, "exists_list")
    assert sql == (f'SELECT a.* FROM "{ANCHOR}" a WHERE EXISTS '
                  f'(SELECT 1 FROM "{CHILD}" b WHERE b."incident_id" = a."id") LIMIT 100')


def test_build_existence_sql_with_kwargs_projects_business_columns():
    from veda.planning import build_existence_sql
    results = [_FakeResult(f"{ANCHOR}.title", 0.9)]
    sql, tables = build_existence_sql(ANCHOR, _EDGES, "exists_list", sm=_SM,
                                      results=results, query="incidents with comments",
                                      all_cols=_ALL_COLS)
    assert "a.*" not in sql
    assert 'a."title"' in sql and 'a."status"' in sql
    assert "created_by_id" not in sql and "deleted_at" not in sql


def test_build_existence_sql_count_mode_unaffected():
    """The COUNT(*) mode never had an a.* problem and must stay untouched
    regardless of whether the new kwargs are passed."""
    from veda.planning import build_existence_sql
    results = [_FakeResult(f"{ANCHOR}.title", 0.9)]
    sql, tables = build_existence_sql(ANCHOR, _EDGES, "exists_count", sm=_SM,
                                      results=results, query="how many incidents have comments",
                                      all_cols=_ALL_COLS)
    assert sql.startswith(f'SELECT COUNT(*) AS n FROM "{ANCHOR}"')


def test_generate_join_sql_omits_recommended_block_without_results():
    """Regression guard: no `results` -> byte-for-byte the old prompt (no
    Recommended Projection section at all)."""
    from veda.generation import generate_join_sql
    alias_map = {"t0": ANCHOR, "t1": CHILD}
    sm = {"columns": {f"{ANCHOR}.title": {}, f"{CHILD}.body": {}}}
    captured = {}

    def fake_call_slm(user, **kwargs):
        captured["user"] = user
        return "SELECT t0.title FROM x"

    import slm
    import unittest.mock as mock
    with mock.patch.object(slm, "call_slm", fake_call_slm):
        generate_join_sql("incidents and comments", "FROM x", alias_map, sm, None)
    assert "Recommended Projection" not in captured["user"]


def test_generate_join_sql_renders_per_alias_recommended_projection():
    from veda.generation import generate_join_sql
    alias_map = {"t0": ANCHOR, "t1": "users_user"}
    sm = {"columns": {
        f"{ANCHOR}.title": {"importance_class": "HIGH"},
        f"{ANCHOR}.status": {"importance_class": "HIGH"},
        f"{ANCHOR}.created_by_id": {"importance_class": "LOW"},
        "users_user.name": {"importance_class": "HIGH"},
        "users_user.email": {"importance_class": "LOW"},
    }}
    results = [_FakeResult(f"{ANCHOR}.title", 0.9), _FakeResult("users_user.name", 0.85)]
    captured = {}

    def fake_call_slm(user, **kwargs):
        captured["user"] = user
        return "SELECT t0.title FROM x"

    import slm
    import unittest.mock as mock
    with mock.patch.object(slm, "call_slm", fake_call_slm):
        generate_join_sql("incidents and their handler names", "FROM x", alias_map, sm, None,
                          results=results)

    user = captured["user"]
    assert "Recommended Projection" in user
    assert "t0: title, status" in user
    assert "t1: name" in user
    # The full, unfiltered per-alias column lists remain present too (never
    # replaced, only supplemented) — validation/correctness is unaffected.
    assert "t0 = worklists_incident: title, status, created_by_id" in user
    assert "t1 = users_user: name, email" in user


def test_generate_sql_rules_never_reference_recommended_projection_when_absent():
    """Regression guard for a bug caught while strengthening the Rules wording:
    the firmer instruction text must be gated on whether a Recommended
    Projection section was ACTUALLY rendered, not on whether the raw
    `recommended_projection` argument was merely non-empty — a caller passing
    a list equal to `columns` (nothing to narrow) must not get Rules text
    that references a section the prompt doesn't contain."""
    from veda.generation import generate_sql
    import unittest.mock as mock
    import slm

    captured = {}

    def fake_call_slm(user, **kwargs):
        captured["user"] = user
        return "SELECT id FROM t"

    columns = ["id", "title", "status"]
    with mock.patch.object(slm, "call_slm", fake_call_slm):
        generate_sql("list incidents", "worklists_incident", columns, None,
                     recommended_projection=list(columns))   # == columns -> no narrowing

    user = captured["user"]
    assert "Recommended Projection" not in user
    assert "Do not add other Available Columns" not in user
