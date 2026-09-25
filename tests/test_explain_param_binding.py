import traceback
"""Filter values must stay attached to the filters they were bound to.

Measured 2026-09-24, DRILLDOWN_QUERY_MATRIX scenario 23 (three-level drill: Nagpur →
FULL → gated). The executed SQL was correct —

    WHERE furnishing = %s AND city_name = %s AND is_gated = %s   params [full, nagpur, true]

— but conversation memory stored is_gated='full', furnishing='nagpur',
city_name='true', and explainability showed the user the same. `_extract` paired
params with placeholders by walking the tree, and sqlglot's find_all() is
breadth-first: ((a AND b) AND c) reaches `c` first. Two filters happened to be
safe (both at the same depth), which is why every earlier drill test passed.

Pure: needs sqlglot, no DB/SLM/network.
Run: `PYTHONPATH=.:veda_core python tests/test_explain_param_binding.py`
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from veda.business_explain import _extract  # noqa: E402
from veda.validation import validate_and_parameterize  # noqa: E402


def _bound(sql, params):
    return {c: v for c, _op, v in _extract(sql, params=params)["filters"]}


def _via_executor(literal_sql, tables, columns):
    """Parameterize exactly as the pipeline does, then read the values back."""
    psql, params, err = validate_and_parameterize(literal_sql, tables, columns)
    assert err is None, err
    return _bound(psql, list(params))


# ── the measured failure ──────────────────────────────────────────────────────────────
def test_three_filters_keep_their_own_values():
    sql = ('SELECT "facing", COUNT(*) FROM "assets_asset" '
           'WHERE LOWER(CAST("furnishing" AS TEXT)) = %s '
           'AND LOWER(CAST("city_name" AS TEXT)) = %s '
           'AND LOWER(CAST("is_gated" AS TEXT)) = %s GROUP BY "facing"')
    got = _bound(sql, ["full", "nagpur", "true"])
    assert got == {"furnishing": "full", "city_name": "nagpur", "is_gated": "true"}, got


def test_four_filters():
    sql = 'SELECT a FROM t WHERE w = %s AND x = %s AND y = %s AND z = %s'
    assert _bound(sql, [1, 2, 3, 4]) == {"w": 1, "x": 2, "y": 3, "z": 4}


# ── shapes where depth-first order would ALSO be wrong ────────────────────────────────
def test_filter_inside_a_cte_is_rendered_first():
    """A CTE renders first but is the last child walked — depth-first gets this wrong."""
    sql = ('WITH c AS (SELECT id FROM u WHERE m = %s) '
           'SELECT a FROM t JOIN c ON c.id = t.id WHERE x = %s AND y = %s')
    got = {c: v for c, _o, v in _extract(sql, params=["m1", "x1", "y1"])["filters"]}
    # _extract reports the OUTER WHERE; whatever it reports must carry its own value.
    assert got.get("x") == "x1" and got.get("y") == "y1", got


def test_literal_in_select_list_does_not_shift_the_filters():
    sql = "SELECT CASE WHEN s = %s THEN 1 END FROM t WHERE x = %s AND y = %s AND z = %s"
    got = _bound(sql, ["k", "p", "q", "r"])
    assert got.get("x") == "p" and got.get("y") == "q" and got.get("z") == "r", got


# ── end to end through the real parameterizer ─────────────────────────────────────────
def test_round_trip_through_validate_and_parameterize():
    sql = ("SELECT facing, COUNT(*) FROM assets_asset WHERE furnishing = 'full' "
           "AND city_name = 'nagpur' AND is_gated = 'true' GROUP BY facing")
    got = _via_executor(sql, {"assets_asset"}, {"facing", "furnishing", "city_name", "is_gated"})
    assert got == {"furnishing": "full", "city_name": "nagpur", "is_gated": "true"}, got


# ── unchanged behaviour ───────────────────────────────────────────────────────────────
def test_two_filters_unchanged():
    assert _bound("SELECT a FROM t WHERE x = %s AND y = %s", ["p", "q"]) == {"x": "p", "y": "q"}


def test_no_params_leaves_values_unknown():
    assert _bound("SELECT a FROM t WHERE x = %s AND y = %s AND z = %s", None) == \
        {"x": None, "y": None, "z": None}


def test_count_mismatch_is_unknown_not_wrong():
    """If the params cannot be accounted for one-to-one, say nothing rather than guess."""
    got = _bound("SELECT a FROM t WHERE x = %s AND y = %s AND z = %s", ["p", "q"])
    assert got == {"x": None, "y": None, "z": None}, got


def test_the_tree_is_left_as_it_was():
    import sqlglot
    from veda.business_explain import _bind_placeholders
    sql = "SELECT a FROM t WHERE x = %s AND y = %s AND z = %s"
    tree = sqlglot.parse_one(sql, read="postgres")
    before = tree.sql(dialect="postgres")
    _bind_placeholders(tree, [1, 2, 3])
    assert tree.sql(dialect="postgres") == before


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
