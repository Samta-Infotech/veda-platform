"""The inference route's wire boundary must not hand a raw engine error to a caller.

  inference/routes/hybrid.py :: _sanitise_error, _serialize

Measured on the direct endpoint (`POST /api/v1/query`): `refuse_reason` and
`result.error` carried a full DuckDB failure — raw relational table names, a SQL
fragment with column aliases, and a hint naming the storage engine. The chat path
shows safe copy for the same failure; this route did not.

Pure python — no Django, no network.
"""
import importlib.util
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _load():
    spec = importlib.util.spec_from_file_location(
        "_standalone_hybrid_route",
        os.path.join(_ROOT, "inference", "routes", "hybrid.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_standalone_hybrid_route"] = mod
    spec.loader.exec_module(mod)
    return mod


h = _load()

INTERNAL = [
    'Catalog Error: Table with name assets_amenitycategory does not exist! '
    'Did you mean "amenities_catalog"?',
    'COUNT(DISTINCT "id") AS "assets_amenitycategory_count" ... GROUP BY "x"',
    'Did you mean "pg_settings"?',
    'relation "assets_asset" does not exist',
    'SELECT amount FROM accounts_generalledger',
    'psycopg2.OperationalError: could not translate host name "pgbouncer"',
    'Binder Error: Referenced column "maintenance_amount" not found',
]

# Business-level refusals: already written for a user, must survive untouched.
USER_FACING = [
    "I couldn't map 'maintenance' to any column or value in the data.",
    "You don't have permission to access this data. Contact your Admin to request access.",
    "this asks for a count, total or average, but what came back is a list of "
    "individual records rather than the single figure requested",
    "more than one grouping fits what you asked for — did you mean status or loe status?",
]


@pytest.mark.parametrize("msg", INTERNAL)
def test_an_engine_internal_error_never_reaches_a_caller(msg):
    out = h._sanitise_error(msg)
    assert out != msg, f"leaked verbatim: {msg[:70]}"
    low = out.lower()
    for banned in ("assets_", "accounts_", "select ", "group by", "pg_",
                   "catalog error", "psycopg2", "did you mean", "pgbouncer"):
        assert banned not in low, f"{banned!r} survived sanitisation: {out}"


@pytest.mark.parametrize("msg", USER_FACING)
def test_a_refusal_already_written_for_a_user_is_passed_through(msg):
    assert h._sanitise_error(msg) == msg, (
        "a business refusal is the answer the user is meant to read — replacing it "
        "with generic copy would lose the guidance it carries")


def test_sanitisation_applies_wherever_the_key_appears():
    """_serialize is the ONE boundary every head result crosses, at any depth."""
    payload = {"items": [{"refuse_reason": 'Catalog Error: no table "assets_asset"',
                          "result": {"error": 'relation "accounts_x" does not exist',
                                     "rows": [[1]], "answer": "fine"}}]}
    out = h._serialize(payload)
    blob = str(out).lower()
    assert "assets_asset" not in blob and "accounts_x" not in blob
    assert "catalog error" not in blob and "relation " not in blob
    assert out["items"][0]["result"]["answer"] == "fine", "other keys untouched"
    assert out["items"][0]["result"]["rows"] == [[1]]


def test_non_string_and_empty_values_are_left_alone():
    for value in (None, 0, "", [], {}, False):
        assert h._sanitise_error(value) == value


# --------------------------------------------------------------------------- explain
def test_an_empty_explain_becomes_null_not_a_truthy_empty_object():
    """`explain: {}` is truthy in JS, so a client's `if (explain)` rendered an
    explainability panel with every block missing (observed on `exec_error`)."""
    out = h._serialize({"status": "tier2_exec_error", "ok": False, "explain": {}})
    assert out["explain"] is None
    assert "explain" in out, "the key must stay — a client is built against the shape"


def test_a_real_explain_is_untouched():
    ex = {"version": "1.0", "understanding": {"summary": "s"}}
    assert h._serialize({"explain": ex})["explain"] == ex


def test_empty_explain_is_normalised_at_any_depth():
    out = h._serialize({"items": [{"result": {"explain": {}}}]})
    assert out["items"][0]["result"]["explain"] is None
