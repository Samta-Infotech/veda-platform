"""No raw table/column names in the answer text (query/business_wording.py).

Measured 2026-09-26: "The corner_property field shows that 64% of assets are in corners
... The assets_asset_count ranges widely from 1 to 471".

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_business_wording.py -q`
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from query.business_wording import business_wording, column_label  # noqa: E402

SM = {"tables": {"assets_asset": {"primary_entity": "Asset"},
                 "assets_leaselisting": {"primary_entity": "A lease listing for an asset."},
                 "users_user": {"primary_entity": "A user of the system."},
                 "accounts_generalledger": {"primary_entity": "A single financial transaction."}},
      "columns": {"assets_asset.corner_property": {"business_role": "Location Feature"},
                  "assets_asset.carpet_area": {"business_role": "Measure"},
                  "assets_asset.facing": {}}}
COLS = ["facing", "corner_property", "assets_asset_count"]
MEASURED = ("The corner_property field shows that 64% of assets are in corners, while the "
            "facing field indicates that only two out of the fourteen groups face east or "
            "west, both above average in terms of assets_asset_count. The "
            "assets_asset_count ranges widely from 1 to 471.")


def test_the_measured_summary():
    out = business_wording(MEASURED, COLS, "assets_asset", SM)
    assert "_" not in out
    assert "The corner property field shows" in out
    assert "in terms of number of assets" in out
    assert "The number of assets ranges widely from 1 to 471." in out
    assert "the facing field" in out.lower()          # one-word column left as written


@pytest.mark.parametrize("col,label", [
    ("assets_asset_count", "number of assets"),
    ("vendor_count", "number of vendors"),
    ("count_of_tickets", "number of tickets"),
    ("total_amount", "total amount"),
    ("avg_expected_monthly_rent", "average expected monthly rent"),
    ("amount_sum", "total amount"),
    ("max_rating", "maximum rating"),
    ("carpet_area", "carpet area"),                  # NOT the business_role "Measure"
    ("expected_monthly_rent", "expected monthly rent"),
])
def test_labels_are_derived_from_the_name(col, label):
    assert column_label(col, SM["tables"], SM) == label


def test_a_table_name_becomes_its_business_name():
    out = business_wording("Rows come from assets_leaselisting.", [], "assets_asset", SM)
    assert out == "Rows come from lease listings."


def test_a_known_column_not_in_the_select_is_also_worded():
    out = business_wording("Consider carpet_area too.", COLS, "assets_asset", SM)
    assert out == "Consider carpet area too."


def test_quotes_and_backticks_around_an_identifier_go_with_it():
    out = business_wording("The `corner_property` and 'assets_asset_count' columns.",
                           COLS, "assets_asset", SM)
    assert out == "The corner property and number of assets columns."


@pytest.mark.parametrize("text", [
    "EAST has 471 assets.",                               # nothing to replace
    "Payment pay_IMf7qjiSiB6PvV was collected.",          # a value, not schema
    "the facing_direction_x value",                        # unknown identifier
    "",
    None,
])
def test_anything_that_is_not_schema_is_untouched(text):
    assert business_wording(text, COLS, "assets_asset", SM) == text


def test_an_identifier_inside_a_longer_word_is_untouched():
    t = "prefix_corner_property_suffix"
    assert business_wording(t, COLS, "assets_asset", SM) == t


def test_numbers_are_never_changed():
    t = "assets_asset_count ranges from 1,222,441,350.14 to 471 (64%)."
    assert business_wording(t, COLS, "assets_asset", SM) == \
        "number of assets ranges from 1,222,441,350.14 to 471 (64%)."


@pytest.mark.parametrize("table,label", [("assets_asset", "assets"),
                                         ("assets_leaselisting", "lease listings"),
                                         ("users_user", "users"),
                                         ("accounts_generalledger", "financial transactions"),
                                         ("worklists_ticketcategory", "ticketcategories")])
def test_table_labels_come_from_the_models_entity_sentence(table, label):
    from query.business_wording import _table_label
    assert _table_label(table, SM) == label


# ── the front-door hook (veda_hybrid._business_wording) ──────────────────────────────
def test_the_front_door_rewords_only_the_answer(monkeypatch):
    import veda_hybrid
    from query.multi_result import MultiResult, SubResult
    monkeypatch.setattr(veda_hybrid, "_load_semantic_model", lambda: (SM, []))
    payload = {"answer": MEASURED, "cols": list(COLS), "table": "assets_asset",
               "rows": [{"assets_asset_count": 471}], "sql": 'SELECT "corner_property"'}
    doc = {"answer": "See the policy_notes section. Sources: (maintenance_policy.docx)"}
    res = MultiResult(items=[SubResult(sub_query="q", route="sql", status="ok",
                                       result=payload),
                             SubResult(sub_query="d", route="rag", status="ok", result=doc)])
    veda_hybrid._business_wording(res)
    assert "_" not in payload["answer"] and "number of assets" in payload["answer"]
    assert payload["sql"] == 'SELECT "corner_property"'           # untouched
    assert payload["rows"] == [{"assets_asset_count": 471}]        # untouched
    assert doc["answer"].endswith("(maintenance_policy.docx)")     # no schema names in it


def test_the_front_door_never_raises(monkeypatch):
    import veda_hybrid
    monkeypatch.setattr(veda_hybrid, "_load_semantic_model",
                        lambda: (_ for _ in ()).throw(RuntimeError("no sm")))
    veda_hybrid._business_wording(object())                        # nothing to do, no error
