"""Picking a displayed record by NAME, by the EXTREME of a shown column, or as a SET.

The row is still identified by its key (what goes to the engine); these only decide which
key. Additive: a message none of them recognises resolves exactly as before.

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_result_reference_selectors.py -q`
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.memory import reference as R  # noqa: E402

TOP = {"status": "answered", "table": "assets_salelisting",
       "cols": ["listing_id", "project_name", "expected_price", "carpet_area", "created_at"],
       "rows": [[101, "ABC Residency", 9000, 1200, "2025-01-03T10:00:00"],
                [102, "Green Valley", 7000, 900, "2025-03-01T10:00:00"],
                [103, "One & Only House", 5000, 1500, "2024-11-20T10:00:00"],
                [104, "Sky Tower", 8000, 700, "2025-06-09T10:00:00"]],
       "result_key": {"kind": "rows", "table": "assets_salelisting",
                      "column": "id", "result_column": "listing_id"}}
FRAME = {"entity": "assets_salelisting", "source_id": 2,
         "order_by": [{"field": "expected_price", "desc": True}]}


def _ref(er=TOP):
    return R.build_reference(er, 2)


def _pick(msg, ref=None, frame=FRAME, referential=True):
    hit = R.resolve_reference(ref or _ref(), msg, frame=frame, referential=referential)
    if hit and hit[0] == "filters":
        return [f["value"] for f in hit[1]]
    return hit


# ── what is kept ──────────────────────────────────────────────────────────────────────
def test_the_naming_column_and_orderable_values_are_kept():
    ref = _ref()
    assert ref["label_column"] == "project_name"
    assert ref["labels"][2] == "One & Only House"
    assert set(ref["values"]) == {"expected_price", "carpet_area"}
    assert set(ref["dates"]) == {"created_at"}


# ── by name ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("msg", ["Show details of One & Only House",
                                 "what about one & only house?"])
def test_a_record_named_by_its_label(msg):
    assert _pick(msg) == ["103"]
    assert R.names_a_row(_ref(), msg, FRAME)


def test_a_duplicate_label_asks_instead_of_picking():
    er = dict(TOP, rows=TOP["rows"] + [[105, "Green Valley", 6000, 950, "2025-02-01"]])
    hit = _pick("details of Green Valley", ref=_ref(er))
    assert hit[0] == "refuse" and "which one" in hit[1]


def test_a_name_not_in_the_result_is_not_a_reference():
    assert _pick("details of Palm Grove Villas") is None
    assert not R.names_a_row(_ref(), "details of Palm Grove Villas", FRAME)


def test_a_label_of_another_table_is_not_matched():
    assert not R.names_a_row(_ref(), "details of Green Valley", {"entity": "users_user"})


# ── by the extreme of a shown column ─────────────────────────────────────────────────
@pytest.mark.parametrize("msg,expected", [
    ("which one is the cheapest", ["103"]),          # ranked by price → price
    ("the most expensive one", ["101"]),
    ("which one has the largest carpet area", ["103"]),   # the message names the column
    ("the smallest carpet area one", ["104"]),
    ("the latest one", ["104"]),                      # the one date column
    ("the oldest one", ["103"]),
])
def test_rankings(msg, expected):
    assert _pick(msg) == expected


def test_no_way_to_choose_the_column_is_not_resolved():
    """Two numeric columns, the result not ranked by either, none named."""
    assert _pick("which one is the cheapest", frame={"entity": "assets_salelisting",
                                                      "source_id": 2}) is None


def test_a_tie_asks():
    er = dict(TOP, rows=[r[:2] + [5000] + r[3:] for r in TOP["rows"]])
    hit = _pick("which one is the cheapest", ref=_ref(er))
    assert hit[0] == "refuse"


# ── sets ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("msg,expected", [
    ("show the first three", ["101", "102", "103"]),
    ("the last two", ["103", "104"]),
    ("compare the top 2 ones", ["101", "102"]),
    ("those four", ["101", "102", "103", "104"]),
])
def test_sets(msg, expected):
    assert _pick(msg) == expected


@pytest.mark.parametrize("msg", ["show the first five", "those three"])
def test_a_count_that_was_not_shown_is_refused(msg):
    assert _pick(msg)[0] == "refuse"


def test_a_new_ranking_question_is_not_a_set():
    """"the first 5 payments" names its own subject — the engine's ranking, as before."""
    assert _pick("show the first 5 payments") is None


# ── additive: nothing changes for what worked before ─────────────────────────────────
def test_positions_still_win():
    assert _pick("the second one") == ["102"]


def test_not_a_follow_up_nothing_is_resolved():
    assert _pick("which one is the cheapest", referential=False) is None


def test_groups_are_untouched():
    groups = {"status": "answered", "table": "assets_asset",
              "cols": ["facing", "assets_asset_count"], "rows": [["EAST", 40], ["WEST", 12]],
              "result_key": {"kind": "groups", "table": "assets_asset",
                             "columns": [{"column": "facing", "result_column": "facing"}]}}
    ref = R.build_reference(groups, 2)
    assert R.resolve_reference(ref, "only the EAST ones", frame={"entity": "assets_asset",
                               "source_id": 2}, referential=True) is None


# ── the shown list survives a pick; a pick is not a redraw ───────────────────────────
def test_selects_rows():
    ref = _ref()
    assert R.selects_rows(ref, "show the first three", FRAME)
    assert R.selects_rows(ref, "details of One & Only House", FRAME)
    assert not R.selects_rows(ref, "show that as a table", FRAME)
    assert not R.selects_rows(ref, "show the first three", {"entity": "users_user"})


def test_a_pick_keeps_the_list_as_the_reference(monkeypatch):
    import chatbot.nodes as N
    written = []
    monkeypatch.setattr(N.MemoryStore, "write_reference",
                        staticmethod(lambda *a, **k: written.append(a[2])))
    for name in ("write_frame", "write_stack", "push_episodic_turn", "write_topics",
                 "write_comparison", "write_user_sessions"):
        if hasattr(N.MemoryStore, name):
            monkeypatch.setattr(N.MemoryStore, name, staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(N.MemoryStore, "read_user_sessions", staticmethod(lambda *a, **k: []),
                        raising=False)
    listing = _ref()
    one_row = dict(TOP, rows=TOP["rows"][2:3])
    er = {**one_row, "explain": {"data_used": {"datasets": ["Sale Listings"]},
                                 "filters": {"applied": []}, "operations": []}}
    base = {"status": "answered", "action": "followup", "tenant": "t", "session_id": "s",
            "frame": dict(FRAME, filters=[]), "drill_stack": [], "delta_type": "refine",
            "engine_result": er, "message": "details of the 3rd one",
            "result_reference": listing}
    out = N.memory_write_node({**base, "conversation_context": {"resolved_terms": ["3rd"]}})
    # the LIST is (re)written — never the one-row answer the pick produced
    assert out["result_reference"] is listing and written == [listing]
    out = N.memory_write_node({**base, "conversation_context": {}})   # not a pick
    assert written and written[-1]["items"] == ["103"]


def test_identifier_columns_are_never_ranked():
    er = dict(TOP, cols=TOP["cols"] + ["asset_id"], rows=[r + [7] for r in TOP["rows"]])
    assert "asset_id" not in (_ref(er).get("values") or {})


def test_a_column_that_was_not_shown_is_not_ranked():
    """"highest amount" when no amount column was displayed → not resolved (the engine
    answers, as before) rather than ranked on something else."""
    assert _pick("which one has the highest amount", frame={"entity": "assets_salelisting",
                                                           "source_id": 2}) is None


@pytest.mark.parametrize("msg", ["the 3rd one", "show the first three"])
def test_picking_a_shown_row_is_never_a_clarification_answer(monkeypatch, msg):
    """Measured 2026-09-26: a refused turn armed a clarification and "the 3rd one" was
    glued onto it ("only the ones on the moon for the 3rd one")."""
    import chatbot.nodes as N
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action": "clarify_reply"}'
                        if k.get("purpose") == "classify" else "standalone")
    out = N.classify_node({"message": msg, "history": [{"role": "user", "content": "x"}],
                           "frame": dict(FRAME, filters=[]), "result_reference": _ref(),
                           "pending_clarification": {"original_query": "only the ones on the moon"}},
                          config={})
    assert out["action"] == "followup" and not out.get("pending_clarification")


def test_a_superlative_pointed_at_with_one_is_a_selection():
    ref = _ref()
    assert R.selects_rows(ref, "Which one is the cheapest?", FRAME)
    assert R.selects_rows(ref, "the most expensive one", FRAME)
    assert not R.selects_rows(ref, "Which city is the cheapest?", FRAME)
    assert not R.selects_rows(ref, "which one has the highest rating", FRAME)  # not shown


@pytest.mark.parametrize("msg,expected", [
    ("the lowest expected price one", ["103"]),        # names a shown column
    ("which one has the highest rating", None),         # names one that was not shown
    ("the lowest amount one", None),
])
def test_the_superlative_is_about_what_follows_it(msg, expected):
    assert _pick(msg) == expected


def test_most_passes_its_quality_word_to_the_engine():
    hit = R.resolve_reference(_ref(), "the most expensive one", frame=FRAME, referential=True)
    assert hit[0] == "filters" and "expensive" in hit[2]



# ── earlier results ──────────────────────────────────────────────────────────────────
AREA = dict(TOP, rows=sorted(TOP["rows"], key=lambda r: -r[3]))


def _history():
    by_price = R.build_reference(TOP, 2)
    by_area = R.build_reference(AREA, 2)
    h = R.remember_result([], by_price, "Show the top 4 sale listings by expected price")
    h = R.remember_result(h, by_area, "Show the top 4 sale listings by carpet area")
    return h, by_price, by_area


def test_history_is_bounded_and_most_recent_first():
    h, _p, area = _history()
    assert h[0]["result_id"] == area["result_id"] and len(h) == 2
    for i in range(10):
        h = R.remember_result(h, R.build_reference(TOP, 2), f"q{i}")
    assert len(h) == R._MAX_RESULTS


@pytest.mark.parametrize("msg,which", [
    ("the 1st one from the price list", "price"),
    ("the 1st one from the expected price result", "price"),
    ("the 2nd one in the carpet area list", "area"),
    ("the 1st one from the earlier list", "price"),     # not the current (area) one
    ("the 1st one from the first list", "price"),
])
def test_a_qualified_reference_picks_that_result(msg, which):
    h, price, area = _history()
    kind, entry, terms = R.earlier_result(h, msg, area)
    assert kind == "ref"
    assert entry["result_id"] == (price if which == "price" else area)["result_id"]
    assert "list" in terms or "result" in terms


def test_no_qualifier_means_the_current_result():
    h, _p, area = _history()
    assert R.earlier_result(h, "the 2nd one", area) is None
    assert R.earlier_result(h, "the 2nd one from the list", area) is None


def test_an_unknown_or_ambiguous_qualifier_asks():
    h, _p, area = _history()
    assert R.earlier_result(h, "the 1st one from the rent list", area)[0] == "refuse"
    assert R.earlier_result(h, "the 1st one from the sale listings list", area)[0] == "refuse"
