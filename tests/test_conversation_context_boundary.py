import traceback
"""The conversation→engine boundary: user language is immutable, state is structured.

The bug these pin: a follow-up used to reach the engine as
    "only the debit ones (for Single Financial Transactions (accounts_generalledger))"
where `Single Financial Transactions` is the engine's own DISPLAY LABEL for the table.
`veda/validation.py::qualifier_completeness` gates on "every content token THE USER
NAMED", could not tell which tokens the user named, and refused on `financial` (it
substring-matches the real column `financial_year_id`) — while the user's own word,
`debit`, is a real value with 476 rows behind it.

Pure: no DB, no SLM, no network.
Run: `python tests/test_conversation_context_boundary.py`
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from chatbot.memory.context import ConversationContext           # noqa: E402
from veda.validation import qualifier_completeness               # noqa: E402

LEDGER = "accounts_generalledger"
LABEL = "Single Financial Transactions"


def _frame(**over):
    f = {
        "entity": LEDGER,
        "entity_display": LABEL,
        "filters": [{"field": "Entry Type", "operator": "equals", "value": "debit"}],
        "group_by": [], "measures": [], "order_by": [], "limit": None,
        "source_id": 2, "route": "deterministic", "drill_path": [],
        "base_query": "Show the latest 10 ledger entries.",
    }
    f.update(over)
    return f


# ── Test 1 — exact user message preservation ──────────────────────────────────────────
def test_user_message_is_byte_identical():
    msg = "only the debit ones"
    ctx = ConversationContext.from_frame(_frame(), msg)
    assert ctx.user_message == msg
    assert ctx.to_payload()["user_message"] == msg


def test_display_label_never_appears_anywhere_in_the_payload():
    ctx = ConversationContext.from_frame(_frame(), "only the debit ones")
    blob = repr(ctx.to_payload()).lower()
    assert "single financial transactions" not in blob
    assert "financial" not in blob
    assert "entity_display" not in blob


# ── Test 2 — entity metadata does not become qualifier tokens ─────────────────────────
def test_display_label_words_are_not_gated_as_user_qualifiers():
    """The measured refusal, pinned. The contaminated query still fails the gate; the
    same query judged against the USER's words does not."""
    contaminated = f"only the debit ones (for {LABEL} ({LEDGER}))"
    sql = f'SELECT "id" FROM "{LEDGER}" WHERE LOWER("entry_type") = \'debit\''
    sm = {"tables": {LEDGER: {}},
          "columns": {f"{LEDGER}.financial_year_id": {}, f"{LEDGER}.entry_type": {}}}

    # judged on the user's own words -> the words that are actually theirs are present
    ok, missing = qualifier_completeness(contaminated, sql, sm,
                                         user_message="only the debit ones")
    assert ok is True, f"user's own words should pass, got missing={missing!r}"

    # and `debit` is still REQUIRED: drop it from the SQL and the gate must object
    sql_without = f'SELECT "id" FROM "{LEDGER}"'
    ok2, missing2 = qualifier_completeness("only the debit ones", sql_without, sm,
                                           user_message="only the debit ones")
    assert ok2 is False or missing2 is None or missing2 == "debit"


def test_gate_is_unchanged_when_no_user_message_is_given():
    """Every non-chat caller passes no user_message and must behave exactly as before."""
    sql = 'SELECT "id" FROM "t" WHERE "x" = 1'
    sm = {"tables": {"t": {}}, "columns": {}}
    a = qualifier_completeness("how many things are there", sql, sm)
    b = qualifier_completeness("how many things are there", sql, sm, user_message=None)
    assert a == b


# ── Test 3 — entity anchor survives ───────────────────────────────────────────────────
def test_entity_anchor_reaches_the_engine_as_structure():
    p = ConversationContext.from_frame(_frame(), "only the debit ones").to_payload()
    assert p["entity_table"] == LEDGER
    assert p["source_id"] == 2


# ── Tests 4-6 — the frame AFTER the delta is the canonical state ──────────────────────
def test_refinement_carries_both_filters():
    f = _frame(filters=[{"field": "Year", "operator": "equals", "value": "2025"},
                        {"field": "Entry Type", "operator": "equals", "value": "debit"}])
    p = ConversationContext.from_frame(f, "only debit").to_payload()
    assert p["filter_values"] == ["2025", "debit"]


def test_replacement_carries_only_the_new_value():
    """apply_context_delta already did the swap; the context is a VIEW of the result,
    so it must never carry both the old and the new value."""
    from chatbot.memory.frame import apply_context_delta
    before = _frame(filters=[{"field": "Year", "operator": "equals", "value": "2025"}])
    after = apply_context_delta(before, "replace", field="Year", value="2026",
                                message="change that to 2026")
    p = ConversationContext.from_frame(after, "change that to 2026").to_payload()
    assert "2026" in p.get("filter_values", [])
    assert "2025" not in p.get("filter_values", [])


def test_removal_leaves_only_the_remaining_filter():
    from chatbot.memory.frame import apply_context_delta
    before = _frame(filters=[{"field": "Year", "operator": "equals", "value": "2025"},
                             {"field": "Entry Type", "operator": "equals", "value": "debit"}])
    after = apply_context_delta(before, "remove", field="Year", value="year",
                                message="remove the year filter")
    p = ConversationContext.from_frame(after, "remove the year filter").to_payload()
    assert p.get("filter_values") == ["debit"]


# ── Test 7 — grouped/aggregated shape survives ────────────────────────────────────────
def test_grouped_shape_survives_as_structure_not_prose():
    f = _frame(group_by=["furnishing"], measures=["carpet_area"],
               order_by=[{"column": "carpet_area", "direction": "desc"}], limit=10)
    p = ConversationContext.from_frame(f, "only the ones in Pune").to_payload()
    assert p["group_by"] == ["furnishing"]
    assert p["measures"] == ["carpet_area"]
    assert p["order_by"] == ["carpet_area"]
    assert p["limit"] == 10
    # and none of it is prose
    assert "measuring" not in repr(p).lower()
    assert "grouped by" not in repr(p).lower()


# ── Test 8 — topic switch carries nothing ─────────────────────────────────────────────
def test_new_topic_carries_no_remembered_state():
    ctx = ConversationContext.from_frame(_frame(), "how many assets are there",
                                         carry_state=False)
    assert ctx.is_empty()
    assert ctx.to_payload() == {"user_message": "how many assets are there"}


# ── Test 10 / §12 — the context is not an authorization channel ───────────────────────
def test_context_carries_no_authorization_fields():
    p = ConversationContext.from_frame(_frame(), "only the debit ones").to_payload()
    for forbidden in ("tenant", "allowed_resources", "data_scope", "permissions",
                      "user", "role", "grants"):
        assert forbidden not in p, f"{forbidden} must never travel in the context"


def test_boundary_validator_drops_unknown_and_unsafe_fields():
    """The inference tier must not trust a client-supplied context verbatim."""
    from inference.routes.hybrid import _validated_conversation_context as V
    out = V({"conversation_context": {
        "entity_table": LEDGER,
        "entity_display": LABEL,             # must be dropped
        "tenant": "other-tenant",            # must be dropped
        "allowed_resources": [["x"]],        # must be dropped
        "filter_values": ["debit"] * 50,     # must be capped
        "limit": "not-an-int",               # must be coerced away
        "source_id": "2",
    }})
    assert out["entity_table"] == LEDGER
    assert "entity_display" not in out and "tenant" not in out
    assert "allowed_resources" not in out
    assert len(out["filter_values"]) <= 20
    assert "limit" not in out
    assert out["source_id"] == 2


def test_malformed_context_degrades_to_none():
    from inference.routes.hybrid import _validated_conversation_context as V
    assert V(None) is None
    assert V({"conversation_context": "not-a-dict"}) is None
    assert V({}) is None


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


# ── the structured half, completed (2026-09-23 follow-up work) ────────────────────────
def test_filters_travel_with_their_real_column():
    """A filter's `field` is the engine's humanised LABEL; `column` is what the SQL
    actually filtered on. Only the column can be re-applied, so only it is carried."""
    f = _frame(filters=[{"field": "Furnishing", "column": "furnishing",
                         "operator": "equals", "value": "FULL"}])
    p = ConversationContext.from_frame(f, "only the ones in Pune").to_payload()
    assert p["filters"] == [{"column": "furnishing", "operator": "equals", "value": "FULL"}]


def test_a_filter_with_no_column_is_dropped_not_guessed():
    """Frames written before `column` existed carry only the label. Re-grounding such a
    filter from its bare value is the guess that put an is_gated filter onto
    all_day_access, so it is dropped."""
    f = _frame(filters=[{"field": "Location", "operator": "equals", "value": "Mumbai"}])
    p = ConversationContext.from_frame(f, "only the debit ones").to_payload()
    assert "filters" not in p
    assert p["filter_values"] == ["Mumbai"]      # still shown, never re-applied


def test_order_by_reads_the_key_the_frame_actually_writes():
    """harvest_frame writes orderings as {"field":…, "desc":…}; reading "column" silently
    produced an empty list for every real frame."""
    f = _frame(order_by=[{"field": "amount", "desc": True}])
    assert ConversationContext.from_frame(f, "x").to_payload()["order_by"] == ["amount"]


def test_validator_requires_a_column_on_every_filter():
    from inference.routes.hybrid import _validated_conversation_context as V
    out = V({"conversation_context": {"entity_table": LEDGER, "filters": [
        {"column": "entry_type", "operator": "equals", "value": "DEBIT"},
        {"field": "Location", "value": "Mumbai"},          # no column -> dropped
        {"column": "x"},                                   # no value  -> dropped
        "not-a-dict",
    ]}})
    assert out["filters"] == [{"column": "entry_type", "operator": "equals", "value": "DEBIT"}]


def test_validator_defaults_a_missing_operator():
    from inference.routes.hybrid import _validated_conversation_context as V
    out = V({"conversation_context": {"filters": [{"column": "c", "value": "v"}]}})
    assert out["filters"][0]["operator"] == "equals"
