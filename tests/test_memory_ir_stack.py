"""The M4 IR-stack memory: deterministic delta detection, delta application, and the
IR-derived surfaces (Checkpoint C.1-C.3/C.5/C.7).

These are pure-function tests — chatbot/memory/{delta,frame}.py do no I/O, make no LLM
call and import no veda_core, so the whole layer is testable without a stack.

What each group is actually guarding:
  delta rules    — that the closed-class follow-up constructions resolve WITHOUT an SLM
                   call (the turn-budget claim), and that anything else returns
                   `ambiguous` rather than a guess.
  grounding      — that a filter value is only ever one the previous result contained.
                   This is the anti-hallucination property the whole design rests on.
  apply_delta    — that editing one slot leaves every other slot intact; the failure the
                   pre-M4 text-restatement path had was silently dropping an applied
                   filter when the question was reworded.
  ir_partial     — that an unstructured IR is NOT edited slot-wise but falls back.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.memory import delta as D          # noqa: E402
from chatbot.memory import frame as F          # noqa: E402


_DEFAULT = object()


def _entry(ir=_DEFAULT, **over):
    base = {
        "ir": ir if ir is not _DEFAULT else {
            "anchor": "maintenance",
            "measure": {"aggregation": "count", "column": None},
            "filters": [], "group_keys": [], "order": None, "limit": None,
            "ir_partial": False,
        },
        "drill_options": {
            "dimensions": ["category", "status", "city", "property type"],
            "measures": ["amount", "rating"],
            "filters_applicable": ["category", "status", "city"],
        },
        "result": {
            "row_count": 8,
            "top_values": {"category": ["Repair", "Plumbing"],
                           "status": ["OPEN", "CLOSED"],
                           "city": ["Kochi", "Mumbai", "Pune"]},
        },
        "question": "how many maintenance records are there",
        "scope": [4],
    }
    base.update(over)
    return base


# ── delta rules: the closed-class constructions resolve with no SLM ──────────
def test_value_narrowing_is_add_filter():
    d = D.detect("how many of those are repairs", _entry(), [])
    assert d["op"] == D.OP_ADD_FILTER
    assert d["concept"] == "category"
    assert d["value"] == "Repair"          # the stored value, not the user's plural
    assert D.is_confident(d)


def test_plural_message_matches_singular_stored_value():
    """Users type "repairs"; the column holds "Repair". Substring matching missed every
    such pair and sent a perfectly ordinary narrowing to the SLM."""
    assert D.detect("only the repairs", _entry(), [])["value"] == "Repair"


def test_grouping_change_names_a_real_dimension():
    d = D.detect("break that down by status", _entry(), [])
    assert d["op"] == D.OP_CHANGE_GROUP and d["concept"] == "status"
    assert D.is_confident(d)


def test_longest_dimension_wins():
    """"property type" must beat the substring "type" — otherwise the delta groups by the
    wrong column and the answer is confidently about something else."""
    d = D.detect("break that down by property type", _entry(), [])
    assert d["concept"] == "property type"


def test_measure_change():
    d = D.detect("what is the total amount for those", _entry(), [])
    assert d["op"] == D.OP_CHANGE_MEASURE and d["value"] == "sum" and d["concept"] == "amount"


def test_limit_change():
    d = D.detect("show the top 3 by amount", _entry(), [])
    assert d["op"] == D.OP_CHANGE_ORDER and d["value"] == 3


def test_drill_up_phrase():
    assert D.detect("go back", _entry(), [])["op"] == D.OP_DRILL_UP


def test_standalone_sentence_is_new_topic_not_a_measure_change():
    """"how many properties are there" starts with a measure word but is a NEW question
    about a different entity. Reading it as a measure change on the previous frame keeps
    the user silently inside the old filters."""
    assert D.detect("how many properties are there", _entry(), [])["op"] == D.OP_NEW_TOPIC


def test_no_frame_is_new_topic():
    assert D.detect("how many maintenance records", None, [])["op"] == D.OP_NEW_TOPIC


# ── grounding: never invent a value ──────────────────────────────────────────
def test_unknown_value_is_ambiguous_not_a_guessed_filter():
    """"only the Bangalore ones" — Bangalore is NOT in this result's values. The rules
    must decline (→ SLM), never fabricate a filter on an unseen literal."""
    d = D.detect("only the Bangalore ones", _entry(), [])
    assert d["op"] == D.OP_AMBIGUOUS
    assert not D.is_confident(d)


def test_ambiguous_pronoun_without_a_resolvable_slot():
    d = D.detect("what about that one", _entry(), [])
    assert d["op"] == D.OP_AMBIGUOUS


# ── apply_delta: edit one slot, preserve the rest ────────────────────────────
def test_add_filter_preserves_existing_filters_and_grouping():
    ir = {"anchor": "maintenance", "measure": {"aggregation": "count", "column": None},
          "filters": [{"column": "category", "op": "=", "value": "Repair"}],
          "group_keys": ["status"], "order": None, "limit": None, "ir_partial": False}
    nxt = F.apply_delta(ir, {"op": "add_filter", "concept": "city", "value": "Kochi"})
    cols = {f["column"] for f in nxt["filters"]}
    assert cols == {"category", "city"}            # the earlier filter SURVIVES
    assert nxt["group_keys"] == ["status"]         # unrelated slot untouched
    assert ir["filters"] == [{"column": "category", "op": "=", "value": "Repair"}]  # no mutation


def test_change_group_replaces_rather_than_appends():
    """"by city instead" means instead — appending would group by both."""
    ir = {"anchor": "a", "filters": [], "group_keys": ["status"], "ir_partial": False}
    assert F.apply_delta(ir, {"op": "change_group", "concept": "city"})["group_keys"] == ["city"]


def test_repeated_filter_on_same_column_replaces_not_duplicates():
    ir = {"anchor": "a", "filters": [{"column": "city", "op": "=", "value": "Kochi"}],
          "group_keys": [], "ir_partial": False}
    nxt = F.apply_delta(ir, {"op": "add_filter", "concept": "city", "value": "Pune"})
    assert nxt["filters"] == [{"table": "a", "column": "city", "op": "=", "value": "Pune",
                               "grounding": "session_top_values", "concept": "city"}]


def test_unknown_op_returns_none_so_caller_falls_back():
    assert F.apply_delta({"anchor": "a"}, {"op": "switch_frame"}) is None


# ── ir_partial gating ────────────────────────────────────────────────────────
def test_partial_ir_is_not_structured():
    """A reverse-engineered IR (veda/ir.from_sql_facts) has real slots but unknown
    provenance, so it must NOT be edited slot-wise — the caller falls back to the text
    restatement instead."""
    assert not F.entry_is_structured(_entry(ir={"anchor": "a", "ir_partial": True}))
    assert not F.entry_is_structured(_entry(ir=None))
    assert F.entry_is_structured(_entry())


# ── stack mechanics ──────────────────────────────────────────────────────────
def test_stack_compacts_beyond_ten_but_keeps_ir_and_scope():
    stack = []
    for i in range(14):
        stack = F.push_entry(stack, _entry(question=f"q{i}", turn_index=i))
    assert len(stack) == 14                       # nothing is dropped
    old = stack[0]
    assert old.get("compacted") is True
    assert old["ir"] is not None and old["scope"] == [4]     # still reachable
    assert old["result"].get("top_values") is None           # detail shed
    assert stack[-1].get("compacted") is not True            # recent entries stay whole


def test_stack_top_follows_the_cursor():
    st = [_entry(question="a"), _entry(question="b"), _entry(question="c")]
    assert F.stack_top({"stack": st, "cursor": -1})["question"] == "c"
    assert F.stack_top({"stack": st, "cursor": -2})["question"] == "b"
    assert F.stack_top({"stack": [], "cursor": -1}) is None


def test_compact_stack_is_prompt_sized():
    """The classifier prompt gets the stack; each entry must stay small enough that ten
    of them fit the plan's ~250-tokens-per-entry budget."""
    frame = {"stack": [_entry(question="how many maintenance records are there")] * 10}
    import json
    per_entry = len(json.dumps(F.compact_stack(frame))) / 10 / 4      # ~4 chars/token
    assert per_entry < 250


# ── IR-derived surfaces (C.7) ────────────────────────────────────────────────
def test_ir_to_question_is_built_from_slots():
    ir = {"anchor": "maintenance", "measure": {"aggregation": "sum", "column": "amount"},
          "filters": [{"column": "category", "op": "=", "value": "Repair"}],
          "group_keys": ["status"], "limit": 3, "order": {"column": "amount"},
          "ir_partial": False}
    q = F.ir_to_question(ir)
    assert "total amount" in q and "category is Repair" in q and "per status" in q and "top 3" in q


def test_context_strip_names_the_current_slice():
    ir = {"anchor": "vendors", "measure": {"aggregation": "count"},
          "filters": [{"column": "city", "op": "=", "value": "Kochi"}],
          "group_keys": ["category"]}
    strip = F.describe_ir(ir)
    assert "Kochi" in strip and "grouped by category" in strip


def test_follow_ups_exclude_already_grouped_and_filtered_dimensions():
    e = _entry(ir={"anchor": "m", "filters": [{"column": "category", "value": "Repair"}],
                   "group_keys": ["status"], "ir_partial": False})
    fups = F.follow_up_questions(e)
    assert not any("category" in f for f in fups)   # pinned to one value — useless to group by
    assert not any("status" in f for f in fups)     # already grouped


# ── document (RAG) answers ───────────────────────────────────────────────────
# A document turn has no `explain`, no `analytics` and no SQL. The harvest guards were
# written for the SQL path and rejected it outright, which left document sources with no
# session memory at all — every follow-up re-routed as a new topic (session script s3,
# inherited 0/9). These pin the minimum a document turn must still remember.
_DOC_RESULT = {
    "status": "answered",
    "source_id": 3,
    "answer": "The handbook says employees must give 30 days notice.",
    "citations": ["Samta-Employee Handbook April_2026.pdf (p.4)",
                  "maintenance_policy.docx (p.1)"],
}


def test_document_answer_is_harvestable_without_explain():
    h = F.harvest_frame(_DOC_RESULT)
    assert h is not None
    assert h["entity"] == "Samta-Employee Handbook April_2026.pdf"   # page suffix stripped
    assert h["source_ids"] == [3]
    assert h["filters"] == [] and h["last_sql"] is None


def test_document_entry_carries_scope_but_is_not_structured():
    e = F.harvest_entry(_DOC_RESULT, "what does the handbook say about leave", turn_index=1)
    assert e is not None
    assert e["scope"] == [3]
    assert e["result"]["shape"] == "document"
    # no IR → the caller must use the text restatement, not slot editing
    assert not F.entry_is_structured(e)


def test_document_follow_up_grounds_against_the_document():
    """The whole point of harvesting a document turn: the next fragment is resolved
    against the document that just answered, instead of being routed cold."""
    h = F.harvest_frame(_DOC_RESULT)
    frame = {**F.empty_frame("default", "s"), **h}
    resolved = F.render_frame_as_query(frame, "what about the notice period", "refine")
    assert "notice period" in resolved
    assert "Handbook" in resolved


def test_refused_document_answer_is_still_not_harvested():
    assert F.harvest_frame({**_DOC_RESULT, "status": "refuse"}) is None
