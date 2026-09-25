"""Document conversations remember the RIGHT document, and can be returned to.

Measured 2026-09-25 (evaluation/demo, verdicts.json):
  D2 — the answer said "Sources: (msa_green_tower.pdf)" but the frame filed the employee
       handbook (the document of the most similar passage), so the follow-up searched the
       handbook and answered "no information about locations".
  X1 — maintenance policy -> one database question -> "what is the fee for repair in the
       maintenance policy?" was answered by the federated route from a structured fee
       table: documents were never indexed as topics, and the engine had no document-lane
       continuity to mirror its SQL-lane one.

Run: `PYTHONPATH=.:veda_core python -m pytest tests/test_document_memory.py -q`
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import chatbot.nodes as N  # noqa: E402
from chatbot.memory import frame as F  # noqa: E402
from chatbot.memory import topics as T  # noqa: E402

DATASETS = ["Samta-Employee Handbook April 2026", "msa green tower"]


def _rag(answer, datasets=DATASETS):
    return {"status": "answered", "route": "rag", "answer": answer,
            "explain": {"data_used": {"datasets": list(datasets)}}}


# ── D2: the cited document, not the first retrieved ──────────────────────────────────
def test_the_frame_files_the_document_the_answer_cites():
    h = F.harvest_frame(_rag("Asset 20 and Asset 21. Sources: (msa_green_tower.pdf)"))
    assert h["entity"] == "msa green tower" and h["entity_display"] == "msa green tower"
    assert h["datasets"] == ["msa green tower", "Samta-Employee Handbook April 2026"]


def test_citation_order_is_kept_when_several_are_cited():
    ans = "x. Sources: (Samta-Employee Handbook April_2026.pdf, page 3), (msa_green_tower.pdf)"
    assert F.harvest_frame(_rag(ans, list(reversed(DATASETS))))["entity"] == DATASETS[0]


@pytest.mark.parametrize("answer", [
    "Asset 20 and Asset 21.",                              # no Sources line
    "Asset 20. Sources: (some_other_file.pdf)",            # cites nothing we retrieved
    None,
])
def test_without_a_usable_citation_nothing_changes(answer):
    assert F.harvest_frame(_rag(answer))["entity"] == DATASETS[0]


def test_a_name_in_the_body_is_not_a_citation():
    ans = "Unlike the msa green tower terms, the handbook says 30 days. Sources: " \
          "(Samta-Employee Handbook April_2026.pdf)"
    assert F.harvest_frame(_rag(ans, list(reversed(DATASETS))))["entity"] == DATASETS[0]


def test_a_table_answer_is_untouched():
    er = {**_rag("Sources: (msa_green_tower.pdf)"), "table": "assets_asset"}
    assert F.harvest_frame(er)["entity"] == "assets_asset"


# ── X1: documents are topics, returned to by naming them ─────────────────────────────
DOC_FRAME = {"entity": "maintenance policy", "entity_display": "maintenance policy",
             "entity_is_document": True, "route": "rag", "source_id": 2,
             "base_query": "Which assets are covered under the maintenance arrangement?"}
TABLE_FRAME = {"entity": "assets_leaselisting", "source_id": 2, "route": "deterministic",
               "base_query": "What is the distribution of lease listings by furnishing?",
               "filters": []}


def _index():
    return T.upsert(T.upsert([], T.snapshot(DOC_FRAME, [])), T.snapshot(TABLE_FRAME, []))


def test_a_document_topic_round_trips_as_a_document():
    cand = T.frame_from_entry(T.snapshot(DOC_FRAME, []))
    assert F.is_document_frame(cand) and cand["datasets"] == ["maintenance policy"]


@pytest.mark.parametrize("msg", ["what is the fee for repair in the maintenance policy?",
                                 "What does the Maintenance Policy say about response times"])
def test_naming_a_remembered_document_returns_to_it(msg):
    hit = N._match_named_document(msg, _index(), TABLE_FRAME)
    assert hit["kind"] == "document" and hit["topic"]["entity"] == "maintenance policy"


@pytest.mark.parametrize("msg", [
    "how many maintenance records are repairs?",          # not every word of the name
    "what is the fee for repair?",
])
def test_not_naming_it_is_not_a_return(msg):
    assert N._match_named_document(msg, _index(), TABLE_FRAME) is None


def test_already_in_that_document_is_left_to_the_normal_path():
    assert N._match_named_document("what does the maintenance policy say about fees",
                                   _index(), {**DOC_FRAME, "version": 3}) is None


def test_two_named_documents_are_not_guessed():
    other = {**DOC_FRAME, "entity": "maintenance policy annex", "entity_display": None}
    idx = T.upsert(_index(), T.snapshot(other, []))
    assert N._match_named_document("the maintenance policy annex fee", idx, TABLE_FRAME) is None


def test_a_table_topic_is_never_matched_by_this_rule():
    assert N._match_named_document("lease listings by furnishing",
                                   [T.snapshot(TABLE_FRAME, [])], {}) is None


def test_classify_routes_it_as_a_followup_into_the_document(monkeypatch):
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: '{"action":"answer","delta_type":"new_topic"}')
    out = N.classify_node({"message": "what is the fee for repair in the maintenance policy?",
                           "history": [], "frame": TABLE_FRAME, "topic_index": _index()},
                          config={})
    assert out["action"] == "followup"
    assert out["topic_restore"]["kind"] == "document"


def test_context_resolve_sends_the_question_on_the_document_lane(monkeypatch):
    monkeypatch.setattr(N.MemoryStore, "read_frame", lambda *a, **k: {"version": 7})
    restore = {"kind": "document", "topic": T.snapshot(DOC_FRAME, [])}
    msg = "what is the fee for repair in the maintenance policy?"
    out = N._return_to_topic({"message": msg, "frame": TABLE_FRAME, "source_ids": [2]},
                             restore)
    ctx = out["conversation_context"]
    assert out["resolved_query"] == msg                      # names the document already
    assert ctx["entity_table"] == "maintenance policy" and ctx["route"] == "rag"
    assert F.is_document_frame(out["frame"]) and out["drill_stack"] == []


def test_a_pure_return_re_asks_the_last_question_anchored(monkeypatch):
    monkeypatch.setattr(N.MemoryStore, "read_frame", lambda *a, **k: {})
    restore = {"kind": "restore", "topic": T.snapshot(DOC_FRAME, [])}
    out = N._return_to_topic({"message": "go back to the maintenance policy",
                              "frame": TABLE_FRAME, "source_ids": [2]}, restore)
    assert out["resolved_query"] == ("Which assets are covered under the maintenance "
                                     "arrangement? (in maintenance policy)")


def test_a_revoked_document_is_refused_not_guessed(monkeypatch):
    monkeypatch.setattr(N.MemoryStore, "read_frame", lambda *a, **k: {})
    restore = {"kind": "document", "topic": T.snapshot(DOC_FRAME, [])}
    out = N._return_to_topic({"message": "fee in the maintenance policy?",
                              "frame": TABLE_FRAME, "source_ids": [5]}, restore)
    assert out.get("status") == "refuse" or not out.get("conversation_context")


# ── "go back to <the document we're already in>" ─────────────────────────────────────
@pytest.mark.parametrize("msg", ["go back to the maintenance policy",
                                 "let's return to the maintenance policy"])
def test_returning_to_the_current_document_is_answered_from_memory(monkeypatch, msg):
    monkeypatch.setattr(N, "call_slm", lambda *a, **k: pytest.fail("no model call"))
    out = N.classify_node({"message": msg, "history": [], "frame": DOC_FRAME,
                           "topic_index": _index()}, config={})
    assert out["action"] == "recall" and out["recall_kind"] == "already_on_document"


@pytest.mark.parametrize("msg", ["what does the maintenance policy say again about fees",
                                 "go back to the lease listings",
                                 "what is the fee in the maintenance policy?"])
def test_a_question_or_another_topic_is_not_that(msg):
    assert N._returns_to_current_document(msg, DOC_FRAME) is False


def test_its_reply_names_the_document():
    out = N.recall_node({"message": "go back to the maintenance policy", "frame": DOC_FRAME,
                         "recall_kind": "already_on_document", "history": []})
    assert "already on maintenance policy" in out["reply_text"]
