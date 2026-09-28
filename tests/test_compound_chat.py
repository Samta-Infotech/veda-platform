"""Chat tier for compound messages (front-door decomposition, 2026-09-25).

A compound answer is ONE turn: the IR stack gets one entry per ANSWERED part, each with
its own source_id, and a follow-up naming a part ("the vendor one — top 3") resolves to
that part's entry. Pure: MemoryStore writes are stubbed; no engine, no LLM.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import chatbot.nodes as nodes                   # noqa: E402
from chatbot.memory import delta as D           # noqa: E402
from chatbot.memory import frame as F           # noqa: E402


def _sql_part(table, question, sid, dims, answer, cols, rows):
    return {
        "status": "ok", "route": "deterministic", "part": question, "outcome": "answered",
        "lane": "tabular" if sid in (4, 5) else "sql", "source_id": sid,
        "result": {
            "ok": True, "status": "answered", "answer": answer, "table": table,
            "cols": cols, "rows": rows, "sql": f"SELECT * FROM {table}", "source_id": sid,
            "ir": {"anchor": table, "measure": None, "filters": [], "group_keys": [],
                   "order": None, "limit": None, "ir_partial": False, "head": "frame.list"},
            "explain": {"data_used": {"datasets": [table.title()]}, "filters": {"applied": []},
                        "operations": [], "understanding": {"summary": question},
                        "sources": [{"id": str(sid)}]},
            "analytics": {"row_count": len(rows), "result_shape": "list",
                          "available_dimensions": dims, "available_measures": ["rating"],
                          "column_stats": [{"name": d, "role": "dimension",
                                            "top_values": ["Kochi", "Pune"]} for d in dims]},
        },
    }


def _payload():
    ledger = _sql_part("accounts_generalledger", "list the latest payments for our properties", 2,
                       ["entry_type"], "Here are the latest 20 payments.",
                       ["id", "amount", "entry_type"], [[1, 100, "DEBIT"], [2, 50, "CREDIT"]])
    doc = {"status": "ok", "route": "rag", "part": "what are the office timings", "outcome": "answered",
           "lane": "rag", "source_id": 3,
           "result": {"answer": "Monday to Friday, 9am to 6pm.", "citations": ["Handbook.pdf (p.30)"],
                      "chunks": [], "source_id": 3}}
    vendors = _sql_part("vendors", "which vendor has the highest rating", 4, ["city"],
                        "V-4 has the highest rating, 4.8.", ["vendor_id", "city", "rating"],
                        [["V-4", "Kochi", 4.8], ["V-1", "Pune", 4.5]])
    return {"result": {"compound": True, "relation": "independent",
                       "summary": "**1. List the latest payments?** ...\n\nAll 3 parts answered.",
                       "items": [ledger, doc, vendors]}}


def _write(monkeypatch):
    for name in ("write_frame", "write_stack", "push_episodic_turn", "write_comparison"):
        monkeypatch.setattr(nodes.MemoryStore, name, staticmethod(lambda *a, **k: None))
    res0, status = nodes._extract_engine_result(_payload())
    assert status == "answered" and res0["is_compound"] and len(res0["compound_parts"]) == 3
    state = {"status": "answered", "engine_result": res0, "tenant": "default",
             "session_id": "s1", "message": "one message, three questions", "frame": {},
             "drill_stack": [], "source_id": 2}
    return nodes.memory_write_node(state)


def test_three_part_result_pushes_three_stack_entries(monkeypatch):
    out = _write(monkeypatch)
    st = out["frame"]["stack"]
    assert len(st) == 3
    assert [e["source_id"] for e in st] == [2, 3, 4]
    assert [e["part_index"] for e in st] == [0, 1, 2]
    assert len({e["compound_turn"] for e in st}) == 1
    # the flat frame follows the LAST answered part
    assert out["frame"]["entity"] and out["frame"]["source_id"] == 4


def test_vendor_one_resolves_to_the_tabular_entry(monkeypatch):
    frame = _write(monkeypatch)["frame"]
    stack = frame["stack"]
    d = D.detect("the vendor one — top 3", F.stack_top(frame), stack)
    assert d["target_frame_index"] == 2 and stack[2]["lane"] == "tabular"
    assert d["op"] == D.OP_CHANGE_ORDER and d["value"] == 3
    d = D.detect("the second one", F.stack_top(frame), stack)
    assert d["op"] == D.OP_SWITCH_FRAME and d["target_frame_index"] == 1
    # the conversation moves onto the part: its source becomes the thread's source
    fr0 = {**frame, "cursor": -1}
    moved = F.switch_to_entry(fr0, 0)
    assert moved["cursor"] == 0 and moved["source_id"] == 2 and moved["source_ids"] == [2]


def test_format_reply_lists_the_parts_with_their_tables(monkeypatch):
    out = _write(monkeypatch)
    res0, _ = nodes._extract_engine_result(_payload())
    reply = nodes.format_reply_node({"engine_result": res0, "frame": out["frame"],
                                     "message": "m", "history": []})
    assert [p["index"] for p in reply["parts"]] == [0, 1, 2]
    assert reply["parts"][2]["rows"][0][0] == "V-4"
    assert reply["context_strip"].count("|") == 2
