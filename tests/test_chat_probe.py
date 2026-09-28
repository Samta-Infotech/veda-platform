"""Unit tests for scripts/chat_probe.py — the SSE parser and the engine-trace
lookup, both pure/network-free. No django setup needed and no network calls;
run with:

    pytest tests/test_chat_probe.py

`scripts/` has no `__init__.py` (nothing else in the repo imports from it as a
package), so the module is loaded directly from its file path via importlib
rather than via a `scripts.chat_probe` import.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "chat_probe.py")

_spec = importlib.util.spec_from_file_location("chat_probe", _SCRIPT_PATH)
chat_probe = importlib.util.module_from_spec(_spec)
sys.modules["chat_probe"] = chat_probe
_spec.loader.exec_module(chat_probe)


# --------------------------------------------------------------------------- #
# parse_sse_lines
# --------------------------------------------------------------------------- #

def test_parse_sse_lines_full_turn_sequence():
    """The typical sequence from CHAT_API_CONTRACT.md §1b, byte for byte."""
    raw = (
        'event: thinking\n'
        'data: {"phase": "visualization_prep", "message": "Preparing your chart..."}\n'
        '\n'
        'event: content\n'
        'data: {"type": "text", "content": "The 5 most recent entries..."}\n'
        '\n'
        'event: explainability\n'
        'data: {"version": "1.0", "confidence": 0.87, "sql": {"enabled": true, "query": "SELECT 1"}}\n'
        '\n'
        'event: usage\n'
        'data: {"prompt_tokens": 1240, "completion_tokens": 312, "total_tokens": 1552, "latency_ms": 2680}\n'
        '\n'
        'event: completed\n'
        'data: {"chat_id": 42, "message_id": 501, "summary": "done", "is_complete": true}\n'
        '\n'
    ).splitlines(keepends=True)

    frames = list(chat_probe.parse_sse_lines(raw))
    events = [e for e, _ in frames]
    assert events == ["thinking", "content", "explainability", "usage", "completed"]

    data_by_event = dict(frames)
    assert data_by_event["thinking"]["phase"] == "visualization_prep"
    assert data_by_event["content"]["content"] == "The 5 most recent entries..."
    assert data_by_event["explainability"]["sql"]["query"] == "SELECT 1"
    assert data_by_event["usage"]["latency_ms"] == 2680
    assert data_by_event["completed"]["chat_id"] == 42
    assert data_by_event["completed"]["is_complete"] is True


def test_parse_sse_lines_error_event_terminates_stream():
    raw = (
        'event: error\n'
        'data: {"code": "LLM_UNAVAILABLE", "message": "The AI assistant is temporarily unavailable."}\n'
        '\n'
    ).splitlines(keepends=True)
    frames = list(chat_probe.parse_sse_lines(raw))
    assert len(frames) == 1
    event, data = frames[0]
    assert event == "error"
    assert data["code"] == "LLM_UNAVAILABLE"


def test_parse_sse_lines_multiple_content_blocks_preserve_order():
    raw = (
        'event: content\ndata: {"content": "first"}\n\n'
        'event: content\ndata: {"content": "second"}\n\n'
        'event: visualization\ndata: {"type": "line"}\n\n'
    ).splitlines(keepends=True)
    frames = list(chat_probe.parse_sse_lines(raw))
    assert [e for e, _ in frames] == ["content", "content", "visualization"]
    assert [d["content"] for _, d in frames[:2]] == ["first", "second"]


def test_parse_sse_lines_accepts_bytes_input():
    raw = [b'event: usage\n', b'data: {"latency_ms": 5}\n', b'\n']
    frames = list(chat_probe.parse_sse_lines(raw))
    assert frames == [("usage", {"latency_ms": 5})]


def test_parse_sse_lines_multiline_data_is_joined_before_json_decode():
    # SSE allows a payload split across several `data:` lines; the JSON only
    # becomes valid once they're joined.
    raw = ['event: content\n', 'data: {"content":\n', 'data: "joined"}\n', '\n']
    frames = list(chat_probe.parse_sse_lines(raw))
    assert frames == [("content", {"content": "joined"})]


def test_parse_sse_lines_malformed_json_becomes_raw_not_an_exception():
    raw = ['event: content\n', 'data: not json at all\n', '\n']
    frames = list(chat_probe.parse_sse_lines(raw))
    assert frames == [("content", {"_raw": "not json at all"})]


def test_parse_sse_lines_ignores_comment_lines():
    raw = [': keepalive\n', 'event: usage\n', 'data: {"a": 1}\n', '\n']
    frames = list(chat_probe.parse_sse_lines(raw))
    assert frames == [("usage", {"a": 1})]


def test_parse_sse_lines_defaults_to_message_event_when_unnamed():
    raw = ['data: {"x": 1}\n', '\n']
    frames = list(chat_probe.parse_sse_lines(raw))
    assert frames == [("message", {"x": 1})]


def test_parse_sse_lines_no_trailing_blank_line_still_flushes_last_frame():
    # A stream that ends right after the last data line, with no closing blank
    # line, must not silently drop that frame.
    raw = ['event: completed\n', 'data: {"chat_id": 1}\n']
    frames = list(chat_probe.parse_sse_lines(raw))
    assert frames == [("completed", {"chat_id": 1})]


# --------------------------------------------------------------------------- #
# find_trace
# --------------------------------------------------------------------------- #

def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_find_trace_matches_by_exact_trace_id(tmp_path):
    p = tmp_path / "explain_trace.jsonl"
    _write_jsonl(p, [
        {"trace_id": "aaaa1111", "classify_lane": "router"},
        {"trace_id": "bbbb2222", "classify_lane": "continuity"},
    ])
    rec = chat_probe.find_trace(str(p), "bbbb2222")
    assert rec is not None
    assert rec["classify_lane"] == "continuity"


def test_find_trace_substring_of_another_id_does_not_false_match(tmp_path):
    # "aaaa1" is a substring of "aaaa11" but must not match — equality, not substring.
    p = tmp_path / "explain_trace.jsonl"
    _write_jsonl(p, [{"trace_id": "aaaa11", "classify_lane": "router"}])
    assert chat_probe.find_trace(str(p), "aaaa1") is None


def test_find_trace_returns_none_for_unknown_trace_id(tmp_path):
    p = tmp_path / "explain_trace.jsonl"
    _write_jsonl(p, [{"trace_id": "aaaa1111"}])
    assert chat_probe.find_trace(str(p), "does-not-exist") is None


def test_find_trace_returns_none_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    assert chat_probe.find_trace(str(missing), "anything") is None


def test_find_trace_returns_none_for_empty_trace_id():
    assert chat_probe.find_trace("/dev/null", "") is None
    assert chat_probe.find_trace("/dev/null", None) is None


def test_find_trace_prefers_last_match_on_duplicate_trace_id(tmp_path):
    p = tmp_path / "explain_trace.jsonl"
    _write_jsonl(p, [
        {"trace_id": "dup", "total_ms": 100},
        {"trace_id": "dup", "total_ms": 200},
    ])
    rec = chat_probe.find_trace(str(p), "dup")
    assert rec["total_ms"] == 200


def test_find_trace_skips_malformed_lines(tmp_path):
    p = tmp_path / "explain_trace.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("not json at all, but contains dup\n")
        fh.write(json.dumps({"trace_id": "dup", "total_ms": 5}) + "\n")
    rec = chat_probe.find_trace(str(p), "dup")
    assert rec["total_ms"] == 5


def test_find_trace_absent_continuity_op_is_just_a_missing_key(tmp_path):
    """continuity_op may not exist in a given trace record yet (added by a
    separate, concurrent change) — the lookup itself must not treat that as
    a reason to skip the record."""
    p = tmp_path / "explain_trace.jsonl"
    _write_jsonl(p, [{"trace_id": "t1", "classify_lane": "router"}])
    rec = chat_probe.find_trace(str(p), "t1")
    assert rec is not None
    assert rec.get("continuity_op") is None


# --------------------------------------------------------------------------- #
# small helpers used by the CLI
# --------------------------------------------------------------------------- #

def test_parse_sources_comma_separated():
    assert chat_probe.parse_sources("2,3,4,5") == [2, 3, 4, 5]


def test_parse_sources_none_when_absent():
    assert chat_probe.parse_sources(None) is None
    assert chat_probe.parse_sources("") is None


def test_parse_sources_rejects_non_int():
    with pytest.raises(SystemExit):
        chat_probe.parse_sources("2,x,4")


def test_turn_result_answer_text_prefers_completed_summary():
    r = chat_probe.TurnResult()
    r.content = [{"type": "text", "content": "raw block text"}]
    r.completed = {"summary": "the real summary"}
    assert r.answer_text == "the real summary"


def test_turn_result_answer_text_falls_back_to_content_blocks():
    r = chat_probe.TurnResult()
    r.content = [{"type": "text", "content": "first"}, {"type": "text", "content": "second"}]
    r.completed = None
    assert r.answer_text == "first\nsecond"


def test_turn_result_sql_absent_when_no_explainability():
    r = chat_probe.TurnResult()
    assert r.sql is None


def test_turn_result_sql_from_explainability():
    r = chat_probe.TurnResult()
    r.explainability = {"sql": {"enabled": True, "query": "SELECT 1"}}
    assert r.sql == "SELECT 1"
