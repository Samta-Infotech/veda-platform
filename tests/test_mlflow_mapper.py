"""Model-free tests for the MLflow exporter mapper enrichment
(mlflow_observability/mapper.py) — provenance/attribution tags + outcome +
refusal taxonomy. No MLflow server, no engine, no SLM — pure record→RunSpec mapping.

Run from repo root: ``pytest tests/test_mlflow_mapper.py``
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "mlflow_observability"))


def _answered_record():
    return {
        "status": "answered", "route": "full:SIMPLE", "intent": "SIMPLE",
        "table": "assets_asset", "action": "single_table", "total_ms": 2400,
        "total_tokens": 366, "tenant": "default", "source_id": 1, "cache_hit": False,
        "full": {"query": "how many assets per city", "sections": {
            "output": {"status": "answered", "sql_model": "qwen2.5-coder:7b"},
            "nl_summary": {"summary_model": "qwen2.5:7b-instruct"}}},
    }


def test_provenance_tags_present(monkeypatch):
    monkeypatch.setenv("VEDA_GIT_SHA", "abc123def")
    from mapper import map_record
    spec = map_record(_answered_record(), raw_line="x", environment="prod")
    assert spec.tags["veda.git_sha"] == "abc123def"
    assert spec.tags["veda.model.sql"] == "qwen2.5-coder:7b"
    assert spec.tags["veda.model.summary"] == "qwen2.5:7b-instruct"
    assert spec.tags["veda.tenant"] == "default"
    assert spec.tags["veda.source_id"] == "1"
    assert spec.tags["veda.cache_hit"] == "False"
    assert spec.tags["veda.environment"] == "prod"


def test_answered_outcome_no_refusal_category():
    from mapper import map_record
    spec = map_record(_answered_record(), raw_line="x")
    assert spec.tags["veda.outcome"] == "answered"
    assert "veda.refusal_category" not in spec.tags


def test_git_sha_absent_when_env_unset(monkeypatch):
    for k in ("VEDA_GIT_SHA", "GIT_SHA", "SOURCE_COMMIT"):
        monkeypatch.delenv(k, raising=False)
    from mapper import map_record
    spec = map_record(_answered_record(), raw_line="x")
    assert "veda.git_sha" not in spec.tags   # best-effort — absent, not crashing


def test_refused_outcome_and_category():
    from mapper import map_record
    rec = {"status": "qualifier_dropped", "refusal": "achieve",
           "full": {"query": "q", "sections": {"output": {"status": "qualifier_dropped"}}}}
    spec = map_record(rec, raw_line="y")
    assert spec.tags["veda.outcome"] == "refused"
    assert spec.tags["veda.refusal_category"] == "dropped_qualifier"


def test_clarify_outcome():
    from mapper import map_record
    rec = {"status": "clarify", "full": {"query": "q", "sections": {"output": {"status": "clarify"}}}}
    spec = map_record(rec, raw_line="y")
    assert spec.tags["veda.outcome"] == "clarify"
    assert spec.tags["veda.refusal_category"] == "clarify_ambiguous"


import pytest


@pytest.mark.parametrize("status,refusal,expected", [
    ("clarify", "", "clarify_ambiguous"),
    ("no_table", "", "no_table_matched"),
    ("ungrounded", "", "ungrounded_value"),
    ("qualifier_dropped", "achieve", "dropped_qualifier"),
    ("ir_mismatch", "", "ir_mismatch"),
    ("invalid", "", "invalid_or_exec_error"),
    ("federated_refused", "", "federation_refused"),
    ("something_new", "", "other"),
])
def test_refusal_taxonomy(status, refusal, expected):
    from mapper import _refusal_category
    assert _refusal_category(status, refusal) == expected


def test_missing_provenance_fields_do_not_break():
    # a minimal record (no sections, no tenant/source/model) must still map cleanly
    from mapper import map_record
    spec = map_record({"status": "answered", "full": {"query": "hi"}}, raw_line="z")
    assert spec.tags["veda.outcome"] == "answered"
    assert "veda.model.sql" not in spec.tags and "veda.tenant" not in spec.tags


# ── span waterfall (MLflow Tracing) ─────────────────────────────────────────────
def _sectioned_record():
    # sections stamp `_ms` = elapsed-at-first-touch; durations = gap to next stage.
    return {"status": "answered", "total_ms": 2000,
            "full": {"query": "revenue by city", "sections": {
                "query_understanding": {"_ms": 0},
                "retrieval":           {"_ms": 200},
                "schema_linking":      {"_ms": 900},
                "sql_planning":        {"_ms": 1100},
                "validation":          {"_ms": 1600},
                "output":              {"_ms": 1950}}}}


def test_build_spans_shape_and_ordering():
    from mapper import build_spans
    secs = _sectioned_record()["full"]["sections"]
    spans = build_spans(secs, 2000)
    names = [s["name"] for s in spans]
    assert names == ["query_understanding", "retrieval", "schema_linking",
                     "sql_planning", "validation", "output"]  # sorted by start offset
    # durations = gap to next stage; last runs to total_ms
    d = {s["name"]: s["duration_ms"] for s in spans}
    assert d["query_understanding"] == 200 and d["retrieval"] == 700
    assert d["output"] == 50  # 2000 - 1950
    # semantic span types
    ty = {s["name"]: s["span_type"] for s in spans}
    assert ty["retrieval"] == "RETRIEVER" and ty["sql_planning"] == "LLM"
    assert ty["query_understanding"] == "PARSER"


def test_build_spans_populated_on_map_record():
    from mapper import map_record
    spec = map_record(_sectioned_record(), raw_line="s")
    assert spec.spans and len(spec.spans) == 6
    assert all({"name", "span_type", "start_offset_ms", "duration_ms"} <= set(s) for s in spec.spans)


def test_build_spans_empty_when_no_timing():
    from mapper import build_spans
    assert build_spans({}, None) == []
    assert build_spans({"retrieval": {"n_columns": 5}}, 100) == []  # no _ms → no span
    assert build_spans({"retrieval": {"_ms": 10}}, None) == []      # no total_ms → []
