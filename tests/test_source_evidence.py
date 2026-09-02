"""Tests for query.source_evidence — SourceEvidence contract + group-by-source (routing Phase 2.3).

Pure/duck-typed, no DB. Run: `pytest tests/test_source_evidence.py` (or python tests/test_source_evidence.py).
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.abspath(os.path.join(ROOT, "veda_core"))
sys.path.insert(0, CORE)

from query.source_evidence import (  # noqa: E402
    group_evidence_by_source, TIER_STRONG, TIER_WEAK, TIER_NONE,
)


class _Col:
    def __init__(self, col_id, col_name, table_name, source_id, similarity):
        self.col_id, self.col_name, self.table_name = col_id, col_name, table_name
        self.source_id, self.similarity = source_id, similarity


class _Chunk:
    def __init__(self, chunk_id, doc_name, source_id, similarity):
        self.chunk_id, self.doc_name, self.source_id, self.similarity = (
            chunk_id, doc_name, source_id, similarity)


def test_groups_columns_and_chunks_by_source():
    cols = [_Col("t1.rev", "actual_revenue", "monthly_revenue", "5", 0.82),
            _Col("t2.a", "amount", "invoices", "7", 0.34)]
    chunks = [_Chunk("c1", "FY2026 Plan", "9", 0.71)]
    g = group_evidence_by_source(cols, chunks)
    assert set(g.keys()) == {"5", "7", "9"}
    assert g["5"].column_count == 1 and g["9"].chunk_count == 1


def test_blank_source_id_is_dropped():
    cols = [_Col("t9.x", "x", "misc", "", 0.9)]      # blank source -> unattributable
    g = group_evidence_by_source(cols, [])
    assert g == {}


def test_dicts_are_accepted_too():
    cols = [{"col_id": "t1.c", "col_name": "c", "table_name": "t", "source_id": "3",
             "similarity": 0.6}]
    g = group_evidence_by_source(cols, [])
    assert "3" in g and g["3"].items[0].name == "c"


def test_placeholder_tiers():
    # strong column similarity -> STRONG
    strong = group_evidence_by_source([_Col("a.b", "b", "a", "1", 0.80)], [])
    assert strong["1"].presence_tier == TIER_STRONG
    # low-but-present -> WEAK
    weak = group_evidence_by_source([_Col("a.b", "b", "a", "2", 0.31)], [])
    assert weak["2"].presence_tier == TIER_WEAK
    # a chunk hit tiers on chunk score independently of columns
    chunky = group_evidence_by_source([], [_Chunk("c", "doc", "3", 0.70)])
    assert chunky["3"].presence_tier == TIER_STRONG
    # NONE only when there is genuinely nothing
    assert group_evidence_by_source([], []) == {}


def test_per_kind_floors_are_independent():
    # Same 0.52 score tiers DIFFERENTLY by kind: chunk floor STRONG=0.50 -> STRONG,
    # tabular floor STRONG=0.55 -> WEAK. This is the whole point: never one cross-type cutoff.
    chunk = group_evidence_by_source([], [_Chunk("c", "d", "9", 0.52)])
    col = group_evidence_by_source([_Col("t.c", "c", "t", "5", 0.52)], [])
    assert chunk["9"].presence_tier == TIER_STRONG
    assert col["5"].presence_tier == TIER_WEAK


def test_config_floor_override_is_respected():
    import config
    prev = config.ROUTING_TIER_TABULAR_STRONG
    try:
        config.ROUTING_TIER_TABULAR_STRONG = 0.90
        g = group_evidence_by_source([_Col("t.c", "c", "t", "5", 0.82)], [])
        assert g["5"].presence_tier == TIER_WEAK  # 0.82 < 0.90 floor
    finally:
        config.ROUTING_TIER_TABULAR_STRONG = prev


def test_summary_shape():
    g = group_evidence_by_source(
        [_Col("t.rev", "actual_revenue", "monthly_revenue", "5", 0.8)],
        [_Chunk("c", "Plan.pdf", "5", 0.6)])
    s = g["5"].summary()
    assert s["source_id"] == "5"
    assert "actual_revenue" in s["columns"]
    assert "monthly_revenue" in s["tables"]
    assert "Plan.pdf" in s["documents"]


if __name__ == "__main__":  # allow `python tests/test_source_evidence.py`
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception:
            failed += 1; print("FAIL", fn.__name__); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
