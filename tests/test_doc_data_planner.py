import traceback
"""Tests for the bounded DOCUMENT_FACT + DATA_GROUNDING planner (query/doc_data_planner.py).

Pure — the SLM is injected, no DB/model. Run: `python tests/test_doc_data_planner.py`.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
from query import doc_data_planner as DD  # noqa: E402

CHUNKS = ["Maintenance Policy",
          "Scope: this policy governs Football court and Basket Ball court upkeep in Muddanahalli."]
DCOLS = [{"source_id": "2", "table": "assets_amenity", "col": "name"},
         {"source_id": "2", "table": "assets_amenity", "col": "id"}]


def _slm(obj):
    return lambda system, user: json.dumps(obj)


def test_intersect_token_subset_match():
    # "Football court" (doc) ↔ "Football" (data); canonical DATA value returned
    out = DD.intersect(["Football court", "Basket Ball court"],
                       ["Football", "Basket Ball", "Swimming Pool"])
    assert out == ["Football", "Basket Ball"]


def test_intersect_no_false_positive():
    # disjoint words → no match
    assert DD.intersect(["Swimming Pool"], ["Football", "Cricket"]) == []


def test_intersect_dedupe_and_canonical():
    out = DD.intersect(["Football court", "football"], ["Football"])
    assert out == ["Football"]                      # deduped, DATA form


def test_classify_grounded_entities_and_column():
    res = DD.classify("which amenities in the policy exist in our assets", CHUNKS, DCOLS,
                      slm_call=_slm({"entities": ["Football court", "Basket Ball court"],
                                     "data_column": 1}))
    assert res is not None
    ents, col = res
    assert ents == ["Football court", "Basket Ball court"]     # both appear in chunks
    assert col["table"] == "assets_amenity" and col["col"] == "name"


def test_classify_drops_hallucinated_entity():
    # "Cricket" is NOT in the chunks → must be dropped (anti-hallucination)
    res = DD.classify("q", CHUNKS, DCOLS,
                      slm_call=_slm({"entities": ["Football court", "Cricket"], "data_column": 1}))
    assert res is not None
    ents, _ = res
    assert "Football court" in ents and "Cricket" not in ents


def test_classify_all_hallucinated_defers():
    res = DD.classify("q", CHUNKS, DCOLS,
                      slm_call=_slm({"entities": ["Tennis", "Golf"], "data_column": 1}))
    assert res is None                              # none grounded → defer


def test_classify_empty_entities_defers():
    res = DD.classify("q", CHUNKS, DCOLS, slm_call=_slm({"entities": [], "data_column": 1}))
    assert res is None


def test_classify_out_of_range_column_defers():
    res = DD.classify("q", CHUNKS, DCOLS,
                      slm_call=_slm({"entities": ["Football court"], "data_column": 99}))
    assert res is None                              # column-set escape rejected


def test_classify_bad_json_defers():
    res = DD.classify("q", CHUNKS, DCOLS, slm_call=lambda s, u: "not json")
    assert res is None


def test_classify_no_chunks_or_cols_defers():
    assert DD.classify("q", [], DCOLS, slm_call=_slm({"entities": ["X"], "data_column": 1})) is None
    assert DD.classify("q", CHUNKS, [], slm_call=_slm({"entities": ["X"], "data_column": 1})) is None


def test_flag_off_by_default():
    _config.DOC_DATA_GROUNDING_ENABLED = False
    assert DD.doc_data_enabled() is False


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
